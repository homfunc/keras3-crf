import numpy as np
import keras
from keras import layers
from typing import Tuple, Dict, List, Optional

from keras_crf.train_utils import make_crf_tagger, prepare_crf_targets
from examples.utils.data import read_conll, build_maps, encode_and_pad, make_varlen_dataset
from examples.utils.data_multiconer import read_multiconer_en_splits
from examples.utils.metrics import MaskedTokenAccuracy
from examples.utils.ner_metrics import EntityF1


def build_bilstm_crf_models(vocab_size: int,
                            num_tags: int,
                            embedding_dim: int = 64,
                            lstm_units: int = 64,
                            loss: str = "nll",
                            dice_smooth: float = 1.0,
                            joint_nll_weight: Optional[float] = None):
    tokens_in = keras.Input(shape=(None,), dtype="int32", name="tokens")
    x = layers.Embedding(vocab_size + 1, embedding_dim, mask_zero=True)(tokens_in)
    x = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True))(x)
    model = make_crf_tagger(tokens_in, x, num_tags,
                            metrics=[MaskedTokenAccuracy()],
                            loss=loss,
                            dice_smooth=dice_smooth,
                            joint_nll_weight=joint_nll_weight)
    decoded_model = keras.Model(tokens_in, model.get_layer('decoded_output').output)
    return model, decoded_model


def load_conll(train_path: str, val_path: str, test_path: str,
               token_col: int = 0, tag_col: int = -1, lowercase: bool = False):
    train_s, train_t = read_conll(train_path, token_col, tag_col, lowercase)
    val_s, val_t = read_conll(val_path, token_col, tag_col, lowercase)
    test_s, test_t = read_conll(test_path, token_col, tag_col, lowercase)
    tok2id, tag2id = build_maps(train_s, train_t)
    X_train, Y_train = encode_and_pad(train_s, train_t, tok2id, tag2id)
    X_val, Y_val = encode_and_pad(val_s, val_t, tok2id, tag2id, max_len=X_train.shape[1])
    X_test, Y_test = encode_and_pad(test_s, test_t, tok2id, tag2id, max_len=X_train.shape[1])
    id2tag = [None] * len(tag2id)
    for t, i in tag2id.items():
        id2tag[i] = t
    vocab_size = len(tok2id) - 1
    num_tags = len(tag2id)
    return (X_train, Y_train, X_val, Y_val, X_test, Y_test, vocab_size, num_tags, id2tag)


def load_multiconer_en(dir_path: str, token_col: int = 0, tag_col: int = -1):
    train_s, train_t, val_s, val_t, test_s, test_t = read_multiconer_en_splits(dir_path, token_col, tag_col)
    tok2id, tag2id = build_maps(train_s, train_t)
    X_train, Y_train = encode_and_pad(train_s, train_t, tok2id, tag2id)
    X_val, Y_val = encode_and_pad(val_s, val_t, tok2id, tag2id, max_len=X_train.shape[1])
    X_test, Y_test = encode_and_pad(test_s, test_t, tok2id, tag2id, max_len=X_train.shape[1])
    id2tag = [None] * len(tag2id)
    for t, i in tag2id.items():
        id2tag[i] = t
    vocab_size = len(tok2id) - 1
    num_tags = len(tag2id)
    return (X_train, Y_train, X_val, Y_val, X_test, Y_test, vocab_size, num_tags, id2tag)


def load_synthetic(samples: int = 3000, max_len: int = 50, vocab: int = 300, tags: int = 4):
    X_train, Y_train, _ = make_varlen_dataset(samples, max_len, vocab, tags, seed=1)
    X_val, Y_val, _ = make_varlen_dataset(max(samples // 5, 1), max_len, vocab, tags, seed=2)
    X_test, Y_test, _ = make_varlen_dataset(max(samples // 5, 1), max_len, vocab, tags, seed=3)
    id2tag = [str(i) for i in range(tags)]
    return (X_train, Y_train, X_val, Y_val, X_test, Y_test, vocab, tags, id2tag)


def train_and_evaluate(model, decoded_model,
                       X_train, Y_train, X_val, Y_val, X_test, Y_test,
                       epochs: int = 5, batch_size: int = 64,
                       id2tag: Optional[List[str]] = None, scheme: str = "BIO") -> Dict[str, float]:
    y_train_dict, sw_train_dict = prepare_crf_targets(Y_train, mask=(X_train != 0).astype(np.float32))
    y_val_dict, sw_val_dict = prepare_crf_targets(Y_val, mask=(X_val != 0).astype(np.float32))
    _ = model.fit({"tokens": X_train, "labels": Y_train},
                  y_train_dict,
                  sample_weight=sw_train_dict,
                  validation_data=({"tokens": X_val, "labels": Y_val}, y_val_dict, sw_val_dict),
                  epochs=epochs, batch_size=batch_size, verbose=2)
    decoded = decoded_model.predict(X_test, batch_size=batch_size, verbose=0)
    mask = (X_test != 0)
    acc = float((decoded[mask] == Y_test[mask]).mean())
    results = {"token_acc": acc}
    if id2tag is not None:
        ent_f1 = EntityF1(id2tag, scheme)
        ent_f1.update_state(Y_test, decoded, sample_weight=mask.astype(np.float32))
        results["entity_f1"] = float(ent_f1.result())
    return results

