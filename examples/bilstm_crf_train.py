#!/usr/bin/env python3
import argparse
import os
import sys

# Ensure package root is importable for `examples.*` imports when running as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import keras
from keras import layers, ops as K

from keras_crf import CRF
from keras_crf.train_utils import make_crf_tagger, prepare_crf_targets
from examples.utils.data import make_varlen_dataset, read_conll, build_maps, encode_and_pad
from examples.utils.metrics import MaskedTokenAccuracy
from examples.utils.ner_metrics import EntityF1


def build_models(vocab_size, num_tags, embedding_dim=64, lstm_units=64, loss="nll", dice_smooth=1.0, joint_nll_weight=None):
    # Variable-length sequences; rely on compute_output_shape in CRF
    tokens_in = keras.Input(shape=(None,), dtype="int32", name="tokens")

    x = layers.Embedding(input_dim=vocab_size + 1, output_dim=embedding_dim, mask_zero=True)(tokens_in)
    x = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True))(x)

    model = make_crf_tagger(tokens_in, x, num_tags, metrics=[MaskedTokenAccuracy()], loss=loss, dice_smooth=dice_smooth, joint_nll_weight=joint_nll_weight)

    # Inference model: decoded tags
    decoded_model = keras.Model(tokens_in, model.get_layer('decoded_output').output)

    return model, decoded_model


def parse_args():
    p = argparse.ArgumentParser(description="Train BiLSTM-CRF for sequence tagging (Keras Core backend-agnostic)")
    p.add_argument("--dataset", choices=["synthetic", "conll"], default="synthetic")
    p.add_argument("--train", type=str, help="Path to CoNLL train file (for conll)")
    p.add_argument("--val", type=str, help="Path to CoNLL validation file (for conll)")
    p.add_argument("--test", type=str, help="Path to CoNLL test file (for conll)")
    p.add_argument("--token-col", type=int, default=0)
    p.add_argument("--tag-col", type=int, default=-1)
    p.add_argument("--lowercase", action="store_true")
    p.add_argument("--synthetic-max-len", type=int, default=50)
    p.add_argument("--synthetic-vocab", type=int, default=300)
    p.add_argument("--synthetic-tags", type=int, default=4)
    p.add_argument("--synthetic-samples", type=int, default=3000)
    p.add_argument("--embedding-dim", type=int, default=64)
    p.add_argument("--lstm-units", type=int, default=64)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--scheme", choices=["BIO","BILOU"], default="BIO")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--loss", choices=["nll", "dice", "dice+nll"], default="nll", help="Training loss: NLL, Dice, or joint Dice+NLL")
    p.add_argument("--dice-smooth", type=float, default=1.0, help="Smoothing constant for Dice loss")
    p.add_argument("--joint-nll-weight", type=float, default=None, help="If using dice+nll, weight for NLL term (default 0.2)")
    # Note: run_eagerly is backend-dependent; safe default False
    return p.parse_args()


def main():
    args = parse_args()

    if args.dataset == "synthetic":
        X_train, Y_train, _ = make_varlen_dataset(args.synthetic_samples, args.synthetic_max_len,
                                                 args.synthetic_vocab, args.synthetic_tags, seed=1)
        X_val, Y_val, _ = make_varlen_dataset(max(args.synthetic_samples // 5, 1), args.synthetic_max_len,
                                             args.synthetic_vocab, args.synthetic_tags, seed=2)
        X_test, Y_test, _ = make_varlen_dataset(max(args.synthetic_samples // 5, 1), args.synthetic_max_len,
                                               args.synthetic_vocab, args.synthetic_tags, seed=3)
        num_tags = args.synthetic_tags
        vocab_size = args.synthetic_vocab
        id2tag = None
    else:
        if not (args.train and args.val and args.test):
            raise SystemExit("--train/--val/--test must be provided for conll dataset")
        train_s, train_t = read_conll(args.train, args.token_col, args.tag_col, args.lowercase)
        val_s, val_t = read_conll(args.val, args.token_col, args.tag_col, args.lowercase)
        test_s, test_t = read_conll(args.test, args.token_col, args.tag_col, args.lowercase)
        tok2id, tag2id = build_maps(train_s, train_t)
        X_train, Y_train = encode_and_pad(train_s, train_t, tok2id, tag2id)
        X_val, Y_val = encode_and_pad(val_s, val_t, tok2id, tag2id, max_len=X_train.shape[1])
        X_test, Y_test = encode_and_pad(test_s, test_t, tok2id, tag2id, max_len=X_train.shape[1])
        num_tags = len(tag2id)
        vocab_size = len(tok2id) - 1
        id2tag = [None] * num_tags
        for t, i in tag2id.items():
            id2tag[i] = t

    model, model_pred = build_models(vocab_size=vocab_size, num_tags=num_tags,
                                   embedding_dim=args.embedding_dim, lstm_units=args.lstm_units,
                                   loss=args.loss, dice_smooth=args.dice_smooth, joint_nll_weight=args.joint_nll_weight)

    # Train using helper targets
    y_train_dict, sw_train_dict = prepare_crf_targets(Y_train, mask=(X_train != 0).astype(np.float32))
    y_val_dict, sw_val_dict = prepare_crf_targets(Y_val, mask=(X_val != 0).astype(np.float32))

    _ = model.fit({"tokens": X_train, "labels": Y_train},
                  y_train_dict,
                  sample_weight=sw_train_dict,
                  validation_data=({"tokens": X_val, "labels": Y_val}, y_val_dict, sw_val_dict),
                  epochs=args.epochs, batch_size=args.batch_size, verbose=2)

    # Evaluate masked token accuracy and per-entity metrics using inference model
    decoded = model_pred.predict(X_test, batch_size=args.batch_size, verbose=0)
    mask = (X_test != 0)
    acc = (decoded[mask] == Y_test[mask]).mean()
    print(f"Masked token accuracy on test: {acc:.4f}")

    if 'id2tag' in locals() and id2tag is not None:
        # Compute EntityF1 via metric class
        ent_f1 = EntityF1(id2tag, args.scheme)
        ent_f1.update_state(Y_test, decoded, sample_weight=mask.astype(np.float32))
        print(f"Entity F1 (micro): {float(ent_f1.result()):.4f}")


if __name__ == "__main__":
    main()
