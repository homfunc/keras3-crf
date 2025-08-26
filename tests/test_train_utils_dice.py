import numpy as np
import keras
from keras import ops as K

from keras_crf.train_utils import make_crf_tagger, prepare_crf_targets


def _tiny_data(B=8, T=6, V=20, C=5, seed=0):
    rng = np.random.default_rng(seed)
    lens = rng.integers(low=T//2, high=T, size=B, dtype=np.int32)
    X = np.zeros((B, T), dtype=np.int32)
    Y = np.zeros((B, T), dtype=np.int32)
    for i in range(B):
        L = lens[i]
        seq = rng.integers(1, V, size=L, dtype=np.int32)
        X[i, :L] = seq
        Y[i, :L] = (seq % C).astype(np.int32)
    return X, Y, lens


def _build_encoder(tokens, V=20, D=16, H=16):
    x = keras.layers.Embedding(V + 1, D, mask_zero=True)(tokens)
    x = keras.layers.Bidirectional(keras.layers.LSTM(H, return_sequences=True))(x)
    return x


def test_make_crf_tagger_with_dice_builds_and_trains_one_step():
    V, C = 30, 6
    tokens = keras.Input(shape=(None,), dtype="int32", name="tokens")
    features = _build_encoder(tokens, V=V)
    model = make_crf_tagger(tokens, features, num_tags=C, loss="dice")
    X, Y, lens = _tiny_data(B=8, T=8, V=V, C=C)
    y_dict, sw = prepare_crf_targets(Y, mask=(X != 0).astype(np.float32))
    # One small training step to ensure it runs
    model.fit({"tokens": X, "labels": Y}, y_dict, sample_weight=sw, epochs=1, batch_size=4, verbose=0)


def test_make_crf_tagger_with_joint_builds_and_trains_one_step():
    V, C = 30, 6
    tokens = keras.Input(shape=(None,), dtype="int32", name="tokens")
    features = _build_encoder(tokens, V=V)
    model = make_crf_tagger(tokens, features, num_tags=C, loss="dice+nll", joint_nll_weight=0.3)
    X, Y, lens = _tiny_data(B=8, T=8, V=V, C=C, seed=1)
    y_dict, sw = prepare_crf_targets(Y, mask=(X != 0).astype(np.float32))
    model.fit({"tokens": X, "labels": Y}, y_dict, sample_weight=sw, epochs=1, batch_size=4, verbose=0)

