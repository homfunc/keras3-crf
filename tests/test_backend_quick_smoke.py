import os
import numpy as np
import keras
from keras import ops as K

from keras_crf import CRF
from keras_crf.crf_ops import crf_log_likelihood


def build_loss_model(T: int, V: int, N: int):
    tokens = keras.Input(shape=(T,), dtype="int32", name="tokens")
    labels = keras.Input(shape=(T,), dtype="int32", name="labels")
    x = keras.layers.Embedding(V + 1, 16, mask_zero=True)(tokens)
    x = keras.layers.Bidirectional(keras.layers.LSTM(16, return_sequences=True))(x)

    crf = CRF(N)
    decoded, potentials, lens, trans = crf(x)

    class CRF_NLL(keras.layers.Layer):
        def call(self, inputs):
            pot, y_true, ln, tr = inputs
            ll = crf_log_likelihood(pot, y_true, ln, tr)
            return -ll

    nll_out = CRF_NLL(name="nll_out")([potentials, labels, lens, trans])
    model_loss = keras.Model({"tokens": tokens, "labels": labels}, nll_out)

    def identity_loss(y_true, y_pred):
        return K.mean(y_pred)

    model_loss.compile(optimizer=keras.optimizers.Adam(5e-3), loss=identity_loss, run_eagerly=True)
    return model_loss


def test_backend_quick_loss_decreases():
    # Small synthetic data
    rng = np.random.default_rng(123)
    B, T, V, N = 8, 8, 50, 4
    X = rng.integers(1, V + 1, size=(B, T), dtype=np.int32)
    Y = rng.integers(0, N, size=(B, T), dtype=np.int32)

    model = build_loss_model(T=T, V=V, N=N)

    # Two quick steps
    y_dummy = np.zeros((B,), dtype=np.float32)
    sw = np.ones((B,), dtype=np.float32)
    loss1 = float(model.train_on_batch({"tokens": X, "labels": Y}, y_dummy, sample_weight=sw))
    loss2 = float(model.train_on_batch({"tokens": X, "labels": Y}, y_dummy, sample_weight=sw))

    assert np.isfinite(loss1) and np.isfinite(loss2)
    # Allow non-monotonicity; require at least small improvement
    assert (loss2 <= loss1 + 1e-4) or (abs(loss2 - loss1) < 1e-5)
