import os
import sys

# Ensure parent directory is importable as package root for `examples.*` imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Ensure torch backend before importing keras
os.environ.setdefault("KERAS_BACKEND", "torch")

import numpy as np
import keras
from keras import layers, ops as K

from keras_crf import CRF
from keras_crf.crf_ops import crf_log_likelihood
from examples.utils.metrics import MaskedTokenAccuracy


def main():
    # Tiny synthetic data
    B, T, V, N = 32, 20, 100, 4
    rng = np.random.default_rng(0)
    X = rng.integers(1, V + 1, size=(B, T), dtype=np.int32)
    Y = rng.integers(0, N, size=(B, T), dtype=np.int32)
    mask = (X != 0).astype("float32")

    tokens = keras.Input(shape=(T,), dtype="int32", name="tokens")
    labels = keras.Input(shape=(T,), dtype="int32", name="labels")

    x = layers.Embedding(V + 1, 32, mask_zero=True)(tokens)
    x = layers.Bidirectional(layers.LSTM(32, return_sequences=True))(x)
    crf = CRF(N)
    decoded, potentials, lens, trans = crf(x)

    class CRF_NLL(keras.layers.Layer):
        def call(self, inputs):
            pot, y_true, ln, tr = inputs
            ll = crf_log_likelihood(pot, y_true, ln, tr)  # [B]
            return -ll

        def compute_output_shape(self, input_shapes):
            pot_shape = input_shapes[0]
            B = pot_shape[0]
            return (B,)

    nll_out = CRF_NLL(name="nll_out")([potentials, labels, lens, trans])

    # Loss-only training graph (robust across backends)
    model_loss = keras.Model(inputs={"tokens": tokens, "labels": labels}, outputs=nll_out)

    def identity_loss(y_true, y_pred):
        return K.mean(y_pred)

    model_loss.compile(optimizer=keras.optimizers.Adam(1e-3), loss=identity_loss, run_eagerly=True)

    # One short epoch optimizing CRF NLL
    _ = model_loss.fit(
        {"tokens": X, "labels": Y},
        np.zeros((B,), dtype=np.float32),
        sample_weight=np.ones((B,), dtype=np.float32),
        epochs=1,
        batch_size=B,
        verbose=2,
    )

    # Separate inference model for decoded tags (for reporting only)
    model_pred = keras.Model(tokens, decoded)
    preds = model_pred.predict(X, batch_size=B, verbose=0)
    acc = (preds[mask.astype(bool)] == Y[mask.astype(bool)]).mean()
    print(f"Done (torch). Masked token accuracy (reported post-train): {acc:.4f}")


if __name__ == "__main__":
    main()

