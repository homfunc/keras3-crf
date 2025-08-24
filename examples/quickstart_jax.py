import os
import sys

# Ensure parent directory is importable as package root for `examples.*` imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Ensure jax backend before importing keras
os.environ.setdefault("KERAS_BACKEND", "jax")

import numpy as np
import keras
from keras import layers, ops as K

from keras_crf import CRF
from keras_crf.crf_ops import crf_log_likelihood
from examples.utils.metrics import MaskedTokenAccuracy


def main():
    # Ensure jax backend
    os.environ.setdefault("KERAS_BACKEND", "jax")

    # Tiny synthetic data
    B, T, V, N = 32, 16, 80, 5
    rng = np.random.default_rng(1)
    X = rng.integers(1, V + 1, size=(B, T), dtype=np.int32)
    Y = rng.integers(0, N, size=(B, T), dtype=np.int32)
    mask = (X != 0).astype("float32")

    tokens = keras.Input(shape=(T,), dtype="int32", name="tokens")
    labels = keras.Input(shape=(T,), dtype="int32", name="labels")
    x = layers.Embedding(V + 1, 32, mask_zero=True)(tokens)
    x = layers.Bidirectional(layers.LSTM(32, return_sequences=True))(x)
    crf = CRF(N)
    decoded, potentials, lens, trans = crf(x)

    # Add CRF negative log-likelihood as a model loss
    ll = crf_log_likelihood(potentials, labels, lens, trans)

    model = keras.Model({"tokens": tokens, "labels": labels}, decoded)
    model.add_loss(-K.mean(ll))

    def zero_loss(y_true, y_pred):
        return K.mean(K.zeros_like(y_pred[..., :1]))

    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=zero_loss, metrics=[MaskedTokenAccuracy()])

    model.fit({"tokens": X, "labels": Y}, y=Y, epochs=1, batch_size=16, sample_weight=mask, verbose=2)
    print("Done (jax).")


if __name__ == "__main__":
    main()

