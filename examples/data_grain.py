"""
Example: training with Grain data loader feeding a Keras Core model.
Requires: grain installed (pip install grain). Keras backend can be torch/tensorflow/jax.
"""
import os
import sys
from typing import Tuple

import numpy as np

# Optional: auto-select a backend if not set
if "KERAS_BACKEND" not in os.environ:
    for mod in ("jax", "tensorflow", "torch"):
        try:
            __import__(mod)
            os.environ["KERAS_BACKEND"] = mod
            break
        except Exception:
            pass

import keras
from keras import layers, ops as K

try:
    import grain.python as grain
except Exception as e:
    print("Grain is not installed: pip install grain; skipping example.")
    raise SystemExit(0)

from keras_crf import CRF
from keras_crf.crf_ops import crf_log_likelihood


def build_models(V: int, N: int):
    tokens = keras.Input(shape=(None,), dtype="int32")
    x = layers.Embedding(V + 1, 32, mask_zero=True)(tokens)
    x = layers.Bidirectional(layers.LSTM(32, return_sequences=True))(x)
    crf = CRF(N)
    decoded, potentials, lens, trans = crf(x)

    class CRF_NLL(keras.layers.Layer):
        def call(self, inputs):
            pot, y_true, ln, tr = inputs
            ll = crf_log_likelihood(pot, y_true, ln, tr)
            return -ll

        def compute_output_shape(self, input_shapes):
            B = input_shapes[0][0]
            return (B,)

    labels = keras.Input(shape=(None,), dtype="int32")
    nll = CRF_NLL()([potentials, labels, lens, trans])
    model_loss = keras.Model(inputs={"tokens": tokens, "labels": labels}, outputs=nll)
    model_pred = keras.Model(tokens, decoded)
    return model_loss, model_pred


def main():
    B, T, V, N = 32, 20, 200, 5
    rng = np.random.default_rng(0)
    X = rng.integers(1, V + 1, size=(512, T), dtype=np.int32)
    Y = rng.integers(0, N, size=(512, T), dtype=np.int32)

    def iterator():
        for i in range(0, len(X), B):
            yield {"tokens": X[i:i+B], "labels": Y[i:i+B]}, np.zeros((min(B, len(X)-i),), dtype=np.float32)

    dataset = grain.MapDataset(iterator)
    loader = grain.Loader(dataset)

    model_loss, model_pred = build_models(V=V, N=N)

    def identity_loss(y_true, y_pred):
        return K.mean(y_pred)

    model_loss.compile(optimizer=keras.optimizers.Adam(1e-3), loss=identity_loss)

    model_loss.fit(loader, steps_per_epoch=max(1, len(X)//B), epochs=1, verbose=2)

    preds = model_pred.predict(X[:B], verbose=0)
    mask = (X[:B] != 0)
    acc = (preds[mask] == Y[:B][mask]).mean()
    print(f"Masked token accuracy (Grain): {acc:.4f}")


if __name__ == "__main__":
    main()

