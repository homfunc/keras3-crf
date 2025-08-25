"""
Example: training with PyTorch DataLoader feeding a Keras Core model.
Requires: torch installed. Keras backend can be torch, tensorflow, or jax.
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
from torch.utils.data import Dataset, DataLoader  # type: ignore

from keras_crf import CRF
from keras_crf.crf_ops import crf_log_likelihood


class ToyTaggingDataset(Dataset):
    def __init__(self, n: int = 512, T: int = 20, V: int = 200, N: int = 5, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.X = rng.integers(1, V + 1, size=(n, T), dtype=np.int32)
        self.Y = rng.integers(0, N, size=(n, T), dtype=np.int32)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        return self.X[idx], self.Y[idx]


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
    ds = ToyTaggingDataset(n=256, T=T, V=V, N=N)
    dl = DataLoader(ds, batch_size=B, shuffle=True)

    model_loss, model_pred = build_models(V=V, N=N)

    def identity_loss(y_true, y_pred):
        return K.mean(y_pred)

    model_loss.compile(optimizer=keras.optimizers.Adam(1e-3), loss=identity_loss)

    # Wrap DataLoader into a Python generator yielding np arrays
    def gen():
        for xb, yb in dl:
            yield {"tokens": np.asarray(xb), "labels": np.asarray(yb)}, np.zeros((len(xb),), dtype=np.float32)

    model_loss.fit(gen(), steps_per_epoch=max(1, len(ds) // B), epochs=1, verbose=2)

    # Predict with the inference model
    xb, yb = next(iter(dl))
    preds = model_pred.predict(np.asarray(xb), verbose=0)
    mask = (np.asarray(xb) != 0)
    acc = (preds[mask] == np.asarray(yb)[mask]).mean()
    print(f"Masked token accuracy (DataLoader): {acc:.4f}")


if __name__ == "__main__":
    main()

