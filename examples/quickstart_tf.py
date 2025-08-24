import os
import sys

# Ensure examples package imports work when run directly
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Ensure TF backend before importing keras
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import numpy as np
import keras
from keras import layers, ops as K

from keras_crf import CRF
from keras_crf.crf_ops import crf_log_likelihood


def main():
    B, T, V, N = 16, 12, 80, 5
    rng = np.random.default_rng(42)
    X = rng.integers(1, V + 1, size=(B, T), dtype=np.int32)
    Y = rng.integers(0, N, size=(B, T), dtype=np.int32)

    tokens = keras.Input(shape=(T,), dtype="int32", name="tokens")
    labels = keras.Input(shape=(T,), dtype="int32", name="labels")
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
            pot_shape = input_shapes[0]
            B = pot_shape[0]
            return (B,)

    nll = CRF_NLL()([potentials, labels, lens, trans])

    model = keras.Model(inputs={"tokens": tokens, "labels": labels}, outputs=nll)

    def identity_loss(y_true, y_pred):
        return K.mean(y_pred)

    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=identity_loss)
    model.fit({"tokens": X, "labels": Y}, np.zeros((B,), dtype=np.float32), epochs=1, batch_size=B, verbose=2)

    print("Done (tf).")


if __name__ == "__main__":
    main()

