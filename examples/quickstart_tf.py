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

    nll = CRF_NLL(name="nll_out")([potentials, labels, lens, trans])

    decoded_named = keras.layers.Lambda(lambda z: z, name="decoded_output")(decoded)
    nll_named = keras.layers.Lambda(lambda z: z, name="crf_log_likelihood_output")(nll)

    model = keras.Model(
        inputs={"tokens": tokens, "labels": labels},
        outputs={"decoded_output": decoded_named, "crf_log_likelihood_output": nll_named},
    )

    def zero_loss(y_true, y_pred):
        return K.mean(K.zeros_like(y_pred[..., :1]))

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss={"decoded_output": zero_loss, "crf_log_likelihood_output": lambda y_true, y_pred: K.mean(y_pred)},
        metrics={"decoded_output": []},
    )

    y_dummy = np.zeros((B,), dtype=np.float32)
    sw_dummy = np.ones((B,), dtype=np.float32)

    model.fit(
        {"tokens": X, "labels": Y},
        {"decoded_output": Y, "crf_log_likelihood_output": y_dummy},
        epochs=1,
        batch_size=B,
        sample_weight={"decoded_output": np.ones_like(Y, dtype=np.float32), "crf_log_likelihood_output": sw_dummy},
        verbose=2,
    )

    print("Done (tf).")


if __name__ == "__main__":
    main()

