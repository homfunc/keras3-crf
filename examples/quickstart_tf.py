import os
import sys

# Ensure examples package imports work when run directly
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Ensure TF backend before importing keras
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import numpy as np
import keras
from keras import layers

from keras_crf.train_utils import make_crf_tagger, prepare_crf_targets


def main():
    B, T, V, N = 16, 12, 80, 5
    rng = np.random.default_rng(42)
    X = rng.integers(1, V + 1, size=(B, T), dtype=np.int32)
    Y = rng.integers(0, N, size=(B, T), dtype=np.int32)

    tokens = keras.Input(shape=(T,), dtype="int32", name="tokens")
    x = layers.Embedding(V + 1, 32, mask_zero=True)(tokens)
    x = layers.Bidirectional(layers.LSTM(32, return_sequences=True))(x)

    model = make_crf_tagger(tokens, x, N)

    y_dict, sw_dict = prepare_crf_targets(Y, mask=(X != 0).astype(np.float32))

    model.fit(
        {"tokens": X, "labels": Y},
        y_dict,
        epochs=1,
        batch_size=B,
        sample_weight=sw_dict,
        verbose=2,
    )

    print("Done (tf).")


if __name__ == "__main__":
    main()

