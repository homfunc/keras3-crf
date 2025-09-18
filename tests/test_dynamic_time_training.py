import numpy as np
import pytest
import keras

from keras_crf.layers import CRF
from keras_crf.losses import CRFNLLHead

def test_graph_scan_dynamic_time_training():
    # Build a small CRF model that returns per-sample NLL and trains over a generator
    B = 2
    F = 6  # feature dim
    Ntags = 5

    x_in = keras.Input(shape=(None, F), dtype="float32")  # [B,T,F]
    y_in = keras.Input(shape=(None,), dtype="int32")      # [B,T]

    # CRF projects internally if use_kernel=True
    crf = CRF(units=Ntags, use_boundary=False, use_kernel=True)
    decoded, potentials, lengths, trans = crf(x_in)

    # Per-sample NLL vector
    nll_vec = CRFNLLHead(name="crf_nll")([potentials, y_in, lengths, trans])

    model = keras.Model([x_in, y_in], nll_vec)
    # Loss: reduce the precomputed per-sample vector
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=lambda yt, yp: keras.ops.mean(yp))

    # Build a Python generator (backend-agnostic) yielding varying time lengths, including T=1
    def gen_np():
        rng = np.random.default_rng(123)
        while True:
            for T in [1, 3, 2, 5]:
                x = rng.normal(size=(B, T, F)).astype("float32")
                y = rng.integers(low=0, high=Ntags, size=(B, T), dtype=np.int32)
                # dummy y_true for loss signature (ignored)
                ytrue = np.zeros((B,), dtype="float32")
                yield (x, y), ytrue

    history = model.fit(gen_np(), epochs=1, steps_per_epoch=4, verbose=0)
    assert history is not None
