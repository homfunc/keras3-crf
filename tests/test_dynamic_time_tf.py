import numpy as np
import keras
import pytest
import keras_crf

from keras_crf.crf_ops import crf_log_likelihood


@pytest.mark.parametrize("T", [1, 2, 5])
def test_crf_log_likelihood_dynamic_time_scan_tf_safe(T):
    # This test builds dynamic-time inputs and runs through crf_log_likelihood
    # to ensure no backend assertion errors in scan for small T, including T=1.
    B, N = 3, 6
    # Create placeholders with dynamic T by using None in Input and then slicing
    x_in = keras.Input(shape=(None, N), dtype="float32")  # [B, None, N]
    tags_in = keras.Input(shape=(None,), dtype="int32")   # [B, None]
    lens_in = keras.Input(shape=(), dtype="int32")        # [B]

    # Build symbolic graph using ops functions
    def ll_layer(args):
        x_sym, y_sym, l_sym, trans = args
        return crf_log_likelihood(x_sym, y_sym, l_sym, trans)

    # Fixed transitions for determinism
    trans = keras.Variable(np.random.uniform(-0.1, 0.1, size=(N, N)).astype("float32"))

    out = keras.layers.Lambda(ll_layer, output_shape=(None,))([x_in, tags_in, lens_in, trans])
    model = keras.Model([x_in, tags_in, lens_in], out)

    # Eager inputs with fixed T controlling length
    rng = np.random.default_rng(0)
    potentials = rng.normal(size=(B, T, N)).astype("float32")
    tags = rng.integers(low=0, high=N, size=(B, T), dtype=np.int32)
    lens = np.full((B,), T, dtype=np.int32)

    # Run forward pass (shouldn't crash)
    ll_val = model.predict([potentials, tags, lens], verbose=0)
    assert ll_val.shape == (B,), f"Expected per-sample log-likelihood, got {ll_val.shape}"


def test_crf_layer_end_to_end_T1_dynamic_ok():
    # Tiny end-to-end: CRF layer used in a model where T is dynamically sized and equal to 1.
    B, N = 2, 4
    inputs = keras.Input(shape=(None, N), dtype="float32")
    crf = keras_crf.layers.CRF(units=N, use_boundary=False, use_kernel=False)
    decoded, potentials, lens, trans = crf(inputs)
    model = keras.Model(inputs, [decoded, potentials, lens, trans])

    x = np.random.randn(B, 1, N).astype("float32")
    y = model.predict(x, verbose=0)
    # decoded: [B,1], lens: [B]
    assert y[0].shape == (B, 1)
    assert y[2].shape == (B,)
