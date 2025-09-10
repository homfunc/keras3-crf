# Tests for keras_crf.layers_core CRF layer behaviors (backend-agnostic)
import numpy as np
import pytest
import keras
from keras import ops as K

from keras_crf import CRF


def test_layer_mask_right_and_left_padding_supported():
    # Two sequences, second with right padding at last timestep
    x = np.array(
        [
            [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
            [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        ],
        dtype=np.float32,
    )
    y = np.array([[1, 2, 2], [1, 1, 1]], dtype=np.int32)
    mask_right = np.array([[1, 1, 1], [1, 1, 0]], dtype=bool)

    layer = CRF(units=3)

    decoded, potentials, seq_len, kernel = layer(x, mask=mask_right)
    assert K.shape(decoded) == (2, 3)
    assert list(K.convert_to_numpy(seq_len)) == [3, 2]

    # Left padding should be supported as well
    mask_left = np.array([[0, 1, 1], [1, 1, 1]], dtype=bool)
    decoded2, potentials2, seq_len2, kernel2 = layer(x, mask=mask_left)
    assert K.shape(decoded2) == (2, 3)
    assert list(K.convert_to_numpy(seq_len2)) == [2, 3]


# Note: Training/gradients are backend-specific; we cover training in integration tests elsewhere.
# Here we focus on shape and config roundtrips for backend-agnostic behavior.


def test_layer_config_and_weights_roundtrip():
    x = np.random.randn(2, 3, 4).astype(np.float32)
    layer = CRF(units=4)

    out1 = layer(x)
    cfg = layer.get_config()
    new_layer = CRF.from_config(cfg)
    # Build the new layer to create Dense weights as needed
    _ = new_layer(x)
    new_layer.set_weights(layer.get_weights())
    out2 = new_layer(x)

    for a, b in zip(out1, out2):
        np.testing.assert_allclose(K.convert_to_numpy(a), K.convert_to_numpy(b), rtol=1e-6, atol=1e-6)

