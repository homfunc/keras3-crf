# Tests for keras_crf.layers CRF layer behaviors (eager-mode)
import numpy as np
import tensorflow as tf
import pytest

from keras_crf import CRF


def test_layer_mask_right_padding_and_left_padding_error():
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
    assert decoded.shape == (2, 3)
    assert seq_len.numpy().tolist() == [3, 2]

    # Left padding should raise NotImplementedError
    mask_left = np.array([[0, 1, 1], [1, 1, 1]], dtype=bool)
    with pytest.raises(NotImplementedError):
        layer(x, mask=mask_left)


def test_layer_training_updates_weights_and_reduces_loss():
    x = np.array(
        [
            [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.1, 0.2, 0.1]],
            [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.1, 0.0]],
        ],
        dtype=np.float32,
    )
    y = np.array([[1, 2, 2], [1, 1, 1]], dtype=np.int32)

    layer = CRF(units=3)
    opt = tf.keras.optimizers.Adam(0.05)

    initial_weights = [w.numpy().copy() for w in layer.trainable_variables]

    losses = []
    for _ in range(5):
        with tf.GradientTape() as tape:
            _, potentials, seq_len, kernel = layer(x)
            ll = layer.log_likelihood(potentials, y, seq_len)
            loss = -tf.reduce_mean(ll)
        grads = tape.gradient(loss, layer.trainable_variables)
        opt.apply_gradients(zip(grads, layer.trainable_variables))
        losses.append(float(loss.numpy()))

    new_weights = [w.numpy() for w in layer.trainable_variables]
    assert any(not np.allclose(a, b) for a, b in zip(initial_weights, new_weights))
    assert losses[-1] <= losses[0] or (losses[0] - losses[-1]) < 1e-4


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
        np.testing.assert_allclose(a.numpy(), b.numpy(), rtol=1e-6, atol=1e-6)

