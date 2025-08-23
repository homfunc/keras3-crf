# Eager-mode smoke tests for the standalone keras_crf package
import numpy as np
import tensorflow as tf

from keras_crf import CRF


def test_smoke_decode_and_loss():
    # 2 examples, seq_len=3, num_tags=4
    x = np.array(
        [
            [[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.1, 0.2, 0.1, 0.0]],
            [[0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 1.1, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    y = np.array([[1, 2, 2], [1, 1, 1]], dtype=np.int32)

    layer = CRF(units=4)

    decoded, potentials, seq_len, kernel = layer(x)
    assert decoded.shape == (2, 3)
    assert potentials.shape == (2, 3, 4)

    ll = layer.log_likelihood(potentials, y, seq_len)
    loss = -tf.reduce_mean(ll)
    assert tf.math.is_finite(loss)


def test_single_timestep_decode_and_loss():
    # Explicitly test max_seq_len == 1 to hit the single-sequence codepath
    # Use use_kernel=False and use_boundary=False to make potentials == inputs
    # so we can assert exact values for decode and scores
    x = np.array(
        [
            [[0.2, 1.0, -0.5]],  # argmax -> 1, best score 1.0
            [[-0.1, 0.0, 0.3]],  # argmax -> 2, best score 0.3
        ],
        dtype=np.float32,
    )  # shape (batch=2, time=1, num_tags=3)
    y = np.array([[1], [2]], dtype=np.int32)

    layer = CRF(units=3, use_kernel=False, use_boundary=False)

    decoded, potentials, seq_len, kernel = layer(x)

    # Shapes
    assert decoded.shape == (2, 1)
    assert potentials.shape == (2, 1, 3)
    assert seq_len.shape == (2,)

    # Decode should be simple argmax at t=0 for each batch element
    expected_decoded = np.array([[1], [2]], dtype=np.int32)
    np.testing.assert_array_equal(decoded.numpy(), expected_decoded)

    # best_score in single-step path should be max over last dim at t=0
    # Use layer decode path again, best score derived from t=0 unary in single-step
    dec2, _, _, _ = layer(x)
    best_score = tf.reduce_max(potentials[:, 0, :], axis=-1)
    np.testing.assert_array_equal(dec2.numpy(), expected_decoded)
    expected_best = np.array([1.0, 0.3], dtype=np.float32)
    np.testing.assert_allclose(best_score.numpy(), expected_best, rtol=1e-6, atol=1e-6)

    # Log-likelihood equals unary(tag) - logsumexp(unary) for each sample, since transitions unused
    ll = layer.log_likelihood(potentials, y, seq_len)
    # Compute expected manually
    def lse(v):
        m = np.max(v)
        return m + np.log(np.sum(np.exp(v - m)))

    expected_ll = np.array(
        [x[0, 0, 1] - lse(x[0, 0]), x[1, 0, 2] - lse(x[1, 0])], dtype=np.float32
    )
    np.testing.assert_allclose(ll.numpy(), expected_ll, rtol=1e-6, atol=1e-6)

