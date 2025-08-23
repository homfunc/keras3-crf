# Tests for keras_crf.text ops (eager-mode)
import itertools
import numpy as np
import tensorflow as tf
import pytest

pytestmark = pytest.mark.tf_only

from keras_crf import text as kcrf


def brute_force_best_sequence(inputs, transition_params, seq_len):
    # inputs: [T, num_tags], numpy array
    T = seq_len
    num_tags = inputs.shape[1]
    best_score = -1e30
    best_seq = None
    for tags in itertools.product(range(num_tags), repeat=T):
        score = 0.0
        # unary
        for t in range(T):
            score += inputs[t, tags[t]]
        # transitions
        for t in range(T - 1):
            score += transition_params[tags[t], tags[t + 1]]
        if score > best_score:
            best_score = score
            best_seq = list(tags)
    return best_seq, best_score


def test_crf_decode_matches_bruteforce_multi_and_single():
    # Multi-step case
    inputs = np.array([[4, 5, -3], [3, -1, 3], [-1, 2, 1], [0, 0, 0]], dtype=np.float32)
    trans = np.array([[-3, 5, -2], [3, 4, 1], [1, 2, 1]], dtype=np.float32)
    seq_len = np.array(3, dtype=np.int32)

    expected_seq, expected_score = brute_force_best_sequence(inputs[:seq_len], trans, seq_len)

    decode, best = kcrf.crf_decode(
        tf.expand_dims(inputs, 0), tf.constant(trans), tf.expand_dims(seq_len, 0)
    )
    decode = tf.squeeze(decode, 0).numpy().tolist()
    best = float(tf.squeeze(best, 0).numpy())

    assert decode[: int(seq_len)] == expected_seq
    np.testing.assert_allclose(best, expected_score, rtol=1e-6, atol=1e-6)

    # Single-step case T == 1
    inputs1 = np.array([[0.2, 1.0, -0.5]], dtype=np.float32)
    seq_len1 = np.array(1, dtype=np.int32)
    expected_seq1, expected_score1 = brute_force_best_sequence(inputs1[:seq_len1], trans, seq_len1)

    decode1, best1 = kcrf.crf_decode(
        tf.expand_dims(inputs1, 0), tf.constant(trans), tf.expand_dims(seq_len1, 0)
    )
    decode1 = tf.squeeze(decode1, 0).numpy().tolist()
    best1 = float(tf.squeeze(best1, 0).numpy())

    assert decode1[: int(seq_len1)] == expected_seq1
    np.testing.assert_allclose(best1, expected_score1, rtol=1e-6, atol=1e-6)


def test_constrained_decode_matches_filtered_decode():
    trans = np.array([[-3, 5, -2], [3, 4, 1], [1, 2, 1]], dtype=np.float32)
    seq_len = np.array(3, dtype=np.int32)
    inputs = np.array([[4, 5, -3], [3, -1, 3], [-1, 2, 1], [0, 0, 0]], dtype=np.float32)
    tag_bitmap = np.array(
        [
            [True, False, False],
            [False, True, True],
            [False, True, True],
            [False, True, True],
        ],
        dtype=bool,
    )

    filtered = kcrf.crf_filtered_inputs(tf.expand_dims(inputs, 0), tf.expand_dims(tag_bitmap, 0))
    exp_seq, exp_best = kcrf.crf_decode(
        filtered, tf.constant(trans), tf.expand_dims(seq_len, 0)
    )
    exp_seq, exp_best = tf.squeeze(exp_seq, 0), tf.squeeze(exp_best, 0)

    act_seq, act_best = kcrf.crf_constrained_decode(
        tf.expand_dims(inputs, 0), tf.expand_dims(tag_bitmap, 0), tf.constant(trans), tf.expand_dims(seq_len, 0)
    )
    act_seq, act_best = tf.squeeze(act_seq, 0), tf.squeeze(act_best, 0)

    np.testing.assert_array_equal(act_seq.numpy()[: int(seq_len)], exp_seq.numpy()[: int(seq_len)])
    np.testing.assert_allclose(act_best.numpy(), exp_best.numpy(), rtol=1e-6, atol=1e-6)


def test_crf_decode_forward_mask_behavior():
    # This verifies that masked timesteps produce zeroed backpointers (as designed)
    batch = 3
    T = 6
    n_tags = 5

    potentials = tf.random.normal([batch, T, n_tags])
    trans = tf.random.normal([n_tags, n_tags])
    seq_lens = tf.constant([3, 6, 2], dtype=tf.int32)

    initial_state = tf.squeeze(tf.slice(potentials, [0, 0, 0], [-1, 1, -1]), axis=1)
    inputs = tf.slice(potentials, [0, 1, 0], [-1, -1, -1])

    seq_less_one = tf.maximum(tf.constant(0, dtype=tf.int32), seq_lens - 1)
    backpointers, _ = kcrf.crf_decode_forward(inputs, initial_state, trans, seq_less_one)

    mask = tf.sequence_mask(seq_less_one, tf.shape(inputs)[1])
    masked_indices = tf.cast(tf.logical_not(mask), tf.int32)

    # Mask sum by row equals T-1 - (seq_len -1)
    exp_mask_sums = tf.repeat(tf.shape(inputs)[1], batch) - seq_less_one
    mask_sums = tf.reduce_sum(masked_indices, axis=1)
    np.testing.assert_array_equal(exp_mask_sums.numpy(), mask_sums.numpy())

    masked_indices = tf.expand_dims(masked_indices, 2)
    zeros = masked_indices * backpointers
    assert tf.reduce_all(zeros == 0).numpy()

