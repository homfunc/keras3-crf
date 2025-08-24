# Tests for backend-agnostic core ops (keras_crf.crf_ops)
import itertools
import numpy as np
import keras
from keras import ops as K

from keras_crf.crf_ops import crf_decode, crf_constrained_decode


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

    # Core decode expects tensors from keras.ops
    decode, best = crf_decode(
        K.convert_to_tensor(np.expand_dims(inputs, 0)),
        K.convert_to_tensor(np.expand_dims(seq_len, 0)),
        K.convert_to_tensor(trans),
    )
    decode = K.convert_to_numpy(decode[0]).tolist()
    best = float(K.convert_to_numpy(best[0]))

    assert decode[: int(seq_len)] == expected_seq
    np.testing.assert_allclose(best, expected_score, rtol=1e-6, atol=1e-6)

    # Single-step case T == 1
    inputs1 = np.array([[0.2, 1.0, -0.5]], dtype=np.float32)
    seq_len1 = np.array(1, dtype=np.int32)
    expected_seq1, expected_score1 = brute_force_best_sequence(inputs1[:seq_len1], trans, seq_len1)

    decode1, best1 = crf_decode(
        K.convert_to_tensor(np.expand_dims(inputs1, 0)),
        K.convert_to_tensor(np.expand_dims(seq_len1, 0)),
        K.convert_to_tensor(trans),
    )
    decode1 = K.convert_to_numpy(decode1[0]).tolist()
    best1 = float(K.convert_to_numpy(best1[0]))

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

    # emulate filtered by replacing -inf outside bitmap
    filtered = np.where(tag_bitmap, inputs, -np.inf).astype(np.float32)
    exp_seq, exp_best = crf_decode(
        K.convert_to_tensor(np.expand_dims(filtered, 0)),
        K.convert_to_tensor(np.expand_dims(seq_len, 0)),
        K.convert_to_tensor(trans),
    )
    exp_seq, exp_best = exp_seq[0], exp_best[0]

    act_seq, act_best = crf_constrained_decode(
        K.convert_to_tensor(np.expand_dims(inputs, 0)),
        K.convert_to_tensor(np.expand_dims(tag_bitmap, 0)),
        K.convert_to_tensor(np.expand_dims(seq_len, 0)),
        K.convert_to_tensor(trans),
    )
    act_seq, act_best = act_seq[0], act_best[0]

    np.testing.assert_array_equal(K.convert_to_numpy(act_seq)[: int(seq_len)], K.convert_to_numpy(exp_seq)[: int(seq_len)])
    np.testing.assert_allclose(K.convert_to_numpy(act_best), K.convert_to_numpy(exp_best), rtol=1e-6, atol=1e-6)

