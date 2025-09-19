import numpy as np
import itertools
import keras
from keras import ops as K

from keras_crf.crf_ops import crf_decode, crf_marginals


def brute_force_best_sequence(inputs, trans, L):
    N = inputs.shape[1]
    best_s = -1e30
    best_seq = None
    for seq in itertools.product(range(N), repeat=L):
        s = 0.0
        for t in range(L):
            s += inputs[t, seq[t]]
        for t in range(L - 1):
            s += trans[seq[t], seq[t + 1]]
        if s > best_s:
            best_s = s
            best_seq = np.array(seq, dtype=np.int32)
    return best_seq, best_s


def test_ragged_batch_decode_and_marginals():
    # Mixed lengths in batch
    B, T, N = 4, 7, 4
    rng = np.random.default_rng(123)
    potentials = rng.normal(size=(B, T, N)).astype("float32")
    lens = np.array([1, 7, 3, 5], dtype=np.int32)
    trans = rng.normal(size=(N, N)).astype("float32")

    pot_t = K.convert_to_tensor(potentials)
    lens_t = K.convert_to_tensor(lens)
    trans_t = K.convert_to_tensor(trans)

    # Decode and check first L tokens match brute force
    tags_t, score_t = crf_decode(pot_t, lens_t, trans_t)
    tags = K.convert_to_numpy(tags_t)

    for b in range(B):
        L = int(lens[b])
        exp_tags, _ = brute_force_best_sequence(potentials[b], trans, L)
        np.testing.assert_array_equal(tags[b, :L], exp_tags)
        # positions >= L equal last valid tag
        if L < T:
            assert np.all(tags[b, L:] == tags[b, L - 1])

    # Marginals: each valid step sums to 1; padded steps are zeros
    probs_t = crf_marginals(pot_t, lens_t, trans_t)
    probs = K.convert_to_numpy(probs_t)

    for b in range(B):
        L = int(lens[b])
        for t in range(L):
            s = probs[b, t].sum()
            assert np.allclose(s, 1.0, rtol=1e-5, atol=1e-5)
        for t in range(L, T):
            assert np.allclose(probs[b, t], 0.0, atol=1e-7)
