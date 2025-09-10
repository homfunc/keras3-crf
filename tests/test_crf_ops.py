import os
import sys
import itertools
import numpy as np
import pytest

# Ensure src is on path
REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
sys.path.insert(0, SRC_ROOT)

import keras
from keras import ops as K

from keras_crf.crf_ops import (
    crf_log_likelihood,
    crf_decode,
    crf_marginals,
    crf_sequence_score,
)


def np_logsumexp(a, axis=None):
    a = np.asarray(a)
    a_max = np.max(a, axis=axis, keepdims=True)
    res = a_max + np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=True))
    if axis is not None:
        res = np.squeeze(res, axis=axis)
    return res


def np_crf_sequence_score(potentials, tags, lens, trans):
    # potentials [T,N], tags [T], trans [N,N]
    L = int(lens)
    unary = potentials[np.arange(L), tags[:L]].sum()
    if L <= 1:
        return unary
    trans_sum = trans[tags[:L-1], tags[1:L]].sum()
    return unary + trans_sum


def np_crf_log_norm(potentials, lens, trans):
    # potentials [T,N], trans [N,N]
    L, N = int(lens), potentials.shape[1]
    alpha = potentials[0]
    for t in range(1, L):
        # alpha_next[j] = potentials[t,j] + logsumexp_i(alpha[i] + trans[i,j])
        scores = alpha[:, None] + trans
        alpha = potentials[t] + np_logsumexp(scores, axis=0)
    return np_logsumexp(alpha, axis=0)


def make_random_batch(B=4, T=5, N=3, seed=0):
    rng = np.random.default_rng(seed)
    potentials = rng.normal(size=(B, T, N)).astype("float32")
    trans = rng.normal(size=(N, N)).astype("float32")
    lens = rng.integers(low=1, high=T+1, size=(B,), endpoint=False).astype("int32")
    tags = rng.integers(low=0, high=N, size=(B, T)).astype("int32")
    return potentials, trans, lens, tags


@pytest.mark.parametrize("B,T,N", [(3, 4, 3), (2, 5, 4)])
def test_log_likelihood_matches_numpy_reference(B, T, N):
    potentials, trans, lens, tags = make_random_batch(B, T, N, seed=1)

    pot_t = K.convert_to_tensor(potentials)
    trans_t = K.convert_to_tensor(trans)
    lens_t = K.convert_to_tensor(lens)
    tags_t = K.convert_to_tensor(tags)

    ll = crf_log_likelihood(pot_t, tags_t, lens_t, trans_t)
    # Convert to numpy in a backend-agnostic way
    try:
        ll_np = K.convert_to_numpy(ll)
    except Exception:
        ll_np = ll.numpy() if hasattr(ll, "numpy") else np.array(ll)

    # Compare per-batch
    for b in range(B):
        score_np = np_crf_sequence_score(potentials[b], tags[b], lens[b], trans)
        logz_np = np_crf_log_norm(potentials[b], lens[b], trans)
        ll_ref = score_np - logz_np
        assert np.allclose(ll_np[b], ll_ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("B,T,N", [(2, 4, 3)])
def test_decode_matches_bruteforce_for_small(B, T, N):
    potentials, trans, lens, _ = make_random_batch(B, T, N, seed=2)

    pot_t = K.convert_to_tensor(potentials)
    trans_t = K.convert_to_tensor(trans)
    lens_t = K.convert_to_tensor(lens)

    tags_pred_t, best_score_t = crf_decode(pot_t, lens_t, trans_t)
    try:
        tags_pred = K.convert_to_numpy(tags_pred_t)
    except Exception:
        tags_pred = tags_pred_t.numpy() if hasattr(tags_pred_t, "numpy") else np.array(tags_pred_t)
    try:
        best_score = K.convert_to_numpy(best_score_t)
    except Exception:
        best_score = best_score_t.numpy() if hasattr(best_score_t, "numpy") else np.array(best_score_t)

    for b in range(B):
        L = int(lens[b])
        best_s = -1e9
        best_seq = None
        for seq in itertools.product(range(N), repeat=L):
            seq = np.array(seq, dtype=np.int32)
            s = np_crf_sequence_score(potentials[b], seq, L, trans)
            if s > best_s:
                best_s = s
                best_seq = seq
        # Compare
        assert np.all(tags_pred[b, :L] == best_seq)
        # path score from prediction equals best_s
        # compute score of predicted path
        pred_score = np_crf_sequence_score(potentials[b], tags_pred[b], L, trans)
        assert np.allclose(best_score[b], pred_score, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("B,T,N", [(2, 6, 4)])
def test_marginals_properties(B, T, N):
    potentials, trans, lens, _ = make_random_batch(B, T, N, seed=3)

    # Zero transitions: marginals should reduce to softmax over N at each t < len
    trans_zero = np.zeros_like(trans).astype("float32")

    pot_t = K.convert_to_tensor(potentials)
    trans0_t = K.convert_to_tensor(trans_zero)
    lens_t = K.convert_to_tensor(lens)

    probs_t = crf_marginals(pot_t, lens_t, trans0_t)
    try:
        probs = K.convert_to_numpy(probs_t)
    except Exception:
        probs = probs_t.numpy() if hasattr(probs_t, "numpy") else np.array(probs_t)

    for b in range(B):
        L = int(lens[b])
        # Normalization and softmax equivalence for valid steps
        for t in range(L):
            p = probs[b, t]
            assert np.allclose(p.sum(), 1.0, rtol=1e-4, atol=1e-4)
            # softmax of potentials[b, t]
            x = potentials[b, t]
            x_max = x.max()
            sm = np.exp(x - x_max)
            sm = sm / sm.sum()
            assert np.allclose(p, sm, rtol=1e-4, atol=1e-4)
        # Padded steps are zeros
        for t in range(L, T):
            assert np.allclose(probs[b, t], 0.0, atol=1e-7)
