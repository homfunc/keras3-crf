import numpy as np
import keras
from keras import ops as K

from keras_crf.crf_ops import crf_marginals


def test_crf_marginals_uniform_distribution():
    # If potentials and transitions are all zeros, marginals should be uniform over classes
    B, T, N = 2, 4, 5
    potentials = K.zeros((B, T, N))
    lens = K.convert_to_tensor(np.array([4, 3], dtype=np.int32))
    trans = K.zeros((N, N))
    probs = crf_marginals(potentials, lens, trans)
    probs_np = K.convert_to_numpy(probs)
    # Valid positions sum to 1 and approx uniform 1/N
    lens_np = K.convert_to_numpy(lens)
    for b in range(B):
        for t in range(int(lens_np[b])):
            row = probs_np[b, t]
            assert np.allclose(row.sum(), 1.0, atol=1e-5)
            assert np.allclose(row, np.full((N,), 1.0 / N), atol=1e-5)
        # Padded positions are zero
        for t in range(int(lens_np[b]), T):
            assert np.allclose(probs_np[b, t], 0.0, atol=1e-6)


def test_crf_marginals_probability_simple():
    # Random small case: check probabilities sum to 1 across classes on valid steps
    B, T, N = 1, 3, 4
    rng = np.random.default_rng(0)
    potentials = K.convert_to_tensor(rng.normal(size=(B, T, N)).astype("float32"))
    lens = K.convert_to_tensor(np.array([3], dtype=np.int32))
    trans = K.convert_to_tensor(rng.normal(size=(N, N)).astype("float32"))
    probs = crf_marginals(potentials, lens, trans)
    s = K.sum(probs, axis=-1)
    s_np = K.convert_to_numpy(s)
    assert np.allclose(s_np[0], np.ones((T,)), atol=1e-5)

