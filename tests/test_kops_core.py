import numpy as np
import keras
from keras import ops as K
from keras_crf.core_kops import crf_log_likelihood, crf_decode


def test_kops_log_likelihood_small():
    B, T, N = 2, 3, 3
    rng = np.random.default_rng(0)
    potentials = K.convert_to_tensor(rng.normal(size=(B, T, N)).astype("float32"))
    lens = K.convert_to_tensor(np.array([3, 2], dtype=np.int32))
    tags = K.convert_to_tensor(np.array([[0,1,2],[2,1,0]], dtype=np.int32))
    trans = K.convert_to_tensor(rng.normal(size=(N, N)).astype("float32"))
    ll = crf_log_likelihood(potentials, tags, lens, trans)
    assert K.shape(ll)[0] == B


def test_kops_decode_shapes():
    B, T, N = 2, 4, 3
    rng = np.random.default_rng(1)
    potentials = K.convert_to_tensor(rng.normal(size=(B, T, N)).astype("float32"))
    lens = K.convert_to_tensor(np.array([4, 3], dtype=np.int32))
    trans = K.convert_to_tensor(rng.normal(size=(N, N)).astype("float32"))
    tags, score = crf_decode(potentials, lens, trans)
    assert K.shape(tags) == (B, T)
    assert K.shape(score)[0] == B
