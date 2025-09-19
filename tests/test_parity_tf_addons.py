import os
import numpy as np
import pytest

# Skip module unless TensorFlow is the selected Keras backend
if os.environ.get("KERAS_BACKEND") != "tensorflow":
    pytest.skip(
        "tests/test_parity_tf_addons.py requires TensorFlow backend (set KERAS_BACKEND=tensorflow).",
        allow_module_level=True,
    )

# Import TensorFlow and Addons only when running under TF backend
tf = pytest.importorskip("tensorflow")
tfa = pytest.importorskip("tensorflow_addons")
import keras
from keras_crf.crf_ops import crf_decode as kcrf_decode, crf_log_likelihood as kcrf_ll

rng = np.random.default_rng(1234)


def make_random_batch(B=4, T=7, N=5, seed=0):
    r = np.random.RandomState(seed)
    potentials = r.randn(B, T, N).astype("float32")
    transitions = r.randn(N, N).astype("float32")
    # lengths in [1..T]
    lengths = r.randint(1, T + 1, size=(B,), dtype="int32")
    # random tag paths (ignored beyond length)
    tags = r.randint(0, N, size=(B, T), dtype="int32")
    return potentials, transitions, lengths, tags


@pytest.mark.parametrize("shape", [(4, 7, 5), (2, 3, 4), (1, 1, 3)])
def test_decode_parity_with_tfa(shape):
    B, T, N = shape
    pot, trans, lens, _ = make_random_batch(B=B, T=T, N=N, seed=42 + B + T + N)

    # TFA decode
    pot_tf = tf.convert_to_tensor(pot)
    trans_tf = tf.convert_to_tensor(trans)
    lens_tf = tf.convert_to_tensor(lens)
    t_tags, t_scores = tfa.text.crf_decode(pot_tf, trans_tf, lens_tf)

    # Keras CRF decode
    k_tags, k_scores = kcrf_decode(pot, lens, trans)

    for i in range(B):
        np.testing.assert_array_equal(k_tags[i, :lens[i]], t_tags.numpy()[i, :lens[i]],
                                      err_msg=f"Decoded tags mismatch with tfa.text.crf_decode for batch item {i}")
    # Best scores can differ by tiny fp error; use tolerance
    np.testing.assert_allclose(k_scores, t_scores.numpy(), rtol=1e-5, atol=1e-5,
                               err_msg="Best path scores mismatch with tfa.text.crf_decode")


@pytest.mark.parametrize("shape", [(4, 7, 5), (3, 5, 6)])
def test_log_likelihood_parity_with_tfa(shape):
    B, T, N = shape
    pot, trans, lens, tags = make_random_batch(B=B, T=T, N=N, seed=99 + B + T + N)

    # Mask tags beyond lengths to keep them in-range; CRF LL ignores them via lengths
    tags_masked = tags.copy()

    # TFA log-likelihood
    ll_tfa, _ = tfa.text.crf_log_likelihood(
        inputs=tf.convert_to_tensor(pot),
        tag_indices=tf.convert_to_tensor(tags_masked),
        sequence_lengths=tf.convert_to_tensor(lens),
        transition_params=tf.convert_to_tensor(trans)
    )

    # Keras CRF log-likelihood
    ll_k = kcrf_ll(pot, tags_masked, lens, trans)

    np.testing.assert_allclose(ll_k, ll_tfa.numpy(), rtol=1e-5, atol=1e-5,
                               err_msg="Log-likelihood parity failed vs tfa.text.crf_log_likelihood")
