import numpy as np
import tensorflow as tf
import keras
from keras import ops as K
import pytest

pytestmark = pytest.mark.tf_only

from keras_crf.core_kops import crf_log_likelihood as core_ll, crf_decode as core_decode
from keras_crf import text as tf_text
from keras_crf.layers import CRF as TFCRF
from keras_crf.layers_core import KerasCoreCRF


def test_ops_compatibility_ll_and_decode():
    B, T, N = 2, 4, 3
    rng = np.random.default_rng(123)
    potentials_np = rng.normal(size=(B, T, N)).astype("float32")
    tags_np = rng.integers(0, N, size=(B, T), dtype=np.int32)
    lens_np = np.array([4, 3], dtype=np.int32)
    trans_np = rng.normal(size=(N, N)).astype("float32")

    # TF ops path
    potentials_tf = tf.convert_to_tensor(potentials_np)
    tags_tf = tf.convert_to_tensor(tags_np)
    lens_tf = tf.convert_to_tensor(lens_np)
    trans_tf = tf.convert_to_tensor(trans_np)
    ll_tf, _ = tf_text.crf_log_likelihood(potentials_tf, tags_tf, lens_tf, trans_tf)
    dec_tf, score_tf = tf_text.crf_decode(potentials_tf, trans_tf, lens_tf)

    # Core (Keras Core ops) path
    potentials_k = K.convert_to_tensor(potentials_np)
    tags_k = K.convert_to_tensor(tags_np)
    lens_k = K.convert_to_tensor(lens_np)
    trans_k = K.convert_to_tensor(trans_np)
    ll_core = core_ll(potentials_k, tags_k, lens_k, trans_k)
    dec_core, score_core = core_decode(potentials_k, lens_k, trans_k)

    np.testing.assert_allclose(ll_tf.numpy(), K.convert_to_numpy(ll_core), rtol=1e-5, atol=1e-5)
    # Compare only valid positions up to each sequence length
    dec_tf_np = dec_tf.numpy()
    dec_core_np = K.convert_to_numpy(dec_core)
    lens = lens_np
    for i in range(B):
        L = int(lens[i])
        np.testing.assert_array_equal(dec_tf_np[i, :L], dec_core_np[i, :L])
    np.testing.assert_allclose(score_tf.numpy(), K.convert_to_numpy(score_core), rtol=1e-5, atol=1e-5)


def test_layer_compatibility_no_kernel_no_boundary():
    # Ensure both layers produce same decode given same transitions
    B, T, N = 2, 3, 4
    rng = np.random.default_rng(7)
    x_np = rng.normal(size=(B, T, N)).astype("float32")
    trans_np = rng.normal(size=(N, N)).astype("float32")

    # TF CRF layer
    tf_layer = TFCRF(units=N, use_kernel=False, use_boundary=False)
    decoded_tf, pot_tf, lens_tf, kernel_tf = tf_layer(x_np)

    # Core CRF layer
    core_layer = KerasCoreCRF(units=N, use_kernel=False, use_boundary=False)
    decoded_core, pot_core, lens_core, trans_core = core_layer(x_np)

    # Set both transitions to same values
    tf_layer.set_weights([trans_np])
    core_layer.set_weights([trans_np])

    # Recompute after setting weights
    decoded_tf, pot_tf, lens_tf, kernel_tf = tf_layer(x_np)
    decoded_core, pot_core, lens_core, trans_core = core_layer(x_np)

    np.testing.assert_array_equal(decoded_tf.numpy(), decoded_core.numpy())
    np.testing.assert_allclose(pot_tf.numpy(), pot_core.numpy(), rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(lens_tf.numpy(), lens_core.numpy())
    np.testing.assert_allclose(kernel_tf.numpy(), trans_core.numpy(), rtol=1e-6, atol=1e-6)
