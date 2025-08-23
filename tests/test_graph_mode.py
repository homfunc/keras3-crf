# Graph-mode tests for keras_crf (tf.function / non-eager)
import numpy as np
import tensorflow as tf
import pytest

pytestmark = pytest.mark.tf_only

from keras_crf import CRF, text as kcrf


def get_basic_data():
    x = np.array(
        [
            [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.1, 0.2, 0.1]],
            [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.1, 0.0]],
        ],
        dtype=np.float32,
    )
    y = np.array([[1, 2, 2], [1, 1, 1]], dtype=np.int32)
    return x, y


class ModelWithCRFLoss(tf.keras.Model):
    def __init__(self, units):
        super().__init__()
        self.crf = CRF(units=units)

    def call(self, inputs, training=False, mask=None):
        return self.crf(inputs, mask=mask)

    def train_step(self, data):
        x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
        with tf.GradientTape() as tape:
            decoded, potentials, seq_len, kernel = self(x, training=True)
            ll, _ = kcrf.crf_log_likelihood(potentials, y, seq_len, kernel)
            loss = -tf.reduce_mean(ll)
            if sample_weight is not None:
                # Broadcast weights and apply
                sw = tf.cast(sample_weight, loss.dtype)
                loss = tf.reduce_mean(tf.cast(sw, loss.dtype) * (-ll))
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return {"loss": loss}


def test_graph_mode_train_and_predict():
    # Ensure graph mode
    tf.config.run_functions_eagerly(False)

    x, y = get_basic_data()
    model = ModelWithCRFLoss(units=3)
    model.compile(optimizer=tf.keras.optimizers.Adam(0.05), run_eagerly=False)

    # One training step
    hist = model.train_on_batch(x, y, return_dict=True)
    assert "loss" in hist

    # Predict should produce decoded sequence as first element of tuple
    decoded, potentials, seq_len, kernel = model.predict(x, verbose=0)
    assert decoded.shape == (2, 3)
    assert potentials.shape == (2, 3, 3)
    assert seq_len.shape == (2,)
    assert kernel.shape == (3, 3)

