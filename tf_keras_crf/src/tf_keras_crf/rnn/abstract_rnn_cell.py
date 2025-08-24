# Minimal AbstractRNNCell compatible with Keras 3 (TensorFlow-only)
import tensorflow as tf


def _generate_zero_filled_state_for_cell(cell, inputs, batch_size, dtype):
    if inputs is not None:
        batch_size = tf.shape(inputs)[0]
        dtype = inputs.dtype
    return _generate_zero_filled_state(batch_size, cell.state_size, dtype)


def _generate_zero_filled_state(batch_size_tensor, state_size, dtype):
    if batch_size_tensor is None or dtype is None:
        raise ValueError(
            "batch_size and dtype cannot be None while constructing initial state: "
            f"batch_size={batch_size_tensor}, dtype={dtype}"
        )

    def create_zeros(unnested_state_size):
        flat_dims = tf.TensorShape(unnested_state_size).as_list()
        init_state_size = [batch_size_tensor] + flat_dims
        return tf.zeros(init_state_size, dtype=dtype)

    if tf.nest.is_nested(state_size):
        return tf.nest.map_structure(create_zeros, state_size)
    else:
        return create_zeros(state_size)


class AbstractRNNCell(tf.keras.layers.Layer):
    @property
    def state_size(self):
        raise NotImplementedError

    @property
    def output_size(self):
        raise NotImplementedError

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return _generate_zero_filled_state_for_cell(self, inputs, batch_size, dtype)

