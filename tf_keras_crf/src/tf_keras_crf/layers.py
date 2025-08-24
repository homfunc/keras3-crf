# Standalone CRF Layer adapted for TensorFlow; delegates algorithmic ops to keras_crf.core_kops
from typing import Optional

import tensorflow as tf
from typeguard import typechecked

from tf.keras import Initializer
from .text import crf_decode, crf_log_likelihood

@tf.keras.utils.register_keras_serializable(package="TFKerasCRF")
class CRF(tf.keras.layers.Layer):
    @typechecked
    def __init__(
        self,
        units: int,
        chain_initializer: Initializer = "orthogonal",
        use_boundary: bool = True,
        boundary_initializer: Initializer = "zeros",
        use_kernel: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.units = units
        self.use_boundary = use_boundary
        self.use_kernel = use_kernel
        self.chain_initializer = tf.keras.initializers.get(chain_initializer)
        self.boundary_initializer = tf.keras.initializers.get(boundary_initializer)

        self.chain_kernel = self.add_weight(
            shape=(self.units, self.units), name="chain_kernel", initializer=self.chain_initializer
        )

        if self.use_boundary:
            self.left_boundary = self.add_weight(
                shape=(self.units,), name="left_boundary", initializer=self.boundary_initializer
            )
            self.right_boundary = self.add_weight(
                shape=(self.units,), name="right_boundary", initializer=self.boundary_initializer
            )

        if self.use_kernel:
            self._dense_layer = tf.keras.layers.Dense(units=self.units, dtype=self.dtype)
        else:
            self._dense_layer = lambda x: tf.cast(x, dtype=self.dtype)

    def call(self, inputs, mask: Optional[tf.Tensor] = None):
        if mask is not None and tf.keras.backend.ndim(mask) != 2:
            raise ValueError("Input mask to CRF must have dim 2 if not None")

        if mask is not None:
            left_boundary_mask = self._compute_mask_left_boundary(mask)
            first_mask = left_boundary_mask[:, 0]
            if first_mask is not None and tf.executing_eagerly():
                no_left_padding = tf.math.reduce_all(first_mask)
                if not bool(no_left_padding.numpy()):
                    raise NotImplementedError("Currently, CRF layer do not support left padding")

        potentials = self._dense_layer(inputs)

        if self.use_boundary:
            potentials = self.add_boundary_energy(potentials, mask, self.left_boundary, self.right_boundary)

        sequence_length = self._get_sequence_length(inputs, mask)
        decoded_sequence, _ = self.get_viterbi_decoding(potentials, sequence_length)
        return (decoded_sequence, potentials, sequence_length, self.chain_kernel)

    def _get_sequence_length(self, input_, mask):
        if mask is not None:
            sequence_length = self.mask_to_sequence_length(mask)
        else:
            input_energy_shape = tf.shape(input_)
            raw_input_shape = tf.slice(input_energy_shape, [0], [2])
            alt_mask = tf.ones(raw_input_shape)
            sequence_length = self.mask_to_sequence_length(alt_mask)
        return sequence_length

    def mask_to_sequence_length(self, mask):
        return tf.reduce_sum(tf.cast(mask, tf.int32), 1)

    @staticmethod
    def _compute_mask_right_boundary(mask):
        offset = 1
        left_shifted_mask = tf.concat([mask[:, offset:], tf.zeros_like(mask[:, :offset])], axis=1)
        right_boundary = tf.math.greater(tf.cast(mask, tf.int32), tf.cast(left_shifted_mask, tf.int32))
        return right_boundary

    @staticmethod
    def _compute_mask_left_boundary(mask):
        offset = 1
        right_shifted_mask = tf.concat([tf.zeros_like(mask[:, :offset]), mask[:, :-offset]], axis=1)
        left_boundary = tf.math.greater(tf.cast(mask, tf.int32), tf.cast(right_shifted_mask, tf.int32))
        return left_boundary

    def add_boundary_energy(self, potentials, mask, start, end):
        def expand_scalar_to_3d(x):
            return tf.reshape(x, (1, 1, -1))

        start = tf.cast(expand_scalar_to_3d(start), potentials.dtype)
        end = tf.cast(expand_scalar_to_3d(end), potentials.dtype)
        if mask is None:
            potentials = tf.concat([potentials[:, :1, :] + start, potentials[:, 1:, :]], axis=1)
            potentials = tf.concat([potentials[:, :-1, :], potentials[:, -1:, :] + end], axis=1)
        else:
            mask = tf.keras.backend.expand_dims(tf.cast(mask, start.dtype), axis=-1)
            start_mask = tf.cast(self._compute_mask_left_boundary(mask), start.dtype)
            end_mask = tf.cast(self._compute_mask_right_boundary(mask), end.dtype)
            potentials = potentials + start_mask * start
            potentials = potentials + end_mask * end
        return potentials

    def get_viterbi_decoding(self, potentials, sequence_length):
        decode_tags, best_score = crf_decode(potentials, sequence_length, self.chain_kernel)
        return decode_tags, best_score

    def get_config(self):
        config = {
            "units": self.units,
            "chain_initializer": tf.keras.initializers.serialize(self.chain_initializer),
            "use_boundary": self.use_boundary,
            "boundary_initializer": tf.keras.initializers.serialize(self.boundary_initializer),
            "use_kernel": self.use_kernel,
        }
        base_config = super().get_config()
        return {**base_config, **config}

    def compute_output_shape(self, input_shape):
        return input_shape[:2]

    def log_likelihood(self, potentials, tags, sequence_length):
        return crf_log_likelihood(potentials, tags, sequence_length, self.chain_kernel)

    def compute_mask(self, input_, mask=None):
        return mask

    @property
    def _compute_dtype(self):
        return tf.int32

