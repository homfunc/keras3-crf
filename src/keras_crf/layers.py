# Keras Core (backend-independent) CRF Layer
import numpy as np
import keras
from keras import ops as K
from keras.layers import Layer, Dense
from typing import Optional

from .crf_ops import crf_log_likelihood as k_crf_ll, crf_decode as k_crf_decode


@keras.utils.register_keras_serializable(package="Keras3CRF")
class CRF(Layer):
    def __init__(self, units: int, use_boundary: bool = True, use_kernel: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.units = units
        self.use_boundary = use_boundary
        self.use_kernel = use_kernel
        if self.use_kernel:
            self.proj = Dense(units)
        else:
            self.proj = None

    def build(self, input_shape):
        self.trans = self.add_weight(shape=(self.units, self.units), name="transitions", initializer="glorot_uniform")
        if self.use_boundary:
            self.left_boundary = self.add_weight(shape=(self.units,), name="left_boundary", initializer="zeros")
            self.right_boundary = self.add_weight(shape=(self.units,), name="right_boundary", initializer="zeros")
        super().build(input_shape)

    def call(self, inputs, mask: Optional[keras.KerasTensor] = None):
        # Disallow left-padding masks for simplicity and parity with TF layer
        if mask is not None:
            left_padding_detected = None
            try:
                first_mask = K.convert_to_numpy(mask)[:, 0].astype(bool)
                left_padding_detected = (not np.all(first_mask))
            except Exception:
                # If mask cannot be converted here (symbolic), skip strict check
                left_padding_detected = None
            if left_padding_detected is True:
                raise NotImplementedError("Currently, CRF layer does not support left padding")
        x = inputs
        if self.proj is not None:
            x = self.proj(x)
        potentials = x
        if self.use_boundary:
            potentials = self._add_boundary(potentials, mask)
        lens = self._mask_to_lengths(mask, potentials)
        decoded, _ = k_crf_decode(potentials, lens, self.trans)
        return decoded, potentials, lens, self.trans

    def _mask_to_lengths(self, mask, potentials):
        if mask is None:
            B = K.shape(potentials)[0]
            T = K.shape(potentials)[1]
            return K.full((B,), T, dtype="int32")
        else:
            # mask [B, T] -> lengths [B]
            return K.sum(K.cast(mask, "int32"), axis=1)

    def _add_boundary(self, potentials, mask):
        # add left at t=0 and right at last valid step
        B, T, N = K.shape(potentials)[0], K.shape(potentials)[1], K.shape(potentials)[2]
        start = K.reshape(self.left_boundary, (1, 1, N))
        end = K.reshape(self.right_boundary, (1, 1, N))
        # add start to t=0: rebuild by concat
        first = potentials[:, 0, :] + start[0, 0, :]
        rest = potentials[:, 1:, :]
        potentials = K.concatenate([K.expand_dims(first, 1), rest], axis=1)
        # add end to last time step: rebuild by concat
        last = potentials[:, -1, :] + end[0, 0, :]
        mid = potentials[:, :-1, :]
        potentials = K.concatenate([mid, K.expand_dims(last, 1)], axis=1)
        return potentials

    def log_likelihood(self, potentials, tags, lens):
        return k_crf_ll(potentials, tags, lens, self.trans)

    def compute_output_shape(self, input_shape):
        # input_shape: (batch, time, features)
        B, T, _ = input_shape
        return (
            (B, T),                  # decoded tags
            (B, T, self.units),      # potentials
            (B,),                    # lengths
            (self.units, self.units) # transition matrix
        )

    def get_config(self):
        cfg = {"units": self.units, "use_boundary": self.use_boundary, "use_kernel": self.use_kernel}
        base = super().get_config()
        return {**base, **cfg}
