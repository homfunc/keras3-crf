# Keras Core (backend-independent) CRF Layer
import numpy as np
import keras
from keras import ops as K
from keras.layers import Layer, Dense
from typing import Optional

from .crf_ops import crf_log_likelihood as k_crf_ll, crf_decode as k_crf_decode


def _is_symbolic_tensor(x) -> bool:
    """Return True if x is a Keras symbolic tensor (graph-time), robust across backends.

    Heuristics:
    - isinstance(x, keras.KerasTensor)
    - Class name or repr contains 'KerasTensor'
    - Tensor shape has dynamic dims (None)
    - Fallback to keras.utils.is_keras_tensor
    """
    try:
        KT = getattr(keras, "KerasTensor")
        if isinstance(x, KT):
            return True
    except Exception:
        pass
    # Heuristic: string-based detection
    try:
        tn = x.__class__.__name__
        if "KerasTensor" in tn:
            return True
        rp = repr(x)
        if "KerasTensor" in rp:
            return True
    except Exception:
        pass
    # Dynamic shape heuristic
    try:
        shp = getattr(x, "shape", None)
        if shp is not None:
            for d in shp:
                if d is None:
                    return True
    except Exception:
        pass
    # If a list/tuple was passed, check first element
    if isinstance(x, (list, tuple)) and x:
        return _is_symbolic_tensor(x[0])
    # Fallbacks
    try:
        return bool(keras.utils.is_keras_tensor(x))
    except Exception:
        return False


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
        if self.use_kernel:
            self.proj.build(input_shape)
        super().build(input_shape)

    def call(self, inputs, mask: Optional[keras.KerasTensor] = None, **kwargs):
        # Symbolic-safe lengths and boundary handling; allow left or right padding
        x = inputs[0] if isinstance(inputs, (list, tuple)) else inputs
        if self.proj is not None:
            x = self.proj(x)
        potentials = x
        lens = self._mask_to_lengths(mask, potentials)
        if self.use_boundary:
            potentials = self._add_boundary_mask_aware(potentials, mask, lens)
        # Choose decode path based on symbolic vs eager tensors
        is_symbolic = _is_symbolic_tensor(potentials)
        if is_symbolic:
            # During symbolic tracing/build, avoid non-symbolic-friendly Viterbi; use argmax for shape propagation
            decoded = K.argmax(potentials, axis=-1)
        else:
            decoded, _ = k_crf_decode(potentials, lens, self.trans)

        # Ensure transitions are returned as a graph-tied tensor across backends
        trans_out = K.convert_to_tensor(self.trans)

        return decoded, potentials, lens, trans_out

    def _mask_to_lengths(self, mask, potentials):
        # Normalize mask: Keras may pass a list (one per input). Use first non-None.
        if isinstance(mask, (list, tuple)):
            mask = next((m for m in mask if m is not None), None)
        if mask is None:
            B = K.shape(potentials)[0]
            T = K.shape(potentials)[1]
            return K.full((B,), T, dtype="int32")
        else:
            # mask [B, T] -> lengths [B]
            return K.sum(K.cast(mask, "int32"), axis=1)

    def _add_boundary_mask_aware(self, potentials, mask, lengths):
        # Add left at first valid timestep and right at last valid timestep
        B = K.shape(potentials)[0]
        T = K.shape(potentials)[1]
        N = K.shape(potentials)[2]
        # Compute first/last valid indices
        if mask is None:
            first_idx = K.zeros((B,), dtype="int32")
            last_idx = K.full((B,), T - 1, dtype="int32")
        else:
            first_idx = T - lengths
            last_idx = first_idx + lengths - 1
        t_range = K.arange(T)[None, :]  # [1, T]
        start_sel = K.equal(t_range, K.expand_dims(first_idx, 1))
        end_sel = K.equal(t_range, K.expand_dims(last_idx, 1))
        start_sel = K.cast(K.expand_dims(start_sel, -1), potentials.dtype)  # [B, T, 1]
        end_sel = K.cast(K.expand_dims(end_sel, -1), potentials.dtype)      # [B, T, 1]
        lb = K.reshape(self.left_boundary, (1, 1, N))
        rb = K.reshape(self.right_boundary, (1, 1, N))
        return potentials + start_sel * lb + end_sel * rb

    def log_likelihood(self, potentials, tags, lens):
        return k_crf_ll(potentials, tags, lens, self.trans)

    def compute_output_shape(self, input_shape):
        # input_shape: (batch, time, features)
        # Use static dims where available; None for dynamic dims
        try:
            B, T, _ = input_shape
        except Exception:
            B, T = None, None
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

    # Keras 3 build-less loading support: ensure child layers are built at load time
    def get_build_config(self):
        """Provide input shape info to rebuild variables on deserialization.

        Returns a dict with an input_shape triple (None, None, D_in).
        If use_kernel is True, D_in is inferred from the Dense kernel shape when available.
        Otherwise, D_in defaults to units.
        """
        D_in = None
        if self.use_kernel and getattr(self, "proj", None) is not None:
            # If proj is already built, infer input dim from kernel shape
            kernel = getattr(self.proj, "kernel", None)
            if kernel is not None:
                try:
                    D_in = int(kernel.shape[0])
                except Exception:
                    D_in = None
        if D_in is None:
            D_in = int(self.units) if not self.use_kernel else None
        return {"input_shape": (None, None, D_in)}

    def build_from_config(self, cfg):
        """Recreate variables without tracing call, for safe weight loading during deserialization."""
        shape = cfg.get("input_shape", (None, None, None))
        D_in = shape[-1]
        self.build((None, None, D_in))
        self.built = True
