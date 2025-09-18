import numpy as np
import keras
from keras import ops as K

from .crf_ops import crf_log_likelihood, crf_marginals
from . import CRF


def make_crf_tagger(tokens_input,
                    features,
                    num_tags,
                    optimizer=None,
                    metrics=None,
                    loss: str = "nll",
                    dice_smooth: float = 1.0,
                    joint_nll_weight: float = None):
    """
    Build a training-ready Keras Model for sequence tagging with a CRF head.

    Inputs
    - tokens_input: a Keras Input (shape (None,)) for token IDs. Name will be normalized to 'tokens'.
    - features: a KerasTensor [B, T, D] from your encoder (e.g., Embedding + LSTM).
    - num_tags: number of tag classes.
    - optimizer: optional optimizer (default Adam(1e-3))
    - metrics: list of metrics to apply to the decoded_output head (optional)
    - loss: one of {"nll", "dice", "dice+nll"}. Default "nll".
    - dice_smooth: smoothing constant for dice.
    - joint_nll_weight: if loss == "dice+nll", weight for the NLL term (0..1). Default 0.2.

    Returns
    - model: a compiled Model with two outputs:
        decoded_output: int tags [B, T] (metrics only)
        crf_log_likelihood_output: per-sample loss [B] (drives training; name kept for backward compatibility)
      The model expects inputs {'tokens': tokens, 'labels': labels} and outputs as above.
    """
    # Ensure named 'tokens' output for inputs dict convenience (mask-preserving)
    class _Identity(keras.layers.Layer):
        def __init__(self, **kw):
            super().__init__(**kw)
            self.supports_masking = True
        def call(self, x):
            return x
    if tokens_input.name.split(':')[0] != 'tokens':
        tokens_named = _Identity(name='tokens')(tokens_input)
    else:
        tokens_named = tokens_input

    crf = CRF(num_tags)
    decoded, potentials, lens, trans = crf(features)

    # Labels input
    labels = keras.Input(shape=(None,), dtype="int32", name="labels")

    class CRF_NLL(keras.layers.Layer):
        def call(self, inputs):
            pot, y_true, ln, tr = inputs
            ll = crf_log_likelihood(pot, y_true, ln, tr)
            return -ll  # negative log-likelihood per sample

        def compute_output_shape(self, input_shapes):
            pot_shape = input_shapes[0]
            B = pot_shape[0]
            return (B,)

    class DiceLossLayer(keras.layers.Layer):
        def __init__(self, num_tags: int, smooth: float = 1.0, **kwargs):
            super().__init__(**kwargs)
            self.num_tags = num_tags
            self.smooth = smooth

        def call(self, inputs):
            pot, y_true, ln, tr = inputs  # pot: [B,T,N], y_true: [B,T], ln: [B], tr: [N,N]
            T = K.shape(pot)[1]
            # CRF token marginals
            probs = crf_marginals(pot, ln, tr)  # [B,T,N]
            # One-hot labels
            y_oh = K.one_hot(K.cast(y_true, "int32"), self.num_tags)
            y_oh = K.cast(y_oh, probs.dtype)
            # Mask by lengths
            time_idx = K.cast(K.arange(T), "int32")
            mask = K.expand_dims(time_idx, 0) < K.expand_dims(K.cast(ln, "int32"), -1)  # [B,T]
            mask = K.cast(mask, probs.dtype)
            mask = K.expand_dims(mask, -1)  # [B,T,1]
            y_oh_m = y_oh * mask
            probs_m = probs * mask
            # Micro dice over all tokens and classes
            intersection = K.sum(y_oh_m * probs_m, axis=(1, 2))  # [B]
            sums = K.sum(y_oh_m, axis=(1, 2)) + K.sum(probs_m, axis=(1, 2))  # [B]
            dice = (2.0 * intersection + self.smooth) / (sums + self.smooth)
            loss = 1.0 - dice  # [B]
            return loss

        def compute_output_shape(self, input_shapes):
            pot_shape = input_shapes[0]
            B = pot_shape[0]
            return (B,)

    # Build the chosen loss head
    loss_choice = (loss or "nll").lower()
    nll_out = CRF_NLL(name="nll_out")([potentials, labels, lens, trans])
    if loss_choice == "nll":
        loss_head = keras.layers.Lambda(lambda z: z, name="crf_loss_out")(nll_out)
    elif loss_choice == "dice":
        dice_out = DiceLossLayer(num_tags, smooth=dice_smooth, name="dice_out")([potentials, labels, lens, trans])
        loss_head = keras.layers.Lambda(lambda z: z, name="crf_loss_out")(dice_out)
    elif loss_choice in ("dice+nll", "joint"):
        alpha = 0.2 if joint_nll_weight is None else float(joint_nll_weight)
        dice_out = DiceLossLayer(num_tags, smooth=dice_smooth, name="dice_out")([potentials, labels, lens, trans])
        # Weighted combination per-sample
        combo = keras.layers.Lambda(lambda zs: alpha * zs[0] + (1.0 - alpha) * zs[1], name="combo_loss")([nll_out, dice_out])
        loss_head = keras.layers.Lambda(lambda z: z, name="crf_loss_out")(combo)
    else:
        raise ValueError(f"Unsupported loss option: {loss}")

    # Name outputs to align with compile/loss/metrics dict keys
    decoded_named = _Identity(name="decoded_output")(decoded)
    loss_named = keras.layers.Lambda(lambda z: z, name="crf_loss_output")(loss_head)

    model = keras.Model(
        inputs={"tokens": tokens_named, "labels": labels},
        outputs={"decoded_output": decoded_named, "crf_loss_output": loss_named},
    )

    def zero_loss(y_true, y_pred):
        return K.mean(K.zeros_like(y_pred[..., :1]))

    model.compile(
        optimizer=optimizer or keras.optimizers.Adam(1e-3),
        loss={"decoded_output": zero_loss, "crf_loss_output": lambda y_true, y_pred: K.mean(y_pred)},
        metrics={"decoded_output": metrics or []},
    )
    return model


def prepare_crf_targets(y_true, mask=None):
    """
    Convenience function to build y and sample_weight dicts for a two-output CRF model.

    Args:
      y_true: np.ndarray [B, T] integer tag IDs
      mask: optional np.ndarray [B, T] float/bool mask (1 for valid tokens)

    Returns:
      y_dict, sample_weight_dict suitable for Model.fit
    """
    B = y_true.shape[0]
    y_dummy = np.zeros((B,), dtype=np.float32)
    sw_dummy = np.ones((B,), dtype=np.float32)
    sw_decoded = mask.astype(np.float32) if mask is not None else np.ones_like(y_true, dtype=np.float32)
    y_dict = {"decoded_output": y_true, "crf_loss_output": y_dummy}
    sw_dict = {"decoded_output": sw_decoded, "crf_loss_output": sw_dummy}
    return y_dict, sw_dict
