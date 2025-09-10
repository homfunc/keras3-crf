# Public CRF losses built on top of Keras 3 backend-agnostic ops.
# These operate on CRF internals: potentials [B,T,N], lengths [B], transitions [N,N], y_true [B,T].

from __future__ import annotations

from keras import ops as K
import keras

from .crf_ops import crf_log_likelihood, crf_marginals


def _reduce(loss, reduction: str = "mean"):
    mode = (reduction or "mean").lower()
    if mode == "none":
        return loss
    if mode == "sum":
        return K.sum(loss)
    return K.mean(loss)


def nll_loss(
    potentials,
    lengths,
    transitions,
    y_true,
    sample_weight=None,
    reduction: str = "mean",
):
    """
    Negative log-likelihood for a linear-chain CRF.

    Args:
        potentials: [B, T, N] unary potentials (emission scores).
        lengths: [B] sequence lengths.
        transitions: [N, N] transition matrix.
        y_true: [B, T] integer tag ids.
        sample_weight: optional weights per-example (shape [B] or [B, 1]).
        reduction: "mean" (default), "sum", or "none".

    Returns:
        Loss tensor reduced per `reduction`. If reduction=="none", shape is [B].
    """
    ll = crf_log_likelihood(
        potentials=potentials,
        tags=K.cast(y_true, "int32"),
        lens=lengths,
        trans=transitions,
    )
    loss = -ll
    if sample_weight is not None:
        loss = loss * K.cast(sample_weight, loss.dtype)
    return _reduce(loss, reduction=reduction)


class _LengthsFromFullTime:
    def __init__(self, fixed_time_dim: int | None = None):
        self.fixed_T = fixed_time_dim

    def __call__(self, y_true, y_pred):
        if self.fixed_T is not None:
            B = K.shape(y_pred)[0]
            return K.full((B,), self.fixed_T, dtype="int32")
        T = K.shape(y_pred)[1]
        B = K.shape(y_pred)[0]
        return K.full((B,), T, dtype="int32")


# ----- In-graph "head" layers that produce per-sample loss vectors -----

@keras.utils.register_keras_serializable(package="Keras3CRF")
class CRFNLLHead(keras.layers.Layer):
    """Compute per-sample CRF negative log-likelihood: outputs [B]."""

    def call(self, inputs):
        potentials, y_true, lengths, transitions = inputs
        ll = crf_log_likelihood(
            potentials=potentials,
            tags=K.cast(y_true, "int32"),
            lens=K.cast(lengths, "int32"),
            trans=transitions,
        )
        return -ll


@keras.utils.register_keras_serializable(package="Keras3CRF")
class CRFDiceHead(keras.layers.Layer):
    """Compute per-sample CRF Dice loss from marginals: outputs [B]."""

    def __init__(self, smooth: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.smooth = float(smooth)

    def call(self, inputs):
        potentials, y_true, lengths, transitions = inputs
        probs = crf_marginals(potentials, K.cast(lengths, "int32"), transitions)  # [B,T,N]
        num_tags = K.shape(probs)[-1]
        y_oh = K.one_hot(K.cast(y_true, "int32"), num_tags)
        y_oh = K.cast(y_oh, probs.dtype)

        T = K.shape(probs)[1]
        time_idx = K.arange(T)
        mask_bt = K.expand_dims(time_idx, 0) < K.expand_dims(K.cast(lengths, "int32"), -1)
        mask_bt = K.cast(mask_bt, probs.dtype)
        mask_btn = K.expand_dims(mask_bt, -1)

        y_m = y_oh * mask_btn
        p_m = probs * mask_btn

        inter = K.sum(y_m * p_m, axis=(1, 2))
        sums = K.sum(y_m, axis=(1, 2)) + K.sum(p_m, axis=(1, 2))
        dice = (2.0 * inter + self.smooth) / (sums + self.smooth)
        return 1.0 - dice


@keras.utils.register_keras_serializable(package="Keras3CRF")
class CRFJointDiceNLLHead(keras.layers.Layer):
    """Compute per-sample weighted joint Dice+NLL loss: outputs [B]."""

    def __init__(self, alpha: float = 0.2, smooth: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = float(alpha)
        self.smooth = float(smooth)

    def call(self, inputs):
        potentials, y_true, lengths, transitions = inputs
        nll = CRFNLLHead()(inputs)
        dce = CRFDiceHead(smooth=self.smooth)(inputs)
        alpha_t = K.convert_to_tensor(self.alpha, dtype="float32")
        return alpha_t * nll + (1.0 - alpha_t) * dce


# ----- Loss classes: wrapper reducers over precomputed loss vectors -----

class _VectorLossReducer(keras.losses.Loss):
    """A Loss that assumes y_pred is a per-sample loss vector and just returns it.

    Keras will apply the configured reduction over the batch dimension, and any
    sample_weight passed to Model.fit will be applied automatically.

    For backward compatibility, if constructed with `crf_layer` we fall back to
    the legacy behavior and compute the losses internally (not recommended).
    """

    def __init__(self, reduction="sum_over_batch_size", name=None):
        super().__init__(reduction=reduction, name=name)
        # Legacy fields for optional compute mode
        self._legacy_crf_layer = None
        self._legacy_lengths_fn = None

    # Back-compat constructor adapter
    def _maybe_set_legacy(self, crf_layer=None, lengths_fn=None):
        self._legacy_crf_layer = crf_layer
        self._legacy_lengths_fn = lengths_fn or _LengthsFromFullTime()


class CRFNLLLoss(_VectorLossReducer):
    def __init__(self, crf_layer=None, lengths_fn=None, reduction="sum_over_batch_size", name="crf_nll_loss"):
        super().__init__(reduction=reduction, name=name)
        if crf_layer is not None:
            self._maybe_set_legacy(crf_layer, lengths_fn)

    def call(self, y_true, y_pred):
        if self._legacy_crf_layer is None:
            return y_pred  # already a per-sample vector
        # Legacy compute mode (may require run_eagerly=True under TF backend)
        lengths = self._legacy_lengths_fn(y_true, y_pred)
        trans = self._legacy_crf_layer.trans
        return nll_loss(y_pred, lengths, trans, y_true, reduction="none")


class CRFDiceLoss(_VectorLossReducer):
    def __init__(self, crf_layer=None, lengths_fn=None, smooth: float = 1.0, reduction="sum_over_batch_size", name="crf_dice_loss"):
        super().__init__(reduction=reduction, name=name)
        self.smooth = float(smooth)
        if crf_layer is not None:
            self._maybe_set_legacy(crf_layer, lengths_fn)

    def call(self, y_true, y_pred):
        if self._legacy_crf_layer is None:
            return y_pred
        lengths = self._legacy_lengths_fn(y_true, y_pred)
        trans = self._legacy_crf_layer.trans
        return dice_loss(y_pred, lengths, trans, y_true, smooth=self.smooth, reduction="none")


class CRFJointDiceNLLLoss(_VectorLossReducer):
    def __init__(self, crf_layer=None, lengths_fn=None, alpha: float = 0.2, smooth: float = 1.0, reduction="sum_over_batch_size", name="crf_joint_dice_nll_loss"):
        super().__init__(reduction=reduction, name=name)
        self.alpha = float(alpha)
        self.smooth = float(smooth)
        if crf_layer is not None:
            self._maybe_set_legacy(crf_layer, lengths_fn)

    def call(self, y_true, y_pred):
        if self._legacy_crf_layer is None:
            return y_pred
        lengths = self._legacy_lengths_fn(y_true, y_pred)
        trans = self._legacy_crf_layer.trans
        return joint_dice_nll_loss(y_pred, lengths, trans, y_true, alpha=self.alpha, smooth=self.smooth, reduction="none")


# ----- Functional forms retained for direct use -----

def dice_loss(
    potentials,
    lengths,
    transitions,
    y_true,
    smooth: float = 1.0,
    reduction: str = "mean",
):
    """
    Dice loss computed from CRF marginals.

    Args:
        potentials: [B, T, N]
        lengths: [B]
        transitions: [N, N]
        y_true: [B, T] integer tag ids
        smooth: Laplace smoothing factor for Dice numerator/denominator.
        reduction: "mean" (default), "sum", or "none".

    Returns:
        Loss tensor reduced per `reduction`. If reduction=="none", shape is [B].
    """
    probs = crf_marginals(potentials, lengths, transitions)  # [B,T,N]
    num_tags = K.shape(probs)[-1]
    y_oh = K.one_hot(K.cast(y_true, "int32"), num_tags)
    y_oh = K.cast(y_oh, probs.dtype)

    T = K.shape(probs)[1]
    time_idx = K.arange(T)
    mask_bt = K.expand_dims(time_idx, 0) < K.expand_dims(K.cast(lengths, "int32"), -1)
    mask_bt = K.cast(mask_bt, probs.dtype)
    mask_btn = K.expand_dims(mask_bt, -1)

    y_m = y_oh * mask_btn
    p_m = probs * mask_btn

    inter = K.sum(y_m * p_m, axis=(1, 2))
    sums = K.sum(y_m, axis=(1, 2)) + K.sum(p_m, axis=(1, 2))
    dice = (2.0 * inter + smooth) / (sums + smooth)
    loss = 1.0 - dice
    return _reduce(loss, reduction=reduction)


def joint_dice_nll_loss(
    potentials,
    lengths,
    transitions,
    y_true,
    alpha: float = 0.2,
    smooth: float = 1.0,
    reduction: str = "mean",
    sample_weight=None,
):
    """
    Weighted combination of NLL and Dice losses:
        L = alpha * NLL + (1 - alpha) * Dice

    Args:
        alpha: weight on NLL (0..1). Dice weight is (1 - alpha).
        smooth: smoothing for Dice.
        reduction: "mean" (default), "sum", or "none".

    Returns:
        Loss tensor reduced per `reduction`. If reduction=="none", shape is [B].
    """
    nll = nll_loss(
        potentials,
        lengths,
        transitions,
        y_true,
        sample_weight=sample_weight,
        reduction="none",
    )
    dce = dice_loss(potentials, lengths, transitions, y_true, smooth=smooth, reduction="none")
    alpha_t = K.convert_to_tensor(alpha, dtype="float32")
    loss = alpha_t * nll + (1.0 - alpha_t) * dce
    return _reduce(loss, reduction=reduction)
