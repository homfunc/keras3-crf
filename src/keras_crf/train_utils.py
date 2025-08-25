import numpy as np
import keras
from keras import ops as K

from .crf_ops import crf_log_likelihood
from . import CRF


def make_crf_tagger(tokens_input, features, num_tags, optimizer=None, metrics=None):
    """
    Build a training-ready Keras Model for sequence tagging with a CRF head.

    Inputs
    - tokens_input: a Keras Input (shape (None,)) for token IDs. Name will be normalized to 'tokens'.
    - features: a KerasTensor [B, T, D] from your encoder (e.g., Embedding + LSTM).
    - num_tags: number of tag classes.
    - optimizer: optional optimizer (default Adam(1e-3))
    - metrics: list of metrics to apply to the decoded_output head (optional)

    Returns
    - model: a compiled Model with two outputs:
        decoded_output: int tags [B, T] (metrics only)
        crf_log_likelihood_output: per-sample NLL [B] (drives training)
      The model expects inputs {'tokens': tokens, 'labels': labels} and outputs as above.
    """
    # Ensure named 'tokens' output for inputs dict convenience
    if tokens_input.name.split(':')[0] != 'tokens':
        tokens_named = keras.layers.Lambda(lambda z: z, name='tokens')(tokens_input)
    else:
        tokens_named = tokens_input

    crf = CRF(num_tags)
    decoded, potentials, lens, trans = crf(features)

    # Labels input
    labels = keras.Input(shape=(None,), dtype="int32", name="labels")

    # NLL head
    class CRF_NLL(keras.layers.Layer):
        def call(self, inputs):
            pot, y_true, ln, tr = inputs
            ll = crf_log_likelihood(pot, y_true, ln, tr)
            return -ll

        def compute_output_shape(self, input_shapes):
            pot_shape = input_shapes[0]
            B = pot_shape[0]
            return (B,)

    nll = CRF_NLL(name="nll_out")([potentials, labels, lens, trans])

    # Name outputs to align with compile/loss/metrics dict keys
    decoded_named = keras.layers.Lambda(lambda z: z, name="decoded_output")(decoded)
    nll_named = keras.layers.Lambda(lambda z: z, name="crf_log_likelihood_output")(nll)

    model = keras.Model(
        inputs={"tokens": tokens_named, "labels": labels},
        outputs={"decoded_output": decoded_named, "crf_log_likelihood_output": nll_named},
    )

    def zero_loss(y_true, y_pred):
        return K.mean(K.zeros_like(y_pred[..., :1]))

    model.compile(
        optimizer=optimizer or keras.optimizers.Adam(1e-3),
        loss={"decoded_output": zero_loss, "crf_log_likelihood_output": lambda y_true, y_pred: K.mean(y_pred)},
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
    y_dict = {"decoded_output": y_true, "crf_log_likelihood_output": y_dummy}
    sw_dict = {"decoded_output": sw_decoded, "crf_log_likelihood_output": sw_dummy}
    return y_dict, sw_dict
