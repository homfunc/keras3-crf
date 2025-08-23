# Standalone CRF text ops adapted from TensorFlow Addons
# Minimal edits for Keras 3 compatibility and local imports

import warnings
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf

from .utils.types import TensorLike
from .rnn.abstract_rnn_cell import AbstractRNNCell
from typeguard import typechecked


def crf_filtered_inputs(inputs: TensorLike, tag_bitmap: TensorLike) -> tf.Tensor:
    filtered_inputs = tf.where(
        tag_bitmap,
        inputs,
        tf.fill(tf.shape(inputs), tf.cast(float("-inf"), inputs.dtype)),
    )
    return filtered_inputs


def crf_unary_score(
    tag_indices: TensorLike, sequence_lengths: TensorLike, inputs: TensorLike
) -> tf.Tensor:
    tag_indices = tf.cast(tag_indices, dtype=tf.int32)
    sequence_lengths = tf.cast(sequence_lengths, dtype=tf.int32)

    batch_size = tf.shape(inputs)[0]
    max_seq_len = tf.shape(inputs)[1]
    num_tags = tf.shape(inputs)[2]

    flattened_inputs = tf.reshape(inputs, [-1])

    offsets = tf.expand_dims(tf.range(batch_size) * max_seq_len * num_tags, 1)
    offsets += tf.expand_dims(tf.range(max_seq_len) * num_tags, 0)
    if tag_indices.dtype == tf.int64:
        offsets = tf.cast(offsets, tf.int64)
    flattened_tag_indices = tf.reshape(offsets + tag_indices, [-1])

    unary_scores = tf.reshape(
        tf.gather(flattened_inputs, flattened_tag_indices), [batch_size, max_seq_len]
    )

    masks = tf.sequence_mask(
        sequence_lengths, maxlen=tf.shape(tag_indices)[1], dtype=unary_scores.dtype
    )
    unary_scores = tf.reduce_sum(unary_scores * masks, 1)
    return unary_scores


def crf_binary_score(
    tag_indices: TensorLike, sequence_lengths: TensorLike, transition_params: TensorLike
) -> tf.Tensor:
    tag_indices = tf.cast(tag_indices, dtype=tf.int32)
    sequence_lengths = tf.cast(sequence_lengths, dtype=tf.int32)

    num_tags = tf.shape(transition_params)[0]
    num_transitions = tf.shape(tag_indices)[1] - 1

    start_tag_indices = tf.slice(tag_indices, [0, 0], [-1, num_transitions])
    end_tag_indices = tf.slice(tag_indices, [0, 1], [-1, num_transitions])

    flattened_transition_indices = start_tag_indices * num_tags + end_tag_indices
    flattened_transition_params = tf.reshape(transition_params, [-1])

    binary_scores = tf.gather(flattened_transition_params, flattened_transition_indices)

    masks = tf.sequence_mask(
        sequence_lengths, maxlen=tf.shape(tag_indices)[1], dtype=binary_scores.dtype
    )
    truncated_masks = tf.slice(masks, [0, 1], [-1, -1])
    binary_scores = tf.reduce_sum(binary_scores * truncated_masks, 1)
    return binary_scores


def crf_sequence_score(
    inputs: TensorLike,
    tag_indices: TensorLike,
    sequence_lengths: TensorLike,
    transition_params: TensorLike,
) -> tf.Tensor:
    tag_indices = tf.cast(tag_indices, dtype=tf.int32)
    sequence_lengths = tf.cast(sequence_lengths, dtype=tf.int32)

    def _single_seq_fn():
        batch_size = tf.shape(inputs, out_type=tf.int32)[0]
        batch_inds = tf.reshape(tf.range(batch_size), [-1, 1])
        indices = tf.concat([batch_inds, tf.zeros_like(batch_inds)], axis=1)

        tag_inds = tf.gather_nd(tag_indices, indices)
        tag_inds = tf.reshape(tag_inds, [-1, 1])
        indices = tf.concat([indices, tag_inds], axis=1)

        sequence_scores = tf.gather_nd(inputs, indices)
        sequence_scores = tf.where(
            tf.less_equal(sequence_lengths, 0),
            tf.zeros_like(sequence_scores),
            sequence_scores,
        )
        return sequence_scores

    def _multi_seq_fn():
        unary_scores = crf_unary_score(tag_indices, sequence_lengths, inputs)
        binary_scores = crf_binary_score(
            tag_indices, sequence_lengths, transition_params
        )
        sequence_scores = unary_scores + binary_scores
        return sequence_scores

    return tf.cond(tf.equal(tf.shape(inputs)[1], 1), _single_seq_fn, _multi_seq_fn)


def crf_log_norm(
    inputs: TensorLike, sequence_lengths: TensorLike, transition_params: TensorLike
) -> tf.Tensor:
    sequence_lengths = tf.cast(sequence_lengths, dtype=tf.int32)
    first_input = tf.slice(inputs, [0, 0, 0], [-1, 1, -1])
    first_input = tf.squeeze(first_input, [1])

    def _single_seq_fn():
        log_norm = tf.reduce_logsumexp(first_input, [1])
        log_norm = tf.where(
            tf.less_equal(sequence_lengths, 0), tf.zeros_like(log_norm), log_norm
        )
        return log_norm

    def _multi_seq_fn():
        rest_of_input = tf.slice(inputs, [0, 1, 0], [-1, -1, -1])
        alphas = crf_forward(rest_of_input, first_input, transition_params, sequence_lengths)
        log_norm = tf.reduce_logsumexp(alphas, [1])
        log_norm = tf.where(
            tf.less_equal(sequence_lengths, 0), tf.zeros_like(log_norm), log_norm
        )
        return log_norm

    return tf.cond(tf.equal(tf.shape(inputs)[1], 1), _single_seq_fn, _multi_seq_fn)


def crf_log_likelihood(
    inputs: TensorLike,
    tag_indices: TensorLike,
    sequence_lengths: TensorLike,
    transition_params: Optional[TensorLike] = None,
) -> Tuple[tf.Tensor, tf.Tensor]:
    inputs = tf.convert_to_tensor(inputs)
    num_tags = inputs.shape[2]

    tag_indices = tf.cast(tag_indices, dtype=tf.int32)
    sequence_lengths = tf.cast(sequence_lengths, dtype=tf.int32)

    if transition_params is None:
        initializer = tf.keras.initializers.GlorotUniform()
        transition_params = tf.Variable(initializer([num_tags, num_tags]), "transitions")
    transition_params = tf.cast(transition_params, inputs.dtype)

    sequence_scores = crf_sequence_score(
        inputs, tag_indices, sequence_lengths, transition_params
    )
    log_norm = crf_log_norm(inputs, sequence_lengths, transition_params)
    log_likelihood = sequence_scores - log_norm
    return log_likelihood, transition_params


def crf_forward(
    inputs: TensorLike,
    state: TensorLike,
    transition_params: TensorLike,
    sequence_lengths: TensorLike,
) -> tf.Tensor:
    sequence_lengths = tf.cast(sequence_lengths, dtype=tf.int32)

    last_index = tf.maximum(tf.constant(0, dtype=sequence_lengths.dtype), sequence_lengths - 1)
    inputs = tf.transpose(inputs, [1, 0, 2])
    transition_params = tf.expand_dims(transition_params, 0)

    def _scan_fn(_state, _inputs):
        _state = tf.expand_dims(_state, 2)
        transition_scores = _state + transition_params
        new_alphas = _inputs + tf.reduce_logsumexp(transition_scores, [1])
        return new_alphas

    all_alphas = tf.transpose(tf.scan(_scan_fn, inputs, state), [1, 0, 2])
    all_alphas = tf.concat([tf.expand_dims(state, 1), all_alphas], 1)

    idxs = tf.stack([tf.range(tf.shape(last_index)[0]), last_index], axis=1)
    return tf.gather_nd(all_alphas, idxs)


class CrfDecodeForwardRnnCell(AbstractRNNCell):
    @typechecked
    def __init__(self, transition_params: TensorLike, **kwargs):
        super().__init__(**kwargs)
        self._transition_params = tf.expand_dims(transition_params, 0)
        self._num_tags = transition_params.shape[0]

    @property
    def state_size(self):
        return self._num_tags

    @property
    def output_size(self):
        return self._num_tags

    def call(self, inputs, state):
        state = tf.expand_dims(state[0], 2)
        transition_scores = state + tf.cast(self._transition_params, self.compute_dtype)
        new_state = inputs + tf.reduce_max(transition_scores, [1])
        backpointers = tf.argmax(transition_scores, 1)
        backpointers = tf.cast(backpointers, dtype=tf.int32)
        return backpointers, new_state


def crf_decode_forward(
    inputs: TensorLike,
    state: TensorLike,
    transition_params: TensorLike,
    sequence_lengths: TensorLike,
) -> tf.Tensor:
    sequence_lengths = tf.cast(sequence_lengths, dtype=tf.int32)
    mask = tf.sequence_mask(sequence_lengths, tf.shape(inputs)[1])
    crf_fwd_cell = CrfDecodeForwardRnnCell(transition_params, dtype=inputs.dtype)
    crf_fwd_layer = tf.keras.layers.RNN(
        crf_fwd_cell,
        return_sequences=True,
        return_state=True,
        dtype=inputs.dtype,
        zero_output_for_mask=True,
    )
    backpointers, last_score = crf_fwd_layer(inputs, state, mask=mask)
    backpointers = tf.cast(backpointers, tf.int32)
    return backpointers, last_score


def crf_decode_backward(inputs: TensorLike, state: TensorLike) -> tf.Tensor:
    inputs = tf.transpose(inputs, [1, 0, 2])

    def _scan_fn(state, inputs):
        state = tf.squeeze(state, axis=[1])
        idxs = tf.stack([tf.range(tf.shape(inputs)[0]), state], axis=1)
        new_tags = tf.expand_dims(tf.gather_nd(inputs, idxs), axis=-1)
        return new_tags

    return tf.transpose(tf.scan(_scan_fn, inputs, state), [1, 0, 2])


def crf_decode(
    potentials: TensorLike, transition_params: TensorLike, sequence_length: TensorLike
) -> tf.Tensor:
    if tf.__version__[:3] == "2.4":
        warnings.warn("CRF Decoding is known to have issues with KerasTensors in TF2.4.")

    sequence_length = tf.cast(sequence_length, dtype=tf.int32)

    def _single_seq_fn():
        decode_tags = tf.cast(tf.argmax(potentials, axis=2), dtype=tf.int32)
        best_score = tf.reshape(tf.reduce_max(potentials, axis=2), shape=[-1])
        return decode_tags, best_score

    def _multi_seq_fn():
        initial_state = tf.slice(potentials, [0, 0, 0], [-1, 1, -1])
        initial_state = tf.squeeze(initial_state, axis=[1])
        inputs = tf.slice(potentials, [0, 1, 0], [-1, -1, -1])

        sequence_length_less_one = tf.maximum(tf.constant(0, dtype=tf.int32), sequence_length - 1)

        backpointers, last_score = crf_decode_forward(
            inputs, initial_state, transition_params, sequence_length_less_one
        )

        backpointers = tf.reverse_sequence(backpointers, sequence_length_less_one, seq_axis=1)

        initial_state = tf.cast(tf.argmax(last_score, axis=1), dtype=tf.int32)
        initial_state = tf.expand_dims(initial_state, axis=-1)

        decode_tags = crf_decode_backward(backpointers, initial_state)
        decode_tags = tf.squeeze(decode_tags, axis=[2])
        decode_tags = tf.concat([initial_state, decode_tags], axis=1)
        decode_tags = tf.reverse_sequence(decode_tags, sequence_length, seq_axis=1)

        best_score = tf.reduce_max(last_score, axis=1)
        return decode_tags, best_score

    if potentials.shape[1] is not None:
        if potentials.shape[1] == 1:
            return _single_seq_fn()
        else:
            return _multi_seq_fn()
    else:
        return tf.cond(tf.equal(tf.shape(potentials)[1], 1), _single_seq_fn, _multi_seq_fn)


def crf_constrained_decode(
    potentials: TensorLike,
    tag_bitmap: TensorLike,
    transition_params: TensorLike,
    sequence_length: TensorLike,
) -> tf.Tensor:
    filtered_potentials = crf_filtered_inputs(potentials, tag_bitmap)
    return crf_decode(filtered_potentials, transition_params, sequence_length)

