# Keras 3 backend-independent CRF ops implemented with keras.ops
# These functions operate on potentials [B, T, N], tags [B, T], lens [B],
# and transitions [N, N], and should work across TF/JAX/Torch backends.

import keras
from keras import ops as K


def _length_mask(lens, max_len):
    # lens: [B]
    # returns mask [B, T] with True for t < lens
    lens = K.cast(lens, "int32")
    ar = K.expand_dims(K.cast(K.arange(max_len), "int32"), 0)
    lens = K.expand_dims(lens, -1)
    return K.less(ar, lens)


def _one_hot(x, depth):
    # Keras.ops.one_hot exists in Keras 3
    return K.one_hot(x, depth)


def crf_sequence_score(potentials, tags, lens, trans):
    # potentials: [B, T, N], tags: [B, T], lens: [B], trans: [N, N]
    B, T, N = K.shape(potentials)[0], K.shape(potentials)[1], K.shape(potentials)[2]
    # Unary score: one-hot gather and mask by length
    tags_oh = _one_hot(tags, N)  # [B, T, N]
    unary_scores = K.sum(potentials * K.cast(tags_oh, potentials.dtype), axis=-1)  # [B, T]
    m = _length_mask(lens, T)
    unary_scores = K.sum(unary_scores * K.cast(m, potentials.dtype), axis=-1)  # [B]

    # Binary score over transitions
    prev = tags[:, :-1]  # [B, T-1]
    curr = tags[:, 1:]   # [B, T-1]
    prev_oh = K.cast(_one_hot(prev, N), potentials.dtype)  # [B, T-1, N]
    curr_oh = K.cast(_one_hot(curr, N), potentials.dtype)  # [B, T-1, N]
    # prev @ trans -> [B, T-1, N]
    trans_applied = K.matmul(prev_oh, K.cast(trans, potentials.dtype))  # matmul handles [..., N] x [N, N]
    step_scores = K.sum(trans_applied * curr_oh, axis=-1)  # [B, T-1]
    lens_minus_one = lens - K.ones_like(lens)
    m2 = _length_mask(K.maximum(lens_minus_one, K.zeros_like(lens)), T - 1)  # [B, T-1]
    binary_scores = K.sum(step_scores * K.cast(m2, potentials.dtype), axis=-1)  # [B]

    return unary_scores + binary_scores


def crf_log_norm(potentials, lens, trans):
    """Forward algorithm (log-normalizer) with variable-length support.

    For TensorFlow, use while_loop to improve graph-mode compatibility.
    For JAX and Torch, use scan which performs well and compiles reliably.
    """
    bk = keras.config.backend()

    lens = K.cast(lens, "int32")
    B = K.shape(potentials)[0]
    N = K.shape(potentials)[2]
    T = K.shape(potentials)[1]

    trans_e = K.expand_dims(K.cast(trans, potentials.dtype), 0)  # [1, N, N]

    # Initial alpha at t=0
    alphas0 = potentials[:, 0, :]  # [B, N]

    # Fast path when T == 1
    T_static = potentials.shape[1]
    if isinstance(T_static, int) and T_static == 1:
        return K.logsumexp(alphas0, axis=-1)

    # Precompute emissions for t=1..T-1 (time-major) and validity mask
    ems = potentials[:, 1:, :]     # [B, T-1, N]
    ems_tm = K.transpose(ems, (1, 0, 2))  # [T-1, B, N]

    time_idx = K.arange(T)  # [T]
    valid_bt = (K.expand_dims(time_idx, 0) < K.expand_dims(lens, -1))  # [B, T]
    valid_tm = K.transpose(valid_bt[:, 1:], (1, 0))  # [T-1, B]

    if bk == "tensorflow":
        # Loop from t=1..T-1 using while_loop and gather via take()
        def cond(t, alphas):
            return t < (T - K.ones_like(T))
        def body(t, alphas):  # t counts number of steps processed so far in 1..T-1
            emit_t = K.take(ems_tm, t, axis=0)     # [B, N]
            valid_t = K.take(valid_tm, t, axis=0)  # [B]
            prev = K.expand_dims(alphas, 2)
            scores = prev + trans_e
            new_alphas = K.logsumexp(scores, axis=1) + emit_t
            alphas = K.where(K.expand_dims(valid_t, -1), new_alphas, alphas)
            return (t + 1, alphas)
        t0 = K.zeros((), dtype="int32")
        _, alphasT = K.while_loop(cond, body, (t0, alphas0))
        return K.logsumexp(alphasT, axis=-1)
    else:
        # Use scan for JAX and Torch
        def step(alphas, inputs):
            emit_t, valid_t = inputs  # [B,N], [B]
            prev = K.expand_dims(alphas, 2)  # [B, N, 1]
            scores = prev + trans_e          # [B, N, N]
            new_alphas = K.logsumexp(scores, axis=1) + emit_t  # [B, N]
            alphas = K.where(K.expand_dims(valid_t, -1), new_alphas, alphas)
            return alphas, new_alphas
        alphasT, _ = K.scan(step, alphas0, xs=(ems_tm, valid_tm))
        return K.logsumexp(alphasT, axis=-1)


def crf_log_likelihood(potentials, tags, lens, trans):
    seq_scores = crf_sequence_score(potentials, tags, lens, trans)
    log_norm = crf_log_norm(potentials, lens, trans)
    return seq_scores - log_norm


def crf_decode(potentials, lens, trans):
    """Backend-agnostic Viterbi decode using keras.ops.scan for forward and backtrace.

    Args:
        potentials: [B, T, N]
        lens: [B]
        trans: [N, N]

    Returns:
        tags: [B, T] int32 decoded sequence
        best_score: [B] float score of best path
    """
    lens = K.cast(lens, "int32")
    B = K.shape(potentials)[0]
    T = K.shape(potentials)[1]
    N = K.shape(potentials)[2]

    # Fast path for single-timestep sequences to avoid scan with empty xs on some backends
    T_static = potentials.shape[1]
    if isinstance(T_static, int) and T_static == 1:
        alphas0 = potentials[:, 0, :]
        best_score = K.max(alphas0, axis=-1)
        last_tags = K.argmax(alphas0, axis=-1)
        tags_out = K.expand_dims(K.cast(last_tags, "int32"), 1)
        return tags_out, best_score

    trans_e = K.expand_dims(K.cast(trans, potentials.dtype), 0)  # [1,N,N]

    # Emissions time-major for t=1..T-1 and validity mask
    ems = potentials[:, 1:, :]        # [B,T-1,N]
    ems_tm = K.transpose(ems, (1, 0, 2))  # [T-1,B,N]
    time_idx = K.arange(T)  # [T]
    valid_bt = (K.expand_dims(time_idx, 0) < K.expand_dims(lens, -1))  # [B,T]
    valid_tm = K.transpose(valid_bt[:, 1:], (1, 0))  # [T-1,B]

    # Forward scan to compute alphas and backpointers using a pure functional style
    # to avoid in-place slice updates (which trigger warnings on Torch backend).
    alphas0 = potentials[:, 0, :]  # [B,N]

    # Return backpointers as float ys to keep structure/dtype uniform across backends,
    # cast to int32 after the scan completes.
    pot_dtype = potentials.dtype

    def fwd_step(alphas, inputs):
        emit_t, valid_t = inputs  # [B,N], [B]
        prev = K.expand_dims(alphas, 2)  # [B,N,1]
        scores = prev + trans_e          # [B,N,N]
        best_prev = K.argmax(scores, axis=1)           # [B,N]
        alphas_new = K.max(scores, axis=1) + emit_t    # [B,N]
        alphas = K.where(K.expand_dims(valid_t, -1), alphas_new, alphas)
        return alphas, K.cast(best_prev, pot_dtype)

    alphasT, backp_tm = K.scan(fwd_step, alphas0, xs=(ems_tm, valid_tm))
    backp_tm = K.cast(backp_tm, "int32")

    # Last tags and best score
    best_score = K.max(alphasT, axis=-1)   # [B]
    last_tags = K.argmax(alphasT, axis=-1) # [B]

    # Backtrace with reverse scan over backpointers
    # valid_prev for updating last_tags: times t=0..T-2 where t < lens-1
    prev_bt = (K.expand_dims(time_idx[:-1], 0) < K.expand_dims(lens - K.ones_like(lens), -1))  # [B,T-1]
    prev_tm = K.transpose(prev_bt, (1, 0))  # [T-1,B]

    def bwd_step(last_t, inputs):
        bp_t, valid_prev = inputs  # bp_t: [B,N], valid_prev: [B]
        tags_t = last_t  # record current tag before moving to previous
        one_hot_idx = K.one_hot(last_t, N)  # [B,N]
        prev_tag_f = K.sum(K.cast(bp_t, potentials.dtype) * K.cast(one_hot_idx, potentials.dtype), axis=1)
        prev_tag = K.cast(prev_tag_f, "int32")
        last_t = K.where(valid_prev, prev_tag, last_t)
        return last_t, tags_t  # carry updated, output current

    last0 = last_tags
    last_final, tags_rev_tm = K.scan(bwd_step, last0, xs=(backp_tm, prev_tm), reverse=True)
    # tags_rev_tm is in chronological order for t=1..T-1; prepend tag for t=0 (last_final)
    t0 = K.expand_dims(last_final, 0)           # [1,B]
    tags_tm = K.concatenate([t0, tags_rev_tm], axis=0)  # [T,B]
    tags_out = K.transpose(tags_tm, (1, 0))  # [B,T]

    return tags_out, best_score


def crf_filtered_inputs(potentials, tag_bitmap):
    # Replace disallowed positions with a large negative constant to strictly forbid them
    # without relying on undefined divide-by-zero behavior that can upset some compilers.
    neg_inf = K.convert_to_tensor(-1e30, dtype=potentials.dtype)
    fill = K.zeros_like(potentials) + neg_inf
    return K.where(tag_bitmap, potentials, fill)


def crf_constrained_decode(potentials, tag_bitmap, lens, trans):
    filtered = crf_filtered_inputs(potentials, tag_bitmap)
    return crf_decode(filtered, lens, trans)


def crf_marginals(potentials, lens, trans):
    """Compute per-token marginals p(y_t = k | x) via forward-backward using keras.ops.scan.

    Args:
      potentials: [B, T, N]
      lens: [B]
      trans: [N, N]

    Returns:
      probs: [B, T, N] where probs[b, t, :] sums to 1 for t < lens[b]; zeros for padded steps.
    """
    lens = K.cast(lens, "int32")
    B = K.shape(potentials)[0]
    T = K.shape(potentials)[1]
    N = K.shape(potentials)[2]
    trans_e = K.expand_dims(K.cast(trans, potentials.dtype), 0)  # [1,N,N]

    # Forward scan to compute log-alpha at each time
    a0 = potentials[:, 0, :]  # [B,N]
    ems = potentials[:, 1:, :]  # [B,T-1,N]
    ems_tm = K.transpose(ems, (1, 0, 2))  # [T-1,B,N]

    time_idx = K.arange(T)
    valid_bt = (K.expand_dims(time_idx, 0) < K.expand_dims(lens, -1))  # [B,T]
    valid_tm = K.transpose(valid_bt[:, 1:], (1, 0))  # [T-1,B]

    def fwd_step(a_t, inputs):
        emit_t, valid_t = inputs  # [B,N], [B]
        prev = K.expand_dims(a_t, 2)
        scores = prev + trans_e
        new_a = K.logsumexp(scores, axis=1) + emit_t  # [B,N]
        a_t = K.where(K.expand_dims(valid_t, -1), new_a, a_t)
        return a_t, a_t

    aT, a_seq_tm_tail = K.scan(fwd_step, a0, xs=(ems_tm, valid_tm))
    a0_tm = K.expand_dims(a0, 0)  # [1,B,N]
    log_alpha_tm = K.concatenate([a0_tm, a_seq_tm_tail], axis=0)  # [T,B,N]
    log_alpha = K.transpose(log_alpha_tm, (1, 0, 2))  # [B,T,N]

    # Backward scan to compute log-beta at each time
    # We scan over emit at t+1 (i.e., potentials[:, 1:, :]) in reverse to get beta at t
    emit_tp1_tm = ems_tm  # [T-1,B,N]
    valid_prev_bt = (K.expand_dims(time_idx[:-1], 0) < K.expand_dims(lens - K.ones_like(lens), -1))  # [B,T-1], t < len-1
    valid_prev_tm = K.transpose(valid_prev_bt, (1, 0))  # [T-1,B]

    def bwd_step(b_t, inputs):
        emit_tp1, valid_prev = inputs  # [B,N], [B]
        nxt = b_t + emit_tp1           # [B,N]
        nxt_e = K.expand_dims(nxt, 1)  # [B,1,N]
        scores = trans_e + nxt_e       # [B,N,N]
        new_b = K.logsumexp(scores, axis=2)  # [B,N]
        b_t = K.where(K.expand_dims(valid_prev, -1), new_b, b_t)
        return b_t, b_t

    bTminus1 = K.zeros((B, N), dtype=potentials.dtype)
    _, b_seq_tm_rev = K.scan(bwd_step, bTminus1, xs=(emit_tp1_tm, valid_prev_tm), reverse=True)
    # b_seq_tm_rev has beta at t in chronological order for t=0..T-2; append beta at T-1 zeros at end
    bTminus1_tm = K.expand_dims(bTminus1, 0)  # [1,B,N]
    log_beta_tm = K.concatenate([b_seq_tm_rev, bTminus1_tm], axis=0)  # [T,B,N]
    log_beta = K.transpose(log_beta_tm, (1, 0, 2))  # [B,T,N]

    logZ = crf_log_norm(potentials, lens, trans)  # [B]
    log_m = log_alpha + log_beta - K.reshape(logZ, (B, 1, 1))  # [B,T,N]
    probs = K.exp(log_m)

    # Zero out padded positions
    mask_bt = (K.expand_dims(time_idx, 0) < K.expand_dims(lens, -1))  # [B,T]
    probs = probs * K.cast(K.expand_dims(mask_bt, -1), probs.dtype)
    return probs
