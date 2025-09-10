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
    """Forward algorithm (log-normalizer) with true variable-length support using keras.ops.while_loop.

    Args:
        potentials: [B, T, N]
        lens: [B]
        trans: [N, N]

    Returns:
        logZ: [B]
    """
    lens = K.cast(lens, "int32")
    B = K.shape(potentials)[0]
    T = K.shape(potentials)[1]
    N = K.shape(potentials)[2]

    # Initial alpha at t=0
    alphas0 = potentials[:, 0, :]  # [B, N]
    trans_e = K.expand_dims(K.cast(trans, potentials.dtype), 0)  # [1, N, N]

    # Precompute time ids once to avoid dynamic indexing ops inside the loop
    time_ids = K.arange(T)  # [T]

    def cond(t, _alphas):
        return K.less(t, T)

    def body(t, alphas):  # t: int32 scalar, alphas: [B, N]
        # Build a mask to extract potentials[:, t, :] without tensor indexing
        eq = K.equal(time_ids, t)          # [T]
        step_mask = K.cast(eq, potentials.dtype)  # [T]
        step_mask = K.reshape(step_mask, (1, T, 1))  # [1, T, 1]
        emit_t = K.sum(potentials * step_mask, axis=1)  # [B, N]

        prev = K.expand_dims(alphas, 2)  # [B, N, 1]
        scores = prev + trans_e          # [B, N, N] (i -> j)
        new_alphas = K.logsumexp(scores, axis=(1,)) + emit_t  # [B, N]

        # Only update sequences where t < len
        cond_b = K.expand_dims(K.less(t, lens), -1)  # [B, 1]
        alphas = K.where(cond_b, new_alphas, alphas)
        return (t + K.convert_to_tensor(1, dtype="int32"), alphas)

    _, alphasT = K.while_loop(
        cond=cond,
        body=body,
        loop_vars=(K.convert_to_tensor(1, dtype="int32"), alphas0),
    )

    return K.logsumexp(alphasT, axis=(-1,))  # [B]


def crf_log_likelihood(potentials, tags, lens, trans):
    seq_scores = crf_sequence_score(potentials, tags, lens, trans)
    log_norm = crf_log_norm(potentials, lens, trans)
    return seq_scores - log_norm


def crf_decode(potentials, lens, trans):
    """Backend-agnostic Viterbi decode using keras.ops.while_loop.

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

    trans_e = K.expand_dims(K.cast(trans, potentials.dtype), 0)  # [1,N,N]

    # Forward pass to compute alphas and backpointers for each time
    alphas0 = potentials[:, 0, :]  # [B,N]
    time_ids = K.arange(T)  # [T]

    # Preallocate backpointers: [T, B, N]; t=0 stays zeros
    backp_all = K.zeros((T, B, N), dtype="int32")

    def cond_fwd(t, _alphas, _backp):
        return K.less(t, T)

    def body_fwd(t, alphas, backp):
        # emit at time t
        eq = K.equal(time_ids, t)  # [T]
        step_mask = K.cast(eq, potentials.dtype)  # [T]
        step_mask = K.reshape(step_mask, (1, T, 1))  # [1,T,1]
        emit_t = K.sum(potentials * step_mask, axis=1)  # [B,N]

        prev = K.expand_dims(alphas, 2)   # [B,N,1]
        scores = prev + trans_e           # [B,N,N]
        best_prev = K.argmax(scores, axis=1)           # [B,N] int
        alphas_new = K.max(scores, axis=1) + emit_t    # [B,N]

        cond_b = K.expand_dims(K.less(t, lens), -1)    # [B,1]
        alphas = K.where(cond_b, alphas_new, alphas)   # [B,N]

        # Write backpointer at time t
        mask_t = K.cast(K.reshape(eq, (T, 1, 1)), backp.dtype)  # [T,1,1]
        best_prev_i32 = K.cast(best_prev, backp.dtype)          # [B,N]
        best_prev_e = K.expand_dims(best_prev_i32, 0)           # [1,B,N]
        backp = backp * (K.ones_like(backp) - mask_t) + best_prev_e * mask_t  # [T,B,N]

        return (t + K.convert_to_tensor(1, dtype="int32"), alphas, backp)

    _, alphasT, backp_all = K.while_loop(
        cond=cond_fwd,
        body=body_fwd,
        loop_vars=(K.convert_to_tensor(1, dtype="int32"), alphas0, backp_all),
    )

    # Best score and last tags at T-1
    best_score = K.max(alphasT, axis=-1)   # [B]
    last_tags = K.argmax(alphasT, axis=-1) # [B]

    # Backward pass: build tags_out [B,T]
    tags_out = K.zeros((B, T), dtype="int32")

    def cond_bwd(t, _last_tags, _tags_out):
        return K.greater_equal(t, K.convert_to_tensor(0, dtype="int32"))

    def body_bwd(t, last_t, tags):
        # Write tag at time t
        eq = K.equal(time_ids, t)  # [T]
        mask_t = K.cast(K.reshape(eq, (1, T)), tags.dtype)  # [1,T]
        last_e = K.expand_dims(last_t, 1)  # [B,1]
        tags = tags * (K.ones_like(tags) - mask_t) + last_e * mask_t

        # Update last_t for previous time (t-1)
        # Select backpointer slice at time t without tensor indexing
        eq_t = K.equal(time_ids, t)            # [T]
        mask_t3 = K.cast(K.reshape(eq_t, (T, 1, 1)), backp_all.dtype)  # [T,1,1]
        bp_t = K.sum(backp_all * mask_t3, axis=0)  # [B,N]
        one_hot_idx = K.one_hot(last_t, N)  # [B,N]
        prev_tag_f = K.sum(K.cast(bp_t, "float32") * K.cast(one_hot_idx, "float32"), axis=1)
        prev_tag = K.cast(prev_tag_f, "int32")
        cond_update = K.less(t, lens)  # [B]
        last_t = K.where(cond_update, prev_tag, last_t)

        return (t - K.convert_to_tensor(1, dtype="int32"), last_t, tags)

    _, last_tags, tags_out = K.while_loop(
        cond=cond_bwd,
        body=body_bwd,
        loop_vars=(T - K.convert_to_tensor(1, dtype="int32"), last_tags, tags_out),
    )

    return tags_out, best_score


def crf_filtered_inputs(potentials, tag_bitmap):
    # Replace disallowed positions with -inf to strictly forbid them (matches tests)
    neg_inf = K.convert_to_tensor(-1.0, dtype=potentials.dtype) / K.convert_to_tensor(0.0, dtype=potentials.dtype)
    fill = K.zeros_like(potentials) + neg_inf
    return K.where(tag_bitmap, potentials, fill)


def crf_constrained_decode(potentials, tag_bitmap, lens, trans):
    filtered = crf_filtered_inputs(potentials, tag_bitmap)
    return crf_decode(filtered, lens, trans)


def crf_marginals(potentials, lens, trans):
    """Compute per-token marginals p(y_t = k | x) via forward-backward using keras.ops.while_loop.

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

    time_ids = K.arange(T)  # [T]

    # Forward pass: log-alpha per time, write into alpha_out [B,T,N]
    alpha_out = K.zeros((B, T, N), dtype=potentials.dtype)
    a_t0 = potentials[:, 0, :]  # [B,N]
    # write t=0
    mask0 = K.cast(K.reshape(K.equal(time_ids, 0), (1, T, 1)), potentials.dtype)
    alpha_out = alpha_out * (K.ones_like(alpha_out) - mask0) + K.expand_dims(a_t0, 1) * mask0

    def cond_fwd(t, _a_t, _alpha_out):
        return K.less(t, T)

    def body_fwd(t, a_t, alpha_out):
        # emit at time t
        eq = K.equal(time_ids, t)
        step_mask = K.cast(eq, potentials.dtype)
        step_mask = K.reshape(step_mask, (1, T, 1))
        emit_t = K.sum(potentials * step_mask, axis=1)  # [B,N]

        prev = K.expand_dims(a_t, 2)  # [B,N,1]
        scores = prev + trans_e       # [B,N,N]
        new_a = K.logsumexp(scores, axis=1) + emit_t  # [B,N]
        cond_b = K.expand_dims(K.less(t, lens), -1)   # [B,1]
        a_t = K.where(cond_b, new_a, a_t)

        # write alpha at time t
        mask_t = K.cast(K.reshape(eq, (1, T, 1)), alpha_out.dtype)
        alpha_out = alpha_out * (K.ones_like(alpha_out) - mask_t) + K.expand_dims(a_t, 1) * mask_t
        return (t + K.convert_to_tensor(1, dtype="int32"), a_t, alpha_out)

    _, _, log_alpha = K.while_loop(
        cond=cond_fwd,
        body=body_fwd,
        loop_vars=(K.convert_to_tensor(1, dtype="int32"), a_t0, alpha_out),
    )

    # Backward pass: log-beta per time into beta_out [B,T,N]
    beta_out = K.zeros((B, T, N), dtype=potentials.dtype)
    b_t_Tm1 = K.zeros((B, N), dtype=potentials.dtype)  # beta at T-1
    # write t=T-1
    maskTm1 = K.cast(K.reshape(K.equal(time_ids, T - K.convert_to_tensor(1, dtype="int32")), (1, T, 1)), potentials.dtype)
    beta_out = beta_out * (K.ones_like(beta_out) - maskTm1) + K.expand_dims(b_t_Tm1, 1) * maskTm1

    def cond_bwd(t, _b_t, _beta_out):
        return K.greater_equal(t, K.convert_to_tensor(0, dtype="int32"))

    def body_bwd(t, b_t, beta_out):
        # nxt = b_{t+1} + emit at t+1
        eq_tp1 = K.equal(time_ids, t + K.convert_to_tensor(1, dtype="int32"))
        step_mask = K.cast(eq_tp1, potentials.dtype)
        step_mask = K.reshape(step_mask, (1, T, 1))
        emit_tp1 = K.sum(potentials * step_mask, axis=1)  # [B,N]
        nxt = b_t + emit_tp1

        nxt_e = K.expand_dims(nxt, 1)  # [B,1,N]
        scores = trans_e + nxt_e       # [B,N,N]
        new_b = K.logsumexp(scores, axis=2)  # [B,N]
        cond_b = K.expand_dims(K.less(t, lens - K.ones_like(lens)), -1)  # t < len-1
        b_t = K.where(cond_b, new_b, b_t)

        # write beta at time t
        eq_t = K.equal(time_ids, t)
        mask_t = K.cast(K.reshape(eq_t, (1, T, 1)), beta_out.dtype)
        beta_out = beta_out * (K.ones_like(beta_out) - mask_t) + K.expand_dims(b_t, 1) * mask_t

        return (t - K.convert_to_tensor(1, dtype="int32"), b_t, beta_out)

    _, _, log_beta = K.while_loop(
        cond=cond_bwd,
        body=body_bwd,
        loop_vars=(T - K.convert_to_tensor(2, dtype="int32"), b_t_Tm1, beta_out),
    )

    logZ = crf_log_norm(potentials, lens, trans)  # [B]
    logZ_e = K.reshape(logZ, (B, 1, 1))

    log_m = log_alpha + log_beta - logZ_e  # [B,T,N]
    probs = K.exp(log_m)

    # Zero out padded positions
    time_idx = K.cast(time_ids, "int32")
    mask_bt = K.expand_dims(time_idx, 0) < K.expand_dims(lens, -1)  # [B,T]
    probs = probs * K.cast(K.expand_dims(mask_bt, -1), probs.dtype)
    return probs
