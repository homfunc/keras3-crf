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
    return ar < lens


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
    # Forward algorithm with masking
    lens = K.cast(lens, "int32")
    B, T, N = K.shape(potentials)[0], K.shape(potentials)[1], K.shape(potentials)[2]
    alphas = potentials[:, 0, :]  # [B, N]
    trans_e = K.expand_dims(K.cast(trans, potentials.dtype), 0)  # [1, N, N]
    # Iterate time
    t = 1
    while t < T:
        prev = K.expand_dims(alphas, 2)  # [B, N, 1]
        scores = prev + trans_e  # [B, N, N]
        new_alphas = K.logsumexp(scores, axis=(1,)) + potentials[:, t, :]
        cond = K.expand_dims((t < lens), -1)  # [B, 1]
        alphas = K.where(cond, new_alphas, alphas)
        t = t + 1
    return K.logsumexp(alphas, axis=(-1,))  # [B]


def crf_log_likelihood(potentials, tags, lens, trans):
    seq_scores = crf_sequence_score(potentials, tags, lens, trans)
    log_norm = crf_log_norm(potentials, lens, trans)
    return seq_scores - log_norm


def crf_decode(potentials, lens, trans):
    # Viterbi decode with masking, returns (tags [B, T], best_score [B])
    lens = K.cast(lens, "int32")
    B, T, N = K.shape(potentials)[0], K.shape(potentials)[1], K.shape(potentials)[2]
    trans_e = K.expand_dims(K.cast(trans, potentials.dtype), 0)

    alphas = potentials[:, 0, :]  # [B, N]
    backp_list = [K.zeros((B, N), dtype="int32")]  # placeholder for t=0

    t = 1
    while t < T:
        prev = K.expand_dims(alphas, 2)  # [B, N, 1]
        scores = prev + trans_e  # [B, N, N]
        best_prev = K.argmax(scores, axis=1)  # [B, N]
        alphas_new = K.max(scores, axis=1) + potentials[:, t, :]
        cond = K.expand_dims((t < lens), -1)
        alphas = K.where(cond, alphas_new, alphas)
        backp_list.append(best_prev)
        t = t + 1

    # Backtrack
    last_tags = K.argmax(alphas, axis=-1)  # [B]
    best_score = K.max(alphas, axis=-1)
    decoded_rev = [last_tags]
    t = T - 1
    while t > 0:
        bp_t = backp_list[t]  # [B, N]
        one_hot_idx = K.one_hot(last_tags, N)  # [B, N]
        prev_tag_f = K.sum(K.cast(bp_t, "float32") * K.cast(one_hot_idx, "float32"), axis=1)
        prev_tag = K.cast(prev_tag_f, "int32")
        cond = (t < lens)
        last_tags = K.where(cond, prev_tag, last_tags)
        decoded_rev.append(last_tags)
        t = t - 1
    # Reverse and stack
    decoded_rev = decoded_rev[::-1]
    tags = K.stack(decoded_rev, axis=1)  # [B, T]
    # For positions >= lens, fill with last valid tag (at lens-1)
    time_idx = K.cast(K.arange(T), "int32")
    mask_valid = K.expand_dims(time_idx, 0) < K.expand_dims(lens, -1)  # [B, T]
    last_idx = lens - K.ones_like(lens)
    oh_last = K.one_hot(last_idx, T)  # [B, T]
    last_tag_valid_f = K.sum(K.cast(tags, "float32") * K.cast(oh_last, "float32"), axis=1)  # [B]
    last_tag_valid = K.cast(last_tag_valid_f, "int32")
    rep_last = K.expand_dims(last_tag_valid, 1) + K.zeros_like(tags)
    mask_valid_i = K.cast(mask_valid, "int32")
    mask_invalid_i = K.ones_like(mask_valid_i) - mask_valid_i
    tags = tags * mask_valid_i + rep_last * mask_invalid_i
    return tags, best_score


def crf_filtered_inputs(potentials, tag_bitmap):
    # Replace disallowed positions with a very negative number to effectively mask them
    return K.where(tag_bitmap, potentials, K.full_like(potentials, -1e30))


def crf_constrained_decode(potentials, tag_bitmap, lens, trans):
    filtered = crf_filtered_inputs(potentials, tag_bitmap)
    return crf_decode(filtered, lens, trans)


def crf_marginals(potentials, lens, trans):
    """Compute per-token marginals p(y_t = k | x) via forward-backward.

    Args:
      potentials: [B, T, N]
      lens: [B]
      trans: [N, N]

    Returns:
      probs: [B, T, N] where probs[b, t, :] sums to 1 for t < lens[b]; zeros for padded steps.
    """
    lens = K.cast(lens, "int32")
    B, T, N = K.shape(potentials)[0], K.shape(potentials)[1], K.shape(potentials)[2]
    trans_e = K.expand_dims(K.cast(trans, potentials.dtype), 0)  # [1,N,N]

    # Forward pass: log-alpha per time
    alphas = []
    a_t = potentials[:, 0, :]  # [B,N]
    alphas.append(a_t)
    t = 1
    while t < T:
        prev = K.expand_dims(a_t, 2)  # [B,N,1]
        scores = prev + trans_e       # [B,N,N] (i -> j)
        new_a = K.logsumexp(scores, axis=1) + potentials[:, t, :]  # [B,N]
        # Only update for sequences with length > t
        cond = K.expand_dims((t < lens), -1)  # [B,1]
        a_t = K.where(cond, new_a, a_t)
        alphas.append(a_t)
        t = t + 1
    # Stack -> [B,T,N]
    log_alpha = K.stack(alphas, axis=1)

    # Backward pass: log-beta per time
    betas = [K.zeros((B, N), dtype=potentials.dtype)]  # beta at T-1 is zeros
    b_t = betas[0]
    t = T - 2
    while t >= 0:
        # next term: beta[t+1] + potentials[:, t+1, :]
        nxt = b_t + potentials[:, t + 1, :]  # [B,N] (j)
        # For beta_t(i) = logsumexp_j trans[i,j] + nxt[j]
        nxt_e = K.expand_dims(nxt, 1)  # [B,1,N]
        scores = trans_e + nxt_e        # [B,N,N]
        new_b = K.logsumexp(scores, axis=2)  # [B,N]
        cond = K.expand_dims((t < (lens - 1)), -1)  # update if t < len-1
        b_t = K.where(cond, new_b, b_t)
        betas.append(b_t)
        t = t - 1
    betas = betas[::-1]  # reverse to time order 0..T-1
    log_beta = K.stack(betas, axis=1)  # [B,T,N]

    logZ = crf_log_norm(potentials, lens, trans)  # [B]
    logZ_e = K.reshape(logZ, (B, 1, 1))

    log_m = log_alpha + log_beta - logZ_e  # [B,T,N]
    probs = K.exp(log_m)

    # Zero out padded positions
    time_idx = K.cast(K.arange(T), "int32")
    mask_bt = K.expand_dims(time_idx, 0) < K.expand_dims(lens, -1)  # [B,T]
    probs = probs * K.cast(K.expand_dims(mask_bt, -1), probs.dtype)
    return probs
