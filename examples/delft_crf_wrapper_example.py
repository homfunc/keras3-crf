#!/usr/bin/env python3
"""
Minimal DeLFT-style CRF wrapper example (backend-agnostic, Keras 3).

This demonstrates two safe ways to train and evaluate a CRF with:
- Left padding (pads at the beginning, sequences are right-aligned)
- Right padding (pads at the end, sequences are left-aligned)

Key points
- We compute true sequence lengths from tokens (tokens != 0).
- We compile with keras_crf.losses.CRFNLLLoss so the CRF loss stays in-graph.
- We enable CRF boundary parameters by default for left padding; for right padding
  boundaries are disabled (or you can force-enable via a flag, see below).

Inputs
- tokens: [B, T] integer token ids (0 is pad)
- char_input: [B, T, max_char] integer char ids (dummy in this demo)
- labels: [B, T] gold tag ids

Outputs
- Training model outputs the CRF potentials [B, T, N], and uses CRFNLLLoss.
- Inference model outputs decoded tags [B, T].

How to run (from the keras-crf repo root):
  # Left padding (default) with boundaries enabled
  python examples/delft_crf_wrapper_example.py --epochs 3 --batch-size 64 --pad-direction left

  # Right padding, boundaries auto-disabled (safer)
  python examples/delft_crf_wrapper_example.py --epochs 3 --batch-size 64 --pad-direction right

Notes
- When padding direction is right, boundary placement in the current CRF layer would be wrong,
  so we default to use_boundary=False in that case unless explicitly overridden.
"""

import argparse
import os
import sys

import numpy as np
import keras
from keras import layers, ops as K

# Ensure local examples utils are importable when running as a script
EXAMPLES_ROOT = os.path.dirname(__file__)
REPO_ROOT = os.path.dirname(EXAMPLES_ROOT)
sys.path.insert(0, REPO_ROOT)

from keras_crf.layers import CRF
from keras_crf.losses import (
    CRFNLLHead, CRFDiceHead, CRFJointDiceNLLHead,
    CRFNLLLoss, CRFDiceLoss, CRFJointDiceNLLLoss,
)
from examples.utils.data import make_varlen_dataset


def _left_pad_from_right(X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return left-padded (right-aligned) copies of X, Y based on nonzero tokens in X."""
    X = np.asarray(X)
    Y = np.asarray(Y)
    B, T = X.shape
    Xo = np.zeros_like(X)
    Yo = np.zeros_like(Y)
    lens = (X != 0).sum(axis=1)
    for i, L in enumerate(lens):
        if L > 0:
            Xo[i, T - L : T] = X[i, :L]
            Yo[i, T - L : T] = Y[i, :L]
    return Xo, Yo


def _right_pad_from_any(X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return right-padded (left-aligned) copies of X, Y by compacting nonzero tokens to the left."""
    X = np.asarray(X)
    Y = np.asarray(Y)
    B, T = X.shape
    Xo = np.zeros_like(X)
    Yo = np.zeros_like(Y)
    for i in range(B):
        mask = X[i] != 0
        L = int(mask.sum())
        if L > 0:
            Xo[i, :L] = X[i, mask][:L]
            Yo[i, :L] = Y[i, mask][:L]
    return Xo, Yo


def build_delft_style_models(
    num_tags: int,
    vocab_size: int,
    max_char: int = 30,
    word_dim: int = 128,
    char_dim: int = 16,
    word_lstm_units: int = 64,
    char_lstm_units: int = 16,
    use_boundary: bool = True,
    loss_type: str = "nll",  # one of: nll, dice, joint
    alpha: float = 0.2,
    smooth: float = 1.0,
    run_eagerly: bool = True,
):
    """Build training and inference models.

    - Training model: outputs decoded tags and a per-sample CRF loss vector (for loss).
    - Inference model: outputs decoded tags only.
    """
    # Inputs (names loosely mirror DeLFT)
    tokens_in = keras.Input(shape=(None,), dtype="int32", name="tokens")
    char_input = keras.Input(shape=(None, max_char), dtype="int32", name="char_input")
    labels_in = keras.Input(shape=(None,), dtype="int32", name="labels")

    # Word channel (no built-in masking here; we pass an explicit token mask to CRF)
    w = layers.Embedding(vocab_size + 1, word_dim, mask_zero=False, name="word_embed")(tokens_in)

    # Char channel (TimeDistributed over tokens) — no mask, but merge will pick the word mask
    c = layers.TimeDistributed(
        layers.Embedding(input_dim=128, output_dim=char_dim, mask_zero=False),
        name="char_embed_td",
    )(char_input)
    c = layers.TimeDistributed(
        layers.Bidirectional(layers.LSTM(char_lstm_units, return_sequences=False)),
        name="char_bilstm_td",
    )(c)

    # Concatenate channels
    x = layers.Concatenate(name="concat_word_char")([w, c])
    x = layers.Dropout(0.2)(x)

    # Word-level BiLSTM
    x = layers.Bidirectional(layers.LSTM(word_lstm_units, return_sequences=True), name="bilstm")(x)
    x = layers.Dropout(0.2)(x)

    # Small projection before CRF
    feats = layers.Dense(word_lstm_units, activation="tanh", name="proj")(x)

    # CRF layer
    crf = CRF(num_tags, use_boundary=use_boundary)
    # Explicit token mask (token != 0) to avoid mask-broadcast quirks through merges
    token_mask = keras.layers.Lambda(lambda t: K.not_equal(t, 0), name="token_mask")(tokens_in)
    decoded, potentials, crf_lengths, _ = crf(feats, mask=token_mask)

    # Also compute numeric lengths directly from the token mask for loss
    lengths_num = keras.layers.Lambda(lambda m: K.sum(K.cast(m, "int32"), axis=1), name="lengths_num")(token_mask)

    # Choose loss head
    if loss_type == "nll":
        loss_vec = CRFNLLHead(name="crf_loss_vec")([potentials, labels_in, lengths_num, crf.trans])
        loss_reducer = CRFNLLLoss()
    elif loss_type == "dice":
        loss_vec = CRFDiceHead(smooth=float(smooth), name="crf_loss_vec")([potentials, labels_in, lengths_num, crf.trans])
        loss_reducer = CRFDiceLoss()
    elif loss_type == "joint":
        loss_vec = CRFJointDiceNLLHead(alpha=float(alpha), smooth=float(smooth), name="crf_loss_vec")([potentials, labels_in, lengths_num, crf.trans])
        loss_reducer = CRFJointDiceNLLLoss()
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    decoded_named = keras.layers.Lambda(lambda z: z, name="decoded_output")(decoded)

    # Training model: two outputs (decoded + per-sample loss vector)
    train_model = keras.Model(inputs=[tokens_in, char_input, labels_in], outputs=[decoded_named, loss_vec], name="delft_crf_train")

    # Zero loss for decoded head; reduce the per-sample vector with the configured reducer loss
    def zero_loss(y_true, y_pred):
        return K.mean(K.zeros_like(y_pred[..., :1]))

    train_model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss={"decoded_output": zero_loss, "crf_loss_vec": loss_reducer},
        run_eagerly=run_eagerly,
    )

    # Inference model returning decoded tags only
    infer_model = keras.Model([tokens_in, char_input], decoded_named, name="delft_crf_infer")
    return train_model, infer_model


def parse_args():
    p = argparse.ArgumentParser(description="DeLFT-style CRF example with left/right padding and CRF NLL/Dice/Joint loss heads")
    p.add_argument("--synthetic-samples", type=int, default=2000)
    p.add_argument("--synthetic-max-len", type=int, default=40)
    p.add_argument("--synthetic-vocab", type=int, default=500)
    p.add_argument("--synthetic-tags", type=int, default=5)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-char", type=int, default=30)
    p.add_argument("--pad-direction", choices=["left", "right"], default="left",
                   help="left: sequences right-aligned (pads on the left); right: left-aligned (pads on the right)")
    p.add_argument("--use-boundary", choices=["auto", "true", "false"], default="auto",
                   help="CRF boundary parameters: auto => enabled for left padding, disabled for right padding")
    p.add_argument("--loss-type", choices=["nll", "dice", "joint"], default="nll",
                   help="Which CRF loss head to use")
    p.add_argument("--alpha", type=float, default=0.2, help="Alpha for joint loss (weight on NLL)")
    p.add_argument("--smooth", type=float, default=1.0, help="Smoothing for Dice in dice/joint")
    p.add_argument("--run-eagerly", choices=["true", "false"], default="true",
                   help="Whether to compile the training model with run_eagerly=True (default true)")
    return p.parse_args()


def main():
    a = parse_args()

    # Synthetic variable-length dataset (right-padded by construction)
    X_train, Y_train, _ = make_varlen_dataset(a.synthetic_samples, a.synthetic_max_len,
                                             a.synthetic_vocab, a.synthetic_tags, seed=1)
    X_val, Y_val, _ = make_varlen_dataset(max(a.synthetic_samples // 5, 1), a.synthetic_max_len,
                                         a.synthetic_vocab, a.synthetic_tags, seed=2)

    # Re-pad according to the requested direction
    if a.pad_direction == "left":
        X_train, Y_train = _left_pad_from_right(X_train, Y_train)
        X_val, Y_val = _left_pad_from_right(X_val, Y_val)
    else:
        # Ensure compact right padding even if input was left-padded
        X_train, Y_train = _right_pad_from_any(X_train, Y_train)
        X_val, Y_val = _right_pad_from_any(X_val, Y_val)

    # Decide boundary usage
    if a.use_boundary == "true":
        use_boundary = True
    elif a.use_boundary == "false":
        use_boundary = False
    else:
        use_boundary = (a.pad_direction == "left")

    num_tags = a.synthetic_tags
    vocab_size = a.synthetic_vocab

    # Dummy char inputs (zeros) for demonstration; shape [B, T, max_char]
    C_train = np.zeros((X_train.shape[0], X_train.shape[1], a.max_char), dtype="int32")
    C_val = np.zeros((X_val.shape[0], X_val.shape[1], a.max_char), dtype="int32")

    train_model, infer_model = build_delft_style_models(
        num_tags=num_tags,
        vocab_size=vocab_size,
        max_char=a.max_char,
        use_boundary=use_boundary,
        loss_type=a.loss_type,
        alpha=a.alpha,
        smooth=a.smooth,
        run_eagerly=(a.run_eagerly == "true"),
    )

    # Train
    history = train_model.fit(
        x={"tokens": X_train, "char_input": C_train, "labels": Y_train},
        y={"decoded_output": Y_train, "crf_loss_vec": np.zeros((X_train.shape[0],), dtype="float32")},
        validation_data=(
            {"tokens": X_val, "char_input": C_val, "labels": Y_val},
            {"decoded_output": Y_val, "crf_loss_vec": np.zeros((X_val.shape[0],), dtype="float32")},
        ),
        epochs=a.epochs,
        batch_size=a.batch_size,
        verbose=2,
    )

    # Quick sanity-check inference + masked accuracy
    decoded_val = infer_model.predict([X_val, C_val], batch_size=a.batch_size, verbose=0)
    mask_val = (X_val != 0)
    acc = (decoded_val[mask_val] == Y_val[mask_val]).mean() if mask_val.any() else 0.0
    print(f"Pad={a.pad_direction}, use_boundary={use_boundary} -> Masked token accuracy (val): {acc:.4f}")


if __name__ == "__main__":
    main()

