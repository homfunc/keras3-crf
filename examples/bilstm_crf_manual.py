#!/usr/bin/env python3
import argparse
import os
import sys

# Make repo root importable
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import keras
from keras import layers, ops as K

from keras_crf.layers import CRF
from keras_crf.crf_ops import crf_log_likelihood, crf_marginals
from examples.utils.data import read_conll, build_maps, encode_and_pad, make_varlen_dataset
from examples.utils.metrics import MaskedTokenAccuracy
from examples.utils.ner_metrics import EntityF1


def build_bilstm_crf_manual(num_tags: int,
                            vocab_size: int,
                            embedding_dim: int = 64,
                            lstm_units: int = 64,
                            loss: str = "nll",
                            dice_smooth: float = 1.0,
                            joint_nll_weight: float = None):
    """Build a BiLSTM + CRF model manually without train_utils.

    Demonstrates how to:
      - attach CRF,
      - create a custom loss head (NLL, Dice, or joint),
      - expose a decoded output for metrics.
    """
    tokens_in = keras.Input(shape=(None,), dtype="int32", name="tokens")
    x = layers.Embedding(vocab_size + 1, embedding_dim, mask_zero=True)(tokens_in)
    x = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True))(x)

    crf = CRF(num_tags)
    decoded, potentials, lens, trans = crf(x)

    labels = keras.Input(shape=(None,), dtype="int32", name="labels")

    # Loss heads
    class NLL(keras.layers.Layer):
        def call(self, inputs):
            pot, y_true, ln, tr = inputs
            return -crf_log_likelihood(pot, y_true, ln, tr)

    class DiceLoss(keras.layers.Layer):
        def __init__(self, num_tags: int, smooth: float = 1.0, **kw):
            super().__init__(**kw)
            self.num_tags = num_tags
            self.smooth = smooth
        def call(self, inputs):
            pot, y_true, ln, tr = inputs
            T = K.shape(pot)[1]
            probs = crf_marginals(pot, ln, tr)
            y_oh = K.one_hot(K.cast(y_true, "int32"), self.num_tags)
            y_oh = K.cast(y_oh, probs.dtype)
            time_idx = K.cast(K.arange(T), "int32")
            mask = K.expand_dims(time_idx, 0) < K.expand_dims(K.cast(ln, "int32"), -1)
            mask = K.cast(mask, probs.dtype)
            mask = K.expand_dims(mask, -1)
            y_oh_m = y_oh * mask
            probs_m = probs * mask
            inter = K.sum(y_oh_m * probs_m, axis=(1, 2))
            sums = K.sum(y_oh_m, axis=(1, 2)) + K.sum(probs_m, axis=(1, 2))
            dice = (2.0 * inter + self.smooth) / (sums + self.smooth)
            return 1.0 - dice

    nll = NLL()([potentials, labels, lens, trans])
    loss_choice = (loss or "nll").lower()
    if loss_choice == "nll":
        loss_vec = nll
    elif loss_choice == "dice":
        d = DiceLoss(num_tags, smooth=dice_smooth)([potentials, labels, lens, trans])
        loss_vec = d
    elif loss_choice in ("dice+nll", "joint"):
        alpha = 0.2 if joint_nll_weight is None else float(joint_nll_weight)
        d = DiceLoss(num_tags, smooth=dice_smooth)([potentials, labels, lens, trans])
        loss_vec = keras.layers.Lambda(lambda zs: alpha * zs[0] + (1.0 - alpha) * zs[1])([nll, d])
    else:
        raise ValueError(f"Unsupported loss: {loss}")

    decoded_out = keras.layers.Lambda(lambda z: z, name="decoded_output")(decoded)
    loss_out = keras.layers.Lambda(lambda z: z, name="crf_log_likelihood_output")(loss_vec)

    model = keras.Model(inputs={"tokens": tokens_in, "labels": labels}, outputs={"decoded_output": decoded_out, "crf_log_likelihood_output": loss_out})

    def zero_loss(y_true, y_pred):
        return K.mean(K.zeros_like(y_pred[..., :1]))

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss={"decoded_output": zero_loss, "crf_log_likelihood_output": lambda yt, yp: K.mean(yp)},
        metrics={"decoded_output": [MaskedTokenAccuracy()]},
    )
    # Also return a separate inference model
    infer_model = keras.Model(tokens_in, decoded_out)
    return model, infer_model


def parse_args():
    p = argparse.ArgumentParser(description="Manual BiLSTM-CRF build (no train_utils)")
    p.add_argument("--dataset", choices=["synthetic", "conll"], default="synthetic")
    p.add_argument("--train", type=str)
    p.add_argument("--val", type=str)
    p.add_argument("--test", type=str)
    p.add_argument("--token-col", type=int, default=0)
    p.add_argument("--tag-col", type=int, default=-1)
    p.add_argument("--lowercase", action="store_true")
    p.add_argument("--embedding-dim", type=int, default=64)
    p.add_argument("--lstm-units", type=int, default=64)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--loss", choices=["nll", "dice", "dice+nll"], default="nll")
    p.add_argument("--dice-smooth", type=float, default=1.0)
    p.add_argument("--joint-nll-weight", type=float, default=None)
    return p.parse_args()


def main():
    a = parse_args()

    if a.dataset == "synthetic":
        X_train, Y_train, _ = make_varlen_dataset(3000, 50, 300, 4, seed=1)
        X_val, Y_val, _ = make_varlen_dataset(600, 50, 300, 4, seed=2)
        X_test, Y_test, _ = make_varlen_dataset(600, 50, 300, 4, seed=3)
        num_tags = 4
        vocab_size = 300
        id2tag = None
    else:
        if not (a.train and a.val and a.test):
            raise SystemExit("--train/--val/--test are required for conll")
        train_s, train_t = read_conll(a.train, a.token_col, a.tag_col, a.lowercase)
        val_s, val_t = read_conll(a.val, a.token_col, a.tag_col, a.lowercase)
        test_s, test_t = read_conll(a.test, a.token_col, a.tag_col, a.lowercase)
        tok2id, tag2id = build_maps(train_s, train_t)
        X_train, Y_train = encode_and_pad(train_s, train_t, tok2id, tag2id)
        X_val, Y_val = encode_and_pad(val_s, val_t, tok2id, tag2id, max_len=X_train.shape[1])
        X_test, Y_test = encode_and_pad(test_s, test_t, tok2id, tag2id, max_len=X_train.shape[1])
        num_tags = len(tag2id)
        vocab_size = len(tok2id) - 1
        id2tag = [None] * num_tags
        for t, i in tag2id.items():
            id2tag[i] = t

    model, infer = build_bilstm_crf_manual(num_tags, vocab_size, a.embedding_dim, a.lstm_units, a.loss, a.dice_smooth, a.joint_nll_weight)

    # Prepare targets and masks
    mask_train = (X_train != 0).astype(np.float32)
    y_train = {"decoded_output": Y_train, "crf_log_likelihood_output": np.zeros((X_train.shape[0],), dtype=np.float32)}
    sw_train = {"decoded_output": mask_train, "crf_log_likelihood_output": np.ones((X_train.shape[0],), dtype=np.float32)}

    mask_val = (X_val != 0).astype(np.float32)
    y_val = {"decoded_output": Y_val, "crf_log_likelihood_output": np.zeros((X_val.shape[0],), dtype=np.float32)}
    sw_val = {"decoded_output": mask_val, "crf_log_likelihood_output": np.ones((X_val.shape[0],), dtype=np.float32)}

    model.fit({"tokens": X_train, "labels": Y_train}, y_train, sample_weight=sw_train,
              validation_data=({"tokens": X_val, "labels": Y_val}, y_val, sw_val),
              epochs=a.epochs, batch_size=a.batch_size, verbose=2)

    # Evaluation
    decoded = infer.predict(X_test, batch_size=a.batch_size, verbose=0)
    mask = (X_test != 0)
    acc = (decoded[mask] == Y_test[mask]).mean()
    print(f"Masked token accuracy on test: {acc:.4f}")

    if id2tag is not None:
        ent_f1 = EntityF1(id2tag)
        ent_f1.update_state(Y_test, decoded, sample_weight=mask.astype(np.float32))
        print(f"Entity F1 (micro): {float(ent_f1.result()):.4f}")


if __name__ == "__main__":
    main()

