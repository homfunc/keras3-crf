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
from keras_crf.losses import nll_loss as crf_nll_loss, dice_loss as crf_dice_loss, joint_dice_nll_loss as crf_joint_loss
from examples.utils.data import read_conll, build_maps, encode_and_pad, make_varlen_dataset
from examples.utils.metrics import MaskedTokenAccuracy
from examples.utils.ner_metrics import EntityF1


@keras.saving.register_keras_serializable(package="examples", name="BiLSTMCRF")
class BiLSTMCRF(keras.Model):
    def __init__(self,
                 num_tags:int,
                 vocab_size:int,
                 embedding_dim:int,
                 lstm_units:int,
                 loss_choice:str,
                 dice_smooth:float,
                 joint_nll_weight:float|None):
        super().__init__(name="bilstm_crf_manual")
        self.num_tags = int(num_tags)
        self.vocab_size = int(vocab_size)
        self.embedding_dim = int(embedding_dim)
        self.lstm_units = int(lstm_units)
        self.embed = layers.Embedding(self.vocab_size + 1, self.embedding_dim, mask_zero=True)
        self.bilstm = layers.Bidirectional(layers.LSTM(self.lstm_units, return_sequences=True))
        self.crf = CRF(self.num_tags)
        self.loss_choice = loss_choice
        self.dice_smooth = float(dice_smooth)
        self.joint_alpha = 0.2 if joint_nll_weight is None else float(joint_nll_weight)
    def call(self, tokens):
        x = self.embed(tokens)
        x = self.bilstm(x)
        decoded, potentials, lens, trans = self.crf(x)
        # Cache tensors for compute_loss (avoid storing variables at model root)
        self._last = (potentials, lens, decoded)
        return decoded
    def compute_output_shape(self, input_shape):
        # input_shape: (batch, time)
        try:
            B, T = input_shape
        except Exception:
            B, T = None, None
        return (B, T)
    def compute_output_spec(self, inputs, batch_size=None, dtype=None):
        # Output is integer tag ids with shape (batch, time)
        if isinstance(inputs, (list, tuple)):
            inp = inputs[0]
        else:
            inp = inputs
        shp = getattr(inp, "shape", None)
        if shp is not None:
            B = shp[0]
            T = shp[1]
        else:
            B, T = None, None
        return keras.KerasTensorSpec(shape=(B, T), dtype="int32")
    def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None):
        cache = getattr(self, "_last", (None, None, None))
        pot, lens, decoded = cache
        if pot is None:
            # ensure forward
            decoded = self(x, training=True)
            pot, lens, decoded = self._last
        y_true = y
        trans = self.crf.trans
        if self.loss_choice == "nll":
            loss = crf_nll_loss(pot, lens, trans, y_true, sample_weight=sample_weight, reduction="mean")
        elif self.loss_choice == "dice":
            loss = crf_dice_loss(pot, lens, trans, y_true, smooth=self.dice_smooth, reduction="mean")
        else:
            loss = crf_joint_loss(pot, lens, trans, y_true, alpha=self.joint_alpha, smooth=self.dice_smooth, reduction="mean")
        return loss
    def get_config(self):
        return {
            "num_tags": self.num_tags,
            "vocab_size": self.vocab_size,
            "embedding_dim": self.embedding_dim,
            "lstm_units": self.lstm_units,
            "loss_choice": self.loss_choice,
            "dice_smooth": float(self.dice_smooth),
            "joint_nll_weight": float(self.joint_alpha),
        }
    @classmethod
    def from_config(cls, cfg):
        return cls(
            num_tags=int(cfg.get("num_tags")),
            vocab_size=int(cfg.get("vocab_size")),
            embedding_dim=int(cfg.get("embedding_dim")),
            lstm_units=int(cfg.get("lstm_units")),
            loss_choice=str(cfg.get("loss_choice")),
            dice_smooth=float(cfg.get("dice_smooth", 1.0)),
            joint_nll_weight=float(cfg.get("joint_nll_weight", 0.2)),
        )
    def get_build_config(self):
        return {"input_specs": [{"name": "tokens", "dtype": "int32", "shape": (None,)}]}
    def build_from_config(self, cfg):
        # Build sublayers eagerly without tracing CRF decode.
        # This creates variables so that weights can be loaded, while avoiding a forward pass.
        emb_in_shape = (None, None)  # (batch, time) int32
        lstm_in_shape = (None, None, self.embedding_dim)
        crf_proj_in_shape = (None, None, 2 * self.lstm_units)
        crf_in_shape = (None, None, self.num_tags)
        # Build embedding, BiLSTM, and CRF projection + transition variables
        self.embed.build(emb_in_shape)
        self.bilstm.build(lstm_in_shape)
        if getattr(self.crf, "proj", None) is not None:
            self.crf.proj.build(crf_proj_in_shape)
        self.crf.build(crf_in_shape)
        self.built = True


def build_bilstm_crf_manual(num_tags: int,
                           vocab_size: int,
                           embedding_dim: int = 64,
                           lstm_units: int = 64,
                           loss: str = "nll",
                           dice_smooth: float = 1.0,
                           joint_nll_weight: float = None):
    """Build a BiLSTM + CRF model using a Keras Model subclass with compute_loss.

    This avoids symbolic graph-time execution of CRF ops and keeps things backend-agnostic.
    """
    loss_choice = (loss or "nll").lower()

    model = BiLSTMCRF(num_tags, vocab_size, embedding_dim, lstm_units, loss_choice, dice_smooth, joint_nll_weight)
    # Compile with optimizer and metric on decoded output
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  metrics=[MaskedTokenAccuracy()])
    # Inference model is simply the same model (it returns decoded tags)
    return model, model


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
    # Train directly on the subclassed model: inputs are tokens, targets are labels
    model.fit(X_train, Y_train,
              validation_data=(X_val, Y_val),
              epochs=a.epochs, batch_size=a.batch_size, verbose=2)

    # Evaluation
    decoded = infer.predict(X_test, batch_size=a.batch_size, verbose=0)
    mask = (X_test != 0)
    acc = (decoded[mask] == Y_test[mask]).mean()
    print(f"Masked token accuracy on test: {acc:.4f}")

    # Save and reload roundtrip assertions
    out_dir = os.path.join(os.path.dirname(__file__), 'out')
    os.makedirs(out_dir, exist_ok=True)
    keras_path = os.path.join(out_dir, 'bilstm_crf_manual.keras')
    infer.save(keras_path)
    loaded = keras.models.load_model(keras_path)
    decoded2 = loaded.predict(X_test, batch_size=a.batch_size, verbose=0)
    assert decoded2.shape == decoded.shape, f"Loaded decoded shape mismatch: {decoded2.shape} vs {decoded.shape}"
    # Exact match should hold with deterministic decode
    if not np.array_equal(decoded2, decoded):
        diff = np.mean(decoded2 != decoded)
        raise AssertionError(f"Reloaded predictions differ from original: mismatch rate {diff:.4f}")
    print("Roundtrip save/load OK; predictions identical.")

    if id2tag is not None:
        ent_f1 = EntityF1(id2tag)
        ent_f1.update_state(Y_test, decoded, sample_weight=mask.astype(np.float32))
        print(f"Entity F1 (micro): {float(ent_f1.result()):.4f}")


if __name__ == "__main__":
    main()

