#!/usr/bin/env python3
import argparse
import numpy as np
import tensorflow as tf
import keras
from keras import layers

from keras_crf import CRF
from examples.utils.data import make_varlen_dataset, read_conll, build_maps, encode_and_pad, make_tfdata, write_tfrecord, read_tfrecord_dataset
from examples.utils.metrics import MaskedTokenAccuracy
from examples.utils.ner_metrics import EntityF1


class BiLstmCrfModel(keras.Model):
    def __init__(self, vocab_size, num_tags, embedding_dim=64, lstm_units=64):
        super().__init__()
        self.embedding = layers.Embedding(input_dim=vocab_size + 1, output_dim=embedding_dim, mask_zero=True)
        self.bilstm = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True))
        self.crf = CRF(units=num_tags)

    def call(self, inputs, training=False):
        x = self.embedding(inputs)
        x = self.bilstm(x)
        mask = self.embedding.compute_mask(inputs)
        return self.crf(x, mask=mask)


class ModelWithCRFLoss(keras.Model):
    def __init__(self, core, id2tag=None, scheme="BIO"):
        super().__init__()
        self.core = core
        self.token_acc = MaskedTokenAccuracy()
        self.entity_f1 = EntityF1(id2tag, scheme) if id2tag is not None else None

    def call(self, inputs, training=False):
        return self.core(inputs, training=training)

    def _loss_from_batch(self, data, training=False):
        x, y, sw = keras.utils.unpack_x_y_sample_weight(data)
        decoded, potentials, seq_len, kernel = self(x, training=training)
        ll = self.core.crf.log_likelihood(potentials, y, seq_len)
        loss = -tf.reduce_mean(ll)
        if sw is not None:
            sw = tf.cast(sw, loss.dtype)
            if sw.shape.rank == 0:
                sw = tf.fill(tf.shape(ll), sw)
            loss = tf.reduce_mean(sw * (-ll))
        # Update metric
        mask = tf.cast(tf.not_equal(x, 0), tf.float32)
        self.token_acc.update_state(y, decoded, sample_weight=mask)
        if self.entity_f1 is not None:
            self.entity_f1.update_state(y, decoded, sample_weight=mask)
        return loss

    def train_step(self, data):
        with tf.GradientTape() as tape:
            loss = self._loss_from_batch(data, training=True)
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        logs = {"loss": loss, self.token_acc.name: self.token_acc.result()}
        if self.entity_f1 is not None:
            logs[self.entity_f1.name] = self.entity_f1.result()
            logs["entity_precision"] = self.entity_f1.precision_value()
            logs["entity_recall"] = self.entity_f1.recall_value()
        return logs

    def test_step(self, data):
        loss = self._loss_from_batch(data, training=False)
        logs = {"loss": loss, self.token_acc.name: self.token_acc.result()}
        if self.entity_f1 is not None:
            logs[self.entity_f1.name] = self.entity_f1.result()
            logs["entity_precision"] = self.entity_f1.precision_value()
            logs["entity_recall"] = self.entity_f1.recall_value()
        return logs

    def reset_metrics(self):
        super().reset_metrics()
        self.token_acc.reset_states()
        if self.entity_f1 is not None:
            self.entity_f1.reset_states()


def parse_args():
    p = argparse.ArgumentParser(description="Train BiLSTM-CRF for sequence tagging")
    p.add_argument("--dataset", choices=["synthetic", "conll"], default="synthetic")
    p.add_argument("--train", type=str, help="Path to CoNLL train file (for conll)")
    p.add_argument("--val", type=str, help="Path to CoNLL validation file (for conll)")
    p.add_argument("--test", type=str, help="Path to CoNLL test file (for conll)")
    p.add_argument("--token-col", type=int, default=0)
    p.add_argument("--tag-col", type=int, default=-1)
    p.add_argument("--lowercase", action="store_true")

    p.add_argument("--synthetic-max-len", type=int, default=50)
    p.add_argument("--synthetic-vocab", type=int, default=300)
    p.add_argument("--synthetic-tags", type=int, default=4)
    p.add_argument("--synthetic-samples", type=int, default=3000)

    p.add_argument("--embedding-dim", type=int, default=64)
    p.add_argument("--lstm-units", type=int, default=64)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--scheme", choices=["BIO","BILOU"], default="BIO")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--use-tfdata", action="store_true")
    p.add_argument("--write-train-tfrecord", type=str, default=None)
    p.add_argument("--read-train-tfrecord", type=str, default=None)
    p.add_argument("--run-eagerly", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    tf.config.run_functions_eagerly(bool(args.run_eagerly))

    if args.dataset == "synthetic":
        X_train, Y_train, _ = make_varlen_dataset(args.synthetic_samples, args.synthetic_max_len,
                                                 args.synthetic_vocab, args.synthetic_tags, seed=1)
        X_val, Y_val, _ = make_varlen_dataset(max(args.synthetic_samples // 5, 1), args.synthetic_max_len,
                                             args.synthetic_vocab, args.synthetic_tags, seed=2)
        X_test, Y_test, _ = make_varlen_dataset(max(args.synthetic_samples // 5, 1), args.synthetic_max_len,
                                               args.synthetic_vocab, args.synthetic_tags, seed=3)
        num_tags = args.synthetic_tags
        vocab_size = args.synthetic_vocab
    else:
        if not (args.train and args.val and args.test):
            raise SystemExit("--train/--val/--test must be provided for conll dataset")
        train_s, train_t = read_conll(args.train, args.token_col, args.tag_col, args.lowercase)
        val_s, val_t = read_conll(args.val, args.token_col, args.tag_col, args.lowercase)
        test_s, test_t = read_conll(args.test, args.token_col, args.tag_col, args.lowercase)
        tok2id, tag2id = build_maps(train_s, train_t)
        X_train, Y_train = encode_and_pad(train_s, train_t, tok2id, tag2id)
        X_val, Y_val = encode_and_pad(val_s, val_t, tok2id, tag2id, max_len=X_train.shape[1])
        X_test, Y_test = encode_and_pad(test_s, test_t, tok2id, tag2id, max_len=X_train.shape[1])
        num_tags = len(tag2id)
        vocab_size = len(tok2id) - 1  # exclude PAD=0 from count if you prefer
        id2tag = [None] * num_tags
        for t, i in tag2id.items():
            id2tag[i] = t

    core = BiLstmCrfModel(vocab_size=vocab_size, num_tags=num_tags,
                          embedding_dim=args.embedding_dim, lstm_units=args.lstm_units)
    model = ModelWithCRFLoss(core, id2tag=id2tag if args.dataset == 'conll' else None, scheme=args.scheme)
    model.compile(optimizer=keras.optimizers.Adam(args.lr), run_eagerly=args.run_eagerly)

    # Optionally serialize TFRecord for training
    if args.write_train_tfrecord:
        write_tfrecord(args.write_train_tfrecord, X_train, Y_train)
        print(f"Wrote TFRecord: {args.write_train_tfrecord}")

    if args.read_train_tfrecord:
        train_ds = read_tfrecord_dataset(args.read_train_tfrecord, seq_len=X_train.shape[1], batch_size=args.batch_size)
        val_ds = make_tfdata(X_val, Y_val, batch_size=args.batch_size, shuffle=False)
        history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs)
    elif args.use_tfdata:
        train_ds = make_tfdata(X_train, Y_train, batch_size=args.batch_size, shuffle=True)
        val_ds = make_tfdata(X_val, Y_val, batch_size=args.batch_size, shuffle=False)
        history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs)
    else:
        history = model.fit(X_train, Y_train,
                            validation_data=(X_val, Y_val),
                            epochs=args.epochs, batch_size=args.batch_size)
    print({k: v[-1] for k, v in history.history.items()})

    # Evaluate masked token accuracy and per-entity report (if available)
    decoded, _, _, _ = model.predict(X_test, batch_size=args.batch_size, verbose=0)
    mask = (X_test != 0)
    acc = (decoded[mask] == Y_test[mask]).mean()
    print(f"Masked token accuracy on test: {acc:.4f}")


if __name__ == "__main__":
    main()
