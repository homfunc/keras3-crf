#!/usr/bin/env python3
import argparse
import os
import sys
import pathlib

# Allow running from repo root or installed environment
ROOT = pathlib.Path(__file__).resolve().parents[1].parents[0]
sys.path.insert(0, str(ROOT))

from examples import train_lib as tl  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Run BiLSTM-CRF training on a chosen dataset")
    p.add_argument("dataset", choices=["synthetic", "conll", "multiconer"], help="Dataset to run")
    # Common hyperparams
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--embedding-dim", type=int, default=64)
    p.add_argument("--lstm-units", type=int, default=64)
    p.add_argument("--loss", choices=["nll", "dice", "dice+nll"], default="dice+nll")
    p.add_argument("--joint-nll-weight", type=float, default=0.2)
    p.add_argument("--scheme", choices=["BIO", "BILOU"], default="BIO")
    # Synthetic
    p.add_argument("--samples", type=int, default=3000)
    p.add_argument("--max-len", type=int, default=50)
    p.add_argument("--vocab", type=int, default=300)
    p.add_argument("--tags", type=int, default=4)
    # CoNLL
    p.add_argument("--train", type=str)
    p.add_argument("--val", type=str)
    p.add_argument("--test", type=str)
    p.add_argument("--token-col", type=int, default=0)
    p.add_argument("--tag-col", type=int, default=-1)
    p.add_argument("--lowercase", action="store_true")
    # MultiCoNER
    p.add_argument("--mc-dir", type=str)
    return p.parse_args()


def main():
    a = parse_args()

    if a.dataset == "synthetic":
        X_tr, Y_tr, X_va, Y_va, X_te, Y_te, V, C, id2tag = tl.load_synthetic(a.samples, a.max_len, a.vocab, a.tags)
    elif a.dataset == "conll":
        if not (a.train and a.val and a.test):
            raise SystemExit("--train/--val/--test must be provided for conll dataset")
        X_tr, Y_tr, X_va, Y_va, X_te, Y_te, V, C, id2tag = tl.load_conll(a.train, a.val, a.test, a.token_col, a.tag_col, a.lowercase)
    elif a.dataset == "multiconer":
        if not a.mc_dir:
            raise SystemExit("--mc-dir must be provided for multiconer dataset")
        X_tr, Y_tr, X_va, Y_va, X_te, Y_te, V, C, id2tag = tl.load_multiconer_en(a.mc_dir, a.token_col, a.tag_col)
    else:
        raise SystemExit(f"Unknown dataset: {a.dataset}")

    model, model_pred = tl.build_bilstm_crf_models(V, C, a.embedding_dim, a.lstm_units, a.loss, 1.0, a.joint_nll_weight)
    res = tl.train_and_evaluate(model, model_pred, X_tr, Y_tr, X_va, Y_va, X_te, Y_te, a.epochs, a.batch_size, id2tag=id2tag, scheme=a.scheme)
    print({k: float(v) for k, v in res.items()})


if __name__ == "__main__":
    main()

