from typing import List, Tuple, Dict, Optional
import numpy as np


def make_varlen_dataset(num_samples: int = 2000,
                        max_len: int = 40,
                        vocab_size: int = 200,
                        num_tags: int = 4,
                        seed: int = 7) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create variable-length synthetic sequences with padding (0) and labels.

    Returns (X, Y, lengths).
    """
    rng = np.random.default_rng(seed)
    lens = rng.integers(low=max_len // 3, high=max_len + 1, size=num_samples, dtype=np.int32)
    X = np.zeros((num_samples, max_len), dtype=np.int32)
    Y = np.zeros((num_samples, max_len), dtype=np.int32)
    for i, L in enumerate(lens):
        seq = rng.integers(1, vocab_size, size=L, dtype=np.int32)
        X[i, :L] = seq
        mod = seq % 10
        y = np.zeros(L, dtype=np.int32)
        y[mod >= 7] = min(num_tags - 1, 3)
        y[(mod >= 4) & (mod <= 6)] = min(num_tags - 1, 2)
        y[(mod >= 2) & (mod <= 3)] = min(num_tags - 1, 1)
        flip = rng.random(L) < 0.03
        if flip.any():
            y[flip] = rng.integers(0, num_tags, size=flip.sum())
        Y[i, :L] = y
    return X, Y, lens


def read_conll(path: str,
               token_col: int = 0,
               tag_col: int = -1,
               lowercase: bool = False) -> Tuple[List[List[str]], List[List[str]]]:
    """Read a CoNLL-style file where each non-empty line has columns separated by whitespace
    (token and tag columns specified), and sentences are separated by blank lines.
    """
    sentences: List[List[str]] = []
    tags: List[List[str]] = []
    cur_toks: List[str] = []
    cur_tags: List[str] = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                if cur_toks:
                    sentences.append(cur_toks)
                    tags.append(cur_tags)
                    cur_toks, cur_tags = [], []
                continue
            if line.startswith('-DOCSTART-'):
                continue
            parts = line.split()
            tok = parts[token_col].lower() if lowercase else parts[token_col]
            tag = parts[tag_col]
            cur_toks.append(tok)
            cur_tags.append(tag)
    if cur_toks:
        sentences.append(cur_toks)
        tags.append(cur_tags)
    return sentences, tags


def build_maps(sentences: List[List[str]],
               tags: List[List[str]],
               min_freq: int = 1) -> Tuple[Dict[str, int], Dict[str, int]]:
    from collections import Counter
    c = Counter(tok for sent in sentences for tok in sent)
    tok2id: Dict[str, int] = {"<PAD>": 0, "<UNK>": 1}
    for tok, cnt in c.items():
        if cnt >= min_freq:
            tok2id.setdefault(tok, len(tok2id))
    tagset = sorted(set(t for ts in tags for t in ts))
    tag2id: Dict[str, int] = {t: i for i, t in enumerate(tagset)}
    return tok2id, tag2id


def encode_and_pad(sentences: List[List[str]],
                   tags: List[List[str]],
                   tok2id: Dict[str, int],
                   tag2id: Dict[str, int],
                   max_len: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    if max_len is None:
        max_len = max(len(s) for s in sentences)
    X = np.zeros((len(sentences), max_len), dtype=np.int32)
    Y = np.zeros((len(sentences), max_len), dtype=np.int32)
    for i, (s, t) in enumerate(zip(sentences, tags)):
        ids = [tok2id.get(w, tok2id["<UNK>"]) for w in s][:max_len]
        tg = [tag2id[u] for u in t][:max_len]
        X[i, : len(ids)] = np.array(ids, dtype=np.int32)
        Y[i, : len(tg)] = np.array(tg, dtype=np.int32)
    return X, Y


def make_tfdata(X: np.ndarray, Y: np.ndarray, batch_size: int = 64, shuffle: bool = True) -> 'tf.data.Dataset':
    import tensorflow as tf
    ds = tf.data.Dataset.from_tensor_slices((X, Y))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(X))
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def write_tfrecord(path: str, X: np.ndarray, Y: np.ndarray) -> None:
    import tensorflow as tf
    with tf.io.TFRecordWriter(path) as w:
        for x, y in zip(X, Y):
            feats = {
                'x': tf.train.Feature(int64_list=tf.train.Int64List(value=x.astype(np.int64))),
                'y': tf.train.Feature(int64_list=tf.train.Int64List(value=y.astype(np.int64))),
            }
            ex = tf.train.Example(features=tf.train.Features(feature=feats))
            w.write(ex.SerializeToString())


def read_tfrecord_dataset(path: str, seq_len: int, batch_size: int = 64, shuffle: bool = True) -> 'tf.data.Dataset':
    import tensorflow as tf
    feature_desc = {
        'x': tf.io.FixedLenFeature([seq_len], tf.int64),
        'y': tf.io.FixedLenFeature([seq_len], tf.int64),
    }
    def _parse(rec):
        ex = tf.io.parse_single_example(rec, feature_desc)
        x = tf.cast(ex['x'], tf.int32)
        y = tf.cast(ex['y'], tf.int32)
        return x, y
    ds = tf.data.TFRecordDataset([path]).map(_parse, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(1024)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds
