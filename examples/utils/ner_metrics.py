import numpy as np
import tensorflow as tf
from typing import List, Optional, Tuple, Set


def _extract_entities(tags: List[str], scheme: str = "BIO") -> Set[Tuple[int, int, str]]:
    """Extract entity spans from a tag sequence.
    Returns a set of (start, end_inclusive, type) with 0-based indices.
    Supports BIO and BILOU schemes.
    """
    spans = set()
    start = None
    ent_type = None

    def close_entity(pos_minus_one):
        nonlocal start, ent_type
        if start is not None and ent_type is not None:
            spans.add((start, pos_minus_one, ent_type))
        start = None
        ent_type = None

    for i, tag in enumerate(tags):
        if tag == "O" or tag == "":
            # Outside
            close_entity(i - 1)
            continue
        if scheme == "BIO":
            if tag.startswith("B-"):
                close_entity(i - 1)
                ent_type = tag[2:]
                start = i
            elif tag.startswith("I-"):
                # Continue if same type; if inconsistent, start new entity
                cur_type = tag[2:]
                if start is None or ent_type != cur_type:
                    close_entity(i - 1)
                    ent_type = cur_type
                    start = i
            else:
                # Unknown prefix -> treat as outside
                close_entity(i - 1)
        elif scheme == "BILOU":
            if tag.startswith("B-"):
                close_entity(i - 1)
                ent_type = tag[2:]
                start = i
            elif tag.startswith("I-"):
                cur_type = tag[2:]
                if start is None or ent_type != cur_type:
                    close_entity(i - 1)
                    ent_type = cur_type
                    start = i
            elif tag.startswith("L-"):
                cur_type = tag[2:]
                if start is None or ent_type != cur_type:
                    # treat as single-token entity
                    spans.add((i, i, cur_type))
                else:
                    spans.add((start, i, cur_type))
                start = None
                ent_type = None
            elif tag.startswith("U-"):
                close_entity(i - 1)
                spans.add((i, i, tag[2:]))
            else:
                close_entity(i - 1)
        else:
            raise ValueError(f"Unsupported scheme: {scheme}")
    # close at end
    if scheme == "BIO":
        close_entity(len(tags) - 1)
    # For BILOU, open entities without 'L-' are closed at end as BIO fallback
    if scheme == "BILOU" and start is not None and ent_type is not None:
        spans.add((start, len(tags) - 1, ent_type))
    return spans


class EntityF1(tf.keras.metrics.Metric):
    """Entity-level F1 for BIO/BILOU schemes.

    Provide `id2tag` mapping (list of strings) and scheme in {BIO, BILOU}.
    update_state expects:
      - y_true: int IDs [batch, time]
      - y_pred: int IDs [batch, time]
      - sample_weight: optional mask [batch, time] (1 for valid tokens)
    """

    def __init__(self, id2tag: List[str], scheme: str = "BIO", name: str = "entity_f1", **kwargs):
        super().__init__(name=name, **kwargs)
        self.id2tag = list(id2tag)
        self.scheme = scheme
        self.tp = self.add_weight(name="tp", initializer="zeros")
        self.fp = self.add_weight(name="fp", initializer="zeros")
        self.fn = self.add_weight(name="fn", initializer="zeros")

    def _batch_counts_numpy(self, y_true: np.ndarray, y_pred: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
        # y_true, y_pred: int arrays [B, T]; mask: {0,1} [B, T] or None
        tp = fp = fn = 0
        B, T = y_true.shape
        for b in range(B):
            if mask is not None:
                valid_idx = mask[b].astype(bool)
                true_ids = y_true[b][valid_idx]
                pred_ids = y_pred[b][valid_idx]
            else:
                true_ids = y_true[b]
                pred_ids = y_pred[b]
            true_tags = [self.id2tag[i] if 0 <= i < len(self.id2tag) else "O" for i in true_ids]
            pred_tags = [self.id2tag[i] if 0 <= i < len(self.id2tag) else "O" for i in pred_ids]
            true_spans = _extract_entities(true_tags, self.scheme)
            pred_spans = _extract_entities(pred_tags, self.scheme)
            tp += len(true_spans & pred_spans)
            fp += len(pred_spans - true_spans)
            fn += len(true_spans - pred_spans)
        return np.array([tp, fp, fn], dtype=np.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.cast(y_pred, tf.int32)
        if sample_weight is not None:
            mask = tf.cast(sample_weight, tf.int32)
        else:
            # all valid
            mask = tf.ones_like(y_true, dtype=tf.int32)
        counts = tf.numpy_function(lambda yt, yp, m: self._batch_counts_numpy(yt, yp, m),
                                   [y_true, y_pred, mask], Tout=tf.float32)
        # counts shape [3]
        self.tp.assign_add(counts[0])
        self.fp.assign_add(counts[1])
        self.fn.assign_add(counts[2])

    def precision_value(self):
        return tf.math.divide_no_nan(self.tp, self.tp + self.fp)

    def recall_value(self):
        return tf.math.divide_no_nan(self.tp, self.tp + self.fn)

    def result(self):
        precision = self.precision_value()
        recall = self.recall_value()
        f1 = tf.math.divide_no_nan(2 * precision * recall, precision + recall)
        return f1

    def reset_states(self):
        self.tp.assign(0.0)
        self.fp.assign(0.0)
        self.fn.assign(0.0)
