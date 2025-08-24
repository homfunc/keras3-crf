import numpy as np
import keras
from keras import ops as K
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


class EntityF1(keras.metrics.Metric):
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
        # Convert to numpy for robust backend-agnostic counting
        yt = K.convert_to_numpy(y_true)
        yp = K.convert_to_numpy(y_pred)
        if sample_weight is not None:
            m = K.convert_to_numpy(sample_weight)
        else:
            m = np.ones_like(yt, dtype=np.int32)
        counts = self._batch_counts_numpy(yt.astype(np.int32), yp.astype(np.int32), m.astype(np.int32))
        # counts shape [3]
        self.tp.assign_add(float(counts[0]))
        self.fp.assign_add(float(counts[1]))
        self.fn.assign_add(float(counts[2]))

    def precision_value(self):
        denom = self.tp + self.fp
        return (self.tp / (denom + 1e-8))

    def recall_value(self):
        denom = self.tp + self.fn
        return (self.tp / (denom + 1e-8))

    def result(self):
        precision = self.precision_value()
        recall = self.recall_value()
        denom = (precision + recall)
        return (2.0 * precision * recall) / (denom + 1e-8)

    def reset_states(self):
        self.tp.assign(0.0)
        self.fp.assign(0.0)
        self.fn.assign(0.0)
