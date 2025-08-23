from typing import Dict, List, Optional, Tuple
import numpy as np
from .ner_metrics import _extract_entities


def per_entity_report(id2tag: List[str], scheme: str, y_true: np.ndarray, y_pred: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict[str, Dict[str, float]]:
    """Compute per-entity precision/recall/F1 by entity type.

    Args:
      id2tag: list mapping tag id -> string tag
      scheme: "BIO" or "BILOU"
      y_true: int array [B, T]
      y_pred: int array [B, T]
      mask: optional {0,1} array [B, T] marking valid tokens

    Returns: dict {entity_type: {precision, recall, f1, support}}
    """
    # Aggregate spans per type
    tp: Dict[str, int] = {}
    fp: Dict[str, int] = {}
    fn: Dict[str, int] = {}

    B, T = y_true.shape
    for b in range(B):
        if mask is not None:
            valid = mask[b].astype(bool)
            true_ids = y_true[b][valid]
            pred_ids = y_pred[b][valid]
        else:
            true_ids = y_true[b]
            pred_ids = y_pred[b]
        true_tags = [id2tag[i] if 0 <= i < len(id2tag) else "O" for i in true_ids]
        pred_tags = [id2tag[i] if 0 <= i < len(id2tag) else "O" for i in pred_ids]
        true_spans = _extract_entities(true_tags, scheme)
        pred_spans = _extract_entities(pred_tags, scheme)

        # per type counts
        true_by_type: Dict[str, set] = {}
        pred_by_type: Dict[str, set] = {}
        for s in true_spans:
            true_by_type.setdefault(s[2], set()).add(s)
        for s in pred_spans:
            pred_by_type.setdefault(s[2], set()).add(s)
        types = set(list(true_by_type.keys()) + list(pred_by_type.keys()))
        for t in types:
            inter = true_by_type.get(t, set()) & pred_by_type.get(t, set())
            tp[t] = tp.get(t, 0) + len(inter)
            fp[t] = fp.get(t, 0) + len(pred_by_type.get(t, set()) - inter)
            fn[t] = fn.get(t, 0) + len(true_by_type.get(t, set()) - inter)

    report: Dict[str, Dict[str, float]] = {}
    for t in sorted(set(tp.keys()) | set(fp.keys()) | set(fn.keys())):
        tp_t = tp.get(t, 0)
        fp_t = fp.get(t, 0)
        fn_t = fn.get(t, 0)
        precision = (tp_t / (tp_t + fp_t)) if (tp_t + fp_t) > 0 else 0.0
        recall = (tp_t / (tp_t + fn_t)) if (tp_t + fn_t) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        report[t] = {"precision": precision, "recall": recall, "f1": f1, "support": tp_t + fn_t}
    return report
