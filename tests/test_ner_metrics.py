import pytest
# Backend-agnostic metric; no TF import required
import numpy as np
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from examples.utils.ner_metrics import _extract_entities, EntityF1


def test_extract_entities_bio():
    tags = ["B-PER","I-PER","O","B-LOC","I-LOC","I-LOC","O","B-ORG","I-ORG"]
    spans = _extract_entities(tags, scheme="BIO")
    assert (0,1,"PER") in spans
    assert (3,5,"LOC") in spans
    assert (7,8,"ORG") in spans


def test_extract_entities_bilou():
    tags = ["B-PER","L-PER","O","U-LOC","B-ORG","I-ORG","L-ORG"]
    spans = _extract_entities(tags, scheme="BILOU")
    assert (0,1,"PER") in spans
    assert (3,3,"LOC") in spans
    assert (4,6,"ORG") in spans


def test_entity_f1_counts():
    id2tag = ["O","B-PER","I-PER","B-LOC","I-LOC"]
    # batch of 2
    y_true = np.array([
        [1,2,0,3,4,0],   # PER span (0,1), LOC span (3,4)
        [0,0,0,3,4,0],   # LOC span (3,4)
    ], dtype=np.int32)
    y_pred = np.array([
        [1,2,0,3,4,0],   # perfect
        [0,0,0,0,0,0],   # missed LOC
    ], dtype=np.int32)
    mask = np.ones_like(y_true, dtype=np.int32)
    m = EntityF1(id2tag, scheme="BIO")
    m.update_state(y_true, y_pred, sample_weight=mask)
    f1 = float(m.result().numpy())
    # tp=2 (PER,LOC in first), fn=1 (second LOC), fp=0 -> P=2/2, R=2/3, F1=2*1*(2/3)/(1+2/3)=0.8
    assert abs(f1 - 0.8) < 1e-6
