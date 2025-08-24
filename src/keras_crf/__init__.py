# Backend-agnostic default API
from .layers import CRF
from . import crf_ops as ops

__all__ = ["CRF", "ops"]

