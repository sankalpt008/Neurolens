"""Fingerprint generation and comparison utilities."""

from .builder import VECTOR_SPEC, build_fingerprint
from .similarity import build_alignment, cosine_similarity, diff

__all__ = [
    "VECTOR_SPEC",
    "build_fingerprint",
    "build_alignment",
    "cosine_similarity",
    "diff",
]
