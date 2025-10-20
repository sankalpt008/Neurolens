"""Mathematical helper utilities for NeuroLens."""
from __future__ import annotations

from math import sqrt
from typing import Iterable, Sequence

__all__ = ["safe_div", "percentile", "l2_distance"]


def safe_div(numerator: float | int | None, denominator: float | int | None, default: float = 0.0) -> float:
    """Safely divide ``numerator`` by ``denominator``.

    Returns ``default`` whenever the denominator is zero or when either operand is ``None``.
    """

    try:
        if denominator in (0, 0.0) or denominator is None:
            return float(default)
        if numerator is None:
            return float(default)
        return float(numerator) / float(denominator)
    except (TypeError, ZeroDivisionError):
        return float(default)


def percentile(values: Sequence[float] | Iterable[float], p: float) -> float:
    """Compute the ``p`` percentile (0-100) for ``values``.

    Uses linear interpolation between closest ranks. Returns ``0.0`` for empty inputs.
    """

    seq = list(values)
    if not seq:
        return 0.0
    if p <= 0:
        return float(min(seq))
    if p >= 100:
        return float(max(seq))
    seq.sort()
    rank = (len(seq) - 1) * (p / 100.0)
    lower = int(rank)
    upper = min(lower + 1, len(seq) - 1)
    weight = rank - lower
    lower_val = float(seq[lower])
    upper_val = float(seq[upper])
    return lower_val + (upper_val - lower_val) * weight


def l2_distance(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    """Return the L2 distance between two equally sized vectors."""

    length = min(len(vec_a), len(vec_b))
    if length == 0:
        return 0.0
    total = 0.0
    for idx in range(length):
        diff = float(vec_a[idx]) - float(vec_b[idx])
        total += diff * diff
    return sqrt(total)
