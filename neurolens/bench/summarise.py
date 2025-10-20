"""Optional helpers for summarising bench results."""
from __future__ import annotations

from typing import Iterable, Mapping

try:  # pragma: no cover - pandas optional
    import pandas as _pd
except Exception:  # pragma: no cover - optional dependency missing
    _pd = None

__all__ = ["rows_to_dataframe"]


def rows_to_dataframe(rows: Iterable[Mapping[str, object]]):
    """Return a pandas DataFrame when pandas is available."""

    if _pd is None:
        return list(rows)
    return _pd.DataFrame(list(rows))
