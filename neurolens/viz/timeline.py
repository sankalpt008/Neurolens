"""Timeline aggregation helpers for NeuroLens visualizations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence

from neurolens.fingerprint import build_fingerprint

__all__ = ["TimelineSeries", "build_timeline_series"]

_COMPUTE_TYPES = {"MatMul", "Gemm", "Conv", "Attention", "FusedMatMul"}
_MEMORY_TYPES = {"Memcpy", "Memset", "Transpose", "Reshape", "Concat"}


@dataclass
class TimelineSeries:
    """Container for per-op latency visualization data."""

    ops: List[Dict[str, Any]]
    aggregates: List[Dict[str, Any]]
    total_latency_ms: float


def _ensure_fingerprint(data: Mapping[str, Any]) -> Mapping[str, Any]:
    if "vector_spec" in data and "ops" in data:
        return data
    return build_fingerprint(data)


def _lat_norm_index(vector_spec: Sequence[str]) -> int:
    try:
        return list(vector_spec).index("lat_norm")
    except ValueError:
        return 0


def _classify_op(op_type: str) -> str:
    if op_type in _COMPUTE_TYPES:
        return "compute"
    if op_type in _MEMORY_TYPES:
        return "memory"
    if op_type.lower().startswith("memcpy"):
        return "memory"
    return "other"


def _color_for_class(kind: str) -> str:
    if kind == "compute":
        return "#1f77b4"  # blue
    if kind == "memory":
        return "#ff7f0e"  # orange
    return "#7f7f7f"  # gray


def build_timeline_series(data: Mapping[str, Any]) -> TimelineSeries:
    """Return timeline visualization data from a run or fingerprint mapping."""

    fingerprint = _ensure_fingerprint(data)
    vector_spec = fingerprint.get("vector_spec", [])
    lat_norm_idx = _lat_norm_index(vector_spec)
    summary = fingerprint.get("summary", {})
    total_latency = float(summary.get("total_latency_ms", 0.0))

    ops_view: List[Dict[str, Any]] = []
    aggregates: Dict[str, float] = {}

    for position, op in enumerate(fingerprint.get("ops", [])):
        vector: Sequence[Any] = op.get("vector", [])  # type: ignore[assignment]
        lat_norm = float(vector[lat_norm_idx]) if len(vector) > lat_norm_idx else 0.0
        latency_ms = lat_norm * total_latency
        op_type = str(op.get("type", "unknown"))
        category = _classify_op(op_type)
        color = _color_for_class(category)
        ops_view.append(
            {
                "index": int(op.get("index", position)),
                "name": op.get("name", f"op_{position}"),
                "type": op_type,
                "latency_ms": latency_ms,
                "latency_pct": lat_norm * 100.0,
                "category": category,
                "color": color,
            }
        )
        aggregates[category] = aggregates.get(category, 0.0) + lat_norm

    aggregate_rows = [
        {
            "category": key,
            "latency_pct": value * 100.0,
            "color": _color_for_class(key),
        }
        for key, value in aggregates.items()
    ]

    ops_view.sort(key=lambda item: item["index"])
    aggregate_rows.sort(key=lambda item: item["latency_pct"], reverse=True)

    return TimelineSeries(ops=ops_view, aggregates=aggregate_rows, total_latency_ms=total_latency)
