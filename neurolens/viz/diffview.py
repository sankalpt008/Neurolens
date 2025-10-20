"""Diff alignment helpers for NeuroLens fingerprint visualization."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence

from neurolens.fingerprint.similarity import build_alignment, diff as fingerprint_diff

__all__ = ["DiffDataset", "build_diff_dataset"]


@dataclass
class DiffDataset:
    """Data structure containing aligned fingerprint vectors for visualization."""

    vector_spec: Sequence[str]
    rows: List[Dict[str, Any]]
    summary: Dict[str, Any]


def build_diff_dataset(
    fingerprint_a: Mapping[str, Any],
    fingerprint_b: Mapping[str, Any],
    *,
    topk: int = 10,
) -> DiffDataset:
    """Align two fingerprints and compute vector deltas for visualization."""

    spec = list(fingerprint_a.get("vector_spec", []))
    ops_a = fingerprint_a.get("ops", [])
    ops_b = fingerprint_b.get("ops", [])
    aligned_a, aligned_b, unmatched_a, unmatched_b = build_alignment(ops_a, ops_b)

    rows: List[Dict[str, Any]] = []
    for index, (op_a, op_b) in enumerate(zip(aligned_a, aligned_b)):
        vector_a = list(op_a.get("vector", []))
        vector_b = list(op_b.get("vector", []))
        deltas = [float(vector_b[i]) - float(vector_a[i]) if i < len(vector_a) and i < len(vector_b) else 0.0 for i in range(len(spec))]
        rows.append(
            {
                "sig": op_a.get("sig"),
                "name": op_a.get("name"),
                "type": op_a.get("type"),
                "vector_a": vector_a,
                "vector_b": vector_b,
                "delta": deltas,
                "index": index,
            }
        )

    summary = fingerprint_diff(fingerprint_a, fingerprint_b, topk=topk)
    summary["unmatched_a"] = len(unmatched_a)
    summary["unmatched_b"] = len(unmatched_b)

    return DiffDataset(vector_spec=spec, rows=rows, summary=summary)
