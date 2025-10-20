"""Similarity and diff utilities for NeuroLens fingerprints."""
from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Sequence, Tuple

Vector = Sequence[float]
Op = Mapping[str, Any]

__all__ = ["build_alignment", "cosine_similarity", "diff"]


def build_alignment(
    ops_a: Sequence[Op], ops_b: Sequence[Op]
) -> Tuple[List[Op], List[Op], List[Op], List[Op]]:
    """Align operations by signature and return matched/unmatched lists."""

    index_b = {op.get("sig"): op for op in ops_b}
    aligned_a: List[Op] = []
    aligned_b: List[Op] = []
    unmatched_a: List[Op] = []

    matched_signatures = set()
    for op in ops_a:
        signature = op.get("sig")
        match = index_b.get(signature)
        if match is None:
            unmatched_a.append(op)
            continue
        aligned_a.append(op)
        aligned_b.append(match)
        matched_signatures.add(signature)

    unmatched_b = [op for op in ops_b if op.get("sig") not in matched_signatures]
    return aligned_a, aligned_b, unmatched_a, unmatched_b


def cosine_similarity(vectors_a: Sequence[Vector], vectors_b: Sequence[Vector]) -> float:
    """Compute cosine similarity over two equally sized sequences of vectors."""

    if len(vectors_a) != len(vectors_b):
        raise ValueError("Aligned vectors must share the same length")
    if not vectors_a:
        return 0.0

    flat_a = _flatten(vectors_a)
    flat_b = _flatten(vectors_b)

    dot = sum(a * b for a, b in zip(flat_a, flat_b))
    norm_a = math.sqrt(sum(a * a for a in flat_a))
    norm_b = math.sqrt(sum(b * b for b in flat_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def diff(
    fingerprint_a: Mapping[str, Any],
    fingerprint_b: Mapping[str, Any],
    *,
    topk: int = 5,
) -> Dict[str, Any]:
    """Return similarity and divergence statistics between two fingerprints."""

    ops_a = fingerprint_a.get("ops", [])
    ops_b = fingerprint_b.get("ops", [])
    aligned_a, aligned_b, unmatched_a, unmatched_b = build_alignment(ops_a, ops_b)

    vectors_a = [op.get("vector", []) for op in aligned_a]
    vectors_b = [op.get("vector", []) for op in aligned_b]
    similarity_score = cosine_similarity(vectors_a, vectors_b) if aligned_a else 0.0

    divergences = _compute_divergences(
        fingerprint_a,
        fingerprint_b,
        aligned_a,
        aligned_b,
    )
    divergences = sorted(divergences, key=lambda item: item["distance"], reverse=True)
    top_entries = divergences[:topk]

    return {
        "similarity": similarity_score,
        "top_divergences": [entry["report"] for entry in top_entries],
        "unmatched_a": len(unmatched_a),
        "unmatched_b": len(unmatched_b),
    }


def _flatten(vectors: Sequence[Vector]) -> List[float]:
    flattened: List[float] = []
    for vector in vectors:
        flattened.extend(float(value) for value in vector)
    return flattened


def _compute_divergences(
    fingerprint_a: Mapping[str, Any],
    fingerprint_b: Mapping[str, Any],
    aligned_a: Sequence[Op],
    aligned_b: Sequence[Op],
) -> List[Dict[str, Any]]:
    spec = fingerprint_a.get("vector_spec", [])
    total_latency_a = float(fingerprint_a.get("summary", {}).get("total_latency_ms", 0.0))
    total_latency_b = float(fingerprint_b.get("summary", {}).get("total_latency_ms", 0.0))

    divergences: List[Dict[str, Any]] = []
    for op_a, op_b in zip(aligned_a, aligned_b):
        vector_a = list(op_a.get("vector", []))
        vector_b = list(op_b.get("vector", []))
        length = min(len(vector_a), len(vector_b), len(spec))
        deltas = {}
        distance = 0.0
        for idx in range(length):
            feature = spec[idx]
            delta = float(vector_b[idx]) - float(vector_a[idx])
            deltas[feature] = delta
            distance += delta * delta
        distance = math.sqrt(distance)

        lat_norm_a = vector_a[0] if vector_a else 0.0
        lat_norm_b = vector_b[0] if vector_b else 0.0
        latency_a = lat_norm_a * total_latency_a
        latency_b = lat_norm_b * total_latency_b
        if latency_a > 0:
            latency_pct = ((latency_b - latency_a) / latency_a) * 100.0
        else:
            latency_pct = None

        divergences.append(
            {
                "distance": distance,
                "report": {
                    "sig": op_a.get("sig"),
                    "name": op_a.get("name"),
                    "type": op_a.get("type"),
                    "delta": deltas,
                    "latency_pct_change": latency_pct,
                },
            }
        )
    return divergences
