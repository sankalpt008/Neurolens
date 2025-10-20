"""Build normalized fingerprints from NeuroLens run artifacts."""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha1
from typing import Any, Dict, List, Mapping, Sequence

from neurolens.utils.io import sha256_json

VECTOR_SPEC: List[str] = [
    "lat_norm",
    "ai",
    "occ",
    "warp_eff",
    "l2_hit",
    "dram_norm",
]

__all__ = ["VECTOR_SPEC", "build_fingerprint"]


def build_fingerprint(
    run_dict: Mapping[str, Any],
    peaks: Mapping[str, float | None] | None = None,
) -> Dict[str, Any]:
    """Return a fingerprint dictionary derived from ``run_dict``."""

    timeline = list(run_dict.get("timeline", []))
    summary = run_dict.get("summary", {})
    total_latency = float(summary.get("total_duration_ms", 0.0))
    total_latency = max(total_latency, 0.0)
    op_latency_total = sum(float(entry.get("duration_ms", 0.0)) for entry in timeline)
    normalizer = op_latency_total if op_latency_total > 0 else total_latency

    if normalizer <= 0 and timeline:
        # Fallback to avoid division by zero when durations are missing
        normalizer = float(len(timeline))
    peaks = peaks or {}

    fingerprint_ops: List[Dict[str, Any]] = []
    for entry in timeline:
        vector = _build_vector(entry, normalizer, peaks)
        fingerprint_ops.append(
            {
                "sig": _op_signature(entry),
                "name": entry.get("op_name", "unknown"),
                "type": entry.get("op_type", "unknown"),
                "index": int(entry.get("op_index", len(fingerprint_ops))),
                "vector": vector,
            }
        )

    fingerprint = {
        "run_id": run_dict.get("run_id"),
        "source_run_sha": sha256_json(run_dict),
        "hardware_norm": {
            "peak_dram_gbps": peaks.get("peak_dram_gbps"),
            "sm_count": peaks.get("sm_count"),
        },
        "vector_spec": VECTOR_SPEC,
        "ops": fingerprint_ops,
        "summary": {
            "total_latency_ms": total_latency,
            "num_ops": len(timeline),
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    return fingerprint


def _op_signature(entry: Mapping[str, Any]) -> str:
    op_type = str(entry.get("op_type", "unknown"))
    index = int(entry.get("op_index", 0))
    shape = _extract_shape(entry)
    payload = f"{op_type}|{shape}|{index}".encode("utf-8")
    return sha1(payload).hexdigest()


def _extract_shape(entry: Mapping[str, Any]) -> str:
    annotations = entry.get("annotations")
    if isinstance(annotations, Mapping):
        shape = annotations.get("shape")
        if shape:
            return str(shape)
    metrics = entry.get("metrics")
    if isinstance(metrics, Mapping):
        shape = metrics.get("shape")
        if shape:
            return str(shape)
    return "unknown"


def _build_vector(
    entry: Mapping[str, Any],
    normalizer: float,
    peaks: Mapping[str, float | None],
) -> List[float]:
    duration_ms = float(entry.get("duration_ms", 0.0))
    lat_norm = duration_ms / normalizer if normalizer > 0 else 0.0

    ai = _compute_ai(entry)
    occ = _mean_kernel_metric(entry, "achieved_occupancy")
    warp_eff = _mean_kernel_metric(entry, "warp_execution_efficiency")
    l2_hit = _mean_kernel_metric(entry, "l2_hit_rate")
    dram_norm = _compute_dram_norm(entry, duration_ms, peaks.get("peak_dram_gbps"))

    return [float(lat_norm), float(ai), float(occ), float(warp_eff), float(l2_hit), float(dram_norm)]


def _compute_ai(entry: Mapping[str, Any]) -> float:
    metrics = entry.get("metrics")
    if isinstance(metrics, Mapping):
        ai_value = metrics.get("ai_flops_per_byte")
        if ai_value is not None:
            return float(ai_value)
        flops = metrics.get("flops")
        bytes_moved = metrics.get("bytes_moved")
        if flops is not None and bytes_moved:
            try:
                return float(flops) / float(bytes_moved)
            except ZeroDivisionError:  # pragma: no cover - guard
                return 0.0
    kernels = entry.get("kernels")
    if isinstance(kernels, Sequence):
        total_bytes = 0.0
        for kernel in kernels:
            if isinstance(kernel, Mapping):
                total_bytes += float(kernel.get("bytes_read", 0.0))
                total_bytes += float(kernel.get("bytes_write", 0.0))
        if total_bytes > 0.0:
            return 0.0  # FLOPs unavailable; treat as 0 but keep normalization consistent
    return 0.0


def _mean_kernel_metric(entry: Mapping[str, Any], key: str) -> float:
    kernels = entry.get("kernels")
    if not isinstance(kernels, Sequence) or not kernels:
        return 0.0
    values: List[float] = []
    for kernel in kernels:
        if isinstance(kernel, Mapping):
            value = kernel.get(key)
            if value is not None:
                values.append(float(value))
    if not values:
        return 0.0
    return sum(values) / len(values)


def _compute_dram_norm(
    entry: Mapping[str, Any],
    duration_ms: float,
    peak_dram: float | None,
) -> float:
    if not peak_dram:
        return 0.0
    kernels = entry.get("kernels")
    if not isinstance(kernels, Sequence) or not kernels:
        return 0.0
    total_gb = 0.0
    for kernel in kernels:
        if isinstance(kernel, Mapping):
            total_gb += float(kernel.get("dram_read_gb", 0.0))
            total_gb += float(kernel.get("dram_write_gb", 0.0))
    if duration_ms <= 0.0 or total_gb <= 0.0:
        return 0.0
    try:
        throughput = total_gb / duration_ms
        return float(throughput / peak_dram)
    except ZeroDivisionError:  # pragma: no cover - peak guard
        return 0.0
