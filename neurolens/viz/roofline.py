"""Roofline computation utilities for NeuroLens visualizations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence

from neurolens.fingerprint import build_fingerprint

DEFAULT_PEAK_BW_GBPS = 900.0
DEFAULT_PEAK_GFLOPS = 18000.0

__all__ = ["RooflinePoints", "build_roofline_points"]


@dataclass
class RooflinePoints:
    """Container for roofline scatter plot information."""

    ops: List[Dict[str, Any]]
    summary: Dict[str, Any]


def _ensure_run_and_fingerprint(data: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    if "timeline" in data:
        return data, build_fingerprint(data)
    return {}, data


def _vector_index(vector_spec: Sequence[str], name: str) -> int:
    try:
        return list(vector_spec).index(name)
    except ValueError:
        return -1


def _extract_peak_bw(run: Mapping[str, Any], fingerprint: Mapping[str, Any]) -> float:
    hardware_norm = fingerprint.get("hardware_norm", {})
    peak_norm = hardware_norm.get("peak_dram_gbps") if isinstance(hardware_norm, Mapping) else None
    if peak_norm:
        return float(peak_norm)
    hardware = run.get("hardware") if isinstance(run, Mapping) else None
    if isinstance(hardware, Mapping):
        value = hardware.get("peak_dram_gbps")
        if value:
            return float(value)
    summary = run.get("summary") if isinstance(run, Mapping) else None
    if isinstance(summary, Mapping):
        metrics = summary.get("metrics")
        if isinstance(metrics, Mapping):
            throughput = metrics.get("dram_throughput_gbps")
            if throughput:
                return float(throughput)
    return DEFAULT_PEAK_BW_GBPS


def _extract_peak_flops(run: Mapping[str, Any], fingerprint: Mapping[str, Any]) -> float:
    hardware_norm = fingerprint.get("hardware_norm", {})
    if isinstance(hardware_norm, Mapping):
        value = hardware_norm.get("peak_flops_gflops")
        if value:
            return float(value)
    hardware = run.get("hardware") if isinstance(run, Mapping) else None
    if isinstance(hardware, Mapping):
        value = hardware.get("peak_flops_gflops")
        if value:
            return float(value)
    return DEFAULT_PEAK_GFLOPS


def _bytes_from_kernels(entry: Mapping[str, Any]) -> float:
    kernels = entry.get("kernels")
    if not isinstance(kernels, Sequence):
        return 0.0
    total_gb = 0.0
    for kernel in kernels:
        if isinstance(kernel, Mapping):
            total_gb += float(kernel.get("dram_read_gb", 0.0))
            total_gb += float(kernel.get("dram_write_gb", 0.0))
    return total_gb


def _ai_from_entry(entry: Mapping[str, Any]) -> float:
    metrics = entry.get("metrics")
    if isinstance(metrics, Mapping):
        value = metrics.get("ai_flops_per_byte")
        if value is not None:
            return float(value)
        flops = metrics.get("flops")
        bytes_moved = metrics.get("bytes_moved")
        if flops is not None and bytes_moved:
            try:
                return float(flops) / float(bytes_moved)
            except ZeroDivisionError:  # pragma: no cover - guard
                return 0.0
    return 0.0


def _collect_from_run(run: Mapping[str, Any], peak_bw: float, peak_flops: float) -> List[Dict[str, Any]]:
    points: List[Dict[str, Any]] = []
    for entry in run.get("timeline", []):
        if not isinstance(entry, Mapping):
            continue
        duration_ms = float(entry.get("duration_ms", 0.0))
        if duration_ms <= 0.0:
            continue
        ai = _ai_from_entry(entry)
        if ai <= 0.0:
            ai = 0.0
        total_gb = _bytes_from_kernels(entry)
        if total_gb <= 0.0:
            metrics = entry.get("metrics")
            if isinstance(metrics, Mapping):
                bytes_moved = metrics.get("bytes_moved_gb")
                if bytes_moved:
                    total_gb = float(bytes_moved)
        latency_s = duration_ms / 1_000.0
        bytes_total = total_gb * 1_000_000_000.0
        flops = ai * bytes_total
        perf_gflops = (flops / latency_s) / 1_000_000_000.0 if latency_s > 0 else 0.0
        threshold = ai * peak_bw
        bound = "memory" if threshold < peak_flops else "compute"
        points.append(
            {
                "name": entry.get("op_name", "op"),
                "type": entry.get("op_type", "unknown"),
                "ai": ai,
                "perf_gflops": perf_gflops,
                "bound": bound,
                "latency_ms": duration_ms,
            }
        )
    return points


def _collect_from_fingerprint(
    fingerprint: Mapping[str, Any],
    peak_bw: float,
    peak_flops: float,
) -> List[Dict[str, Any]]:
    vector_spec = fingerprint.get("vector_spec", [])
    lat_idx = _vector_index(vector_spec, "lat_norm")
    ai_idx = _vector_index(vector_spec, "ai")
    dram_idx = _vector_index(vector_spec, "dram_norm")
    total_latency = float(fingerprint.get("summary", {}).get("total_latency_ms", 0.0))

    points: List[Dict[str, Any]] = []
    for position, op in enumerate(fingerprint.get("ops", [])):
        vector: Sequence[Any] = op.get("vector", [])  # type: ignore[assignment]
        if lat_idx < 0 or lat_idx >= len(vector):
            continue
        lat_norm = float(vector[lat_idx])
        duration_ms = lat_norm * total_latency
        ai = float(vector[ai_idx]) if 0 <= ai_idx < len(vector) else 0.0
        dram_norm = float(vector[dram_idx]) if 0 <= dram_idx < len(vector) else 0.0

        total_gb = 0.0
        if peak_bw and dram_norm > 0.0:
            total_gb = dram_norm * peak_bw * duration_ms / 1_000.0
        latency_s = duration_ms / 1_000.0
        flops = ai * total_gb * 1_000_000_000.0
        perf_gflops = (flops / latency_s) / 1_000_000_000.0 if latency_s > 0 else 0.0
        threshold = ai * peak_bw
        bound = "memory" if threshold < peak_flops else "compute"
        points.append(
            {
                "name": op.get("name", f"op_{position}"),
                "type": op.get("type", "unknown"),
                "ai": ai,
                "perf_gflops": perf_gflops,
                "bound": bound,
                "latency_ms": duration_ms,
            }
        )
    return points


def build_roofline_points(data: Mapping[str, Any]) -> RooflinePoints:
    """Return roofline data derived from a run.json or fingerprint."""

    run, fingerprint = _ensure_run_and_fingerprint(data)
    if not fingerprint:
        fingerprint = build_fingerprint(data)
    peak_bw = _extract_peak_bw(run, fingerprint)
    peak_flops = _extract_peak_flops(run, fingerprint)

    if run:
        ops = _collect_from_run(run, peak_bw, peak_flops)
    else:
        ops = _collect_from_fingerprint(fingerprint, peak_bw, peak_flops)

    summary = {
        "peak_bw_gbps": peak_bw,
        "peak_flops_gflops": peak_flops,
        "total_ops": len(ops),
    }

    return RooflinePoints(ops=ops, summary=summary)
