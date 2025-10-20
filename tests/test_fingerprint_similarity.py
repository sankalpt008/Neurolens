from __future__ import annotations

from typing import List

from neurolens.fingerprint import build_fingerprint, diff


def _make_run(latencies: List[float]) -> dict:
    timeline = []
    start = 0.0
    for idx, duration in enumerate(latencies):
        end = start + duration
        timeline.append(
            {
                "op_index": idx,
                "op_name": f"Op_{idx}",
                "op_type": "Linear",
                "start_ms": start,
                "end_ms": end,
                "duration_ms": duration,
                "api_launch_overhead_ms": 0.0,
                "kernels": [
                    {
                        "name": f"kernel_{idx}",
                        "start_ms": start,
                        "duration_ms": duration,
                        "achieved_occupancy": 0.5,
                        "warp_execution_efficiency": 0.6,
                        "sm_efficiency": 0.5,
                        "gpu_utilization": 0.4,
                        "l2_hit_rate": 0.7,
                        "dram_read_gb": 0.1,
                        "dram_write_gb": 0.05,
                        "bytes_read": 1.0,
                        "bytes_write": 1.0,
                    }
                ],
                "metrics": {
                    "latency_ms": duration,
                    "total_latency_ms": duration,
                    "ai_flops_per_byte": 0.0,
                },
            }
        )
        start = end

    total_latency = sum(latencies)
    run = {
        "run_id": "00000000-0000-0000-0000-000000000000",
        "created_at": "2025-01-01T00:00:00Z",
        "hardware": {
            "vendor": "Stub",
            "gpu_model": "StubGPU",
            "gpu_count": 1,
            "memory_gb": 1.0,
            "driver_version": "0",
        },
        "software": {
            "backend": "onnxrt",
            "version": "stub",
            "precision": "fp32",
        },
        "model": {
            "name": "stub",
            "framework_opset": "stub",
            "batch_size": 1,
            "parameters_millions": 0.0,
        },
        "timeline": timeline,
        "summary": {
            "total_duration_ms": total_latency,
            "total_kernels": len(timeline),
            "top_bottleneck": timeline[0]["op_name"],
            "metrics": {
                "sm_efficiency": 0.5,
                "gpu_utilization": 0.4,
            },
        },
    }
    return run


def test_similarity_and_divergence_ranking() -> None:
    base_run = _make_run([1.0, 2.0, 1.5])
    variant_run = _make_run([1.2, 2.5, 1.0])

    fp_base = build_fingerprint(base_run)
    fp_variant = build_fingerprint(variant_run)

    report = diff(fp_base, fp_variant, topk=3)

    assert report["similarity"] < 1.0
    assert report["unmatched_a"] == 0
    assert report["unmatched_b"] == 0
    assert len(report["top_divergences"]) == 3

    top_names = [entry["name"] for entry in report["top_divergences"]]
    assert top_names == ["Op_2", "Op_1", "Op_0"]

    deltas = report["top_divergences"][0]["delta"]
    assert "lat_norm" in deltas
    assert abs(deltas["lat_norm"]) > 0.0
