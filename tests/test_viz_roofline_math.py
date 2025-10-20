from __future__ import annotations

from neurolens.viz.roofline import build_roofline_points


def test_roofline_classification() -> None:
    run = {
        "timeline": [
            {
                "op_name": "op_memory",
                "op_type": "MatMul",
                "duration_ms": 1.0,
                "kernels": [
                    {"dram_read_gb": 0.1, "dram_write_gb": 0.1},
                ],
                "metrics": {"ai_flops_per_byte": 2.0},
            },
            {
                "op_name": "op_compute",
                "op_type": "MatMul",
                "duration_ms": 1.0,
                "kernels": [
                    {"dram_read_gb": 0.1, "dram_write_gb": 0.1},
                ],
                "metrics": {"ai_flops_per_byte": 120.0},
            },
        ],
        "summary": {"total_duration_ms": 2.0},
    }

    points = build_roofline_points(run)
    bounds = {item["name"]: item["bound"] for item in points.ops}

    assert bounds["op_memory"] == "memory"
    assert bounds["op_compute"] == "compute"
    assert all(item["perf_gflops"] >= 0.0 for item in points.ops)
