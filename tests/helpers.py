from __future__ import annotations

from typing import Any, Dict, List


def build_insights_sample_run() -> Dict[str, Any]:
    total_latency = 10.0
    summary_metrics = {
        "gpu_utilization": 0.35,
        "sm_efficiency": 0.5,
        "dram_throughput_gbps": 0.8,
        "ai_flops_per_byte": 4.2,
    }

    timeline: List[Dict[str, Any]] = [
        {
            "op_index": 0,
            "op_name": "EncoderBlock_0.MatMul_QK",
            "op_type": "MatMul",
            "start_ms": 0.0,
            "end_ms": 3.5,
            "duration_ms": 3.5,
            "api_launch_overhead_ms": 0.3,
            "kernels": [
                {
                    "name": "matmul_main",
                    "start_ms": 0.0,
                    "duration_ms": 3.2,
                    "achieved_occupancy": 0.65,
                    "warp_execution_efficiency": 0.68,
                    "sm_efficiency": 0.72,
                    "gpu_utilization": 0.6,
                    "l2_hit_rate": 0.3,
                    "dram_read_gb": 2.1,
                    "dram_write_gb": 0.8,
                    "bytes_read": 2.1e9,
                    "bytes_write": 0.8e9,
                }
            ],
            "metrics": {
                "latency_ms": 3.5,
            },
        },
        {
            "op_index": 1,
            "op_name": "EncoderBlock_0.LayerNorm",
            "op_type": "LayerNormalization",
            "start_ms": 3.5,
            "end_ms": 6.0,
            "duration_ms": 2.5,
            "api_launch_overhead_ms": 0.4,
            "kernels": [
                {
                    "name": "layernorm_kernel",
                    "start_ms": 3.6,
                    "duration_ms": 2.2,
                    "achieved_occupancy": 0.3,
                    "warp_execution_efficiency": 0.5,
                    "sm_efficiency": 0.45,
                    "gpu_utilization": 0.4,
                    "l2_hit_rate": 0.6,
                    "dram_read_gb": 0.6,
                    "dram_write_gb": 0.2,
                    "bytes_read": 0.6e9,
                    "bytes_write": 0.2e9,
                }
            ],
            "metrics": {
                "latency_ms": 2.5,
            },
        },
        {
            "op_index": 2,
            "op_name": "EncoderBlock_0.MLP_GEMM",
            "op_type": "Gemm",
            "start_ms": 6.0,
            "end_ms": 10.0,
            "duration_ms": 4.0,
            "api_launch_overhead_ms": 0.2,
            "kernels": [
                {
                    "name": "gemm_main",
                    "start_ms": 6.1,
                    "duration_ms": 3.6,
                    "achieved_occupancy": 0.78,
                    "warp_execution_efficiency": 0.82,
                    "sm_efficiency": 0.8,
                    "gpu_utilization": 0.7,
                    "l2_hit_rate": 0.72,
                    "dram_read_gb": 1.0,
                    "dram_write_gb": 0.4,
                    "bytes_read": 1.0e9,
                    "bytes_write": 0.4e9,
                }
            ],
            "metrics": {
                "latency_ms": 4.0,
            },
        },
    ]

    return {
        "run_id": "00000000-0000-0000-0000-000000000001",
        "created_at": "2025-01-01T00:00:00Z",
        "hardware": {
            "vendor": "NVIDIA",
            "gpu_model": "TestGPU",
            "gpu_count": 1,
            "memory_gb": 16,
            "driver_version": "000.000",
        },
        "software": {
            "backend": "onnxrt",
            "version": "1.0",
            "precision": "fp16",
        },
        "model": {
            "name": "SyntheticModel",
            "framework_opset": "onnx-18",
            "batch_size": 8,
            "parameters_millions": 100.0,
        },
        "timeline": timeline,
        "summary": {
            "total_duration_ms": total_latency,
            "total_kernels": 3,
            "top_bottleneck": "EncoderBlock_0.MatMul_QK",
            "metrics": {
                "sm_efficiency": summary_metrics["sm_efficiency"],
                "gpu_utilization": summary_metrics["gpu_utilization"],
                "dram_throughput_gbps": summary_metrics["dram_throughput_gbps"],
                "ai_flops_per_byte": summary_metrics["ai_flops_per_byte"],
            },
        },
    }
