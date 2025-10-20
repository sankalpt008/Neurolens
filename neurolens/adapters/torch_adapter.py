"""PyTorch adapter leveraging the built-in profiler."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from neurolens.adapters.base import register
from neurolens.utils import get_hardware_info, get_software_info

OP_TYPE_MAP = {
    "aten::mm": "MatMul",
    "aten::add": "Add",
    "aten::relu": "Relu",
    "aten::softmax": "Softmax",
    "aten::linear": "Linear",
}


@register
class TorchAdapter:
    """PyTorch runtime adapter."""

    name = "torch"

    def run(self, model_spec: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        try:
            import torch
            from torch import nn
            from torch.profiler import ProfilerActivity, profile
        except Exception as exc:  # pragma: no cover - executed when torch missing
            raise RuntimeError("PyTorch is required for the torch adapter") from exc

        model = self._load_model(torch, nn, model_spec)
        model.eval()

        batch_size = int(config.get("batch_size") or 1)
        precision = config.get("precision", "fp32")
        input_dim = int(config.get("input_dim") or 4)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        dummy_input = torch.randn(batch_size, input_dim, device=device)

        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():  # pragma: no cover - depends on CI hardware
            activities.append(ProfilerActivity.CUDA)

        with profile(activities=activities, record_shapes=True, with_stack=False) as prof:
            with torch.no_grad():
                model(dummy_input)

        events = [evt for evt in prof.key_averages() if evt.key.startswith("aten::")]
        if not events:
            raise RuntimeError("PyTorch profiler did not capture aten ops")

        timeline = self._build_timeline(events)
        summary = self._build_summary(timeline)

        run_payload: Dict[str, Any] = {
            "run_id": str(uuid.uuid4()),
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "hardware": get_hardware_info(),
            "software": get_software_info("torch", precision),
            "model": {
                "name": self._model_name(model_spec),
                "framework_opset": "torch",  # torch has no opset concept
                "batch_size": batch_size,
                "parameters_millions": 0,
            },
            "timeline": timeline,
            "summary": summary,
        }
        return run_payload

    # ------------------------------------------------------------------
    # Model helpers
    # ------------------------------------------------------------------
    def _load_model(self, torch: Any, nn: Any, model_spec: Dict[str, Any]):
        builtin = model_spec.get("builtin")
        if builtin == "tiny_linear":
            return self._build_tiny_linear(nn)

        path_val = model_spec.get("path")
        if not path_val:
            raise ValueError("PyTorch adapter requires 'path' or builtin model")
        path = Path(path_val)
        if not path.exists():
            raise FileNotFoundError(f"PyTorch model not found: {path}")
        try:
            return torch.jit.load(str(path))
        except Exception:
            return torch.load(path)

    def _build_tiny_linear(self, nn: Any):
        class TinyLinear(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = nn.Linear(4, 2)

            def forward(self, x):  # type: ignore[override]
                return self.linear(x)

        return TinyLinear()

    def _model_name(self, model_spec: Dict[str, Any]) -> str:
        if model_spec.get("builtin"):
            return model_spec["builtin"]
        return Path(model_spec.get("path", "unknown")).stem

    # ------------------------------------------------------------------
    # Timeline + summary
    # ------------------------------------------------------------------
    def _build_timeline(self, events: List[Any]) -> List[Dict[str, Any]]:
        timeline: List[Dict[str, Any]] = []
        cursor = 0.0
        for index, evt in enumerate(events):
            dur_ms = max(evt.cpu_time_total / 1_000_000.0, 0.000_001)
            op_name = evt.key
            op_type = OP_TYPE_MAP.get(op_name, op_name.split("::")[-1].title())
            start_ms = cursor
            end_ms = start_ms + dur_ms
            cursor = end_ms

            kernel_entry = {
                "name": f"{op_name}_kernel",
                "start_ms": start_ms,
                "duration_ms": dur_ms,
                "achieved_occupancy": 0.0,
                "warp_execution_efficiency": 0.0,
                "sm_efficiency": 0.0,
                "gpu_utilization": 0.0,
                "l2_hit_rate": 0.0,
                "dram_read_gb": 0.0,
                "dram_write_gb": 0.0,
                "bytes_read": 0.0,
                "bytes_write": 0.0,
            }

            metrics = {
                "latency_ms": dur_ms,
                "total_latency_ms": dur_ms,
                "p50_ms": dur_ms,
                "p95_ms": dur_ms,
            }

            timeline.append(
                {
                    "op_index": index,
                    "op_name": op_name,
                    "op_type": op_type,
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "duration_ms": dur_ms,
                    "api_launch_overhead_ms": 0.0,
                    "kernels": [kernel_entry],
                    "metrics": metrics,
                }
            )
        return timeline

    def _build_summary(self, timeline: List[Dict[str, Any]]) -> Dict[str, Any]:
        total_latency = sum(entry["duration_ms"] for entry in timeline)
        top_entry = max(timeline, key=lambda entry: entry.get("duration_ms", 0.0))
        total_kernels = sum(len(entry.get("kernels", [])) for entry in timeline)
        return {
            "total_duration_ms": total_latency,
            "total_kernels": total_kernels or 1,
            "top_bottleneck": top_entry["op_name"],
            "metrics": {
                "sm_efficiency": 0.0,
                "gpu_utilization": 0.0,
                "dram_throughput_gbps": 0.0,
                "ai_flops_per_byte": 0.0,
            },
        }
