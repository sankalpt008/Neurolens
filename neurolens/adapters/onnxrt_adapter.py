"""ONNX Runtime adapter producing NeuroLens timeline artifacts."""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from neurolens.adapters.base import register
from neurolens.utils import get_hardware_info, get_software_info

ProfileEvents = List[Dict[str, Any]]


class OnnxRuntimeNotAvailable(RuntimeError):
    """Raised when ``onnxruntime`` is required but not installed."""


@register
class OnnxRuntimeAdapter:
    """Adapter that normalises ONNX Runtime profiling traces."""

    name = "onnxrt"

    def __init__(
        self,
        session_factory: Optional[Callable[[Path], Any]] = None,
        profile_loader: Optional[Callable[[Path], ProfileEvents]] = None,
        input_provider: Optional[Callable[[Any, Optional[int], Optional[int]], Dict[str, Any]]] = None,
    ) -> None:
        self._session_factory = session_factory or self._default_session_factory
        self._profile_loader = profile_loader or self._default_profile_loader
        self._input_provider = input_provider

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self, model_spec: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        model_path = Path(model_spec.get("path") or model_spec.get("model"))
        if not model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")

        batch_size = config.get("batch_size")
        sequence_length = config.get("sequence_length")
        precision = config.get("precision", "fp32")

        session = self._session_factory(model_path)
        feeds = self._prepare_inputs(session, batch_size=batch_size, sequence_length=sequence_length)
        session.run(None, feeds)
        profile_path = Path(session.end_profiling())
        events = self._profile_loader(profile_path)

        timeline = self._build_timeline(events)
        summary = self._build_summary(timeline)

        run_payload: Dict[str, Any] = {
            "run_id": str(uuid.uuid4()),
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "hardware": get_hardware_info(),
            "software": get_software_info("onnxrt", precision),
            "model": self._model_info(session, model_path, batch_size),
            "timeline": timeline,
            "summary": summary,
        }
        return run_payload

    # ------------------------------------------------------------------
    # Session helpers
    # ------------------------------------------------------------------
    def _default_session_factory(self, model_path: Path) -> Any:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise OnnxRuntimeNotAvailable(
                "onnxruntime must be installed to execute profiling runs"
            ) from exc

        sess_options = ort.SessionOptions()
        sess_options.enable_profiling = True
        return ort.InferenceSession(str(model_path), sess_options=sess_options)

    def _prepare_inputs(
        self,
        session: Any,
        *,
        batch_size: Optional[int],
        sequence_length: Optional[int],
    ) -> Dict[str, Any]:
        if self._input_provider is not None:
            return self._input_provider(session, batch_size, sequence_length)

        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError(
                "numpy is required to prepare dummy inputs for profiling"
            ) from exc

        feeds: Dict[str, Any] = {}
        for input_meta in session.get_inputs():
            name = getattr(input_meta, "name", f"input_{len(feeds)}")
            raw_shape = list(getattr(input_meta, "shape", []))
            dtype = self._resolve_dtype(getattr(input_meta, "type", "tensor(float)"))
            shape = [
                self._resolve_dim(dim, batch_size=batch_size, sequence_length=sequence_length)
                for dim in raw_shape
            ]
            feeds[name] = np.random.rand(*shape).astype(dtype)
        return feeds

    def _resolve_dtype(self, ort_type: str) -> Any:
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError(
                "numpy is required to resolve ONNX Runtime tensor dtypes"
            ) from exc

        mapping = {
            "tensor(float)": np.float32,
            "tensor(float16)": np.float16,
            "tensor(double)": np.float64,
            "tensor(int32)": np.int32,
            "tensor(int64)": np.int64,
        }
        return mapping.get(ort_type, np.float32)

    def _resolve_dim(
        self,
        dim: Any,
        *,
        batch_size: Optional[int],
        sequence_length: Optional[int],
    ) -> int:
        if isinstance(dim, int) and dim > 0:
            return dim
        if isinstance(dim, str):
            key = dim.lower()
            if "batch" in key or key in {"b", "n"}:
                return batch_size or 1
            if key in {"seq", "sequence", "s"}:
                return sequence_length or 1
        if dim is None:
            return batch_size or 1
        return 1

    def _default_profile_loader(self, profile_path: Path) -> ProfileEvents:
        with profile_path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        if isinstance(payload, dict):
            return payload.get("traceEvents", [])
        return payload

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------
    def _model_info(self, session: Any, model_path: Path, batch_size: Optional[int]) -> Dict[str, Any]:
        name = model_path.stem
        opset = self._infer_opset(model_path)
        try:
            metadata = session.get_modelmeta()
            graph_name = getattr(metadata, "graph_name", None)
            if graph_name:
                name = graph_name
        except Exception:
            pass
        return {
            "name": name,
            "framework_opset": opset,
            "batch_size": int(batch_size or 1),
            "parameters_millions": 0,
        }

    def _infer_opset(self, model_path: Path) -> str:
        try:
            import onnx

            model = onnx.load(str(model_path))
            if model.opset_import:
                return f"onnx-opset{model.opset_import[0].version}"
        except Exception:
            pass
        return "onnx-opset-unknown"

    # ------------------------------------------------------------------
    # Timeline + summary
    # ------------------------------------------------------------------
    def _build_timeline(self, events: ProfileEvents) -> List[Dict[str, Any]]:
        node_events = [event for event in events if event.get("cat") == "Node"]
        node_events.sort(key=lambda evt: evt.get("ts", 0))
        if not node_events:
            raise RuntimeError("No node events found in ONNX Runtime profile output")

        origin = node_events[0].get("ts", 0)
        timeline: List[Dict[str, Any]] = []
        for index, event in enumerate(node_events):
            args = event.get("args", {})
            name = args.get("op_name") or event.get("name") or f"node_{index}"
            op_type = args.get("op_type") or args.get("op", "Unknown")
            start_us = event.get("ts", 0) - origin
            dur_us = event.get("dur", 0)
            duration_ms = max(dur_us / 1000.0, 0.000_001)
            start_ms = max(start_us / 1000.0, 0.0)
            end_ms = start_ms + duration_ms

            kernel_entry = {
                "name": f"{name}_kernel",
                "start_ms": start_ms,
                "duration_ms": duration_ms,
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
                "latency_ms": duration_ms,
                "total_latency_ms": duration_ms,
                "p50_ms": duration_ms,
                "p95_ms": duration_ms,
            }

            timeline.append(
                {
                    "op_index": index,
                    "op_name": name,
                    "op_type": op_type,
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "duration_ms": duration_ms,
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


__all__ = ["OnnxRuntimeAdapter", "OnnxRuntimeNotAvailable"]
