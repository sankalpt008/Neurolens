from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from neurolens.adapters import get_adapter
from neurolens.utils.validate import validate_run_schema

BACKENDS_READY = all(importlib.util.find_spec(name) is not None for name in ("onnxruntime", "torch"))
GOLDEN_DIR = Path(__file__).resolve().parents[1] / "golden"


def _load_golden(name: str) -> dict:
    path = GOLDEN_DIR / name
    if not path.exists():
        pytest.skip(f"Golden file missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@pytest.mark.skipif(not BACKENDS_READY, reason="Backends unavailable; skipping golden parity test")
def test_golden_trace_matches(torch_model_paths) -> None:  # pragma: no cover - depends on runtime deps
    onnx_adapter = get_adapter("onnxrt")
    torch_adapter = get_adapter("torch")

    onnx_path = torch_model_paths.get("onnx")
    if onnx_path is None:
        pytest.skip("Tiny linear ONNX export unavailable; skipping golden trace comparison")

    onnx_run = onnx_adapter.run({"path": str(onnx_path)}, {"batch_size": 1, "precision": "fp32"})
    torch_run = torch_adapter.run({"path": str(torch_model_paths["pt"])}, {"batch_size": 1, "precision": "fp32"})

    validate_run_schema(onnx_run)
    validate_run_schema(torch_run)

    golden_onnx = _load_golden("tiny_linear_onnxrt.json")
    golden_torch = _load_golden("tiny_linear_torch.json")

    def timeline_signature(run: dict) -> list[tuple[str, str]]:
        return [(entry["op_name"], entry["op_type"]) for entry in run["timeline"]]

    assert {key for key in onnx_run.keys()} == {key for key in golden_onnx.keys()}
    assert {key for key in torch_run.keys()} == {key for key in golden_torch.keys()}

    assert timeline_signature(onnx_run) == timeline_signature(torch_run)

    def top_k_ops(run: dict, k: int = 3) -> list[str]:
        entries = sorted(run["timeline"], key=lambda item: item["duration_ms"], reverse=True)
        return [entry["op_name"] for entry in entries[:k]]

    assert top_k_ops(onnx_run) == top_k_ops(torch_run)
