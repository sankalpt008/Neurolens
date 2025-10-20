from __future__ import annotations

import pytest

from neurolens.adapters import get_adapter
from neurolens.utils.validate import validate_run_schema

REQUIRED_TOP_LEVEL_KEYS = {"run_id", "created_at", "hardware", "software", "model", "timeline", "summary"}


@pytest.mark.parametrize("backend", ["onnxrt", "torch", "tensorrt"])
def test_adapter_outputs_validate(backend: str, request: pytest.FixtureRequest) -> None:
    if backend == "onnxrt":
        pytest.importorskip("onnxruntime", reason="onnxruntime not installed; skipping ONNX Runtime adapter test")
    if backend == "torch":
        pytest.importorskip("torch", reason="PyTorch not installed; skipping torch adapter test")

    adapter = get_adapter(backend)

    if backend == "onnxrt":
        onnx_model_path = request.getfixturevalue("onnx_model_path")
        model_spec = {"path": str(onnx_model_path)}
        config = {"batch_size": 1, "sequence_length": 1, "precision": "fp32"}
    elif backend == "torch":
        model_spec = {"builtin": "tiny_linear"}
        config = {"batch_size": 1, "precision": "fp32"}
    else:  # tensorrt
        onnx_model_path = request.getfixturevalue("onnx_model_path")
        model_spec = {"path": str(onnx_model_path)}
        config = {"precision": "fp32", "allow_stub_trt": True}

    run_payload = adapter.run(model_spec, config)
    validate_run_schema(run_payload)

    assert set(run_payload.keys()) == REQUIRED_TOP_LEVEL_KEYS
    for entry in run_payload["timeline"]:
        assert set(entry.keys()).issuperset({"op_index", "op_name", "op_type", "duration_ms", "kernels"})
        for kernel in entry["kernels"]:
            assert set(kernel.keys()).issuperset({"name", "duration_ms"})
