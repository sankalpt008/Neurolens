from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Dict

import pytest
"""Pytest configuration for NeuroLens tests."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def has_torch() -> bool:
    return importlib.util.find_spec("torch") is not None


def has_onnxrt() -> bool:
    return importlib.util.find_spec("onnxruntime") is not None


def has_trt() -> bool:
    return importlib.util.find_spec("tensorrt") is not None


def gen_add_onnx(dst: Path) -> Path:
    onnx = pytest.importorskip("onnx", reason="onnx package required for generating models")
    from onnx import TensorProto, helper

    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", 4])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["batch", 4])
    bias = helper.make_tensor("bias", TensorProto.FLOAT, [4], [0.1, 0.2, 0.3, 0.4])
    add_node = helper.make_node("Add", ["input", "bias"], ["output"], name="AddBias")
    graph = helper.make_graph([add_node], "NeurolensAdd", [input_tensor], [output_tensor], [bias])
    model = helper.make_model(graph, producer_name="neurolens.tests.gen_add_onnx")
    onnx.checker.check_model(model)
    dst.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, dst)
    return dst


def gen_tiny_linear(dst_dir: Path) -> Dict[str, Path]:  # pragma: no cover - depends on PyTorch availability
    torch = pytest.importorskip("torch", reason="PyTorch not installed; skipping torch adapter tests")
    from torch import nn

    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))
    model.eval()

    dst_dir.mkdir(parents=True, exist_ok=True)
    pt_path = dst_dir / "tiny_linear.pt"
    torch.save(model, pt_path)

    paths: Dict[str, Path] = {"pt": pt_path}

    try:
        import onnx  # noqa: F401

        dummy_input = torch.randn(1, 4)
        onnx_path = dst_dir / "tiny_linear.onnx"
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            input_names=["input"],
            output_names=["output"],
            opset_version=16,
        )
        paths["onnx"] = onnx_path
    except Exception:
        pass

    return paths


@pytest.fixture
def onnx_model_path(tmp_path: Path) -> Path:
    return gen_add_onnx(tmp_path / "add.onnx")


@pytest.fixture
def torch_model_paths(tmp_path: Path) -> Dict[str, Path]:  # pragma: no cover - depends on PyTorch availability
    return gen_tiny_linear(tmp_path / "tiny")


__all__ = ["has_torch", "has_onnxrt", "has_trt", "gen_add_onnx", "gen_tiny_linear"]
