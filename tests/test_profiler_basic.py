"""Basic sanity checks for the ONNX Runtime profiling pipeline."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from neurolens.adapters.onnxrt_adapter import OnnxRuntimeAdapter
from neurolens.core.profiler import Profiler, validate_run_schema


class _FakeModelMeta:
    graph_name = "UnitTestGraph"
    graph_description = ""
    graph_version = 17
    producer_name = "unit-tests"


class _FakeSession:
    def __init__(self, profile_path: Path) -> None:
        self._profile_path = profile_path

    def get_inputs(self):
        return []

    def run(self, outputs, feeds):
        self.last_feeds = feeds

    def end_profiling(self):
        return str(self._profile_path)

    def get_modelmeta(self):
        return _FakeModelMeta()


def _build_adapter(tmp_path: Path, events: list[dict]) -> tuple[OnnxRuntimeAdapter, Path]:
    profile_file = tmp_path / "profile.json"
    profile_file.write_text(json.dumps(events), encoding="utf-8")
    session = _FakeSession(profile_file)
    adapter = OnnxRuntimeAdapter(
        session_factory=lambda path: session,
        profile_loader=lambda path: events,
        input_provider=lambda session, batch, seq: {},
    )
    model_path = tmp_path / "dummy.onnx"
    model_path.write_text("placeholder", encoding="utf-8")
    return adapter, model_path


def test_profiler_creates_schema_valid_artifact(tmp_path):
    events = [
        {"cat": "Node", "name": "Add_0", "ts": 0, "dur": 2000, "args": {"op_name": "Add_0", "op_type": "Add"}},
        {
            "cat": "Node",
            "name": "Relu_1",
            "ts": 2000,
            "dur": 1000,
            "args": {"op_name": "Relu_1", "op_type": "Relu"},
        },
    ]
    adapter, model_path = _build_adapter(tmp_path, events)
    profiler = Profiler(adapter, output_dir=tmp_path / "runs")

    result = profiler.profile(model_path)

    assert result.output_path.exists()
    payload = json.loads(result.output_path.read_text(encoding="utf-8"))
    assert validate_run_schema(payload)
    assert payload["summary"]["total_duration_ms"] == pytest.approx(
        sum(entry["duration_ms"] for entry in payload["timeline"])
    )
    assert payload["summary"]["total_kernels"] == sum(len(entry["kernels"]) for entry in payload["timeline"])
    assert payload["summary"]["top_bottleneck"] == "Add_0"
