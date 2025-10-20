"""Summary-specific validations for the profiler."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from neurolens.adapters.onnxrt_adapter import OnnxRuntimeAdapter
from neurolens.core.profiler import Profiler


class _FakeModelMeta:
    graph_name = "SummaryGraph"
    graph_description = ""
    graph_version = 16
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
    model_path = tmp_path / "summary.onnx"
    model_path.write_text("placeholder", encoding="utf-8")
    return adapter, model_path


def test_summary_tracks_top_three_ops(tmp_path):
    events = [
        {"cat": "Node", "name": "MatMul_0", "ts": 0, "dur": 5000, "args": {"op_name": "MatMul_0", "op_type": "MatMul"}},
        {"cat": "Node", "name": "Softmax_1", "ts": 5000, "dur": 2000, "args": {"op_name": "Softmax_1", "op_type": "Softmax"}},
        {"cat": "Node", "name": "Add_2", "ts": 7000, "dur": 3000, "args": {"op_name": "Add_2", "op_type": "Add"}},
    ]
    adapter, model_path = _build_adapter(tmp_path, events)
    profiler = Profiler(adapter, output_dir=tmp_path / "runs")

    result = profiler.profile(model_path)
    timeline = result.run_dict["timeline"]
    summary = result.run_dict["summary"]

    durations = [entry["duration_ms"] for entry in timeline]
    assert summary["total_duration_ms"] == pytest.approx(sum(durations))

    ordered = sorted(timeline, key=lambda entry: entry["duration_ms"], reverse=True)
    top_names = [entry["op_name"] for entry in ordered[:3]]
    assert top_names == ["MatMul_0", "Add_2", "Softmax_1"]
    assert summary["top_bottleneck"] == "MatMul_0"
    assert summary["total_kernels"] == len(timeline)

