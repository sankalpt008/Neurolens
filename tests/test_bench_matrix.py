import copy
import importlib
import sys
from pathlib import Path
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

bench_module = importlib.import_module("neurolens.bench.run_matrix")
from neurolens.core.profiler import ProfilerResult  # noqa: E402
from neurolens.utils.io import read_json  # noqa: E402
from neurolens.core.profiler import ProfilerResult
from neurolens.utils.io import read_json

SAMPLE_RUN = read_json(PROJECT_ROOT / "samples" / "trace_minimal.json")


def _stub_profile_model(output_dir, backend, cfg):
    run_copy = copy.deepcopy(SAMPLE_RUN)
    run_copy["run_id"] = str(uuid4())
    run_copy["software"]["backend"] = backend
    run_copy["software"]["precision"] = cfg["precision"]
    run_copy["model"]["batch_size"] = cfg["batch_size"]
    summary = run_copy.setdefault("summary", {})
    summary["total_duration_ms"] = 3.14
    metrics = summary.setdefault("metrics", {})
    metrics["gpu_utilization"] = 0.42
    path = Path(output_dir) / "placeholder.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}", encoding="utf-8")
    return ProfilerResult(run_dict=run_copy, output_path=path)


def test_run_matrix_generates_artifacts(tmp_path, monkeypatch):
    def fake_profile_model(backend, model_spec, cfg, output_dir=None):
        return _stub_profile_model(output_dir or tmp_path / "runs", backend, cfg)

    monkeypatch.setattr(bench_module, "profile_model", fake_profile_model)
    monkeypatch.setattr(
        bench_module,
        "collect_env",
        lambda: {
            "driver_version": "unknown",
            "cuda_version": "unknown",
            "gpu_name": "mock-gpu",
            "os": "Linux",
            "python_version": "3.x",
        },
    )

    grid = {"batch_size": [1, 2], "seq_len": [16, 32], "precision": ["fp32"], "repeats": 1, "tags": {"suite": "unit"}}
    rows = bench_module.run_matrix("./model.onnx", "onnxrt", grid, tmp_path)

    assert len(rows) == 4
    runs_dir = tmp_path / "runs"
    fp_dir = tmp_path / "fp"
    assert len(list(runs_dir.glob("*.json"))) == 4
    assert len(list(fp_dir.glob("*.fp.json"))) == 4

    summary_path = tmp_path / "summary.csv"
    assert summary_path.exists()
    contents = summary_path.read_text(encoding="utf-8")
    assert "run_id" in contents and "batch_size" in contents and "seq_len" in contents

    sample_run = read_json(next(runs_dir.glob("*.json")))
    assert "meta" in sample_run
    assert sample_run["meta"]["tags"]["suite"] == "unit"
    assert sample_run["meta"]["config"]["batch_size"] in {1, 2}
