from __future__ import annotations

from pathlib import Path

from neurolens.storage.store import query_index, rebuild_index_from_manifest, write_run

from tests.helpers import build_insights_sample_run


def _sample_env() -> dict:
    return {
        "driver_version": "550.00",
        "cuda_version": "12.4",
        "gpu_name": "TestGPU",
        "os": "Linux 6.0",
        "python_version": "3.11.0",
    }


def test_write_run_creates_partition_and_index(tmp_path: Path) -> None:
    run = build_insights_sample_run()
    run["created_at"] = "2025-10-20T05:00:00Z"
    run["run_id"] = "11111111-1111-1111-1111-111111111111"
    run.setdefault("meta", {})["env"] = _sample_env()
    run["meta"]["tags"] = {"experiment": "unit", "note": "smoke"}
    run["meta"]["config"] = {
        "batch_size": run["model"].get("batch_size", 8),
        "sequence_length": 128,
        "precision": run["software"].get("precision", "fp16"),
        "backend": run["software"].get("backend", "onnxrt"),
        "repeat": 1,
    }

    result_paths = write_run(tmp_path, run)

    partition_path = tmp_path / "2025" / "10" / "20" / "onnxrt"
    assert partition_path.exists()
    run_path = partition_path / f"{run['run_id']}.json"
    assert run_path.exists()
    assert result_paths["run_json"] == str(run_path.relative_to(tmp_path))

    manifest_path = tmp_path / "_manifest.jsonl"
    assert manifest_path.exists()

    index_path = Path(rebuild_index_from_manifest(tmp_path))
    assert index_path.exists()

    rows = query_index(tmp_path, backend="onnxrt")
    assert len(rows) == 1
    row = rows[0]
    assert row["run_id"] == run["run_id"]
    assert row["path_run_json"] == result_paths["run_json"]
    assert row["gpu_name"] == "TestGPU"
    assert row["tags"]
