from __future__ import annotations

from pathlib import Path
from typing import Dict

import pytest

from neurolens.storage import store

from tests.helpers import build_insights_sample_run


def _make_run(run_id: str, created_at: str, backend: str, model_name: str, batch: int, seq: int, precision: str, tags: Dict[str, str]) -> Dict:
    run = build_insights_sample_run()
    run["run_id"] = run_id
    run["created_at"] = created_at
    run.setdefault("software", {})["backend"] = backend
    run["software"]["precision"] = precision
    run.setdefault("model", {})["name"] = model_name
    run["model"]["batch_size"] = batch
    run.setdefault("meta", {})["env"] = {
        "driver_version": "550.00",
        "cuda_version": "12.4",
        "gpu_name": f"GPU-{backend}",
        "os": "Linux 6.0",
        "python_version": "3.11.0",
    }
    run["meta"]["tags"] = tags
    run["meta"]["config"] = {
        "batch_size": batch,
        "sequence_length": seq,
        "precision": precision,
        "backend": backend,
        "repeat": 1,
    }
    return run


@pytest.mark.parametrize("parquet_enabled", [True, False])
def test_query_filters(tmp_path: Path, monkeypatch, parquet_enabled: bool) -> None:
    original = store.PARQUET_AVAILABLE
    if parquet_enabled and not original:
        pytest.skip("Parquet dependencies not available")
    monkeypatch.setattr(store, "PARQUET_AVAILABLE", parquet_enabled)
    runs = [
        _make_run("00000000-0000-0000-0000-00000000000a", "2025-10-20T01:00:00Z", "onnxrt", "gpt2", 1, 64, "fp16", {"experiment": "smoke"}),
        _make_run("00000000-0000-0000-0000-00000000000b", "2025-10-20T02:00:00Z", "onnxrt", "gpt2", 2, 128, "fp32", {"experiment": "sweep"}),
        _make_run("00000000-0000-0000-0000-00000000000c", "2025-10-21T03:00:00Z", "torch", "bert", 4, 128, "fp16", {"note": "baseline"}),
    ]

    for payload in runs:
        store.write_run(tmp_path, payload)

    index_path = store.rebuild_index_from_manifest(tmp_path)
    assert Path(index_path).exists()

    all_rows = store.query_index(tmp_path)
    assert len(all_rows) == 3

    onnx_rows = store.query_index(tmp_path, backend="onnxrt")
    assert len(onnx_rows) == 2

    fp16_rows = store.query_index(tmp_path, precision="fp16")
    assert {row["run_id"] for row in fp16_rows} == {
        "00000000-0000-0000-0000-00000000000a",
        "00000000-0000-0000-0000-00000000000c",
    }

    day_rows = store.query_index(tmp_path, day="2025-10-20")
    assert {row["run_id"] for row in day_rows} == {
        "00000000-0000-0000-0000-00000000000a",
        "00000000-0000-0000-0000-00000000000b",
    }

    tag_rows = store.query_index(tmp_path, tags=["experiment=smoke"])
    assert len(tag_rows) == 1
    assert tag_rows[0]["run_id"] == "00000000-0000-0000-0000-00000000000a"

    limited = store.query_index(tmp_path, backend="onnxrt", limit=1)
    assert len(limited) == 1
