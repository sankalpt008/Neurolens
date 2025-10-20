from __future__ import annotations

import pytest

from neurolens.viz.diffview import build_diff_dataset


def test_diff_dataset_alignment_orders_by_index() -> None:
    spec = ["lat_norm", "ai", "occ"]
    fingerprint_a = {
        "vector_spec": spec,
        "summary": {"total_latency_ms": 10.0},
        "ops": [
            {"sig": "a", "name": "opA", "type": "MatMul", "vector": [0.4, 2.0, 0.5], "index": 0},
            {"sig": "b", "name": "opB", "type": "Add", "vector": [0.6, 1.0, 0.3], "index": 1},
        ],
    }
    fingerprint_b = {
        "vector_spec": spec,
        "summary": {"total_latency_ms": 11.0},
        "ops": [
            {"sig": "a", "name": "opA", "type": "MatMul", "vector": [0.5, 2.5, 0.55], "index": 0},
            {"sig": "b", "name": "opB", "type": "Add", "vector": [0.5, 1.2, 0.25], "index": 1},
        ],
    }

    dataset = build_diff_dataset(fingerprint_a, fingerprint_b, topk=2)

    assert len(dataset.rows) == 2
    assert dataset.rows[0]["delta"][0] == pytest.approx(0.1)
    assert dataset.rows[1]["delta"][1] == pytest.approx(0.2)
    assert dataset.summary["unmatched_a"] == 0
    assert dataset.summary["unmatched_b"] == 0
    assert dataset.summary["top_divergences"][0]["name"] in {"opA", "opB"}
