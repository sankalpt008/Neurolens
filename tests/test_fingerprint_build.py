from __future__ import annotations

from pathlib import Path

import pytest

from neurolens.fingerprint import VECTOR_SPEC, build_fingerprint
from neurolens.utils.io import read_json


def test_build_fingerprint_from_sample() -> None:
    run_path = Path("samples/trace_minimal.json")
    run_dict = read_json(run_path)

    fingerprint_a = build_fingerprint(run_dict)
    fingerprint_b = build_fingerprint(run_dict)

    assert fingerprint_a["vector_spec"] == VECTOR_SPEC
    assert len(fingerprint_a["ops"]) == len(run_dict["timeline"])
    assert fingerprint_a["source_run_sha"] == fingerprint_b["source_run_sha"]

    lat_sum = sum(op["vector"][0] for op in fingerprint_a["ops"])
    assert lat_sum == pytest.approx(1.0, rel=1e-6, abs=1e-6)

    assert fingerprint_a["summary"]["num_ops"] == len(run_dict["timeline"])
    assert fingerprint_a["summary"]["total_latency_ms"] == pytest.approx(
        run_dict["summary"]["total_duration_ms"]
    )
