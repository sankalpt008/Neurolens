from __future__ import annotations

from neurolens.insights import evaluate_run

from helpers import build_insights_sample_run


def test_insights_engine_triggers_expected_rules() -> None:
    run_dict = build_insights_sample_run()
    insights = evaluate_run(run_dict)

    op_findings = {(item["id"], item["op_index"]): item for item in insights["op_findings"]}
    assert ("memory_bound_low_ai", 0) in op_findings
    assert ("poor_l2_locality", 0) in op_findings
    assert ("low_occupancy", 1) in op_findings

    memory_score = op_findings[("memory_bound_low_ai", 0)]["score"]
    low_occ_score = op_findings[("low_occupancy", 1)]["score"]
    assert memory_score > low_occ_score

    global_ids = {item["id"] for item in insights["global_findings"]}
    assert "host_bound_launch" in global_ids
    assert "gpu_idle" in global_ids

    ranking_ops = insights["ranking"]["top_ops"]
    if ranking_ops:
        assert ranking_ops[0]["score"] >= ranking_ops[-1]["score"]
