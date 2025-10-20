# NeuroLens Rules Guide

## Overview

NeuroLens insights are driven by a YAML-based rules DSL (the project stores the rules using JSON syntax for maximum portability) that inspects per-op vectors, kernel counters, and global run statistics. Rules produce actionable findings that surface in the `neurolens report` CLI as global or per-op suggestions.

## Rule Structure

Each rule entry in `neurolens/insights/rules.yaml` follows this schema:

```yaml
- id: unique_rule_identifier
  scope: "op" | "global"
  where:
    all:
      - "boolean expression"
      - "..."
    any:
      - "optional disjunction"
  severity: "low" | "medium" | "high"
  message: "Human-readable summary"
  suggest:
    - "Actionable next step"
  refs:
    - "docs/cookbook.md#section"
  impact: "optional expression used for scoring"
```

* **scope** — `op` rules evaluate per operation, `global` rules evaluate run-level metrics.
* **where** — Boolean expressions composed of comparisons, arithmetic, and `all`/`any` lists. If omitted, the rule always fires.
* **severity** — Determines base weight (low=1, medium=2, high=3) when scoring.
* **suggest** — Bullet list of remediation ideas rendered in Markdown/HTML reports.
* **impact** — Optional expression (same language as `where`) that yields a numeric multiplier representing impact share. If omitted, per-op rules default to `features.lat_norm` and global rules use heuristics (launch overhead share, memcpy share, idle fraction).

## Expression Language

Expressions are parsed safely (no `eval`) and may reference the following namespaces:

* `features.*` — normalized per-op values (`lat_norm`, `ai`, `occ`, `warp_eff`, `l2_hit`, `dram_norm`, etc.). Missing features resolve to `null`.
* `op.*` — metadata about the current operation (`name`, `type`, `latency_ms`, `index`).
* `global.*` — run-wide metrics (e.g., `total_latency_ms`, `api_launch_overhead_share`, `gpu_utilization`, `memcpy_share`, `num_ops`, `median_latency_ms`).
* `delta.*` — difference metrics when a baseline fingerprint is supplied (`lat_ms`, `lat_ms_pct`, `lat_norm`).
* Literals include `null`, `true`, `false`, numeric constants, and arithmetic `+ - * /`. Division is safe and returns `0.0` on zero denominators.

Comparisons follow Python semantics (`==`, `!=`, `<`, `<=`, `>`, `>=`). Any comparison involving `null` evaluates to `false` unless explicitly guarded (`features.ai != null`).

## Current Rules

| ID | Scope | Trigger Summary | Suggestions |
| --- | --- | --- | --- |
| `memory_bound_low_ai` | op | AI < 8, high DRAM pressure, >2% runtime share | Kernel fusion, lower precision |
| `low_occupancy` | op | Occupancy < 0.35 and warp efficiency < 0.6 | Tune block size, reduce register pressure |
| `poor_l2_locality` | op | L2 hit < 0.4 with high DRAM norm | Improve layouts or fuse kernels |
| `warp_divergence_proxy` | op | Warp efficiency < 0.65 with moderate occupancy | Audit divergent branches |
| `regression_op_latency` | op | Latency grew >10% vs baseline | Inspect recent changes, compiler diffs |
| `host_bound_launch` | global | API launch overhead >5% of runtime | Batch launches, enable CUDA Graphs |
| `gpu_idle` | global | GPU utilization <40% | Overlap compute/transfers, increase batch size |
| `many_tiny_ops` | global | >200 ops with median latency <0.05 ms | Fuse elementwise ops, enable compiler fusion |
| `memcpy_dominates` | global | Memcpy share >25% | Stage tensors earlier, overlap copies |

## Adding a New Rule

1. Edit `neurolens/insights/rules.yaml` and append a new entry following the schema above. Pick a descriptive `id` and fill `message`, `suggest`, and optional `refs` links.
2. Use the available variables (`features`, `op`, `global`, `delta`) to express the trigger condition. Guard nullables with `!= null`.
3. (Optional) Provide an `impact` expression if the default latency share is insufficient.
4. Run `pytest -k insights` to execute unit tests, or `neurolens report --run <run.json>` to manually inspect the output. Add/adjust tests under `tests/test_insights_engine_unit.py` as needed.
5. Update documentation if the new rule introduces fresh terminology or references.

Rules are hot-reloadable at runtime; advanced users can maintain custom rule packs by pointing `evaluate_run(..., rules_path=...)` to alternate YAML (or JSON) files.
