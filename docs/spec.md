# NeuroLens Phase 0 Product Requirements

## Problem Statement
Fragmented GPU and model profiling tooling makes it difficult for ML performance engineers to reproduce, compare, and communicate bottlenecks across teams. Existing solutions emit heterogeneous logs, rely on manual spreadsheet collation, and rarely support cross-backend comparisons. NeuroLens aims to provide a unified profiling ledger that normalizes run metadata, timeline events, and derived metrics so teams can quickly isolate regressions and share reproducible findings.

## Target Users
- **ML performance engineers** who analyze kernel-level bottlenecks and need consistent metrics across projects.
- **Developer technology (DevTech) specialists** who benchmark partner workloads and must deliver auditable reports.
- **Inference infrastructure teams** responsible for production model deployments and capacity planning.

## Non-Goals
- Optimizing or tuning model architectures or training loops automatically.
- Providing correctness validation for model outputs or numerical stability checks.
- Replacing existing low-level profilers; NeuroLens consumes their exports via adapters.

## Success Metrics
- **Time-to-bottleneck:** Median time for engineers to pinpoint the top performance limiter per run decreases by ≥30% compared to current workflows.
- **Report clarity score:** User surveys rate NeuroLens run reports ≥4/5 on clarity and actionability after Phase 2.
- **Adoption:** ≥3 internal teams ingest their profiling runs through NeuroLens within 3 months of Phase 2 launch.

## Phase Plan Overview
- **Phase 0 (Groundwork & Spec):** Establish repository, schema, documentation, and validation harness. *(This deliverable.)*
- **Phase 1 (Core Profiler MVP):** Build ONNX Runtime adapter capturing per-op latency, emit schema-compliant JSON, add CLI entrypoint.
- **Phase 2 (Insights & Fingerprinting):** Implement derived metric calculations, compare runs, surface top anomalies.
- **Phase 3 (Visualization Alpha):** Generate static dashboards and exportables; optional simple web front-end.
- **Phase 4 (Adapter Expansion):** Add PyTorch profiler ingestion and TensorRT timeline support.
- **Phase 5 (Operationalization):** CI pipelines, artifact storage integration, production hardening.
- **Phase 6+ (Optional Web App):** Interactive web UI for browsing runs, user auth, collaborative annotations.

## Scope Summary for Phase 0
- Author strict JSON schema for profiling runs with explicit enumerations and unit conventions.
- Provide minimal example traces and automated tests for validation.
- Document core metrics, architecture roadmap, and success criteria to guide later phases.
- Establish development hygiene (devlog, testing instructions, licensing).
