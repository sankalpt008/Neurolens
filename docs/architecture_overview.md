# Architecture Overview

```
+--------------+     +---------------+     +------------------+
| Profiling    | --> | Backend        | --> | Adapter Normalizer|
| Backend (e.g.|     | Export (JSON)  |     | (core/adapters/)  |
| ORT trace)   |     +---------------+     +------------------+
       |                                            |
       v                                            v
+-----------------+     +-----------------+     +----------------------+
| Schema Validator| --> | Fingerprint Gen | --> | Insights + Reporting  |
| (schema/*.json) |     | (fingerprint/)  |     | (insights/, viz/)    |
+-----------------+     +-----------------+     +----------------------+
```

## Data Flow
1. **Backend exporters** (ONNX Runtime, PyTorch, TensorRT) produce raw profiling dumps.
2. **Adapters** convert backend-specific dumps into the NeuroLens run schema (JSON/Parquet) while enriching with derived counters.
3. **Schema validation** ensures normalized runs feed the fingerprint engine to enable comparisons and insights before visualization.

## Key Modules (Planned)
- `core/`: Shared utilities (schema loader, validation, metric math).
- `adapters/`: Backend-specific parsers (starting with ONNX Runtime Phase 1).
- `fingerprint/`: Run signature generation for regression detection.
- `insights/`: Bottleneck analysis, trend detection, and reporting exports.
- `viz/`: Rendering layer for dashboards and static summaries.

## Extensibility
- Configuration via `rules.yaml` will allow teams to customize bottleneck thresholds and derived metric formulas.
- Adding a new backend requires implementing an adapter that maps the backend export into the canonical schema; tests enforce compliance.
- Outputs are stored as JSON initially, with Parquet writers available for large-scale archival.
