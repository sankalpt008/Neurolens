# Bench Harness & Matrix Sweeps

The NeuroLens bench harness automates profiling across combinations of batch size,
sequence length, and precision. It runs offline, attaches environment metadata to each
profiling artifact, and writes a CSV summary so you can compare sweeps quickly.

## Configuration

Bench jobs are defined in YAML. The [configs/example_bench.yaml](../configs/example_bench.yaml)
file demonstrates the supported keys:

```yaml
backend: onnxrt
model: ./tmp/add.onnx
grid:
  batch_size: [1, 2, 4]
  seq_len: [64, 128]
  precision: [fp32, fp16]
repeats: 1
tags:
  experiment: "smoke"
  note: "initial sweep"
```

* `backend` – profiling backend (`onnxrt`, `torch`, or `tensorrt`).
* `model` – path to the model artifact or `builtin:tiny_linear`.
* `grid` – axes to sweep. Each value can be a scalar or list.
* `repeats` – optional repeat count per grid point (defaults to `1`).
* `tags` – arbitrary key/value metadata stored alongside every run.

## Running the Bench Harness

```bash
python tools/gen_add_onnx.py --out ./tmp/add.onnx  # prepare a sample model
neurolens bench --config configs/example_bench.yaml --out-dir bench_runs
```

The command writes artifacts into `bench_runs/`:

```
bench_runs/
  runs/            # Schema-validated run JSON files with env + tags metadata
  fp/              # Derived fingerprints matching each run
  summary.csv      # Tabular rollup of all runs in the sweep
```

Each `runs/*.json` file includes a `meta` block capturing driver, CUDA, GPU, OS, and
Python version along with the grid parameters (`meta.config`). The summary CSV columns
include run identifier, backend, batch size, sequence length, precision, total latency,
and GPU utilization.

## Tips for Reliable Measurements

* Warm up the model manually before running large sweeps.
* Increase `repeats` and post-process medians or percentiles in Phase 8+.
* Pin GPU clocks and disable competing workloads when measuring latency.
* Record experiment `tags` to keep track of changes between sweeps.

## Next Steps

Phase 8 will introduce an append-only Parquet store and `neurolens ls` for querying
historical runs across machines.
