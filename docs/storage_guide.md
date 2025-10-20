# NeuroLens Run Store

The NeuroLens storage layer captures every profiling run in an append-only layout so
teams can analyse results offline, rebuild indexes deterministically, and share
artifacts across environments.

## Directory layout

By default the store writes to `./runs_store/` and organises files by UTC day and
backend:

```
runs_store/
  2025/10/20/onnxrt/
    4d9f.json           # canonical run
    4d9f.fp.json        # optional fingerprint
  _manifest.jsonl       # append-only ledger
  _index.parquet        # tabular index (falls back to JSONL when Parquet unavailable)
```

The manifest is the source of truth. The index is derived and can always be
reconstructed via `rebuild_index_from_manifest`.

## Stored metadata

Each manifest entry contains:

- Run identifiers: `run_id`, `created_at`, `day`
- Runtime context: backend, precision, batch size, sequence length
- Performance summary: `total_latency_ms`, `gpu_utilization`
- Hardware snapshot: `gpu_name`, driver + CUDA versions
- Paths to the stored JSON artifacts (relative to the root)
- Tags serialised as JSON for quick filtering

## Writing runs programmatically

```python
from neurolens.storage.store import write_run

paths = write_run("runs_store", run_dict, fingerprint=fingerprint_dict)
print(paths["run_json"], paths["fp_json"])
```

`write_run` validates the run against `schema/run.schema.json`, writes the JSON files,
appends the manifest, and updates the index (Parquet when `pyarrow`/`pandas` are
available).

## Listing and filtering runs

Use the CLI to filter by model, backend, day, or tags:

```bash
neurolens ls --backend onnxrt --model gpt2 --precision fp16 --day 2025-10-20
neurolens ls --tag experiment=smoke --limit 10
```

Add `--json` to emit machine-readable rows.

## Rebuilding the index

Should the index be removed or corrupted, rebuild it deterministically:

```python
from neurolens.storage.store import rebuild_index_from_manifest

index_path = rebuild_index_from_manifest("runs_store")
print(f"Rebuilt index at {index_path}")
```

## Custom store roots

Pass a custom root via the CLI `--root` flag or set the `NEUROLENS_STORE_ROOT`
environment variable before invoking commands. All manifest and index files will be
created beneath that directory.

