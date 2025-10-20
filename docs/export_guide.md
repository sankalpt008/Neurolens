# NeuroLens Export Guide

## Why export bundles?

Packaging profiling runs, fingerprints, and human-readable reports into a single archive makes it easy to:

- Attach artifacts to CI build logs or release pipelines.
- Share reproducible diagnostics with teammates without requiring access to the original workspace.
- Preserve a point-in-time snapshot for regression analysis and audits.

Each bundle is a self-contained `.bundle.zip` file that can be unzipped anywhere to inspect the original JSON artifacts and reports.

## Prerequisites

Before exporting, generate the artifacts you want to package:

```bash
neurolens fingerprint --run runs/example.json --out fp/example.fp.json
neurolens report --run runs/example.json --html reports/example.html
```

## Creating a bundle

```bash
neurolens export --run runs/example.json \
  --fingerprint fp/example.fp.json \
  --report reports/example.html \
  --out-dir exports
```

- `--run` points to the profiling artifact that validates against `schema/run.schema.json`.
- `--fingerprint` references the fingerprint generated from the run.
- One or more `--report` flags may be supplied for Markdown or HTML reports.
- `--out-dir` selects (and creates if necessary) the destination folder for the archive.

The CLI prints the path to the generated archive, e.g. `exports/run123.bundle.zip`.

## Manifest format

Every bundle contains a `manifest.json` that records each file's checksum and size:

```json
{
  "bundled_at": "2025-10-20T15:30:45Z",
  "files": [
    { "path": "runs/run123.json", "sha256": "…", "size": 18342 },
    { "path": "fingerprints/run123.fp.json", "sha256": "…", "size": 5421 },
    { "path": "reports/run123.html", "sha256": "…", "size": 9024 }
  ]
}
```

To verify integrity after transfer:

1. Extract the archive.
2. Compute `sha256sum` for each file.
3. Compare against the `sha256` values listed in `manifest.json`.

Matching hashes confirm that no corruption occurred.

## Unpacking a bundle

Use any ZIP utility:

```bash
unzip exports/run123.bundle.zip -d /tmp/run123
```

The extracted directory reproduces the run, fingerprint, reports, and manifest exactly as they were bundled.
