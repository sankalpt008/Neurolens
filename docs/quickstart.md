# NeuroLens Quickstart

This guide walks through installing NeuroLens from source, generating a profiling run,
creating comparison artifacts, and exploring results locally. All commands run fully
offline once dependencies are installed.

## 1. Clone and set up a virtual environment

```bash
git clone https://github.com/neurolens/neurolens.git
cd neurolens
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .[dev]
```

Optional extras:

- `pip install -e .[profiler]` for ONNX Runtime / Torch helpers
- `pip install -e .[viz]` for the Streamlit dashboard

## 2. Generate a tiny model

```bash
python tools/gen_add_onnx.py --out /tmp/add.onnx
```

PyTorch users can also export the builtin tiny model:

```bash
python tools/gen_tiny_linear.py --out-dir /tmp/tiny
```

## 3. Profile the model

```bash
neurolens profile --backend onnxrt --model /tmp/add.onnx --bs 2 --seq 128 --precision fp32
```

The command writes a schema-validated `run_*.json` artifact under `runs/` and prints a
latency summary.

## 4. Build and compare fingerprints

```bash
neurolens fingerprint --run runs/run_onnxrt_*.json --out fp/sample.fp.json
neurolens fingerprint --run samples/trace_minimal.json --out fp/baseline.fp.json
neurolens compare --a fp/baseline.fp.json --b fp/sample.fp.json --topk 5 --markdown out/compare.md
```

## 5. Generate insights and reports

```bash
neurolens report --run runs/run_onnxrt_*.json --html reports/run.html --md reports/run.md
```

Reports highlight global issues, per-op bottlenecks, and (when a baseline fingerprint is
supplied) top regressions.

## 6. List stored runs

```bash
neurolens ingest --run runs/run_onnxrt_*.json --root runs_store
neurolens ls --root runs_store --backend onnxrt --limit 5
```

## 7. Launch the local dashboard

```bash
pip install -e .[viz]  # if not already installed
neurolens view --run runs/run_onnxrt_*.json
```

The Streamlit dashboard provides timeline, roofline, and fingerprint diff views that run
entirely on local data.
