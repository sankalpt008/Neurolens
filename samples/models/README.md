# Sample Models

Binary model artifacts are no longer checked into the repository. Use the helper scripts to generate them when needed:

- `python tools/gen_add_onnx.py --out /tmp/add.onnx`
- `python tools/gen_tiny_linear.py --out-dir /tmp/tiny`

Tests rely on these generators to populate temporary directories on the fly.
