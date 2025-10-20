"""Generate tiny linear Torch model artifacts for NeuroLens samples."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def build_tiny_linear(out_dir: Path) -> None:
    try:
        import torch
        from torch import nn
    except ImportError:  # pragma: no cover - depends on optional deps
        print("[neurolens] PyTorch not installed; skipping tiny model generation.")
        return

    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))
    model.eval()

    out_dir.mkdir(parents=True, exist_ok=True)

    pt_path = out_dir / "tiny_linear.pt"
    torch.save(model, pt_path)

    try:
        import onnx  # noqa: F401

        dummy_input = torch.randn(1, 4)
        onnx_path = out_dir / "tiny_linear.onnx"
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            input_names=["input"],
            output_names=["output"],
            opset_version=16,
        )
        print(f"[neurolens] Wrote ONNX model to {onnx_path}")
    except Exception as exc:  # pragma: no cover - depends on runtime
        print(f"[neurolens] Skipped ONNX export: {exc}")

    print(f"[neurolens] Wrote PyTorch model to {pt_path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a tiny Torch linear model")
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory where the generated artifacts will be stored",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    build_tiny_linear(args.out_dir)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    sys.exit(main())
