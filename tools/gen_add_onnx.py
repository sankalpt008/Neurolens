"""Utility to generate a tiny ONNX add model (y = x + b)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def build_add_model(output_path: Path) -> None:
    try:
        import onnx
        from onnx import TensorProto, helper
    except ImportError as exc:  # pragma: no cover - depends on optional deps
        raise SystemExit(
            "onnx package is required to generate the sample model. "
            "Install it with `pip install onnx`."
        ) from exc

    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", 4])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["batch", 4])

    bias_initializer = helper.make_tensor(
        name="bias",
        data_type=TensorProto.FLOAT,
        dims=[4],
        vals=[0.1, 0.2, 0.3, 0.4],
    )

    add_node = helper.make_node(
        "Add",
        inputs=["input", "bias"],
        outputs=["output"],
        name="AddBias",
    )

    graph = helper.make_graph(
        nodes=[add_node],
        name="NeurolensAdd",
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=[bias_initializer],
    )

    model = helper.make_model(graph, producer_name="neurolens.tools.gen_add_onnx")

    onnx.checker.check_model(model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output_path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a tiny ONNX add model")
    parser.add_argument("--out", type=Path, required=True, help="Destination path for the ONNX model")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    build_add_model(args.out)
    print(f"[neurolens] Wrote ONNX model to {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    sys.exit(main())
