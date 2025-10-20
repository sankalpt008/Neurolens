"""Streamlit dashboard for NeuroLens visualizations."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from neurolens.fingerprint import build_fingerprint
from neurolens.utils.io import read_json
from neurolens.viz.diffview import build_diff_dataset
from neurolens.viz.roofline import build_roofline_points
from neurolens.viz.timeline import build_timeline_series

__all__ = ["PreloadedArtifacts", "load_artifacts", "run_app"]


@dataclass
class PreloadedArtifacts:
    """Paths that should be auto-loaded when the dashboard starts."""

    run_path: Optional[Path] = None
    fingerprint_path: Optional[Path] = None
    baseline_path: Optional[Path] = None
    compare_path: Optional[Path] = None


def _read_optional(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if path is None:
        return None
    try:
        return read_json(path)
    except FileNotFoundError:
        return None


def _read_uploaded(uploaded) -> Dict[str, Any]:  # type: ignore[no-untyped-def]
    """Load JSON content from a Streamlit uploaded file object."""

    if hasattr(uploaded, "seek"):
        uploaded.seek(0)
    data = uploaded.read()
    if isinstance(data, bytes):
        return json.loads(data.decode("utf-8"))
    return json.loads(data)


def load_artifacts(preloaded: Optional[PreloadedArtifacts]) -> Dict[str, Any]:
    """Load JSON artifacts for the dashboard based on ``preloaded`` settings."""

    if preloaded is None:
        preloaded = parse_preloaded_from_env()
    run_data = _read_optional(preloaded.run_path)
    fingerprint = _read_optional(preloaded.fingerprint_path)
    baseline = _read_optional(preloaded.baseline_path)
    candidate = _read_optional(preloaded.compare_path)

    return {
        "run": run_data,
        "fingerprint": fingerprint,
        "baseline": baseline,
        "candidate": candidate,
    }


def parse_preloaded_from_env() -> PreloadedArtifacts:
    """Return ``PreloadedArtifacts`` derived from ``NEUROLENS_VIEW_*`` env vars."""

    def _path_from_env(name: str) -> Optional[Path]:
        value = os.environ.get(name)
        if value:
            return Path(value)
        return None

    return PreloadedArtifacts(
        run_path=_path_from_env("NEUROLENS_VIEW_RUN"),
        fingerprint_path=_path_from_env("NEUROLENS_VIEW_FP"),
        baseline_path=_path_from_env("NEUROLENS_VIEW_BASELINE"),
        compare_path=_path_from_env("NEUROLENS_VIEW_CANDIDATE"),
    )


def _render_run_view(st_mod, run_data: Mapping[str, Any]) -> None:
    fingerprint = build_fingerprint(run_data)
    timeline = build_timeline_series(fingerprint)
    roofline = build_roofline_points(run_data)

    st_mod.subheader("Overview")
    summary = run_data.get("summary", {})
    metrics = summary.get("metrics", {})
    cols = st_mod.columns(3)
    cols[0].metric("Total latency (ms)", f"{summary.get('total_duration_ms', 0.0):.2f}")
    cols[1].metric("Ops", len(run_data.get("timeline", [])))
    cols[2].metric("GPU util", f"{metrics.get('gpu_utilization', 0.0):.2f}")

    st_mod.markdown("### Timeline")
    chart_data = [
        {
            "index": item["index"],
            "latency_pct": item["latency_pct"],
            "name": item["name"],
            "type": item["type"],
            "category": item["category"],
        }
        for item in timeline.ops
    ]
    st_mod.vega_lite_chart(
        chart_data,
        {
            "mark": "bar",
            "encoding": {
                "x": {"field": "index", "type": "ordinal", "title": "Op index"},
                "y": {"field": "latency_pct", "type": "quantitative", "title": "Latency %"},
                "color": {"field": "category", "type": "nominal"},
                "tooltip": ["name", "type", "latency_pct"],
            },
        },
        use_container_width=True,
    )

    st_mod.markdown("### Roofline")
    scatter_data = [
        {
            "ai": op["ai"],
            "perf": op["perf_gflops"],
            "name": op["name"],
            "type": op["type"],
            "bound": op["bound"],
        }
        for op in roofline.ops
    ]
    st_mod.vega_lite_chart(
        scatter_data,
        {
            "mark": {"type": "point", "filled": True, "size": 100},
            "encoding": {
                "x": {"field": "ai", "type": "quantitative", "title": "Arithmetic intensity"},
                "y": {"field": "perf", "type": "quantitative", "title": "GFLOPs"},
                "color": {"field": "bound", "type": "nominal"},
                "tooltip": ["name", "type", "ai", "perf"],
            },
        },
        use_container_width=True,
    )

    st_mod.markdown("### Aggregated latency")
    st_mod.table(timeline.aggregates)


def _render_diff_view(st_mod, baseline: Mapping[str, Any], candidate: Mapping[str, Any]) -> None:
    dataset = build_diff_dataset(baseline, candidate)
    st_mod.subheader("Similarity")
    st_mod.write(f"Similarity: {dataset.summary.get('similarity', 0.0):.3f}")
    st_mod.write(
        f"Unmatched ops → baseline: {dataset.summary.get('unmatched_a', 0)}, "
        f"candidate: {dataset.summary.get('unmatched_b', 0)}",
    )

    st_mod.markdown("### Top divergences")
    for entry in dataset.summary.get("top_divergences", []):
        st_mod.write(
            f"**{entry.get('name','op')}** — Δ latency %: {entry.get('latency_pct_change')}"
        )

    if dataset.rows:
        st_mod.markdown("### Δ vectors")
        diff_rows = []
        for row in dataset.rows:
            diff_rows.append(
                {
                    "name": row.get("name"),
                    "type": row.get("type"),
                    **{
                        feature: row["delta"][idx]
                        for idx, feature in enumerate(dataset.vector_spec)
                    },
                }
            )
        st_mod.dataframe(diff_rows)


def _render_roofline_only(st_mod, fingerprint: Mapping[str, Any]) -> None:
    points = build_roofline_points(fingerprint)
    st_mod.subheader("Roofline overview")
    st_mod.write(
        f"Peak BW {points.summary['peak_bw_gbps']:.1f} GB/s · Peak compute {points.summary['peak_flops_gflops']:.1f} GFLOPs"
    )
    scatter_data = [
        {
            "ai": op["ai"],
            "perf": op["perf_gflops"],
            "name": op["name"],
            "bound": op["bound"],
        }
        for op in points.ops
    ]
    st_mod.vega_lite_chart(
        scatter_data,
        {
            "mark": {"type": "point", "filled": True},
            "encoding": {
                "x": {"field": "ai", "type": "quantitative"},
                "y": {"field": "perf", "type": "quantitative"},
                "color": {"field": "bound", "type": "nominal"},
                "tooltip": ["name", "ai", "perf"],
            },
        },
        use_container_width=True,
    )


def run_app(preloaded: Optional[PreloadedArtifacts] = None) -> None:
    """Launch the Streamlit dashboard in the current process."""

    try:
        import streamlit as st
    except ImportError as exc:  # pragma: no cover - handled in tests
        raise RuntimeError("Streamlit is required to run the NeuroLens dashboard") from exc

    artifacts = load_artifacts(preloaded)
    st.set_page_config(page_title="NeuroLens Dashboard", layout="wide")
    logo_path = Path(__file__).resolve().parents[2] / "assets" / "logo.svg"
    if logo_path.exists():
        st.sidebar.image(str(logo_path))
    st.sidebar.title("NeuroLens")

    mode = st.sidebar.selectbox(
        "Mode",
        ["Run Viewer", "Fingerprint Diff", "Roofline Plot"],
    )

    if mode == "Run Viewer":
        if artifacts["run"] is None:
            st.sidebar.warning("Provide --run when launching or load a file below.")
            uploaded = st.file_uploader("Upload run.json", type=["json"])
            if uploaded is not None:
                artifacts["run"] = _read_uploaded(uploaded)
        if artifacts["run"]:
            _render_run_view(st, artifacts["run"])
        else:
            st.info("No run selected yet.")
    elif mode == "Fingerprint Diff":
        if artifacts["baseline"] is None or artifacts["candidate"] is None:
            st.sidebar.warning("Provide --compare A B when launching or load fingerprints below.")
            baseline_file = st.file_uploader("Baseline fingerprint", type=["json"], key="baseline")
            candidate_file = st.file_uploader("Candidate fingerprint", type=["json"], key="candidate")
            if baseline_file is not None:
                artifacts["baseline"] = _read_uploaded(baseline_file)
            if candidate_file is not None:
                artifacts["candidate"] = _read_uploaded(candidate_file)
        if artifacts["baseline"] and artifacts["candidate"]:
            _render_diff_view(st, artifacts["baseline"], artifacts["candidate"])
        else:
            st.info("Waiting for both fingerprints.")
    else:
        fingerprint = artifacts["fingerprint"] or artifacts["run"]
        if fingerprint is None:
            st.sidebar.warning("Provide --fingerprint or --run for roofline view.")
            uploaded = st.file_uploader("Fingerprint JSON", type=["json"], key="roofline")
            if uploaded is not None:
                fingerprint = _read_uploaded(uploaded)
        if fingerprint:
            _render_roofline_only(st, fingerprint)
        else:
            st.info("No fingerprint available.")
