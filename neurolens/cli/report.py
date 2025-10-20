"""Report generation CLI for NeuroLens insights."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import typer
from jinja2 import Environment, FileSystemLoader, TemplateNotFound, select_autoescape

from neurolens.fingerprint.builder import build_fingerprint
from neurolens.fingerprint.similarity import diff as fingerprint_diff
from neurolens.insights import evaluate_run
from neurolens.utils.io import ensure_parent_dir, read_json
from neurolens.utils.validate import SchemaValidationError, validate_run_schema

app = typer.Typer(add_completion=False, help="Generate NeuroLens insights reports")

_TEMPLATE_DIR = Path(__file__).resolve().parents[1] / "insights" / "templates"
_ENV = Environment(
    loader=FileSystemLoader(_TEMPLATE_DIR),
    autoescape=select_autoescape(["html"]),
    trim_blocks=True,
    lstrip_blocks=True,
)


@app.command()
def main(
    run: Optional[Path] = typer.Option(None, "--run", exists=True, readable=True, help="Path to run.json"),
    fingerprint: Optional[Path] = typer.Option(
        None,
        "--fingerprint",
        exists=True,
        readable=True,
        help="Path to a fingerprint JSON",
    ),
    baseline: Optional[Path] = typer.Option(
        None,
        "--baseline",
        exists=True,
        readable=True,
        help="Optional baseline fingerprint or run JSON",
    ),
    html: Optional[Path] = typer.Option(None, "--html", help="Write HTML report to this path"),
    md: Optional[Path] = typer.Option(None, "--md", help="Write Markdown report to this path"),
) -> None:
    """Generate a report for profiling runs or fingerprints."""

    if (run is None) == (fingerprint is None):
        raise typer.BadParameter("Provide exactly one of --run or --fingerprint")

    try:
        primary_data, primary_type = _load_input(run, fingerprint)
        baseline_data = read_json(baseline) if baseline else None
        if baseline and _looks_like_run(baseline_data):
            validate_run_schema(baseline_data)
    except (SchemaValidationError, ValueError, RuntimeError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)

    insights = evaluate_run(primary_data, baseline_data)
    context = insights.get("context", {})

    primary_fingerprint = context.get("fingerprint")
    if primary_fingerprint is None and primary_type == "run":
        primary_fingerprint = build_fingerprint(primary_data)

    baseline_context = context.get("baseline") or {}
    baseline_fingerprint = baseline_context.get("fingerprint")
    if baseline_fingerprint is None and baseline_data and _looks_like_run(baseline_data):
        baseline_fingerprint = build_fingerprint(baseline_data)

    compare = None
    if primary_fingerprint and baseline_fingerprint:
        compare = fingerprint_diff(baseline_fingerprint, primary_fingerprint, topk=10)

    report_payload = _build_report_payload(
        primary_type=primary_type,
        input_path=str(run or fingerprint),
        baseline_path=str(baseline) if baseline else None,
        insights=insights,
        compare=compare,
    )

    if html:
        _render_template("report.html.j2", report_payload, html)
    if md:
        _render_template("report.md.j2", report_payload, md)

    if not html and not md:
        _print_summary(report_payload)


def _looks_like_run(data: Optional[Dict[str, Any]]) -> bool:
    return bool(data and "timeline" in data and "summary" in data)


def _load_input(run: Optional[Path], fingerprint: Optional[Path]) -> tuple[Dict[str, Any], str]:
    if run is not None:
        run_dict = read_json(run)
        validate_run_schema(run_dict)
        return run_dict, "run"
    assert fingerprint is not None
    fp_dict = read_json(fingerprint)
    if "vector_spec" not in fp_dict or "ops" not in fp_dict:
        raise ValueError("Fingerprint JSON missing required fields")
    return fp_dict, "fingerprint"


def _build_report_payload(
    *,
    primary_type: str,
    input_path: str,
    baseline_path: Optional[str],
    insights: Dict[str, Any],
    compare: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    context = insights.get("context", {})
    generated_at = datetime.now(timezone.utc).isoformat()
    payload = {
        "generated_at": generated_at,
        "input": {
            "type": primary_type,
            "path": input_path,
        },
        "baseline": {
            "path": baseline_path,
            "context": context.get("baseline"),
        },
        "summary": context.get("summary", {}),
        "metadata": context.get("metadata", {}),
        "global": context.get("global", {}),
        "ops": context.get("ops", []),
        "insights": insights,
        "compare": compare,
    }
    return payload


def _render_template(template_name: str, report_payload: Dict[str, Any], destination: Path | str) -> None:
    try:
        template = _ENV.get_template(template_name)
    except TemplateNotFound:
        typer.echo(f"Error: missing template {template_name}", err=True)
        raise typer.Exit(code=1)
    output = template.render(report=report_payload)
    path = ensure_parent_dir(destination)
    path.write_text(output, encoding="utf-8")


def _print_summary(report_payload: Dict[str, Any]) -> None:
    summary = report_payload.get("summary", {})
    total = summary.get("total_latency_ms")
    num_ops = summary.get("num_ops")
    typer.echo(
        f"Report for {report_payload['input']['type']} ({report_payload['input']['path']}): "
        f"total_latency_ms={total}, num_ops={num_ops}"
    )
    ranking = report_payload.get("insights", {}).get("ranking", {})
    global_findings = ranking.get("top_global", [])
    if global_findings:
        typer.echo("Top global findings:")
        for finding in global_findings[:5]:
            typer.echo(f"  - [{finding['severity']}] {finding['id']}: {finding['message']} (score={finding['score']})")
    else:
        typer.echo("No global findings triggered.")
    op_findings = ranking.get("top_ops", [])
    if op_findings:
        typer.echo("Top op findings:")
        for finding in op_findings[:10]:
            typer.echo(
                f"  - op#{finding['op_index']} {finding['op_name']} [{finding['severity']}] "
                f"{finding['id']} (score={finding['score']})"
            )
    else:
        typer.echo("No per-op findings triggered.")


__all__ = ["app", "main"]
