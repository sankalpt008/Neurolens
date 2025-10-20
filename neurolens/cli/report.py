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
    if template_name.endswith(".html.j2"):
        output = _render_html(report_payload)
    elif template_name.endswith(".md.j2"):
        output = _render_markdown(report_payload)
    else:  # pragma: no cover - configuration guard
        raise typer.BadParameter(f"Unsupported template {template_name}")

    try:
        template = _ENV.get_template(template_name)
    except TemplateNotFound:
        typer.echo(f"Error: missing template {template_name}", err=True)
        raise typer.Exit(code=1)
    output = template.render(report=report_payload)
    path = ensure_parent_dir(destination)
    path.write_text(output, encoding="utf-8")


def _render_markdown(report: Dict[str, Any]) -> str:
    lines = ["# NeuroLens Profiling Report", ""]
    lines.append(f"- Generated at: {report.get('generated_at')}")
    lines.append(f"- Input: {report['input']['type']} — `{report['input']['path']}`")
    baseline_path = report.get("baseline", {}).get("path")
    if baseline_path:
        lines.append(f"- Baseline: `{baseline_path}`")
    compare = report.get("compare")
    if compare and compare.get("similarity") is not None:
        lines.append(f"- Similarity vs baseline: {compare['similarity']:.4f}")

    summary = report.get("summary", {})
    lines.extend(
        [
            "",
            "## Run Summary",
            f"- Total latency: {summary.get('total_latency_ms', 'n/a')} ms",
            f"- Operations: {summary.get('num_ops', 'n/a')}",
            f"- GPU utilization: {report.get('global', {}).get('gpu_utilization', 'n/a')}",
        ]
    )

    lines.append("\n## Top Global Findings")
    global_findings = report.get("insights", {}).get("ranking", {}).get("top_global", [])
    if global_findings:
        lines.append("| Severity | Rule | Score | Message |")
        lines.append("| --- | --- | --- | --- |")
        for finding in global_findings:
            lines.append(
                f"| {finding['severity']} | {finding['id']} | {finding['score']:.3f} | {finding['message']} |"
            )
    else:
        lines.append("_No global findings triggered._")

    lines.append("\n## Top Per-Op Findings")
    op_findings = report.get("insights", {}).get("ranking", {}).get("top_ops", [])
    if op_findings:
        lines.append("| Op Index | Name | Rule | Severity | Score | Suggestions |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for finding in op_findings:
            suggestions = "; ".join(finding.get("suggest", [])) or "n/a"
            lines.append(
                f"| {finding['op_index']} | {finding['op_name']} | {finding['id']} | "
                f"{finding['severity']} | {finding['score']:.3f} | {suggestions} |"
            )
    else:
        lines.append("_No per-op findings triggered._")

    lines.append("\n## Per-Op Metrics")
    lines.append("| # | Name | Type | Latency (ms) | Lat % | AI | Occ | Warp Eff | L2 Hit | DRAM Norm |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for op in report.get("ops", []):
        lines.append(
            "| {index} | {name} | {type} | {latency_ms:.3f} | {lat_norm:.3f} | {ai:.3f} | {occ:.3f} | "
            "{warp_eff:.3f} | {l2_hit:.3f} | {dram_norm:.3f} |".format(
                index=op.get("index", "n/a"),
                name=op.get("name", "n/a"),
                type=op.get("type", "n/a"),
                latency_ms=float(op.get("latency_ms", 0.0)),
                lat_norm=float(op.get("lat_norm", 0.0)),
                ai=float(op.get("features", {}).get("ai", 0.0)),
                occ=float(op.get("features", {}).get("occ", 0.0)),
                warp_eff=float(op.get("features", {}).get("warp_eff", 0.0)),
                l2_hit=float(op.get("features", {}).get("l2_hit", 0.0)),
                dram_norm=float(op.get("features", {}).get("dram_norm", 0.0)),
            )
        )

    return "\n".join(lines) + "\n"


def _render_html(report: Dict[str, Any]) -> str:
    lines = ["<!DOCTYPE html>", "<html lang=\"en\">", "<head>", "  <meta charset=\"utf-8\">", "  <title>NeuroLens Report</title>", "  <style>body{font-family:Arial, sans-serif;margin:2rem;}table{border-collapse:collapse;}th,td{border:1px solid #ccc;padding:0.4rem;}th{background:#f3f4f6;}}</style>", "</head>", "<body>"]
    lines.append("  <h1>NeuroLens Profiling Report</h1>")
    lines.append(f"  <p><strong>Generated:</strong> {report.get('generated_at')}</p>")
    lines.append(
        f"  <p><strong>Input:</strong> {report['input']['type']} — <code>{report['input']['path']}</code></p>"
    )
    baseline_path = report.get("baseline", {}).get("path")
    if baseline_path:
        lines.append(f"  <p><strong>Baseline:</strong> <code>{baseline_path}</code></p>")
    compare = report.get("compare")
    if compare and compare.get("similarity") is not None:
        lines.append(f"  <p><strong>Similarity:</strong> {compare['similarity']:.4f}</p>")

    summary = report.get("summary", {})
    global_metrics = report.get("global", {})
    lines.append("  <h2>Run Summary</h2>")
    lines.append("  <ul>")
    lines.append(f"    <li>Total latency: {summary.get('total_latency_ms', 'n/a')} ms</li>")
    lines.append(f"    <li>Operations: {summary.get('num_ops', 'n/a')}</li>")
    lines.append(f"    <li>GPU utilization: {global_metrics.get('gpu_utilization', 'n/a')}</li>")
    lines.append("  </ul>")

    def _render_table(rows: list[dict[str, Any]], headers: list[str]) -> None:
        lines.append("  <table>")
        lines.append("    <tr>" + "".join(f"<th>{header}</th>" for header in headers) + "</tr>")
        for row in rows:
            lines.append(
                "    <tr>"
                + "".join(f"<td>{value}</td>" for value in row.values())
                + "</tr>"
            )
        lines.append("  </table>")

    lines.append("  <h2>Top Global Findings</h2>")
    global_findings = report.get("insights", {}).get("ranking", {}).get("top_global", [])
    if global_findings:
        rows = [
            {
                "Severity": item.get("severity"),
                "Rule": item.get("id"),
                "Score": f"{item.get('score', 0):.3f}",
                "Message": item.get("message"),
            }
            for item in global_findings
        ]
        _render_table(rows, ["Severity", "Rule", "Score", "Message"])
    else:
        lines.append("  <p>No global findings triggered.</p>")

    lines.append("  <h2>Top Per-Op Findings</h2>")
    op_findings = report.get("insights", {}).get("ranking", {}).get("top_ops", [])
    if op_findings:
        rows = []
        for item in op_findings:
            rows.append(
                {
                    "Op": item.get("op_index"),
                    "Name": item.get("op_name"),
                    "Rule": item.get("id"),
                    "Severity": item.get("severity"),
                    "Score": f"{item.get('score', 0):.3f}",
                    "Suggestions": "; ".join(item.get("suggest", [])) or "n/a",
                }
            )
        _render_table(rows, ["Op", "Name", "Rule", "Severity", "Score", "Suggestions"])
    else:
        lines.append("  <p>No per-op findings triggered.</p>")

    lines.append("  <h2>Per-Op Metrics</h2>")
    op_rows = []
    for op in report.get("ops", []):
        features = op.get("features", {})
        op_rows.append(
            {
                "#": op.get("index", "n/a"),
                "Name": op.get("name", "n/a"),
                "Type": op.get("type", "n/a"),
                "Latency (ms)": f"{float(op.get('latency_ms', 0.0)):.3f}",
                "Lat %": f"{float(op.get('lat_norm', 0.0)):.3f}",
                "AI": f"{float(features.get('ai', 0.0)):.3f}",
                "Occ": f"{float(features.get('occ', 0.0)):.3f}",
                "Warp Eff": f"{float(features.get('warp_eff', 0.0)):.3f}",
                "L2 Hit": f"{float(features.get('l2_hit', 0.0)):.3f}",
                "DRAM Norm": f"{float(features.get('dram_norm', 0.0)):.3f}",
            }
        )
    if op_rows:
        _render_table(
            op_rows,
            ["#", "Name", "Type", "Latency (ms)", "Lat %", "AI", "Occ", "Warp Eff", "L2 Hit", "DRAM Norm"],
        )
    else:
        lines.append("  <p>No operations available.</p>")

    lines.append("</body>")
    lines.append("</html>")
    return "\n".join(lines) + "\n"


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
