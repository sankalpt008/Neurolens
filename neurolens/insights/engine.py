"""Rule evaluation engine for NeuroLens insights."""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import yaml

from neurolens.utils.mathx import percentile, safe_div

RULES_PATH = Path(__file__).resolve().with_name("rules.yaml")

__all__ = ["load_rules", "evaluate_run"]


@dataclass
class NormalizedOp:
    index: int
    name: str
    type: str
    latency_ms: float
    lat_norm: float
    features: Dict[str, Any]
    sig: Optional[str] = None
    delta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NormalizedRun:
    source: str
    metadata: Dict[str, Any]
    summary: Dict[str, Any]
    global_metrics: Dict[str, Any]
    ops: List[NormalizedOp]
    fingerprint: Optional[Mapping[str, Any]] = None


class ContextNamespace(dict):
    """Dictionary that provides attribute access with ``None`` defaults."""

    def __getattr__(self, key: str) -> Any:  # pragma: no cover - passthrough
        return self.get(key)


class SafeExpressionEvaluator(ast.NodeVisitor):
    """Evaluate a restricted Python expression against a safe context."""

    def __init__(self, context: Mapping[str, Any]) -> None:
        self.context = context

    def visit(self, node: ast.AST) -> Any:  # type: ignore[override]
        return super().visit(node)

    def generic_visit(self, node: ast.AST) -> Any:  # pragma: no cover - defensive
        raise ValueError(f"Unsupported expression node: {ast.dump(node)}")

    def visit_Expression(self, node: ast.Expression) -> Any:
        return self.visit(node.body)

    def visit_Name(self, node: ast.Name) -> Any:
        if node.id in {"None", "null"}:
            return None
        if node.id in {"True", "true"}:
            return True
        if node.id in {"False", "false"}:
            return False
        if node.id not in self.context:
            raise ValueError(f"Unknown identifier '{node.id}' in expression")
        return self.context[node.id]

    def visit_Constant(self, node: ast.Constant) -> Any:
        return node.value

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        operand = self.visit(node.operand)
        if isinstance(node.op, ast.USub):
            return -float(operand)
        if isinstance(node.op, ast.UAdd):
            return float(operand)
        if isinstance(node.op, ast.Not):
            return not bool(operand)
        raise ValueError("Unsupported unary operator")

    def visit_BoolOp(self, node: ast.BoolOp) -> Any:
        if isinstance(node.op, ast.And):
            for value in node.values:
                if not self.visit(value):
                    return False
            return True
        if isinstance(node.op, ast.Or):
            for value in node.values:
                if self.visit(value):
                    return True
            return False
        raise ValueError("Unsupported boolean operator")

    def visit_Compare(self, node: ast.Compare) -> Any:
        left = self.visit(node.left)
        for operator, comparator in zip(node.ops, node.comparators):
            right = self.visit(comparator)
            try:
                if isinstance(operator, ast.Eq):
                    result = left == right
                elif isinstance(operator, ast.NotEq):
                    result = left != right
                elif isinstance(operator, ast.Lt):
                    result = left < right
                elif isinstance(operator, ast.LtE):
                    result = left <= right
                elif isinstance(operator, ast.Gt):
                    result = left > right
                elif isinstance(operator, ast.GtE):
                    result = left >= right
                else:
                    raise ValueError("Unsupported comparison operator")
            except TypeError:
                return False
            if not result:
                return False
            left = right
        return True

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        left = self.visit(node.left)
        right = self.visit(node.right)
        try:
            if isinstance(node.op, ast.Add):
                return float(left) + float(right)
            if isinstance(node.op, ast.Sub):
                return float(left) - float(right)
            if isinstance(node.op, ast.Mult):
                return float(left) * float(right)
            if isinstance(node.op, ast.Div):
                return safe_div(left, right)
        except (TypeError, ValueError):
            return 0.0
        raise ValueError("Unsupported binary operator")

    def visit_Attribute(self, node: ast.Attribute) -> Any:
        value = self.visit(node.value)
        if isinstance(value, ContextNamespace):
            return value.get(node.attr)
        if isinstance(value, Mapping):
            return value.get(node.attr)
        if value is None:
            return None
        return getattr(value, node.attr, None)


def load_rules(path: str | Path | None = None) -> List[Dict[str, Any]]:
    """Load rules from ``path`` (defaults to package rules)."""

    rules_path = Path(path) if path else RULES_PATH
    with rules_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not data:
        return []
    if not isinstance(data, list):
        raise ValueError("Rules file must define a list of rule entries")
    return data


def evaluate_run(
    run_dict: Mapping[str, Any],
    baseline_run: Mapping[str, Any] | None = None,
    *,
    rules_path: str | Path | None = None,
) -> Dict[str, Any]:
    """Evaluate insights for ``run_dict`` and optional ``baseline_run``."""

    rules = load_rules(rules_path)
    primary = _normalize_input(run_dict)
    baseline = _normalize_input(baseline_run) if baseline_run else None

    if baseline is not None:
        _attach_deltas(primary, baseline)

    findings_global: List[Dict[str, Any]] = []
    findings_ops: Dict[tuple[int, str], Dict[str, Any]] = {}

    global_context = _to_namespace(primary.global_metrics)
    for rule in rules:
        scope = rule.get("scope")
        if scope == "global":
            context = {
                "features": ContextNamespace(),
                "op": ContextNamespace(),
                "global": global_context,
                "delta": ContextNamespace(),
            }
            if _evaluate_rule(rule.get("where"), context):
                finding = _build_finding(rule, context, primary, None)
                findings_global.append(finding)
        elif scope == "op":
            for op in primary.ops:
                context = {
                    "features": _to_namespace(op.features),
                    "op": _to_namespace(
                        {
                            "name": op.name,
                            "type": op.type,
                            "latency_ms": op.latency_ms,
                            "index": op.index,
                        }
                    ),
                    "global": global_context,
                    "delta": _to_namespace(op.delta),
                }
                if _evaluate_rule(rule.get("where"), context):
                    finding = _build_finding(rule, context, primary, op)
                    key = (op.index, rule["id"])
                    existing = findings_ops.get(key)
                    if existing is None or finding["score"] > existing["score"]:
                        findings_ops[key] = finding
        else:  # pragma: no cover - invalid config
            raise ValueError(f"Unsupported rule scope: {scope}")

    sorted_global = sorted(
        findings_global,
        key=lambda item: (-item["score"], item["id"]),
    )
    sorted_ops = sorted(
        findings_ops.values(),
        key=lambda item: (-item["score"], item["id"], item["op_name"]),
    )

    ranking = {
        "top_global": sorted_global[:5],
        "top_ops": sorted_ops[:10],
    }

    return {
        "global_findings": sorted_global,
        "op_findings": sorted_ops,
        "ranking": ranking,
        "context": {
            "metadata": primary.metadata,
            "summary": primary.summary,
            "global": primary.global_metrics,
            "ops": [
                {
                    "index": op.index,
                    "name": op.name,
                    "type": op.type,
                    "latency_ms": op.latency_ms,
                    "lat_norm": op.lat_norm,
                    "features": op.features,
                    "delta": op.delta,
                    "sig": op.sig,
                }
                for op in primary.ops
            ],
            "fingerprint": primary.fingerprint,
            "baseline": None
            if baseline is None
            else {
                "metadata": baseline.metadata,
                "summary": baseline.summary,
                "global": baseline.global_metrics,
                "ops": [
                    {
                        "index": op.index,
                        "name": op.name,
                        "type": op.type,
                        "latency_ms": op.latency_ms,
                        "lat_norm": op.lat_norm,
                        "features": op.features,
                        "delta": op.delta,
                        "sig": op.sig,
                    }
                    for op in baseline.ops
                ],
                "fingerprint": baseline.fingerprint,
            },
        },
    }


def _evaluate_rule(condition: Any, context: Mapping[str, Any]) -> bool:
    if condition is None:
        return True
    if isinstance(condition, str):
        expr, prepared = _normalize_expression(condition, context)
        evaluator = SafeExpressionEvaluator(prepared)
        try:
            return bool(evaluator.visit(ast.parse(expr, mode="eval")))
        except Exception:
            return False
    if isinstance(condition, Mapping):
        result = True
        if "all" in condition:
            result = all(_evaluate_rule(item, context) for item in condition["all"])
        if "any" in condition:
            any_result = any(_evaluate_rule(item, context) for item in condition["any"])
            result = result and any_result
        return result
    if isinstance(condition, Iterable) and not isinstance(condition, (str, bytes)):
        return all(_evaluate_rule(item, context) for item in condition)
    raise ValueError("Unsupported condition format")


def _build_finding(
    rule: Mapping[str, Any],
    context: Mapping[str, Any],
    normalized: NormalizedRun,
    op: NormalizedOp | None,
) -> Dict[str, Any]:
    severity = str(rule.get("severity", "low")).lower()
    severity_weight = {"low": 1.0, "medium": 2.0, "high": 3.0}.get(severity, 1.0)
    impact = _compute_impact(rule, context, op, normalized)
    score = round(severity_weight * impact, 4)

    finding: Dict[str, Any] = {
        "id": rule.get("id"),
        "severity": severity,
        "message": rule.get("message"),
        "suggest": list(rule.get("suggest", [])),
        "refs": list(rule.get("refs", [])),
        "score": score,
    }
    if op is not None:
        finding.update(
            {
                "op_index": op.index,
                "op_name": op.name,
                "op_type": op.type,
            }
        )
    return finding


def _compute_impact(
    rule: Mapping[str, Any],
    context: Mapping[str, Any],
    op: NormalizedOp | None,
    normalized: NormalizedRun,
) -> float:
    expression = rule.get("impact")
    if isinstance(expression, str):
        expr, prepared = _normalize_expression(expression, context)
        evaluator = SafeExpressionEvaluator(prepared)
        try:
            value = evaluator.visit(ast.parse(expr, mode="eval"))
            if value is None:
                return 0.0
            return max(float(value), 0.0)
        except Exception:
            return 0.0

    if op is not None:
        return float(op.lat_norm)

    global_metrics = normalized.global_metrics
    candidates = [
        global_metrics.get("api_launch_overhead_share"),
        global_metrics.get("memcpy_share"),
        1.0 - float(global_metrics.get("gpu_utilization", 0.0)),
        global_metrics.get("tiny_op_fraction"),
    ]
    values = [float(val) for val in candidates if isinstance(val, (int, float))]
    if values:
        return max(values)
    return 1.0


def _normalize_input(data: Mapping[str, Any] | None) -> NormalizedRun:
    if data is None:
        raise ValueError("Input data is required for normalization")
    if "vector_spec" in data and "ops" in data:
        return _normalize_from_fingerprint(data)
    return _normalize_from_run(data)


def _normalize_from_run(run: Mapping[str, Any]) -> NormalizedRun:
    timeline = list(run.get("timeline", []))
    summary = dict(run.get("summary", {}))
    total_latency = float(summary.get("total_duration_ms", 0.0))
    if total_latency <= 0.0:
        total_latency = sum(float(entry.get("duration_ms", 0.0)) for entry in timeline)
    total_latency = max(total_latency, 0.0)

    hardware = dict(run.get("hardware", {}))
    software = dict(run.get("software", {}))
    model = dict(run.get("model", {}))

    ops: List[NormalizedOp] = []
    latencies: List[float] = []
    api_overhead = 0.0
    memcpy_latency = 0.0
    tiny_threshold = 0.05
    for entry in timeline:
        features, latency_ms = _extract_features_from_run(entry, total_latency, summary)
        latencies.append(latency_ms)
        api_overhead += float(entry.get("api_launch_overhead_ms", 0.0))
        if str(entry.get("op_type", "")).lower().startswith("memcpy"):
            memcpy_latency += latency_ms
        if latency_ms < tiny_threshold:
            features.setdefault("is_tiny", True)
        sig = _op_signature(entry)
        op = NormalizedOp(
            index=int(entry.get("op_index", len(ops))),
            name=str(entry.get("op_name", f"op_{len(ops)}")),
            type=str(entry.get("op_type", "unknown")),
            latency_ms=latency_ms,
            lat_norm=float(features.get("lat_norm", 0.0)),
            features=features,
            sig=sig,
        )
        ops.append(op)

    gpu_util = summary.get("metrics", {}).get("gpu_utilization")
    if gpu_util is None:
        gpu_util = _average_kernel_metric(timeline, "gpu_utilization")

    median_latency = percentile(latencies, 50) if latencies else 0.0
    tiny_fraction = safe_div(sum(1 for lat in latencies if lat < tiny_threshold), len(latencies))
    total_latency = total_latency or sum(latencies)

    global_metrics = {
        "total_latency_ms": total_latency,
        "num_ops": len(ops),
        "api_launch_overhead_ms": api_overhead,
        "api_launch_overhead_share": safe_div(api_overhead, total_latency),
        "gpu_utilization": gpu_util,
        "memcpy_latency_ms": memcpy_latency,
        "memcpy_share": safe_div(memcpy_latency, total_latency),
        "median_latency_ms": median_latency,
        "tiny_op_fraction": tiny_fraction,
    }

    summary_struct = {
        "total_latency_ms": total_latency,
        "num_ops": len(ops),
        "source": "run",
        "hardware": hardware,
        "software": software,
        "model": model,
    }

    return NormalizedRun(
        source="run",
        metadata={"hardware": hardware, "software": software, "model": model},
        summary=summary_struct,
        global_metrics=global_metrics,
        ops=ops,
    )


def _normalize_from_fingerprint(fp: Mapping[str, Any]) -> NormalizedRun:
    vector_spec = list(fp.get("vector_spec", []))
    spec_index = {name: idx for idx, name in enumerate(vector_spec)}
    summary = dict(fp.get("summary", {}))
    total_latency = float(summary.get("total_latency_ms", 0.0))
    ops_data = list(fp.get("ops", []))

    ops: List[NormalizedOp] = []
    for idx, entry in enumerate(ops_data):
        vector = list(entry.get("vector", []))
        features: Dict[str, Any] = {}
        for name, position in spec_index.items():
            if position < len(vector):
                features[name] = float(vector[position])
        lat_norm = float(features.get("lat_norm", 0.0))
        latency_ms = lat_norm * total_latency
        features.setdefault("lat_norm", lat_norm)
        features["latency_ms"] = latency_ms
        op = NormalizedOp(
            index=int(entry.get("index", idx)),
            name=str(entry.get("name", f"op_{idx}")),
            type=str(entry.get("type", "unknown")),
            latency_ms=latency_ms,
            lat_norm=lat_norm,
            features=features,
            sig=entry.get("sig"),
        )
        ops.append(op)

    global_metrics = {
        "total_latency_ms": total_latency,
        "num_ops": len(ops),
        "api_launch_overhead_ms": None,
        "api_launch_overhead_share": None,
        "gpu_utilization": None,
        "memcpy_latency_ms": None,
        "memcpy_share": None,
        "median_latency_ms": percentile([op.latency_ms for op in ops], 50) if ops else 0.0,
        "tiny_op_fraction": safe_div(sum(1 for op in ops if op.latency_ms < 0.05), len(ops)),
    }

    summary_struct = {
        "total_latency_ms": total_latency,
        "num_ops": len(ops),
        "source": "fingerprint",
    }

    return NormalizedRun(
        source="fingerprint",
        metadata={},
        summary=summary_struct,
        global_metrics=global_metrics,
        ops=ops,
        fingerprint=fp,
    )


def _attach_deltas(primary: NormalizedRun, baseline: NormalizedRun) -> None:
    index = {op.sig: op for op in baseline.ops if op.sig}
    for op in primary.ops:
        match = index.get(op.sig)
        if match is not None:
            _assign_delta(op, match)


def _assign_delta(current: NormalizedOp, previous: NormalizedOp) -> None:
    delta_latency = current.latency_ms - previous.latency_ms
    pct = safe_div(delta_latency, previous.latency_ms)
    current.delta = {
        "lat_ms": delta_latency,
        "lat_ms_pct": pct,
        "lat_norm": current.lat_norm - previous.lat_norm,
    }


def _extract_features_from_run(
    entry: Mapping[str, Any],
    total_latency: float,
    summary: Mapping[str, Any],
) -> tuple[Dict[str, Any], float]:
    duration = float(entry.get("duration_ms", 0.0))
    metrics = entry.get("metrics", {})
    latency_ms = float(metrics.get("latency_ms", duration))
    lat_norm = safe_div(latency_ms, total_latency)

    features: Dict[str, Any] = {
        "latency_ms": latency_ms,
        "lat_norm": lat_norm,
        "ai": _extract_metric(metrics, entry, "ai_flops_per_byte"),
        "occ": _average_kernel_metric([entry], "achieved_occupancy"),
        "warp_eff": _average_kernel_metric([entry], "warp_execution_efficiency"),
        "l2_hit": _average_kernel_metric([entry], "l2_hit_rate"),
    }
    if features["ai"] is None:
        features["ai"] = _estimate_ai(entry)

    peak_dram = summary.get("metrics", {}).get("peak_dram_gbps")
    if peak_dram is None:
        peak_dram = summary.get("metrics", {}).get("dram_throughput_gbps")
    features["dram_norm"] = _compute_dram_norm(entry, latency_ms, peak_dram)

    return features, latency_ms


def _extract_metric(metrics: Mapping[str, Any], entry: Mapping[str, Any], key: str) -> Any:
    value = None
    if isinstance(metrics, Mapping):
        value = metrics.get(key)
    if value is not None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    annotations = entry.get("annotations")
    if isinstance(annotations, Mapping):
        alt_key = key.split("ai_")[-1]
        if alt_key in annotations:
            try:
                return float(annotations[alt_key])
            except (TypeError, ValueError):
                return None
    return None


def _compute_dram_norm(entry: Mapping[str, Any], duration_ms: float, peak_dram: Any) -> float:
    if peak_dram in (None, 0, 0.0):
        return 0.0
    kernels = entry.get("kernels", [])
    total_gb = 0.0
    for kernel in kernels:
        if not isinstance(kernel, Mapping):
            continue
        total_gb += float(kernel.get("dram_read_gb", 0.0))
        total_gb += float(kernel.get("dram_write_gb", 0.0))
    throughput = safe_div(total_gb, duration_ms)
    return safe_div(throughput, peak_dram)


def _estimate_ai(entry: Mapping[str, Any]) -> float:
    metrics = entry.get("metrics")
    if isinstance(metrics, Mapping):
        flops = metrics.get("flops")
        bytes_moved = metrics.get("bytes_moved")
        if flops is not None and bytes_moved not in (None, 0, 0.0):
            try:
                return float(flops) / float(bytes_moved)
            except (TypeError, ZeroDivisionError):
                return 0.0
    kernels = entry.get("kernels", [])
    total_bytes = 0.0
    for kernel in kernels or []:
        if not isinstance(kernel, Mapping):
            continue
        total_bytes += float(kernel.get("bytes_read", 0.0))
        total_bytes += float(kernel.get("bytes_write", 0.0))
    if total_bytes > 0.0:
        return 0.0
    return 0.0


def _average_kernel_metric(entries: Sequence[Mapping[str, Any]], key: str) -> float:
    values: List[float] = []
    for entry in entries:
        kernels = entry.get("kernels", [])
        for kernel in kernels or []:
            if not isinstance(kernel, Mapping):
                continue
            value = kernel.get(key)
            if value is not None:
                try:
                    values.append(float(value))
                except (TypeError, ValueError):
                    continue
    if not values:
        return 0.0
    return sum(values) / len(values)


def _op_signature(entry: Mapping[str, Any]) -> str:
    from hashlib import sha1

    op_type = str(entry.get("op_type", "unknown"))
    index = int(entry.get("op_index", 0))
    shape = _extract_shape(entry)
    payload = f"{op_type}|{shape}|{index}".encode("utf-8")
    return sha1(payload).hexdigest()


def _extract_shape(entry: Mapping[str, Any]) -> str:
    annotations = entry.get("annotations")
    if isinstance(annotations, Mapping):
        shape = annotations.get("shape")
        if shape:
            return str(shape)
    metrics = entry.get("metrics")
    if isinstance(metrics, Mapping):
        shape = metrics.get("shape")
        if shape:
            return str(shape)
    return "unknown"


def _to_namespace(value: Mapping[str, Any] | None) -> ContextNamespace:
    ns = ContextNamespace()
    if value is None:
        return ns
    for key, item in value.items():
        if isinstance(item, Mapping):
            ns[key] = _to_namespace(item)
        else:
            ns[key] = item
    return ns


def _normalize_expression(
    expression: str, context: Mapping[str, Any]
) -> tuple[str, Dict[str, Any]]:
    prepared: Dict[str, Any] = dict(context)
    substitutions = {"global": "g_ctx"}
    normalized = expression
    for original, alias in substitutions.items():
        token = f"{original}."
        if token in normalized:
            normalized = normalized.replace(token, f"{alias}.")
            prepared.setdefault(alias, prepared.get(original, ContextNamespace()))
    return normalized, prepared
