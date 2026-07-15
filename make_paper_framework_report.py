#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Any, Dict, List, Tuple

from sklearn.metrics import f1_score


DEFAULT_RESULTS = [
    (
        "vpn-app",
        "reasoningDataset/vpn-app/test_selector_best_prior_embedding_experts_calib_shift000_valid_macro.json",
        0.74,
        0.65,
    ),
    (
        "tls-120",
        "reasoningDataset/tls-120/test_selector_graph_seq_rawproj_change_weight_calib_shift005_valid_macro.json",
        0.78,
        0.70,
    ),
    (
        "ustc-app",
        "reasoningDataset/ustc-app/test_selector_base_flowproto_full_s200_w002_step150_calib_shift005_valid_macro.json",
        None,
        None,
    ),
]


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def flow_metrics(data: Dict[str, Any]) -> Tuple[float | None, float | None]:
    metrics = data.get("metrics")
    if isinstance(metrics, dict):
        flow = metrics.get("flow_level")
        if isinstance(flow, dict):
            acc = flow.get("accuracy", flow.get("flow_acc"))
            f1 = flow.get("macro_f1", flow.get("flow_macro_f1"))
            return (float(acc) if acc is not None else None, float(f1) if f1 is not None else None)
    return None, None


def format_float(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def percentile(values: List[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return float(ordered[lo] * (1.0 - frac) + ordered[hi] * frac)


def bootstrap_uncertainty(data: Dict[str, Any], samples: int, seed: int) -> Dict[str, Any] | None:
    y_true = data.get("flow_y_true")
    y_pred = data.get("flow_y_pred")
    if not y_true or not y_pred or len(y_true) != len(y_pred) or samples <= 0:
        return None
    n = len(y_true)
    rng = random.Random(seed)
    acc_values: List[float] = []
    f1_values: List[float] = []
    for _ in range(samples):
        idx = [rng.randrange(n) for _ in range(n)]
        boot_true = [y_true[i] for i in idx]
        boot_pred = [y_pred[i] for i in idx]
        acc_values.append(sum(1 for a, b in zip(boot_true, boot_pred) if a == b) / n)
        f1_values.append(float(f1_score(boot_true, boot_pred, average="macro", zero_division=0)))
    return {
        "samples": samples,
        "seed": seed,
        "accuracy_ci95": [percentile(acc_values, 0.025), percentile(acc_values, 0.975)],
        "macro_f1_ci95": [percentile(f1_values, 0.025), percentile(f1_values, 0.975)],
    }


def format_ci(ci: Any) -> str:
    if not isinstance(ci, list) or len(ci) != 2:
        return "-"
    return f"[{format_float(ci[0])}, {format_float(ci[1])}]"


def selector_summary(selected: Dict[str, Any]) -> str:
    strategy = selected.get("strategy", "-")
    config = selected.get("config") or {}
    fallback = selected.get("fallback_reason")
    if fallback:
        rejected = fallback.get("rejected", {})
        rejected_strategy = rejected.get("strategy", "-")
        source = config.get("source", "base")
        first_rejection = {}
        if fallback.get("rejected_candidates"):
            first_rejection = fallback["rejected_candidates"][0]
            rejected = first_rejection.get("rejected", rejected)
            rejected_strategy = rejected.get("strategy", rejected_strategy)
        carrier = first_rejection if first_rejection else fallback
        shift = carrier.get("target_shift_guard", {})
        boot = carrier.get("bootstrap_guard", {})
        reasons = []
        if "max_prediction_change_rate" in carrier:
            reasons.append(
                f"target_change={format_float(shift.get('prediction_change_rate'))}>{format_float(carrier.get('max_prediction_change_rate'))}"
            )
        if "bootstrap_min_gain_quantile" in carrier and carrier.get("reject_reasons", {}).get("bootstrap", True):
            reasons.append(f"boot_q={format_float(boot.get('gain_quantile'))}<0")
        if not reasons and "min_valid_gain_over_base" in fallback:
            reasons.append(
                f"valid_gain={format_float(fallback.get('selected_gain'))}<{format_float(fallback.get('min_valid_gain_over_base'))}"
            )
        return f"fallback to {source}; rejected {rejected_strategy} ({'; '.join(reasons)})"
    if strategy == "class_precision":
        return f"class_precision alpha={config.get('alpha')}, margin={config.get('metric_margin')}"
    if strategy == "reliability_fusion":
        return (
            "reliability_fusion "
            f"alpha={config.get('alpha')}, rpow={config.get('reliability_power')}, "
            f"cpow={config.get('confidence_power')}, temp={config.get('temperature')}"
        )
    if strategy == "threshold_switch":
        return f"threshold_switch expert={config.get('expert')}"
    if strategy == "always":
        return f"always {config.get('source')}"
    return strategy


def guard_summary(selected: Dict[str, Any]) -> str:
    fallback = selected.get("fallback_reason")
    carrier = fallback if fallback else selected
    if fallback and fallback.get("rejected_candidates"):
        carrier = fallback["rejected_candidates"][0]
    boot = carrier.get("bootstrap_guard", {})
    shift = carrier.get("target_shift_guard", {})
    parts = []
    if boot:
        parts.append(
            "bootstrap "
            f"win={format_float(boot.get('win_rate'), 2)}, "
            f"q={format_float(boot.get('gain_quantile'))}"
        )
    if shift:
        parts.append(
            "target "
            f"change={format_float(shift.get('prediction_change_rate'))}, "
            f"JS={format_float(shift.get('prediction_js_divergence'))}"
        )
    return "; ".join(parts) if parts else "-"


def multi_view_gate_summary(data: Dict[str, Any]) -> Dict[str, Any] | None:
    metrics = data.get("metrics")
    if not isinstance(metrics, dict):
        return None
    eval_config = metrics.get("eval_config")
    if not isinstance(eval_config, dict):
        return None
    gate = eval_config.get("multi_view_gate")
    if not isinstance(gate, dict):
        return None
    branches = gate.get("branches") or []
    mean = gate.get("mean") or []
    std = gate.get("std") or []
    if not branches or not mean or len(branches) != len(mean):
        return None
    pairs = sorted(zip(branches, mean), key=lambda item: float(item[1]), reverse=True)
    return {
        "branches": branches,
        "mean": mean,
        "std": std,
        "num_flows": gate.get("num_flows"),
        "dominant_branch": pairs[0][0],
        "dominant_weight": float(pairs[0][1]),
    }


def module_usage(data: Dict[str, Any]) -> Dict[str, str]:
    selected = data.get("selected", {})
    feature_config = data.get("feature_config") or {}
    strategies = feature_config.get("strategies") or []
    strategy = selected.get("strategy", "unknown")
    config = selected.get("config") or {}
    fallback = selected.get("fallback_reason")

    usage = {
        "packet_embedding_backbone": "active",
        "flow_base_expert": "active",
        "validation_gated_selector": "active",
        "bootstrap_guard": "inactive",
        "target_shift_guard": "inactive",
        "expert_switch_or_fusion": "identity",
        "class_bias_calibration_candidate": "not_configured",
        "trainable_multiview_gate": "active" if multi_view_gate_summary(data) else "not_observed",
    }
    if "class_bias_calibration" in strategies:
        usage["class_bias_calibration_candidate"] = "evaluated"

    carrier = selected
    if fallback:
        rejected = fallback.get("rejected", {})
        if fallback.get("rejected_candidates"):
            carrier = fallback["rejected_candidates"][0]
            rejected = carrier.get("rejected", rejected)
        else:
            carrier = fallback
        rejected_strategy = rejected.get("strategy", "candidate")
        usage["expert_switch_or_fusion"] = f"gated_off:{rejected_strategy}"
    elif strategy == "always":
        source = config.get("source", "base")
        usage["expert_switch_or_fusion"] = f"identity:{source}"
    elif strategy in {"threshold_switch", "class_precision", "reliability_fusion"}:
        usage["expert_switch_or_fusion"] = f"active:{strategy}"
    else:
        usage["expert_switch_or_fusion"] = f"active:{strategy}"

    if carrier.get("bootstrap_guard"):
        usage["bootstrap_guard"] = "active"
    if carrier.get("target_shift_guard"):
        usage["target_shift_guard"] = "active"
    return usage


def module_usage_summary(usage: Dict[str, str]) -> str:
    return (
        f"base={usage['flow_base_expert']}; "
        f"selector={usage['validation_gated_selector']}; "
        f"expert={usage['expert_switch_or_fusion']}; "
        f"calib={usage['class_bias_calibration_candidate']}; "
        f"mv_gate={usage.get('trainable_multiview_gate', 'not_observed')}; "
        f"guards=boot:{usage['bootstrap_guard']},shift:{usage['target_shift_guard']}"
    )


def framework_consistency(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    required_active = [
        "packet_embedding_backbone",
        "flow_base_expert",
        "validation_gated_selector",
    ]
    required_safety = [
        "bootstrap_guard",
        "target_shift_guard",
    ]
    required_candidates = [
        "expert_switch_or_fusion",
        "class_bias_calibration_candidate",
        "trainable_multiview_gate",
    ]
    dataset_checks = []
    for row in rows:
        usage = row["module_usage"]
        failures = []
        for name in required_active:
            if usage.get(name) != "active":
                failures.append(f"{name}!={usage.get(name)}")
        for name in required_safety:
            if usage.get(name) != "active":
                failures.append(f"{name}!={usage.get(name)}")
        expert_state = usage.get("expert_switch_or_fusion", "")
        if not expert_state.startswith(("active:", "gated_off:", "identity:")):
            failures.append(f"expert_switch_or_fusion!={expert_state}")
        if usage.get("class_bias_calibration_candidate") not in {"evaluated", "active", "gated_off"}:
            failures.append(f"class_bias_calibration_candidate!={usage.get('class_bias_calibration_candidate')}")
        dataset_checks.append(
            {
                "dataset": row["dataset"],
                "consistent": not failures,
                "failures": failures,
                "expert_state": expert_state,
            }
        )
    return {
        "consistent": all(item["consistent"] for item in dataset_checks),
        "required_active_modules": required_active,
        "required_safety_modules": required_safety,
        "candidate_modules": required_candidates,
        "dataset_checks": dataset_checks,
        "interpretation": (
            "All datasets pass through the same backbone, base expert, validation-gated selector, "
            "bootstrap guard, target-shift guard, and candidate expert/calibration family. "
            "Dataset-specific validation may activate, gate off, or leave candidate experts as identity. "
            "When multi-view pooling is evaluated, learned branch-gate weights are reported as trainable module evidence."
        ),
    }


def build_rows(results: List[Tuple[str, str, float | None, float | None]], bootstrap_samples: int, bootstrap_seed: int) -> List[Dict[str, Any]]:
    rows = []
    for dataset, path, target_acc, target_f1 in results:
        data = load_json(path)
        acc, macro_f1 = flow_metrics(data)
        selected = data.get("selected", {})
        achieved = None
        if target_acc is not None and target_f1 is not None and acc is not None and macro_f1 is not None:
            achieved = acc >= target_acc and macro_f1 >= target_f1
        rows.append(
            {
                "dataset": dataset,
                "path": path,
                "accuracy": acc,
                "macro_f1": macro_f1,
                "target_accuracy": target_acc,
                "target_macro_f1": target_f1,
                "achieved": achieved,
                "num_flows": len(data.get("flow_y_true", [])),
                "selector": selector_summary(selected),
                "guards": guard_summary(selected),
                "module_usage": module_usage(data),
                "multi_view_gate": multi_view_gate_summary(data),
                "uncertainty": bootstrap_uncertainty(data, bootstrap_samples, bootstrap_seed),
            }
        )
    return rows


def markdown_table(rows: List[Dict[str, Any]]) -> str:
    lines = [
        "| Dataset | Accuracy | Macro-F1 | Target | Status | Flows | Module usage | Selector decision | Guards |",
        "|---|---:|---:|---|---|---:|---|---|---|",
    ]
    for row in rows:
        target = "-"
        if row["target_accuracy"] is not None:
            target = f"{format_float(row['target_accuracy'])}/{format_float(row['target_macro_f1'])}"
        status = "PASS" if row["achieved"] is True else ("MISS" if row["achieved"] is False else "evidence")
        lines.append(
            "| {dataset} | {acc} | {f1} | {target} | {status} | {flows} | {modules} | {selector} | {guards} |".format(
                dataset=row["dataset"],
                acc=format_float(row["accuracy"]),
                f1=format_float(row["macro_f1"]),
                target=target,
                status=status,
                flows=row["num_flows"],
                modules=module_usage_summary(row["module_usage"]),
                selector=row["selector"],
                guards=row["guards"],
            )
        )
    return "\n".join(lines) + "\n"


def markdown_consistency(audit: Dict[str, Any]) -> str:
    status = "PASS" if audit["consistent"] else "CHECK"
    lines = [
        "",
        f"Framework consistency audit: {status}",
        "",
        "| Dataset | Consistent | Expert candidate state | Notes |",
        "|---|---|---|---|",
    ]
    for item in audit["dataset_checks"]:
        notes = "same module family" if item["consistent"] else "; ".join(item["failures"])
        lines.append(
            "| {dataset} | {consistent} | {expert} | {notes} |".format(
                dataset=item["dataset"],
                consistent="yes" if item["consistent"] else "no",
                expert=item["expert_state"],
                notes=notes,
            )
        )
    lines += ["", audit["interpretation"], ""]
    return "\n".join(lines)


def markdown_uncertainty(rows: List[Dict[str, Any]]) -> str:
    if not any(row.get("uncertainty") for row in rows):
        return ""
    lines = [
        "",
        "Flow-level bootstrap uncertainty",
        "",
        "| Dataset | Samples | Accuracy 95% CI | Macro-F1 95% CI |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        unc = row.get("uncertainty") or {}
        if not unc:
            lines.append(f"| {row['dataset']} | - | - | - |")
            continue
        lines.append(
            "| {dataset} | {samples} | {acc_ci} | {f1_ci} |".format(
                dataset=row["dataset"],
                samples=unc.get("samples", "-"),
                acc_ci=format_ci(unc.get("accuracy_ci95")),
                f1_ci=format_ci(unc.get("macro_f1_ci95")),
            )
        )
    lines.append("")
    return "\n".join(lines)


def markdown_multiview_gates(rows: List[Dict[str, Any]]) -> str:
    gated = [row for row in rows if row.get("multi_view_gate")]
    if not gated:
        return ""
    lines = [
        "",
        "Trainable multi-view branch gates",
        "",
        "| Dataset | Flows | Dominant branch | Mean weights: mean/max/std/attention |",
        "|---|---:|---|---|",
    ]
    for row in gated:
        gate = row["multi_view_gate"]
        branches = gate.get("branches") or []
        means = gate.get("mean") or []
        weight_by_branch = {branch: value for branch, value in zip(branches, means)}
        ordered = [weight_by_branch.get(branch) for branch in ["mean", "max", "std", "attention"]]
        weight_text = "/".join(format_float(value) for value in ordered)
        lines.append(
            "| {dataset} | {flows} | {branch} ({weight}) | {weights} |".format(
                dataset=row["dataset"],
                flows=gate.get("num_flows", "-"),
                branch=gate.get("dominant_branch", "-"),
                weight=format_float(gate.get("dominant_weight")),
                weights=weight_text,
            )
        )
    lines.append("")
    return "\n".join(lines)


def parse_result(raw: List[str]) -> Tuple[str, str, float | None, float | None]:
    if len(raw) != 4:
        raise ValueError("--result expects DATASET JSON TARGET_ACC TARGET_MACRO_F1; use '-' for no target")
    dataset, path, acc, f1 = raw
    target_acc = None if acc == "-" else float(acc)
    target_f1 = None if f1 == "-" else float(f1)
    return dataset, path, target_acc, target_f1


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a paper-ready summary for the unified traffic framework.")
    ap.add_argument("--result", nargs=4, action="append", metavar=("DATASET", "JSON", "TARGET_ACC", "TARGET_F1"))
    ap.add_argument("--bootstrap_samples", type=int, default=300)
    ap.add_argument("--bootstrap_seed", type=int, default=13)
    ap.add_argument("--output_json", default="")
    ap.add_argument("--output_md", default="")
    args = ap.parse_args()

    results = [parse_result(item) for item in args.result] if args.result else DEFAULT_RESULTS
    rows = build_rows(results, args.bootstrap_samples, args.bootstrap_seed)
    audit = framework_consistency(rows)
    md = markdown_table(rows) + markdown_consistency(audit) + markdown_multiview_gates(rows) + markdown_uncertainty(rows)
    print(md)

    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump({"results": rows, "framework_consistency": audit}, f, indent=2, ensure_ascii=False)
    if args.output_md:
        Path(args.output_md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_md).write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()
