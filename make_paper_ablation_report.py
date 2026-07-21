#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Any, Dict, List, Tuple

from sklearn.metrics import f1_score


DEFAULT_CASES = [
    (
        "vpn-app",
        "base constrained ensemble",
        "reasoningDataset/vpn-app/test_fusion_best_prior_flow_embedding_experts_minbest90_valid_acc.json",
        "strong base",
    ),
    (
        "vpn-app",
        "unsafe reliability fusion",
        "reasoningDataset/vpn-app/test_selector_best_prior_embedding_experts_reliability_unsafe_valid_macro.json",
        "validation gain, target shift",
    ),
    (
        "vpn-app",
        "calibration-enabled selector",
        "reasoningDataset/vpn-app/test_selector_best_prior_embedding_experts_calib_shift005_valid_macro.json",
        "extra candidate, still unsafe",
    ),
    (
        "vpn-app",
        "safe selector",
        "reasoningDataset/vpn-app/test_selector_best_prior_embedding_experts_calib_shift000_valid_macro.json",
        "target-shift fallback",
    ),
    (
        "tls-120",
        "graph/seq base",
        "reasoningDataset/tls-120/test_fusion_graph_seq_tls120_rawproj_change_weight_valid_acc.json",
        "strong base",
    ),
    (
        "tls-120",
        "strict safe selector",
        "reasoningDataset/tls-120/test_selector_graph_seq_rawproj_change_weight_bootstrap_shift_safe_valid_macro.json",
        "strict bootstrap fallback",
    ),
    (
        "tls-120",
        "tolerant safe selector",
        "reasoningDataset/tls-120/test_selector_graph_seq_rawproj_change_weight_bootstrap_shift_tol001_valid_macro.json",
        "low-shift seq switch",
    ),
    (
        "tls-120",
        "unified-slot stacker",
        "reasoningDataset/tls-120/test_stacker_unified_slot_tls120_confidence_valid_macro.json",
        "trainable slot stacker upper/probe",
    ),
    (
        "tls-120",
        "guarded slot-stacker selector",
        "reasoningDataset/tls-120/test_selector_unified_slot_stacker_tls120_valid_macro.json",
        "low-shift stacker switch",
    ),
    (
        "tls-120",
        "soft expert gate",
        "reasoningDataset/tls-120/test_expert_gate_base_slot_stacker_tls120_e80_seed7.json",
        "trainable expert weighting",
    ),
    (
        "tls-120",
        "soft-gate calibrated selector",
        "reasoningDataset/tls-120/test_selector_soft_gate_tls120_tol0015_calib_family_valid_macro.json",
        "class-bias calibrated soft gate",
    ),
    (
        "tls-120",
        "coverage-audited distill student fusion",
        "reasoningDataset/tls-120/test_fusion_graph_seq_rawproj_flowaware_change_weight_fold2_stage8_flowaware_consensus_distill_student_valid_acc.json",
        "single-student graph/seq distillation; lower than consensus teacher",
    ),
    (
        "tls-120",
        "coverage-audited distill student selector",
        "reasoningDataset/tls-120/test_selector_best_plus_rawproj_flowaware_change_weight_fold2_stage8_flowaware_consensus_distill_student_valid_macro.json",
        "student candidate admitted through unified expert slots",
    ),
]


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def metric_from_payload(data: Dict[str, Any]) -> Tuple[float | None, float | None]:
    nested = data.get("metrics")
    if isinstance(nested, dict):
        flow_level = nested.get("flow_level")
        if isinstance(flow_level, dict):
            acc = flow_level.get("accuracy", flow_level.get("flow_acc"))
            f1 = flow_level.get("macro_f1", flow_level.get("flow_macro_f1"))
            return float(acc) if acc is not None else None, float(f1) if f1 is not None else None
    for acc_key in ("flow_acc", "accuracy", "acc"):
        if acc_key in data:
            f1 = data.get("flow_macro_f1", data.get("macro_f1"))
            return float(data[acc_key]), float(f1) if f1 is not None else None
    return None, None


def format_float(value: Any, digits: int = 4, signed: bool = False) -> str:
    if value is None:
        return "-"
    prefix = "+" if signed and float(value) > 0 else ""
    return f"{prefix}{float(value):.{digits}f}"


def percentile(values: List[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return float(ordered[lo] * (1.0 - frac) + ordered[hi] * frac)


def format_ci(ci: Any, signed: bool = True) -> str:
    if not isinstance(ci, list) or len(ci) != 2:
        return "-"
    return f"[{format_float(ci[0], signed=signed)}, {format_float(ci[1], signed=signed)}]"


def paired_delta_uncertainty(
    base_data: Dict[str, Any],
    data: Dict[str, Any],
    samples: int,
    seed: int,
) -> Dict[str, Any] | None:
    y_true = data.get("flow_y_true")
    y_pred = data.get("flow_y_pred")
    base_true = base_data.get("flow_y_true")
    base_pred = base_data.get("flow_y_pred")
    if (
        not y_true
        or not y_pred
        or not base_true
        or not base_pred
        or len(y_true) != len(y_pred)
        or len(base_true) != len(base_pred)
        or len(y_true) != len(base_true)
        or samples <= 0
    ):
        return None
    if list(y_true) != list(base_true):
        return None
    n = len(y_true)
    rng = random.Random(seed)
    dacc_values: List[float] = []
    df1_values: List[float] = []
    for _ in range(samples):
        idx = [rng.randrange(n) for _ in range(n)]
        boot_true = [y_true[i] for i in idx]
        boot_pred = [y_pred[i] for i in idx]
        boot_base_pred = [base_pred[i] for i in idx]
        acc = sum(1 for a, b in zip(boot_true, boot_pred) if a == b) / n
        base_acc = sum(1 for a, b in zip(boot_true, boot_base_pred) if a == b) / n
        f1 = float(f1_score(boot_true, boot_pred, average="macro", zero_division=0))
        base_f1 = float(f1_score(boot_true, boot_base_pred, average="macro", zero_division=0))
        dacc_values.append(acc - base_acc)
        df1_values.append(f1 - base_f1)
    return {
        "samples": samples,
        "seed": seed,
        "delta_accuracy_ci95": [percentile(dacc_values, 0.025), percentile(dacc_values, 0.975)],
        "delta_macro_f1_ci95": [percentile(df1_values, 0.025), percentile(df1_values, 0.975)],
    }


def selector_label(data: Dict[str, Any]) -> str:
    selected = data.get("selected")
    if not isinstance(selected, dict):
        weights = data.get("selected_weights")
        if isinstance(weights, dict):
            active = [f"{name}={float(value):.2f}" for name, value in weights.items() if float(value) > 0]
            return ", ".join(active) if active else "-"
        return "-"
    fallback = selected.get("fallback_reason")
    if fallback:
        rejected = fallback.get("rejected", {}).get("strategy", "-")
        return f"fallback; reject {rejected}"
    strategy = selected.get("strategy", "-")
    config = selected.get("config") or {}
    if strategy == "threshold_switch":
        return f"threshold_switch:{config.get('expert')}"
    if strategy == "class_precision":
        return f"class_precision:a={config.get('alpha')}"
    if strategy == "reliability_fusion":
        return "reliability_fusion"
    if strategy == "always":
        return f"always:{config.get('source')}"
    return strategy


def build_rows(cases: List[Tuple[str, str, str, str]], bootstrap_samples: int, bootstrap_seed: int) -> List[Dict[str, Any]]:
    rows = []
    baselines: Dict[str, Tuple[float | None, float | None]] = {}
    baseline_payloads: Dict[str, Dict[str, Any]] = {}
    for dataset, stage, path, note in cases:
        data = load_json(path)
        acc, macro_f1 = metric_from_payload(data)
        if dataset not in baselines:
            baselines[dataset] = (acc, macro_f1)
            baseline_payloads[dataset] = data
        base_acc, base_f1 = baselines[dataset]
        uncertainty = paired_delta_uncertainty(baseline_payloads[dataset], data, bootstrap_samples, bootstrap_seed)
        rows.append(
            {
                "dataset": dataset,
                "stage": stage,
                "path": path,
                "note": note,
                "accuracy": acc,
                "macro_f1": macro_f1,
                "delta_accuracy": None if acc is None or base_acc is None else acc - base_acc,
                "delta_macro_f1": None if macro_f1 is None or base_f1 is None else macro_f1 - base_f1,
                "num_flows": len(data.get("flow_y_true", [])),
                "selector": selector_label(data),
                "paired_delta_uncertainty": uncertainty,
            }
        )
    return rows


def markdown_table(rows: List[Dict[str, Any]]) -> str:
    lines = [
        "| Dataset | Stage | Accuracy | Delta Acc | Macro-F1 | Delta F1 | Selector/Fusion | Note |",
        "|---|---|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {dataset} | {stage} | {acc} | {dacc} | {f1} | {df1} | {selector} | {note} |".format(
                dataset=row["dataset"],
                stage=row["stage"],
                acc=format_float(row["accuracy"]),
                dacc=format_float(row["delta_accuracy"], signed=True),
                f1=format_float(row["macro_f1"]),
                df1=format_float(row["delta_macro_f1"], signed=True),
                selector=row["selector"],
                note=row["note"],
            )
        )
    return "\n".join(lines) + "\n"


def markdown_uncertainty(rows: List[Dict[str, Any]]) -> str:
    if not any(row.get("paired_delta_uncertainty") for row in rows):
        return ""
    lines = [
        "",
        "Paired bootstrap delta vs dataset baseline",
        "",
        "| Dataset | Stage | Samples | Delta Acc 95% CI | Delta Macro-F1 95% CI |",
        "|---|---|---:|---:|---:|",
    ]
    for row in rows:
        unc = row.get("paired_delta_uncertainty") or {}
        if not unc:
            lines.append(f"| {row['dataset']} | {row['stage']} | - | - | - |")
            continue
        lines.append(
            "| {dataset} | {stage} | {samples} | {dacc_ci} | {df1_ci} |".format(
                dataset=row["dataset"],
                stage=row["stage"],
                samples=unc.get("samples", "-"),
                dacc_ci=format_ci(unc.get("delta_accuracy_ci95")),
                df1_ci=format_ci(unc.get("delta_macro_f1_ci95")),
            )
        )
    lines.append("")
    return "\n".join(lines)


def parse_case(raw: List[str]) -> Tuple[str, str, str, str]:
    if len(raw) != 4:
        raise ValueError("--case expects DATASET STAGE JSON NOTE")
    return raw[0], raw[1], raw[2], raw[3]


def main() -> None:
    ap = argparse.ArgumentParser(description="Build paper-ready ablation table for unified traffic framework modules.")
    ap.add_argument("--case", nargs=4, action="append", metavar=("DATASET", "STAGE", "JSON", "NOTE"))
    ap.add_argument("--bootstrap_samples", type=int, default=300)
    ap.add_argument("--bootstrap_seed", type=int, default=17)
    ap.add_argument("--output_json", default="")
    ap.add_argument("--output_md", default="")
    args = ap.parse_args()

    cases = [parse_case(item) for item in args.case] if args.case else DEFAULT_CASES
    rows = build_rows(cases, args.bootstrap_samples, args.bootstrap_seed)
    md = markdown_table(rows) + markdown_uncertainty(rows)
    print(md)

    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump({"ablations": rows}, f, indent=2, ensure_ascii=False)
    if args.output_md:
        Path(args.output_md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_md).write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()
