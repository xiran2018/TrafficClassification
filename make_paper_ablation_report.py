#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


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
        "safe selector",
        "reasoningDataset/vpn-app/test_selector_best_prior_embedding_experts_bootstrap_shift_tol001_valid_macro.json",
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
        "ustc-app",
        "base residual",
        "reasoningDataset/ustc-app/test_fusion_graph_seq_emb_rawproj_flowaware_change_weight_s200_pb8_step150_stage8_flowaware_safe_prior_residual.json",
        "base",
    ),
    (
        "ustc-app",
        "proto embedding expert",
        "reasoningDataset/ustc-app/test_flow_embedding_classifier_flowproto_full_s200_w002_step150_message_header_ports_valid_macro.json",
        "Tower-1 prototype",
    ),
    (
        "ustc-app",
        "safe selector",
        "reasoningDataset/ustc-app/test_selector_base_flowproto_full_s200_w002_step150_bootstrap_shift_tol001_valid_macro.json",
        "class-precision gate",
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


def build_rows(cases: List[Tuple[str, str, str, str]]) -> List[Dict[str, Any]]:
    rows = []
    baselines: Dict[str, Tuple[float | None, float | None]] = {}
    for dataset, stage, path, note in cases:
        data = load_json(path)
        acc, macro_f1 = metric_from_payload(data)
        if dataset not in baselines:
            baselines[dataset] = (acc, macro_f1)
        base_acc, base_f1 = baselines[dataset]
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


def parse_case(raw: List[str]) -> Tuple[str, str, str, str]:
    if len(raw) != 4:
        raise ValueError("--case expects DATASET STAGE JSON NOTE")
    return raw[0], raw[1], raw[2], raw[3]


def main() -> None:
    ap = argparse.ArgumentParser(description="Build paper-ready ablation table for unified traffic framework modules.")
    ap.add_argument("--case", nargs=4, action="append", metavar=("DATASET", "STAGE", "JSON", "NOTE"))
    ap.add_argument("--output_json", default="")
    ap.add_argument("--output_md", default="")
    args = ap.parse_args()

    cases = [parse_case(item) for item in args.case] if args.case else DEFAULT_CASES
    rows = build_rows(cases)
    md = markdown_table(rows)
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
