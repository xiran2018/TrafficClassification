#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from summarize_experiment_results import DEFAULT_TARGETS, collect_dataset, metric_from_payload


DEFAULT_PROBES = {
    "vpn-app": [
        (
            "paired CE + consistency seq probe",
            "reasoningDataset/vpn-app/test_seq_metrics_flow_rawproj_change_weight_stage8_flowaware_paired_ipport_oldview_seqprobe_probs.json",
        ),
        (
            "paired consistency-only seq probe",
            "reasoningDataset/vpn-app/test_seq_metrics_flow_rawproj_change_weight_stage8_flowaware_paired_ipport_consistency_seqprobe_probs.json",
        ),
        (
            "best + paired seq constrained residual",
            "reasoningDataset/vpn-app/test_fusion_best_paired_seqprobe_minbase90_valid_acc.json",
        ),
    ],
}


DEFAULT_PAPER_SAFE_RESULTS = {
    "vpn-app": "reasoningDataset/vpn-app/test_selector_best_prior_embedding_experts_calib_shift000_valid_macro.json",
    "tls-120": "reasoningDataset/tls-120/test_selector_unified_slot_stacker_tls120_valid_macro.json",
    "ustc-app": "reasoningDataset/ustc-app/test_selector_base_flowproto_full_s200_w002_step150_calib_shift005_valid_macro.json",
}


def cuda_summary() -> Dict[str, Any]:
    try:
        import torch

        available = bool(torch.cuda.is_available())
        return {
            "available": available,
            "device_count": int(torch.cuda.device_count()) if available else 0,
            "device0": torch.cuda.get_device_name(0) if available else "",
        }
    except Exception as exc:
        return {"available": False, "device_count": 0, "device0": "", "error": str(exc)}


def load_metric(path: str) -> Dict[str, Any] | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"path": path, "error": str(exc)}
    acc, f1 = metric_from_payload(data)
    if acc is None:
        return {"path": path, "error": "no flow-level metric found"}
    return {
        "path": path,
        "accuracy": acc,
        "macro_f1": f1,
        "num_flows": len(data.get("flow_y_true", [])),
    }


def probe_rows(dataset: str, best: Dict[str, Any] | None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for name, path in DEFAULT_PROBES.get(dataset, []):
        metric = load_metric(path)
        if metric is None:
            rows.append({"name": name, "path": path, "status": "missing"})
            continue
        row = {"name": name, "status": "available", **metric}
        if best and "accuracy" in metric:
            row["delta_acc_vs_best"] = metric["accuracy"] - best["accuracy"]
            row["delta_f1_vs_best"] = (metric.get("macro_f1") or 0.0) - (best.get("macro_f1") or 0.0)
        rows.append(row)
    return rows


def paper_safe_overrides(raw_items: List[List[str]]) -> Dict[str, str]:
    return {dataset: path for dataset, path in raw_items}


def paper_safe_path(dataset: str, overrides: Dict[str, str]) -> str:
    return overrides.get(dataset, DEFAULT_PAPER_SAFE_RESULTS.get(dataset, ""))


def decide_recommendation(
    dataset: str,
    raw_best: Dict[str, Any] | None,
    paper_safe: Dict[str, Any] | None,
    probes: List[Dict[str, Any]],
    target: tuple[float, float] | None,
    cuda: Dict[str, Any],
) -> str:
    reference = paper_safe or raw_best
    if reference is None:
        return "No test metric JSON was found; run the dataset pipeline through eval/fusion first."
    target_met = bool(target and reference["accuracy"] >= target[0] and (reference.get("macro_f1") or 0.0) >= target[1])
    harmful_probes = [
        row for row in probes
        if row.get("status") == "available"
        and row.get("delta_acc_vs_best", 0.0) < -0.005
        and row.get("delta_f1_vs_best", 0.0) < -0.005
    ]
    if dataset == "vpn-app" and harmful_probes:
        if cuda.get("available"):
            return (
                "Current paired-view probes are negative on old embeddings. Run the full Stage-8 A800 path next: "
                "regenerate rawproj_flowaware_ipport_rand embeddings from the flow-aware Tower-1 checkpoint, then train with --run_tag paired_ipport."
            )
        return (
            "Current paired-view probes are negative on old embeddings, and CUDA is unavailable in this session. "
            "Do not spend more CPU time on graph paired-view training; run the documented A800 Stage-8 paired-view path in the real llm-factory environment."
        )
    if target_met:
        return (
            "Paper-safe target is met; keep raw-best probes as ablations unless validation-gated selection "
            "and target-shift guards accept them into the framework result."
        )
    return "Target is not met; prioritize representation learning experiments before additional probability-level fusion."


def render_markdown(report: Dict[str, Any]) -> str:
    lines = [
        "# Experiment Diagnosis",
        "",
        f"CUDA available: `{report['cuda']['available']}`; devices: `{report['cuda'].get('device_count', 0)}`",
        "",
        "| Dataset | Raw Best Acc | Raw Best F1 | Paper-Safe Acc | Paper-Safe F1 | Target | Status | Raw Best File | Paper-Safe File | Recommendation |",
        "|---|---:|---:|---:|---:|---|---|---|---|---|",
    ]
    for row in report["datasets"]:
        best = row.get("best") or {}
        paper = row.get("paper_safe") or {}
        target = row.get("target")
        target_text = "-" if not target else f"{target[0]:.4f}/{target[1]:.4f}"
        status = "PASS" if row.get("paper_safe_target_met") else ("MISS" if target else "evidence")
        lines.append(
            "| {dataset} | {raw_acc} | {raw_f1} | {paper_acc} | {paper_f1} | {target} | {status} | {raw_path} | {paper_path} | {rec} |".format(
                dataset=row["dataset"],
                raw_acc=f"{best.get('accuracy', 0.0):.4f}" if best else "-",
                raw_f1=f"{best.get('macro_f1', 0.0):.4f}" if best else "-",
                paper_acc=f"{paper.get('accuracy', 0.0):.4f}" if paper else "-",
                paper_f1=f"{paper.get('macro_f1', 0.0):.4f}" if paper else "-",
                target=target_text,
                status=status,
                raw_path=best.get("path", "-"),
                paper_path=paper.get("path", "-"),
                rec=row["recommendation"],
            )
        )
    lines.append("")
    for row in report["datasets"]:
        probes = row.get("probes") or []
        if not probes:
            continue
        lines.append(f"## {row['dataset']} Probes")
        lines.append("")
        lines.append("| Probe | Acc | Macro-F1 | Delta Acc | Delta F1 | File |")
        lines.append("|---|---:|---:|---:|---:|---|")
        for probe in probes:
            if probe.get("status") != "available":
                lines.append(f"| {probe['name']} | - | - | - | - | {probe['path']} |")
                continue
            lines.append(
                "| {name} | {acc:.4f} | {f1:.4f} | {dacc:.4f} | {df1:.4f} | {path} |".format(
                    name=probe["name"],
                    acc=probe["accuracy"],
                    f1=probe.get("macro_f1") or 0.0,
                    dacc=probe.get("delta_acc_vs_best", 0.0),
                    df1=probe.get("delta_f1_vs_best", 0.0),
                    path=probe["path"],
                )
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Diagnose current experiment results and recommend the next unified-framework run.")
    ap.add_argument("--dataset", action="append", default=[], help="Dataset under reasoningDataset/. Can be repeated.")
    ap.add_argument("--pattern", action="append", default=["test*.json"], help="Glob under each dataset directory.")
    ap.add_argument("--target", action="append", default=[], help="Optional DATASET:ACC:MACRO_F1 target override.")
    ap.add_argument(
        "--paper_safe_result",
        nargs=2,
        action="append",
        default=[],
        metavar=("DATASET", "JSON"),
        help="Override the paper-safe framework result used for target status and recommendations.",
    )
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--output_json", default="")
    ap.add_argument("--output_md", default="")
    args = ap.parse_args()

    targets = DEFAULT_TARGETS.copy()
    for raw in args.target:
        dataset, acc, f1 = raw.split(":")
        targets[dataset] = (float(acc), float(f1))

    cuda = cuda_summary()
    datasets = args.dataset or ["vpn-app", "tls-120", "ustc-app"]
    safe_overrides = paper_safe_overrides(args.paper_safe_result)
    report = {"cuda": cuda, "datasets": []}
    for dataset in datasets:
        rows = collect_dataset(dataset, args.pattern)
        best = rows[0] if rows else None
        target = targets.get(dataset)
        safe_path = paper_safe_path(dataset, safe_overrides)
        paper_safe = load_metric(safe_path) if safe_path else None
        reference = paper_safe or best
        probes = probe_rows(dataset, reference)
        raw_target_met = bool(best and target and best["accuracy"] >= target[0] and (best.get("macro_f1") or 0.0) >= target[1])
        reference_target_met = bool(
            reference and target and reference["accuracy"] >= target[0] and (reference.get("macro_f1") or 0.0) >= target[1]
        )
        report["datasets"].append(
            {
                "dataset": dataset,
                "target": target,
                "target_met": reference_target_met if target else None,
                "raw_target_met": raw_target_met if target else None,
                "paper_safe_target_met": reference_target_met if target else None,
                "best": best,
                "paper_safe": paper_safe,
                "top_results": rows[: args.top_k],
                "num_results": len(rows),
                "probes": probes,
                "recommendation": decide_recommendation(dataset, best, paper_safe, probes, target, cuda),
            }
        )

    md = render_markdown(report)
    print(md)
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.output_md:
        Path(args.output_md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_md).write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()
