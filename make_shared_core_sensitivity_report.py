#!/usr/bin/env python3
"""Summarize identical Packet/Flow shared-core sensitivity diagnostics."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


REQUIRED_DATASETS = ("vpn-app", "tls-120")
REQUIRED_FOLDS = (0, 1, 2)
REQUIRED_TASKS = ("packet", "flow")


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def framework_notes(manifest: dict[str, Any]) -> dict[str, Any]:
    return (manifest.get("framework") or {}).get("notes") or {}


def result_metrics(payload: dict[str, Any], task: str) -> dict[str, float]:
    metrics = payload.get("metrics") or {}
    if task == "flow":
        metrics = metrics.get("flow_level") or {}
    if "accuracy" not in metrics or "macro_f1" not in metrics:
        raise ValueError(f"{task} result lacks accuracy/macro_f1")
    return {
        "accuracy": float(metrics["accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
    }


def baseline_result_path(manifest: dict[str, Any], task: str) -> Path:
    paths = [str(path) for path in framework_notes(manifest).get("result_paths") or []]
    if task == "packet":
        matches = [path for path in paths if "test_unified_packet_single_head" in path]
    else:
        matches = [path for path in paths if "test_seq_metrics" in path]
    if len(matches) != 1:
        raise ValueError(f"{task} manifest must expose exactly one fixed test result")
    return Path(matches[0])


def build_report(summary_paths: list[str], weak_delta: float = 0.002) -> dict[str, Any]:
    summaries = [load_json(path) for path in summary_paths]
    observed = {(row.get("dataset"), int(row.get("fold", -1))) for row in summaries}
    required = {(dataset, fold) for dataset in REQUIRED_DATASETS for fold in REQUIRED_FOLDS}
    if observed != required or len(summaries) != len(required):
        raise ValueError("sensitivity summaries must cover VPN/TLS folds 0,1,2 exactly")
    fingerprints = {str(row.get("shared_core_config_sha256") or "") for row in summaries}
    if len(fingerprints) != 1 or len(next(iter(fingerprints))) != 64:
        raise ValueError("sensitivity summaries do not share one frozen-core fingerprint")
    rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], list[dict[str, float]]] = defaultdict(list)
    for summary in summaries:
        if summary.get("scope") != "inference_only_not_retrained_ablation":
            raise ValueError("sensitivity summary has an invalid scope")
        if summary.get("split") != "test":
            raise ValueError("paper sensitivity report requires fixed test diagnostics")
        manifests = {
            "packet": load_json(summary["packet_manifest"]),
            "flow": load_json(summary["flow_manifest"]),
        }
        baselines = {
            task: result_metrics(load_json(baseline_result_path(manifests[task], task)), task)
            for task in REQUIRED_TASKS
        }
        for diagnostic, task_paths in (summary.get("diagnostics") or {}).items():
            if set(task_paths) != set(REQUIRED_TASKS):
                raise ValueError(f"{diagnostic} does not cover Packet and Flow")
            for task in REQUIRED_TASKS:
                ablated = result_metrics(load_json(task_paths[task]), task)
                delta = {
                    "accuracy": ablated["accuracy"] - baselines[task]["accuracy"],
                    "macro_f1": ablated["macro_f1"] - baselines[task]["macro_f1"],
                }
                grouped[(diagnostic, task)].append(delta)
                rows.append(
                    {
                        "dataset": summary["dataset"],
                        "fold": int(summary["fold"]),
                        "task": task,
                        "diagnostic": diagnostic,
                        "baseline": baselines[task],
                        "ablated": ablated,
                        "delta": delta,
                        "path": task_paths[task],
                    }
                )
    aggregate: list[dict[str, Any]] = []
    by_diagnostic: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (diagnostic, task), values in sorted(grouped.items()):
        if len(values) != 6:
            raise ValueError(f"{diagnostic}/{task} does not contain six folds")
        row = {
            "diagnostic": diagnostic,
            "task": task,
            "num_dataset_folds": len(values),
            "mean_delta_accuracy": sum(item["accuracy"] for item in values) / len(values),
            "mean_delta_macro_f1": sum(item["macro_f1"] for item in values) / len(values),
            "worst_delta_accuracy": min(item["accuracy"] for item in values),
            "worst_delta_macro_f1": min(item["macro_f1"] for item in values),
        }
        aggregate.append(row)
        by_diagnostic[diagnostic].append(row)
    decisions = []
    for diagnostic, task_rows in sorted(by_diagnostic.items()):
        negligible = all(
            abs(row[metric]) < weak_delta
            for row in task_rows
            for metric in ("mean_delta_accuracy", "mean_delta_macro_f1")
        )
        decisions.append(
            {
                "diagnostic": diagnostic,
                "inference_sensitivity_negligible": negligible,
                "decision": (
                    "schedule_retrained_ablation_before_considering_removal"
                    if negligible
                    else "retain_until_matched_retrained_ablation"
                ),
                "automatic_module_removal_allowed": False,
            }
        )
    return {
        "schema": "exact_shared_core_inference_sensitivity_report_v1",
        "scope": "inference_only_not_retrained_ablation",
        "shared_core_config_sha256": next(iter(fingerprints)),
        "coverage": {
            "datasets": list(REQUIRED_DATASETS),
            "folds": list(REQUIRED_FOLDS),
            "tasks": list(REQUIRED_TASKS),
            "num_rows": len(rows),
        },
        "weak_delta_threshold": weak_delta,
        "aggregate": aggregate,
        "module_decisions": decisions,
        "rows": rows,
        "test_labels_used_for_model_selection": False,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Shared-Core Inference Sensitivity",
        "",
        "These are inference-only diagnostics on frozen models, not retrained ablations.",
        "",
        "| Diagnostic | Task | Mean Delta Acc | Mean Delta Macro-F1 | Worst Delta Acc | Worst Delta Macro-F1 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in report["aggregate"]:
        lines.append(
            "| {diagnostic} | {task} | {mean_delta_accuracy:+.4f} | "
            "{mean_delta_macro_f1:+.4f} | {worst_delta_accuracy:+.4f} | "
            "{worst_delta_macro_f1:+.4f} |".format(**row)
        )
    lines += [
        "",
        "No module is removed automatically. A negligible diagnostic only schedules a matched retrained ablation.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", action="append", required=True)
    parser.add_argument("--weak_delta", type=float, default=0.002)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", required=True)
    args = parser.parse_args()
    if args.weak_delta < 0:
        parser.error("--weak_delta must be non-negative")
    report = build_report(args.summary, weak_delta=args.weak_delta)
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    output_md.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"status": "complete", "output_json": str(output_json)}, indent=2))


if __name__ == "__main__":
    main()
