#!/usr/bin/env python3
"""Summarize matched retrained Packet/Flow ablations without test selection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from make_shared_core_sensitivity_report import result_metrics
from run_shared_core_sensitivity import load_json, notes


DATASETS = ("vpn-app", "tls-120")
DIAGNOSTICS = (
    "factual_only",
    "no_semantic",
    "no_content",
    "no_structural",
    "fixed_fusion",
    "row_risk",
)
TASKS = ("packet", "flow")
SPLITS = ("valid", "test")


def unique_result(manifest: dict[str, Any], task: str, split: str) -> Path:
    paths = [Path(str(path)) for path in notes(manifest).get("result_paths") or []]
    if task == "packet":
        matches = [path for path in paths if path.name.startswith(f"{split}_")]
    else:
        matches = [
            path for path in paths
            if path.name.startswith(f"{split}_seq_metrics")
        ]
    if len(matches) != 1:
        raise ValueError(f"{task}/{split} reference must expose exactly one result")
    return matches[0]


def packet_ablation_result(summary: dict[str, Any], split: str) -> Path:
    manifest_path = summary["outputs"]["packet"]["manifest"]
    return unique_result(load_json(manifest_path), "packet", split)


def build_report(
    summary_paths: list[str],
    contribution_delta: float = 0.005,
    simplification_margin: float = 0.005,
) -> dict[str, Any]:
    summaries = [load_json(path) for path in summary_paths]
    observed = {
        (row.get("dataset"), int(row.get("fold", -1)), row.get("diagnostic"))
        for row in summaries
    }
    required = {(dataset, 0, diagnostic) for dataset in DATASETS for diagnostic in DIAGNOSTICS}
    if observed != required or len(summaries) != len(required):
        raise ValueError("fold0 report requires exactly six diagnostics for VPN and TLS")
    fingerprints = {str(row.get("shared_core_config_sha256") or "") for row in summaries}
    if len(fingerprints) != 1 or len(next(iter(fingerprints))) != 64:
        raise ValueError("ablation summaries do not share one frozen-core fingerprint")

    rows = []
    validation_by_diagnostic: dict[str, list[dict[str, float]]] = {
        diagnostic: [] for diagnostic in DIAGNOSTICS
    }
    for summary in summaries:
        if summary.get("scope") != "retrained_ablation" or summary.get("dry_run"):
            raise ValueError("report requires completed retrained ablations")
        if summary.get("test_labels_used_for_model_selection") is not False:
            raise ValueError("ablation used test labels for model selection")
        references = {
            "packet": load_json(summary["packet_reference_manifest"]),
            "flow": load_json(summary["flow_reference_manifest"]),
        }
        for task in TASKS:
            for split in SPLITS:
                baseline_path = unique_result(references[task], task, split)
                if task == "packet":
                    ablated_path = packet_ablation_result(summary, split)
                else:
                    ablated_path = Path(summary["outputs"]["flow"]["results"][split])
                baseline = result_metrics(load_json(baseline_path), task)
                ablated = result_metrics(load_json(ablated_path), task)
                delta = {
                    metric: ablated[metric] - baseline[metric]
                    for metric in ("accuracy", "macro_f1")
                }
                row = {
                    "dataset": summary["dataset"],
                    "fold": 0,
                    "diagnostic": summary["diagnostic"],
                    "task": task,
                    "split": split,
                    "baseline": baseline,
                    "ablated": ablated,
                    "delta": delta,
                    "baseline_path": str(baseline_path),
                    "ablated_path": str(ablated_path),
                }
                rows.append(row)
                if split == "valid":
                    validation_by_diagnostic[summary["diagnostic"]].append(delta)

    decisions = []
    for diagnostic in DIAGNOSTICS:
        values = validation_by_diagnostic[diagnostic]
        if len(values) != 4:
            raise ValueError(f"{diagnostic} lacks four validation task-dataset cells")
        mean_acc = sum(row["accuracy"] for row in values) / len(values)
        mean_f1 = sum(row["macro_f1"] for row in values) / len(values)
        harmed_cells = sum(
            row["macro_f1"] <= -contribution_delta for row in values
        )
        all_noninferior = all(
            row[metric] >= -simplification_margin
            for row in values
            for metric in ("accuracy", "macro_f1")
        )
        if mean_f1 <= -contribution_delta or harmed_cells >= 2:
            decision = "retain_module_and_expand_to_three_folds"
        elif all_noninferior:
            decision = "candidate_simplification_requires_three_fold_noninferiority"
        else:
            decision = "mixed_fold0_evidence_expand_to_three_folds"
        decisions.append(
            {
                "diagnostic": diagnostic,
                "mean_validation_delta_accuracy": mean_acc,
                "mean_validation_delta_macro_f1": mean_f1,
                "validation_cells_harmed": harmed_cells,
                "decision": decision,
                "automatic_module_removal_allowed": False,
            }
        )

    return {
        "schema": "matched_retrained_shared_core_ablation_report_v1",
        "selection_scope": "validation_only_fold0_screen",
        "shared_core_config_sha256": next(iter(fingerprints)),
        "coverage": {
            "datasets": list(DATASETS),
            "folds": [0],
            "tasks": list(TASKS),
            "diagnostics": list(DIAGNOSTICS),
        },
        "thresholds": {
            "contribution_delta": contribution_delta,
            "simplification_margin": simplification_margin,
        },
        "decisions": decisions,
        "rows": rows,
        "test_rows_are_descriptive_only": True,
        "test_labels_used_for_model_selection": False,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Matched Retrained Shared-Core Ablations",
        "",
        "Decisions use fold0 validation only. Test deltas are descriptive and never select modules.",
        "",
        "| Ablation | Mean Valid Delta Acc | Mean Valid Delta Macro-F1 | Harmed Cells | Decision |",
        "|---|---:|---:|---:|---|",
    ]
    for row in report["decisions"]:
        lines.append(
            "| {diagnostic} | {mean_validation_delta_accuracy:+.4f} | "
            "{mean_validation_delta_macro_f1:+.4f} | {validation_cells_harmed} | "
            "{decision} |".format(**row)
        )
    lines += [
        "",
        "No component can be removed from this fold0 screen alone; simplification requires three-fold non-inferiority on both datasets and tasks.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", action="append", required=True)
    parser.add_argument("--contribution_delta", type=float, default=0.005)
    parser.add_argument("--simplification_margin", type=float, default=0.005)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", required=True)
    args = parser.parse_args()
    if args.contribution_delta < 0 or args.simplification_margin < 0:
        parser.error("thresholds must be non-negative")
    report = build_report(
        args.summary, args.contribution_delta, args.simplification_margin
    )
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    output_md.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"status": "complete", "output_json": str(output_json)}, indent=2))


if __name__ == "__main__":
    main()
