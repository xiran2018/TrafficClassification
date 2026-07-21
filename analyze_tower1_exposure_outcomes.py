#!/usr/bin/env python3
"""Relate Tower-1 class exposure to held-out outcomes without claiming causality."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr


def correlation(values: np.ndarray, outcomes: np.ndarray) -> dict[str, float]:
    result = spearmanr(values, outcomes)
    return {
        "rho": float(result.statistic),
        "p_value": float(result.pvalue),
    }


def analyze_exposure_outcomes(
    sampling_report: dict[str, Any], validation_metrics: dict[str, Any]
) -> dict[str, Any]:
    flow_counts = {int(key): int(value) for key, value in sampling_report["flow_counts"].items()}
    per_class = validation_metrics["metrics"]["per_class"]
    rows = []
    seen_labels = set()
    for class_name, metrics in per_class.items():
        label_id = int(metrics["label_id"])
        if label_id in seen_labels:
            raise ValueError(f"Duplicate validation label_id={label_id}")
        if label_id not in flow_counts:
            raise ValueError(f"Validation label_id={label_id} is missing from flow counts")
        seen_labels.add(label_id)
        rows.append(
            {
                "class": str(class_name),
                "label_id": label_id,
                "train_flow_count": flow_counts[label_id],
                "validation_f1": float(metrics["f1"]),
                "validation_recall": float(metrics["recall"]),
                "validation_support": int(metrics["support"]),
            }
        )
    if seen_labels != set(flow_counts):
        missing = sorted(set(flow_counts) - seen_labels)
        raise ValueError(f"Training labels are missing from validation metrics: {missing}")
    rows.sort(key=lambda row: (row["train_flow_count"], row["label_id"]))
    log_exposure = np.log1p([row["train_flow_count"] for row in rows])
    f1 = np.asarray([row["validation_f1"] for row in rows], dtype=np.float64)
    recall = np.asarray([row["validation_recall"] for row in rows], dtype=np.float64)
    return {
        "scope": "heldout_validation_exposure_association",
        "causal_claim": False,
        "exposure_unit": "unique_training_flows_per_class",
        "exposure_transform": "log1p",
        "num_classes": len(rows),
        "association": {
            "validation_f1": correlation(log_exposure, f1),
            "validation_recall": correlation(log_exposure, recall),
        },
        "classes": rows,
        "interpretation_guard": (
            "Association can motivate a validation-screened sampler-aware loss, "
            "but cannot establish that exposure imbalance causes class errors."
        ),
    }


def analyze_exposure_history(
    sampling_report: dict[str, Any], history: list[dict[str, Any]]
) -> dict[str, Any]:
    if not history:
        raise ValueError("Validation history is empty")
    ordered = sorted(history, key=lambda row: int(row["step"]))
    analyses = [
        analyze_exposure_outcomes(sampling_report, {"metrics": row["metrics"]})
        for row in ordered
    ]
    trajectory = []
    for row, analysis in zip(ordered, analyses):
        trajectory.append(
            {
                "step": int(row["step"]),
                "accuracy": float(row["metrics"]["accuracy"]),
                "macro_f1": float(row["metrics"]["macro_f1"]),
                "association": analysis["association"],
            }
        )
    f1_rho = [row["association"]["validation_f1"]["rho"] for row in trajectory]
    recall_rho = [
        row["association"]["validation_recall"]["rho"] for row in trajectory
    ]
    latest = dict(analyses[-1])
    latest["trajectory"] = trajectory
    latest["trajectory_summary"] = {
        "num_validation_points": len(trajectory),
        "f1_rho_min": min(f1_rho),
        "f1_rho_max": max(f1_rho),
        "f1_significant_points_p_lt_0_05": sum(
            row["association"]["validation_f1"]["p_value"] < 0.05
            for row in trajectory
        ),
        "recall_rho_min": min(recall_rho),
        "recall_rho_max": max(recall_rho),
        "recall_significant_points_p_lt_0_05": sum(
            row["association"]["validation_recall"]["p_value"] < 0.05
            for row in trajectory
        ),
        "selection_role": "descriptive_only",
    }
    return latest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sampling_report", required=True)
    parser.add_argument("--report_index", type=int, default=0)
    validation = parser.add_mutually_exclusive_group(required=True)
    validation.add_argument("--validation_metrics")
    validation.add_argument("--validation_history")
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    sampling_payload = json.loads(Path(args.sampling_report).read_text(encoding="utf-8"))
    reports = sampling_payload["reports"]
    if not 0 <= args.report_index < len(reports):
        raise IndexError("--report_index is outside the sampling report")
    if args.validation_history:
        with open(args.validation_history, "r", encoding="utf-8") as handle:
            history = [json.loads(line) for line in handle if line.strip()]
        result = analyze_exposure_history(reports[args.report_index], history)
    else:
        validation_payload = json.loads(
            Path(args.validation_metrics).read_text(encoding="utf-8")
        )
        result = analyze_exposure_outcomes(reports[args.report_index], validation_payload)
    result["sampling_report"] = args.sampling_report
    result["sampling_report_index"] = args.report_index
    result["validation_metrics"] = args.validation_metrics
    result["validation_history"] = args.validation_history

    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result["association"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
