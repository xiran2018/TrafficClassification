#!/usr/bin/env python3
"""Summarize Tower-1 validation dynamics with one dataset-agnostic protocol."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


def load_history(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"Validation history is empty: {path}")
    return rows


def class_f1(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        str(label): float(values["f1"])
        for label, values in (metrics.get("per_class") or {}).items()
    }


def summarize_history(
    rows: list[dict[str, Any]],
    weak_f1_threshold: float = 0.25,
    recovery_delta: float = 0.20,
) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: int(row["step"]))
    best = max(
        ordered,
        key=lambda row: (
            float(row["metrics"]["macro_f1"]),
            float(row["metrics"]["accuracy"]),
            -int(row["step"]),
        ),
    )
    first = ordered[0]
    final = ordered[-1]
    first_f1 = class_f1(first["metrics"])
    best_f1 = class_f1(best["metrics"])
    all_labels = sorted({label for row in ordered for label in class_f1(row["metrics"])})
    max_f1 = {
        label: max(class_f1(row["metrics"]).get(label, 0.0) for row in ordered)
        for label in all_labels
    }
    return {
        "num_evaluations": len(ordered),
        "first": {
            "step": int(first["step"]),
            "accuracy": float(first["metrics"]["accuracy"]),
            "macro_f1": float(first["metrics"]["macro_f1"]),
        },
        "best": {
            "step": int(best["step"]),
            "accuracy": float(best["metrics"]["accuracy"]),
            "macro_f1": float(best["metrics"]["macro_f1"]),
        },
        "final": {
            "step": int(final["step"]),
            "accuracy": float(final["metrics"]["accuracy"]),
            "macro_f1": float(final["metrics"]["macro_f1"]),
        },
        "macro_f1_gain_from_first": float(best["metrics"]["macro_f1"])
        - float(first["metrics"]["macro_f1"]),
        "validation_regression_after_best": float(best["metrics"]["macro_f1"])
        - float(final["metrics"]["macro_f1"]),
        "best_per_class_f1": best_f1,
        "persistently_weak_classes": [
            label for label in all_labels if max_f1[label] < weak_f1_threshold
        ],
        "recovered_classes": [
            label
            for label in all_labels
            if best_f1.get(label, 0.0) - first_f1.get(label, 0.0) >= recovery_delta
        ],
        "thresholds": {
            "weak_f1": weak_f1_threshold,
            "recovery_delta": recovery_delta,
        },
    }


def aggregate_runs(runs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    best_accuracy = [run["best"]["accuracy"] for run in runs.values()]
    best_macro_f1 = [run["best"]["macro_f1"] for run in runs.values()]
    return {
        "num_runs": len(runs),
        "best_accuracy_mean": statistics.fmean(best_accuracy),
        "best_accuracy_population_std": statistics.pstdev(best_accuracy),
        "best_macro_f1_mean": statistics.fmean(best_macro_f1),
        "best_macro_f1_population_std": statistics.pstdev(best_macro_f1),
    }


def _average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + end - 1) / 2.0 + 1.0
        for index in order[start:end]:
            ranks[index] = rank
        start = end
    return ranks


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = sum(
        (x - left_mean) * (y - right_mean) for x, y in zip(left, right)
    )
    left_scale = math.sqrt(sum((x - left_mean) ** 2 for x in left))
    right_scale = math.sqrt(sum((y - right_mean) ** 2 for y in right))
    if left_scale == 0.0 or right_scale == 0.0:
        return None
    return numerator / (left_scale * right_scale)


def compare_matched_histories(
    baseline_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compare validation runs only at checkpoints reached by both runs."""
    baseline = {int(row["step"]): row for row in baseline_rows}
    candidate = {int(row["step"]): row for row in candidate_rows}
    common_steps = sorted(set(baseline) & set(candidate))
    if not common_steps:
        raise ValueError("Validation histories do not contain a common step")

    comparisons = []
    for step in common_steps:
        base_metrics = baseline[step]["metrics"]
        candidate_metrics = candidate[step]["metrics"]
        base_classes = base_metrics.get("per_class") or {}
        candidate_classes = candidate_metrics.get("per_class") or {}
        if set(base_classes) != set(candidate_classes):
            raise ValueError(f"Per-class labels differ at matched step {step}")

        class_rows = []
        for label in sorted(base_classes):
            base_class = base_classes[label]
            candidate_class = candidate_classes[label]
            base_support = int(base_class.get("support", 0))
            candidate_support = int(candidate_class.get("support", 0))
            if base_support != candidate_support:
                raise ValueError(
                    f"Validation support differs for class {label!r} at step {step}"
                )
            class_rows.append(
                {
                    "label": str(label),
                    "support": base_support,
                    "baseline_f1": float(base_class["f1"]),
                    "candidate_f1": float(candidate_class["f1"]),
                    "f1_delta": float(candidate_class["f1"])
                    - float(base_class["f1"]),
                    "recall_delta": float(candidate_class.get("recall", 0.0))
                    - float(base_class.get("recall", 0.0)),
                }
            )
        gains = [row["f1_delta"] for row in class_rows]
        supports = [float(row["support"]) for row in class_rows]
        ranked = sorted(class_rows, key=lambda row: row["f1_delta"], reverse=True)
        positive_ranked = [row for row in ranked if row["f1_delta"] > 0.0]
        negative_ranked = sorted(
            (row for row in class_rows if row["f1_delta"] < 0.0),
            key=lambda row: row["f1_delta"],
        )
        comparisons.append(
            {
                "step": step,
                "baseline_accuracy": float(base_metrics["accuracy"]),
                "candidate_accuracy": float(candidate_metrics["accuracy"]),
                "accuracy_delta": float(candidate_metrics["accuracy"])
                - float(base_metrics["accuracy"]),
                "baseline_macro_f1": float(base_metrics["macro_f1"]),
                "candidate_macro_f1": float(candidate_metrics["macro_f1"]),
                "macro_f1_delta": float(candidate_metrics["macro_f1"])
                - float(base_metrics["macro_f1"]),
                "positive_f1_classes": sum(delta > 0.0 for delta in gains),
                "negative_f1_classes": sum(delta < 0.0 for delta in gains),
                "support_gain_spearman": _pearson(
                    _average_ranks(supports), _average_ranks(gains)
                ),
                "top_f1_gains": positive_ranked[:5],
                "top_f1_losses": negative_ranked[:5],
                "per_class": class_rows,
            }
        )
    accuracy_deltas = [row["accuracy_delta"] for row in comparisons]
    macro_f1_deltas = [row["macro_f1_delta"] for row in comparisons]

    split_index = max(1, len(comparisons) // 2)
    early_comparisons = comparisons[:split_index]
    late_comparisons = comparisons[split_index:] or comparisons[-1:]

    def win_counts(values: list[float]) -> dict[str, int]:
        return {
            "candidate_wins": sum(value > 0.0 for value in values),
            "ties": sum(value == 0.0 for value in values),
            "candidate_losses": sum(value < 0.0 for value in values),
        }

    def phase_summary(metric: str) -> dict[str, Any]:
        key = f"{metric}_delta"
        early = [float(row[key]) for row in early_comparisons]
        late = [float(row[key]) for row in late_comparisons]
        return {
            "early_steps": [int(row["step"]) for row in early_comparisons],
            "late_steps": [int(row["step"]) for row in late_comparisons],
            "early_mean_delta": statistics.fmean(early),
            "late_mean_delta": statistics.fmean(late),
            "late_minus_early_mean_delta": statistics.fmean(late)
            - statistics.fmean(early),
            "first_to_latest_delta_change": float(comparisons[-1][key])
            - float(comparisons[0][key]),
            "late_phase": win_counts(late),
        }

    return {
        "scope": "held_out_validation_only",
        "matched_steps": common_steps,
        "matched_curve_summary": {
            "num_matched_points": len(comparisons),
            "accuracy": {
                **win_counts(accuracy_deltas),
                "mean_delta": statistics.fmean(accuracy_deltas),
                "median_delta": statistics.median(accuracy_deltas),
                "phase_dynamics": phase_summary("accuracy"),
            },
            "macro_f1": {
                **win_counts(macro_f1_deltas),
                "mean_delta": statistics.fmean(macro_f1_deltas),
                "median_delta": statistics.median(macro_f1_deltas),
                "phase_dynamics": phase_summary("macro_f1"),
                "best_matched_step": max(
                    comparisons, key=lambda row: row["macro_f1_delta"]
                )["step"],
                "worst_matched_step": min(
                    comparisons, key=lambda row: row["macro_f1_delta"]
                )["step"],
            },
            "selection_role": "descriptive_only",
        },
        "latest_matched": comparisons[-1],
        "comparisons": comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs=2, action="append", metavar=("NAME", "HISTORY"), required=True)
    parser.add_argument("--weak_f1_threshold", type=float, default=0.25)
    parser.add_argument("--recovery_delta", type=float, default=0.20)
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("BASELINE_NAME", "CANDIDATE_NAME"),
        help="Add matched-step, validation-only class dynamics for two named inputs.",
    )
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    runs = {
        name: summarize_history(
            load_history(path),
            weak_f1_threshold=args.weak_f1_threshold,
            recovery_delta=args.recovery_delta,
        )
        for name, path in args.input
    }
    result = {
        "selection_metric": "held_out_packet_macro_f1",
        "runs": runs,
        "aggregate": aggregate_runs(runs),
    }
    if args.compare:
        baseline_name, candidate_name = args.compare
        input_paths = dict(args.input)
        missing = [name for name in args.compare if name not in input_paths]
        if missing:
            parser.error(f"--compare names are missing from --input: {missing}")
        result["matched_comparison"] = compare_matched_histories(
            load_history(input_paths[baseline_name]),
            load_history(input_paths[candidate_name]),
        )
        result["matched_comparison"]["baseline_name"] = baseline_name
        result["matched_comparison"]["candidate_name"] = candidate_name
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result["aggregate"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
