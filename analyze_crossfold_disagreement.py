#!/usr/bin/env python3
"""Report cross-fold prediction complementarity without selecting a model."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected NAME=JSON")
    name, raw_path = value.split("=", 1)
    if not name or not raw_path:
        raise argparse.ArgumentTypeError("expected non-empty NAME=JSON")
    return name, Path(raw_path)


def load_prediction(name: str, path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    required = ("flow_ids", "flow_y_true", "flow_y_pred")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"{path}: missing fields {missing}")
    flow_ids = [str(value) for value in payload["flow_ids"]]
    y_true = np.asarray(payload["flow_y_true"], dtype=np.int64)
    y_pred = np.asarray(payload["flow_y_pred"], dtype=np.int64)
    if not (len(flow_ids) == len(y_true) == len(y_pred)):
        raise ValueError(f"{path}: flow_ids/y_true/y_pred lengths differ")
    if len(set(flow_ids)) != len(flow_ids):
        raise ValueError(f"{path}: duplicate flow_ids are not allowed")
    return {
        "name": name,
        "path": str(path),
        "flow_ids": flow_ids,
        "y_true": y_true,
        "y_pred": y_pred,
        "label_map": payload.get("label_map"),
    }


def align_predictions(rows: list[dict[str, Any]]) -> tuple[list[str], np.ndarray, np.ndarray]:
    reference = rows[0]
    flow_ids = reference["flow_ids"]
    reference_set = set(flow_ids)
    y_true = reference["y_true"]
    predictions = []
    for row in rows:
        current_set = set(row["flow_ids"])
        if current_set != reference_set:
            missing = sorted(reference_set - current_set)[:5]
            extra = sorted(current_set - reference_set)[:5]
            raise ValueError(
                f"{row['path']}: flow ID set differs; missing={missing}, extra={extra}"
            )
        positions = {flow_id: index for index, flow_id in enumerate(row["flow_ids"])}
        order = np.asarray([positions[flow_id] for flow_id in flow_ids], dtype=np.int64)
        aligned_true = row["y_true"][order]
        if not np.array_equal(aligned_true, y_true):
            mismatch = int(np.flatnonzero(aligned_true != y_true)[0])
            raise ValueError(
                f"{row['path']}: true label mismatch for flow_id={flow_ids[mismatch]}"
            )
        predictions.append(row["y_pred"][order])
    return flow_ids, y_true, np.stack(predictions, axis=0)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "num_samples": int(len(y_true)),
    }


def label_names(rows: list[dict[str, Any]], labels: np.ndarray) -> dict[int, str]:
    mappings = [row["label_map"] for row in rows if isinstance(row.get("label_map"), dict)]
    if mappings and any(mapping != mappings[0] for mapping in mappings[1:]):
        raise ValueError("input label maps differ")
    if not mappings:
        return {int(label): str(int(label)) for label in labels}
    inverse = {int(index): str(name) for name, index in mappings[0].items()}
    return {int(label): inverse.get(int(label), str(int(label))) for label in labels}


def build_report(
    inputs: list[tuple[str, Path]], consensus: tuple[str, Path] | None = None
) -> dict[str, Any]:
    if len(inputs) < 2:
        raise ValueError("at least two prediction inputs are required")
    names = [name for name, _ in inputs]
    if len(set(names)) != len(names):
        raise ValueError("prediction input names must be unique")

    rows = [load_prediction(name, path) for name, path in inputs]
    flow_ids, y_true, y_pred = align_predictions(rows)
    correct = y_pred == y_true[None, :]
    correct_count = correct.sum(axis=0)
    any_correct = correct_count > 0
    unanimous_prediction = np.all(y_pred == y_pred[0:1, :], axis=0)
    unanimous_wrong = unanimous_prediction & ~correct[0]
    labels = np.unique(y_true)
    names_by_label = label_names(rows, labels)

    fold_metrics = {
        row["name"]: metrics(y_true, y_pred[index]) for index, row in enumerate(rows)
    }
    accuracies = [float(value["accuracy"]) for value in fold_metrics.values()]
    macro_f1s = [float(value["macro_f1"]) for value in fold_metrics.values()]

    pairwise = []
    for left, right in combinations(range(len(rows)), 2):
        left_correct = correct[left]
        right_correct = correct[right]
        union_error = (~left_correct) | (~right_correct)
        shared_error = (~left_correct) & (~right_correct)
        pairwise.append(
            {
                "left": rows[left]["name"],
                "right": rows[right]["name"],
                "prediction_agreement_rate": float(np.mean(y_pred[left] == y_pred[right])),
                "both_correct_rate": float(np.mean(left_correct & right_correct)),
                "left_only_correct_rate": float(np.mean(left_correct & ~right_correct)),
                "right_only_correct_rate": float(np.mean(~left_correct & right_correct)),
                "both_wrong_rate": float(np.mean(shared_error)),
                "both_wrong_same_prediction_rate": float(
                    np.mean(shared_error & (y_pred[left] == y_pred[right]))
                ),
                "error_jaccard": float(shared_error.sum() / max(int(union_error.sum()), 1)),
            }
        )

    per_class = []
    for label in labels:
        mask = y_true == label
        class_size = int(mask.sum())
        class_fold_accuracy = {
            rows[index]["name"]: float(correct[index, mask].mean())
            for index in range(len(rows))
        }
        best_fold_accuracy = max(class_fold_accuracy.values())
        oracle_accuracy = float(any_correct[mask].mean())
        per_class.append(
            {
                "label": int(label),
                "label_name": names_by_label[int(label)],
                "num_samples": class_size,
                "fold_accuracy": class_fold_accuracy,
                "best_fold_accuracy": best_fold_accuracy,
                "oracle_any_fold_accuracy": oracle_accuracy,
                "oracle_headroom_over_best_fold": oracle_accuracy - best_fold_accuracy,
                "prediction_disagreement_rate": float(
                    np.mean(~np.all(y_pred[:, mask] == y_pred[0:1, mask], axis=0))
                ),
                "unanimous_wrong_rate": float(unanimous_wrong[mask].mean()),
            }
        )
    per_class.sort(
        key=lambda row: (row["oracle_headroom_over_best_fold"], row["num_samples"]),
        reverse=True,
    )

    report: dict[str, Any] = {
        "analysis_contract": {
            "selection_role": "none",
            "label_usage": "test_labels_diagnostic_only",
            "permitted_use": "post_hoc_error_analysis_and_hypothesis_generation",
            "prohibited_use": "model_selection_hyperparameter_selection_or_test_set_adaptation",
            "oracle_is_deployable": False,
        },
        "inputs": [{"name": row["name"], "path": row["path"]} for row in rows],
        "alignment": {
            "status": "exact_flow_id_and_true_label_match",
            "num_flows": len(flow_ids),
            "num_folds": len(rows),
            "num_classes": len(labels),
        },
        "fold_metrics": fold_metrics,
        "summary": {
            "mean_fold_accuracy": float(np.mean(accuracies)),
            "best_fold_accuracy": max(accuracies),
            "mean_fold_macro_f1": float(np.mean(macro_f1s)),
            "best_fold_macro_f1": max(macro_f1s),
            "oracle_any_fold_accuracy": float(any_correct.mean()),
            "oracle_accuracy_headroom_over_best_fold": float(any_correct.mean()) - max(accuracies),
            "prediction_disagreement_rate": float(np.mean(~unanimous_prediction)),
            "unanimous_wrong_rate": float(unanimous_wrong.mean()),
            "all_folds_wrong_rate": float(np.mean(correct_count == 0)),
        },
        "correct_fold_count": {
            str(count): {
                "count": int(np.sum(correct_count == count)),
                "rate": float(np.mean(correct_count == count)),
            }
            for count in range(len(rows) + 1)
        },
        "pairwise": pairwise,
        "per_class": per_class,
    }

    if consensus is not None:
        consensus_row = load_prediction(*consensus)
        _, consensus_true, consensus_pred_stack = align_predictions([rows[0], consensus_row])
        if not np.array_equal(consensus_true, y_true):
            raise ValueError("consensus labels differ from fold labels")
        consensus_pred = consensus_pred_stack[1]
        consensus_correct = consensus_pred == y_true
        comparisons = {}
        for index, row in enumerate(rows):
            fold_correct = correct[index]
            comparisons[row["name"]] = {
                "corrected_fold_errors_rate": float(np.mean(~fold_correct & consensus_correct)),
                "harmed_fold_correct_rate": float(np.mean(fold_correct & ~consensus_correct)),
                "net_accuracy_delta": float(consensus_correct.mean() - fold_correct.mean()),
            }
        report["consensus"] = {
            "name": consensus_row["name"],
            "path": consensus_row["path"],
            "metrics": metrics(y_true, consensus_pred),
            "captures_oracle_correct_rate": float(
                consensus_correct[any_correct].mean() if np.any(any_correct) else 0.0
            ),
            "correct_when_all_folds_wrong_rate": float(
                consensus_correct[~any_correct].mean() if np.any(~any_correct) else 0.0
            ),
            "vs_folds": comparisons,
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, type=parse_named_path)
    parser.add_argument("--consensus", type=parse_named_path)
    parser.add_argument("--output_json", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(args.input, args.consensus)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps({"output_json": str(args.output_json), **report["summary"]}, indent=2))


if __name__ == "__main__":
    main()
