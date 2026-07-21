#!/usr/bin/env python3
"""Diagnose aligned Packet cross-fold complementarity without selecting a model."""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    name, raw_path = value.split("=", 1)
    if not name or not raw_path:
        raise argparse.ArgumentTypeError("expected non-empty NAME=PATH")
    return name, Path(raw_path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_label_map(path: Path) -> tuple[dict[str, int], dict[int, str]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"{path}: label map must be a non-empty object")
    label_map = {str(name): int(index) for name, index in payload.items()}
    if len(set(label_map.values())) != len(label_map):
        raise ValueError(f"{path}: label IDs must be unique")
    return label_map, {index: name for name, index in label_map.items()}


def load_index_rows(
    path: Path, label_map: dict[str, int]
) -> tuple[np.ndarray, np.ndarray]:
    labels: list[int] = []
    packet_uids: list[str] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "label_id" not in row:
                raise ValueError(f"{path}:{line_number}: missing label_id")
            label_id = int(row["label_id"])
            packet_uid = str(row.get("packet_uid") or "")
            if not packet_uid:
                raise ValueError(f"{path}:{line_number}: missing packet_uid")
            if packet_uid in seen:
                raise ValueError(f"{path}:{line_number}: duplicate packet_uid")
            seen.add(packet_uid)
            label = row.get("label")
            if label is not None and label_map.get(str(label)) != label_id:
                raise ValueError(
                    f"{path}:{line_number}: label/label_id disagrees with label map"
                )
            labels.append(label_id)
            packet_uids.append(packet_uid)
    if not labels:
        raise ValueError(f"{path}: packet index is empty")
    return np.asarray(labels, dtype=np.int64), np.asarray(packet_uids, dtype=np.str_)


def load_source_test_index(path: Path) -> Path:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    config = payload.get("config")
    declared = config.get("test_index") if isinstance(config, dict) else None
    declared = declared or payload.get("packet_index")
    if not declared:
        raise ValueError(
            f"{path}: source JSON must contain config.test_index or packet_index"
        )
    return Path(str(declared)).expanduser().resolve()


def load_prediction(
    name: str,
    path: Path,
    expected_y_true: np.ndarray,
    expected_packet_uids: np.ndarray,
) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as payload:
        missing = [key for key in ("y_true", "probabilities") if key not in payload]
        if missing:
            raise ValueError(f"{path}: missing arrays {missing}")
        y_true = np.asarray(payload["y_true"], dtype=np.int64)
        probabilities = np.asarray(payload["probabilities"], dtype=np.float32)
        packet_uids = (
            np.asarray(payload["packet_uids"], dtype=np.str_)
            if "packet_uids" in payload
            else None
        )
    if y_true.ndim != 1 or probabilities.ndim != 2:
        raise ValueError(f"{path}: expected y_true [N] and probabilities [N,C]")
    if len(y_true) != len(probabilities):
        raise ValueError(f"{path}: y_true/probabilities lengths differ")
    if not np.array_equal(y_true, expected_y_true):
        mismatch = np.flatnonzero(y_true != expected_y_true)
        detail = int(mismatch[0]) if len(mismatch) else "shape"
        raise ValueError(f"{path}: labels do not match packet index at row {detail}")
    if not np.isfinite(probabilities).all():
        raise ValueError(f"{path}: probabilities contain non-finite values")
    if probabilities.shape[1] <= int(y_true.max(initial=-1)):
        raise ValueError(f"{path}: probability class dimension is too small")
    if packet_uids is not None and not np.array_equal(packet_uids, expected_packet_uids):
        raise ValueError(f"{path}: packet_uids do not match packet index row order")
    y_pred = probabilities.argmax(axis=1).astype(np.int64)
    return {
        "name": name,
        "path": str(path),
        "sha256": sha256_file(path),
        "y_true": y_true,
        "y_pred": y_pred,
        "packet_uids_present": packet_uids is not None,
    }


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "num_samples": int(len(y_true)),
    }


def build_report(
    inputs: list[tuple[str, Path]],
    sources: list[tuple[str, Path]],
    sample_index: Path,
    label_map_path: Path,
    consensus: tuple[str, Path] | None = None,
) -> dict[str, Any]:
    if len(inputs) < 2:
        raise ValueError("at least two prediction inputs are required")
    input_names = [name for name, _ in inputs]
    if len(set(input_names)) != len(input_names):
        raise ValueError("prediction input names must be unique")
    source_by_name = dict(sources)
    if set(source_by_name) != set(input_names) or len(source_by_name) != len(sources):
        raise ValueError("--source names must match --input names exactly")

    sample_index = sample_index.expanduser().resolve()
    label_map_path = label_map_path.expanduser().resolve()
    label_map, names_by_label = load_label_map(label_map_path)
    index_y_true, index_packet_uids = load_index_rows(sample_index, label_map)

    source_evidence = []
    for name in input_names:
        source_path = source_by_name[name].expanduser().resolve()
        declared_index = load_source_test_index(source_path)
        if declared_index != sample_index:
            raise ValueError(
                f"{source_path}: test_index {declared_index} does not match {sample_index}"
            )
        source_evidence.append(
            {
                "name": name,
                "path": str(source_path),
                "sha256": sha256_file(source_path),
                "declared_test_index": str(declared_index),
            }
        )

    rows = [
        load_prediction(
            name,
            path.expanduser().resolve(),
            index_y_true,
            index_packet_uids,
        )
        for name, path in inputs
    ]
    uid_availability = {row["packet_uids_present"] for row in rows}
    if len(uid_availability) != 1:
        raise ValueError("prediction inputs differ in packet_uids availability")
    y_true = index_y_true
    y_pred = np.stack([row["y_pred"] for row in rows], axis=0)
    correct = y_pred == y_true[None, :]
    correct_count = correct.sum(axis=0)
    any_correct = correct_count > 0
    unanimous_prediction = np.all(y_pred == y_pred[0:1, :], axis=0)
    unanimous_wrong = unanimous_prediction & ~correct[0]
    labels = np.unique(y_true)

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
        class_fold_accuracy = {
            rows[index]["name"]: float(correct[index, mask].mean())
            for index in range(len(rows))
        }
        best_fold_accuracy = max(class_fold_accuracy.values())
        oracle_accuracy = float(any_correct[mask].mean())
        per_class.append(
            {
                "label": int(label),
                "label_name": names_by_label.get(int(label), str(int(label))),
                "num_samples": int(mask.sum()),
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
            "historical_inputs_are_strict_v2_evidence": False,
        },
        "inputs": [
            {"name": row["name"], "path": row["path"], "sha256": row["sha256"]}
            for row in rows
        ],
        "source_evidence": source_evidence,
        "alignment": {
            "status": "shared_source_index_path_hash_row_count_and_true_labels_verified",
            "sample_index": str(sample_index),
            "sample_index_sha256": sha256_file(sample_index),
            "label_map": str(label_map_path),
            "label_map_sha256": sha256_file(label_map_path),
            "num_packets": int(len(y_true)),
            "num_folds": len(rows),
            "num_classes": len(labels),
            "npz_packet_uids": (
                "exact_packet_index_row_match" if True in uid_availability else "unavailable"
            ),
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
        consensus_row = load_prediction(
            consensus[0],
            consensus[1].expanduser().resolve(),
            index_y_true,
            index_packet_uids,
        )
        if consensus_row["packet_uids_present"] != rows[0]["packet_uids_present"]:
            raise ValueError("consensus differs in packet_uids availability")
        consensus_pred = consensus_row["y_pred"]
        consensus_correct = consensus_pred == y_true
        report["consensus"] = {
            "name": consensus_row["name"],
            "path": consensus_row["path"],
            "sha256": consensus_row["sha256"],
            "metrics": metrics(y_true, consensus_pred),
            "captures_oracle_correct_rate": float(
                consensus_correct[any_correct].mean() if np.any(any_correct) else 0.0
            ),
            "correct_when_all_folds_wrong_rate": float(
                consensus_correct[~any_correct].mean() if np.any(~any_correct) else 0.0
            ),
            "vs_folds": {
                row["name"]: {
                    "corrected_fold_errors_rate": float(
                        np.mean(~correct[index] & consensus_correct)
                    ),
                    "harmed_fold_correct_rate": float(
                        np.mean(correct[index] & ~consensus_correct)
                    ),
                    "net_accuracy_delta": float(
                        consensus_correct.mean() - correct[index].mean()
                    ),
                }
                for index, row in enumerate(rows)
            },
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, type=parse_named_path)
    parser.add_argument("--source", action="append", required=True, type=parse_named_path)
    parser.add_argument("--sample_index", type=Path, required=True)
    parser.add_argument("--label_map", type=Path, required=True)
    parser.add_argument("--consensus", type=parse_named_path)
    parser.add_argument("--output_json", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(
        args.input,
        args.source,
        args.sample_index,
        args.label_map,
        args.consensus,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps({"output_json": str(args.output_json), **report["summary"]}, indent=2))


if __name__ == "__main__":
    main()
