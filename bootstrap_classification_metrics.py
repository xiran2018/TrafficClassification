#!/usr/bin/env python3
"""Cluster-aware confidence intervals for fixed packet/flow predictions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def classification_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(
            f1_score(labels, predictions, average="macro", zero_division=0)
        ),
    }


def load_fixed_predictions(path: str, task: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if task == "packet":
        with np.load(path) as payload:
            required = {"y_true", "probabilities", "flow_ids"}
            missing = sorted(required - set(payload.files))
            if missing:
                raise ValueError(
                    f"packet prediction archive is missing {missing}; "
                    "flow_ids are required for cluster-aware inference"
                )
            labels = np.asarray(payload["y_true"], dtype=np.int64)
            probabilities = np.asarray(payload["probabilities"], dtype=np.float64)
            group_ids = np.asarray(payload["flow_ids"])
    elif task == "flow":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        required = {"flow_y_true", "flow_prob", "flow_ids"}
        missing = sorted(required - set(data))
        if missing:
            raise ValueError(f"flow prediction JSON is missing {missing}")
        labels = np.asarray(data["flow_y_true"], dtype=np.int64)
        probabilities = np.asarray(data["flow_prob"], dtype=np.float64)
        group_ids = np.asarray([str(value) for value in data["flow_ids"]])
    else:
        raise ValueError(f"unknown task: {task}")
    if labels.ndim != 1 or probabilities.ndim != 2 or group_ids.ndim != 1:
        raise ValueError("labels/groups must be vectors and probabilities must be a matrix")
    if len(labels) == 0 or len(labels) != len(probabilities) or len(labels) != len(group_ids):
        raise ValueError("prediction arrays are empty or have inconsistent lengths")
    if not np.isfinite(probabilities).all():
        raise ValueError("probabilities contain non-finite values")
    return labels, probabilities.argmax(axis=1), group_ids


def class_stratified_groups(
    labels: np.ndarray, group_ids: np.ndarray
) -> dict[int, list[np.ndarray]]:
    grouped: dict[Any, list[int]] = {}
    for index, group_id in enumerate(group_ids.tolist()):
        grouped.setdefault(group_id, []).append(index)
    by_class: dict[int, list[np.ndarray]] = {}
    for group_id, indexes in grouped.items():
        index = np.asarray(indexes, dtype=np.int64)
        group_labels = np.unique(labels[index])
        if len(group_labels) != 1:
            raise ValueError(
                f"cluster {group_id!r} spans labels {group_labels.tolist()}; "
                "a flow must have exactly one class"
            )
        by_class.setdefault(int(group_labels[0]), []).append(index)
    return by_class


def cluster_bootstrap(
    labels: np.ndarray,
    predictions: np.ndarray,
    group_ids: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    if samples <= 0:
        raise ValueError("samples must be positive")
    groups = class_stratified_groups(labels, group_ids)
    if len(groups) < 2:
        raise ValueError("class-stratified bootstrap requires at least two classes")
    rng = np.random.default_rng(seed)
    draws = {"accuracy": [], "macro_f1": []}
    for _ in range(samples):
        sampled_indexes = []
        for class_groups in groups.values():
            selected = rng.integers(0, len(class_groups), size=len(class_groups))
            sampled_indexes.extend(class_groups[int(index)] for index in selected)
        index = np.concatenate(sampled_indexes)
        metrics = classification_metrics(labels[index], predictions[index])
        for name in draws:
            draws[name].append(metrics[name])
    point = classification_metrics(labels, predictions)
    intervals = {}
    for name, values in draws.items():
        array = np.asarray(values, dtype=np.float64)
        intervals[name] = {
            "point_estimate": point[name],
            "bootstrap_mean": float(array.mean()),
            "bootstrap_95_ci": [
                float(value) for value in np.percentile(array, [2.5, 97.5])
            ],
            "bootstrap_std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        }
    return {
        "method": "class_stratified_flow_cluster_bootstrap",
        "num_samples": int(len(labels)),
        "num_flow_clusters": int(sum(len(value) for value in groups.values())),
        "num_classes": int(len(groups)),
        "clusters_per_class": {
            str(label): len(value) for label, value in sorted(groups.items())
        },
        "bootstrap_samples": int(samples),
        "seed": int(seed),
        "metrics": intervals,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--task", choices=["packet", "flow"], required=True)
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()
    labels, predictions, group_ids = load_fixed_predictions(args.input, args.task)
    report = cluster_bootstrap(
        labels,
        predictions,
        group_ids,
        samples=args.samples,
        seed=args.seed,
    )
    report.update({"task": args.task, "input": args.input})
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
