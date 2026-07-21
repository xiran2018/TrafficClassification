#!/usr/bin/env python3
"""Report Packet/Flow accuracy under train-seen versus novel session fields."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from packet_eval_utils import packet_classification_metrics
from train_tower1_multitask import stable_flow_id


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_index(path: str | Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise ValueError(f"packet index is empty: {path}")
    return rows


def session_signatures(row: dict[str, Any]) -> dict[str, tuple[Any, ...]]:
    meta = row.get("meta") or {}
    protocol = str(meta.get("l4") or "unknown")
    src_ip = str(meta.get("src_ip") or "unknown")
    dst_ip = str(meta.get("dst_ip") or "unknown")
    sport = int(meta.get("sport", -1))
    dport = int(meta.get("dport", -1))
    endpoints = tuple(sorted((src_ip, dst_ip)))
    ports = tuple(sorted((sport, dport)))
    endpoint_ports = tuple(sorted(((src_ip, sport), (dst_ip, dport))))
    return {
        "endpoint": (protocol, *endpoints),
        "port": (protocol, *ports),
        "session": (protocol, *endpoint_ports[0], *endpoint_ports[1]),
    }


def training_signature_sets(rows: list[dict[str, Any]]) -> dict[str, set[tuple[Any, ...]]]:
    sets = {name: set() for name in ("endpoint", "port", "session")}
    for row in rows:
        for name, value in session_signatures(row).items():
            sets[name].add(value)
    return sets


def novelty_tags(
    row: dict[str, Any], train_sets: dict[str, set[tuple[Any, ...]]]
) -> dict[str, bool]:
    signatures = session_signatures(row)
    return {
        f"{name}_seen": signatures[name] in train_sets[name]
        for name in ("endpoint", "port", "session")
    }


def load_predictions(path: str | Path, task: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = Path(path)
    if task == "packet":
        with np.load(path, allow_pickle=False) as payload:
            y_true = np.asarray(payload["y_true"], dtype=np.int64)
            probabilities = np.asarray(payload["probabilities"], dtype=np.float64)
            flow_ids = np.asarray(payload["flow_ids"], dtype=np.int64)
    elif task == "flow":
        payload = json.loads(path.read_text(encoding="utf-8"))
        y_true = np.asarray(payload["flow_y_true"], dtype=np.int64)
        probabilities = np.asarray(payload["flow_prob"], dtype=np.float64)
        flow_ids = np.asarray(payload["flow_ids"], dtype=str)
    else:
        raise ValueError(f"unknown task: {task}")
    if probabilities.ndim != 2 or len(y_true) != len(probabilities):
        raise ValueError("prediction labels/probabilities are not row aligned")
    if len(flow_ids) != len(y_true):
        raise ValueError("prediction flow_ids are not row aligned")
    if not np.isfinite(probabilities).all():
        raise ValueError("prediction probabilities contain non-finite values")
    if (probabilities < -1e-8).any() or not np.allclose(
        probabilities.sum(axis=1), 1.0, atol=1e-5, rtol=0.0
    ):
        raise ValueError("prediction probabilities must be non-negative and normalized")
    if task == "flow" and len(set(flow_ids.astype(str).tolist())) != len(flow_ids):
        raise ValueError("flow predictions contain duplicate flow_ids")
    return y_true, probabilities, flow_ids


def evaluation_rows(
    test_rows: list[dict[str, Any]],
    task: str,
    prediction_flow_ids: np.ndarray,
) -> list[dict[str, Any]]:
    if task == "packet":
        if len(test_rows) != len(prediction_flow_ids):
            raise ValueError("packet predictions do not match packet-index row count")
        expected = np.asarray(
            [stable_flow_id(str(row["flow_id"])) for row in test_rows], dtype=np.int64
        )
        if not np.array_equal(expected, prediction_flow_ids.astype(np.int64)):
            raise ValueError("packet prediction flow_ids do not align with packet index")
        return test_rows

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in test_rows:
        grouped[str(row["flow_id"])].append(row)
    aligned = []
    for flow_id in prediction_flow_ids.astype(str):
        rows = grouped.get(str(flow_id))
        if not rows:
            raise ValueError(f"flow prediction is missing from packet index: {flow_id}")
        signatures = {session_signatures(row)["session"] for row in rows}
        if len(signatures) != 1:
            raise ValueError(f"flow_id={flow_id} contains multiple session tuples")
        aligned.append(rows[0])
    return aligned


def load_label_names(path: str | Path, num_classes: int) -> list[str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    names = [""] * num_classes
    for label, index in payload.items():
        if 0 <= int(index) < num_classes:
            names[int(index)] = str(label)
    return [name or str(index) for index, name in enumerate(names)]


def subset_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray,
    num_classes: int,
    label_names: list[str],
) -> dict[str, Any]:
    count = int(mask.sum())
    if count == 0:
        return {"num_samples": 0, "metrics": None, "present_class_macro_f1": None}
    metrics = packet_classification_metrics(
        y_true[mask], y_pred[mask], num_classes, label_names
    )
    present = [
        values["f1"]
        for values in metrics["per_class"].values()
        if values["support"] > 0
    ]
    return {
        "num_samples": count,
        "metrics": metrics,
        "present_class_macro_f1": float(np.mean(present)) if present else None,
    }


def evaluate_session_novelty(
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    y_true: np.ndarray,
    probabilities: np.ndarray,
    prediction_flow_ids: np.ndarray,
    task: str,
    label_names: list[str],
) -> dict[str, Any]:
    aligned_rows = evaluation_rows(test_rows, task, prediction_flow_ids)
    if len(aligned_rows) != len(y_true):
        raise ValueError("aligned evaluation rows do not match predictions")
    index_labels = np.asarray([int(row["label_id"]) for row in aligned_rows])
    if not np.array_equal(index_labels, y_true):
        raise ValueError("prediction labels do not align with packet-index labels")
    train_sets = training_signature_sets(train_rows)
    tags = [novelty_tags(row, train_sets) for row in aligned_rows]
    endpoint_seen = np.asarray([tag["endpoint_seen"] for tag in tags], dtype=bool)
    port_seen = np.asarray([tag["port_seen"] for tag in tags], dtype=bool)
    session_seen = np.asarray([tag["session_seen"] for tag in tags], dtype=bool)
    y_pred = probabilities.argmax(axis=1)
    num_classes = probabilities.shape[1]
    groups = {
        "all": np.ones(len(y_true), dtype=bool),
        "endpoint_seen": endpoint_seen,
        "endpoint_novel": ~endpoint_seen,
        "port_seen": port_seen,
        "port_novel": ~port_seen,
        "session_seen": session_seen,
        "session_novel": ~session_seen,
        "endpoint_seen_port_seen": endpoint_seen & port_seen,
        "endpoint_seen_port_novel": endpoint_seen & ~port_seen,
        "endpoint_novel_port_seen": ~endpoint_seen & port_seen,
        "endpoint_novel_port_novel": ~endpoint_seen & ~port_seen,
    }
    evaluated_groups = {
        name: subset_metrics(y_true, y_pred, mask, num_classes, label_names)
        for name, mask in groups.items()
    }

    def conditional_gap(prefix: str) -> dict[str, float | None]:
        seen = evaluated_groups[f"{prefix}_seen"]
        novel = evaluated_groups[f"{prefix}_novel"]
        if seen["metrics"] is None or novel["metrics"] is None:
            return {
                "accuracy_seen_minus_novel": None,
                "macro_f1_seen_minus_novel": None,
                "present_class_macro_f1_seen_minus_novel": None,
            }
        return {
            "accuracy_seen_minus_novel": float(
                seen["metrics"]["accuracy"] - novel["metrics"]["accuracy"]
            ),
            "macro_f1_seen_minus_novel": float(
                seen["metrics"]["macro_f1"] - novel["metrics"]["macro_f1"]
            ),
            "present_class_macro_f1_seen_minus_novel": float(
                seen["present_class_macro_f1"]
                - novel["present_class_macro_f1"]
            ),
        }

    return {
        "schema": "session_novelty_evaluation_v1",
        "task": task,
        "selection_role": "reporting_only_no_training_or_model_selection",
        "group_definition_uses_test_labels": False,
        "training_reference": "union_of_all_supplied_training_signature_sets",
        "num_classes": num_classes,
        "training_signature_counts": {
            name: len(values) for name, values in train_sets.items()
        },
        "conditional_gaps_are_not_causal_effects": True,
        "seen_minus_novel_gaps": {
            prefix: conditional_gap(prefix)
            for prefix in ("endpoint", "port", "session")
        },
        "groups": evaluated_groups,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["packet", "flow"], required=True)
    parser.add_argument(
        "--train_packet_index",
        action="append",
        required=True,
        help="Training packet index; repeat for fixed cross-fold consensus.",
    )
    parser.add_argument("--test_packet_index", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--label_map", required=True)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    train_rows = [
        row
        for path in args.train_packet_index
        for row in load_index(path)
    ]
    test_rows = load_index(args.test_packet_index)
    y_true, probabilities, flow_ids = load_predictions(args.predictions, args.task)
    label_names = load_label_names(args.label_map, probabilities.shape[1])
    result = evaluate_session_novelty(
        train_rows,
        test_rows,
        y_true,
        probabilities,
        flow_ids,
        args.task,
        label_names,
    )
    result["inputs"] = {
        "train_packet_indices": [
            {"path": path, "sha256": sha256_file(path)}
            for path in args.train_packet_index
        ],
        "test_packet_index": {
            "path": args.test_packet_index,
            "sha256": sha256_file(args.test_packet_index),
        },
        "predictions": {
            "path": args.predictions,
            "sha256": sha256_file(args.predictions),
        },
        "label_map": {
            "path": args.label_map,
            "sha256": sha256_file(args.label_map),
        },
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                name: {
                    "num_samples": row["num_samples"],
                    "accuracy": None if row["metrics"] is None else row["metrics"]["accuracy"],
                    "macro_f1": None if row["metrics"] is None else row["metrics"]["macro_f1"],
                }
                for name, row in result["groups"].items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
