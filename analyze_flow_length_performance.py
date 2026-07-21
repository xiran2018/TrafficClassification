#!/usr/bin/env python3
"""Report flow-classification performance stratified by packet count.

This is a reporting-only diagnostic. It never selects a checkpoint, changes a
prediction, or consumes training data.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


FIXED_BINS = (
    (1, 8),
    (9, 16),
    (17, 32),
    (33, 64),
    (65, None),
)


def load_packet_counts(path: str | Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            flow_id = str(row["flow_id"])
            if flow_id in counts:
                raise ValueError(f"duplicate flow_id at line {line_no}: {flow_id}")
            metas = row.get("packet_metas")
            if not isinstance(metas, list) or not metas:
                raise ValueError(f"flow {flow_id} has no packet_metas")
            counts[flow_id] = len(metas)
    if not counts:
        raise ValueError("embedding index is empty")
    return counts


def load_predictions(path: str | Path) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    flow_ids = [str(value) for value in payload.get("flow_ids") or []]
    y_true = np.asarray(payload.get("flow_y_true"), dtype=np.int64)
    probabilities = np.asarray(payload.get("flow_prob"), dtype=np.float64)
    if not flow_ids or y_true.ndim != 1 or probabilities.ndim != 2:
        raise ValueError("prediction JSON must contain flow_ids, flow_y_true, and flow_prob")
    if len(flow_ids) != len(y_true) or len(flow_ids) != len(probabilities):
        raise ValueError("prediction rows are not aligned")
    if len(set(flow_ids)) != len(flow_ids):
        raise ValueError("prediction flow_ids are not unique")
    if not np.isfinite(probabilities).all():
        raise ValueError("prediction probabilities are not finite")
    y_pred = probabilities.argmax(axis=1).astype(np.int64)
    return flow_ids, y_true, y_pred, probabilities


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 2 or np.all(x == x[0]) or np.all(y == y[0]):
        return None
    value = np.corrcoef(rankdata(x), rankdata(y))[0, 1]
    return float(value) if np.isfinite(value) else None


def class_conditional_associations(
    counts: np.ndarray,
    outcome: np.ndarray,
    y_true: np.ndarray,
) -> dict[str, Any]:
    rows = []
    for label in np.unique(y_true):
        mask = y_true == label
        value = spearman(counts[mask].astype(float), outcome[mask].astype(float))
        if value is None:
            continue
        rows.append(
            {
                "label_id": int(label),
                "num_flows": int(mask.sum()),
                "spearman": value,
            }
        )
    values = np.asarray([row["spearman"] for row in rows], dtype=np.float64)
    return {
        "num_eligible_classes": len(rows),
        "macro_mean_spearman": float(values.mean()) if len(values) else None,
        "median_spearman": float(np.median(values)) if len(values) else None,
        "fraction_negative": float((values < 0).mean()) if len(values) else None,
        "per_class": rows,
    }


def subset_metrics(
    mask: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    packet_counts: np.ndarray,
) -> dict[str, Any]:
    indices = np.flatnonzero(mask)
    if len(indices) == 0:
        return {"num_flows": 0}
    truth = y_true[indices]
    pred = y_pred[indices]
    return {
        "num_flows": int(len(indices)),
        "packet_count_min": int(packet_counts[indices].min()),
        "packet_count_median": float(np.median(packet_counts[indices])),
        "packet_count_max": int(packet_counts[indices].max()),
        "accuracy": float(accuracy_score(truth, pred)),
        "present_class_macro_f1": float(
            f1_score(truth, pred, labels=np.unique(truth), average="macro", zero_division=0)
        ),
        "mean_confidence": float(probabilities[indices].max(axis=1).mean()),
        "num_present_classes": int(len(np.unique(truth))),
    }


def fixed_strata(
    counts: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
) -> list[dict[str, Any]]:
    rows = []
    for lower, upper in FIXED_BINS:
        mask = counts >= lower
        if upper is not None:
            mask &= counts <= upper
        label = f"{lower}-{upper}" if upper is not None else f"{lower}+"
        rows.append({"stratum": label, **subset_metrics(mask, y_true, y_pred, probabilities, counts)})
    return rows


def quantile_strata(
    counts: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[list[dict[str, Any]], list[float]]:
    edges = np.quantile(counts, [0.0, 0.25, 0.5, 0.75, 1.0]).astype(float)
    rows = []
    for index in range(4):
        lower, upper = edges[index], edges[index + 1]
        mask = counts >= lower
        mask &= counts <= upper if index == 3 else counts < upper
        rows.append(
            {
                "stratum": f"q{index + 1}",
                "lower_inclusive": lower,
                "upper_inclusive": upper if index == 3 else None,
                "upper_exclusive": None if index == 3 else upper,
                **subset_metrics(mask, y_true, y_pred, probabilities, counts),
            }
        )
    return rows, edges.tolist()


def build_report(predictions: str | Path, embedding_index: str | Path) -> dict[str, Any]:
    flow_ids, y_true, y_pred, probabilities = load_predictions(predictions)
    count_by_flow = load_packet_counts(embedding_index)
    missing = [flow_id for flow_id in flow_ids if flow_id not in count_by_flow]
    if missing:
        raise ValueError(f"embedding index misses {len(missing)} prediction flows")
    counts = np.asarray([count_by_flow[flow_id] for flow_id in flow_ids], dtype=np.int64)
    correctness = (y_true == y_pred).astype(np.float64)
    quantiles, edges = quantile_strata(counts, y_true, y_pred, probabilities)
    return {
        "schema": "flow_length_performance_v1",
        "status": "reporting_only",
        "selection_role": "none",
        "inputs": {
            "predictions": str(predictions),
            "embedding_index": str(embedding_index),
        },
        "num_flows": len(flow_ids),
        "packet_count_summary": {
            "min": int(counts.min()),
            "median": float(np.median(counts)),
            "mean": float(counts.mean()),
            "max": int(counts.max()),
            "quantile_edges": edges,
        },
        "associations": {
            "spearman_packet_count_correctness": spearman(counts.astype(float), correctness),
            "spearman_packet_count_confidence": spearman(
                counts.astype(float), probabilities.max(axis=1)
            ),
            "within_class_correctness": class_conditional_associations(
                counts, correctness, y_true
            ),
            "within_class_confidence": class_conditional_associations(
                counts, probabilities.max(axis=1), y_true
            ),
        },
        "fixed_packet_count_strata": fixed_strata(
            counts, y_true, y_pred, probabilities
        ),
        "quantile_strata": quantiles,
        "interpretation_guard": (
            "Associations are descriptive and do not establish that flow length causes errors."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--embedding_index", required=True)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()
    report = build_report(args.predictions, args.embedding_index)
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output_json": str(output), "num_flows": report["num_flows"]}, indent=2))


if __name__ == "__main__":
    main()
