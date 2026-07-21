#!/usr/bin/env python3
"""Fuse aligned packet probabilities from independently trained folds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from packet_eval_utils import packet_classification_metrics
from train_tower1_multitask import load_label_names


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--label_map", required=True)
    ap.add_argument("--output_json", required=True)
    ap.add_argument("--output_npz", default="")
    ap.add_argument(
        "--output_validation_npz",
        default="",
        help="Optional selected validation blend; requires --validation_inputs.",
    )
    ap.add_argument("--method", choices=["mean", "log_mean"], default="mean")
    ap.add_argument(
        "--weights",
        default="",
        help="Optional comma-separated non-negative input weights; defaults to a uniform mean.",
    )
    ap.add_argument(
        "--validation_inputs",
        nargs=2,
        default=None,
        metavar=("FIRST_VALID", "SECOND_VALID"),
        help="Select two-input arithmetic weights on aligned validation probabilities.",
    )
    ap.add_argument("--weight_grid_size", type=int, default=201)
    ap.add_argument("--select_metric", choices=["accuracy", "macro_f1"], default="macro_f1")
    args = ap.parse_args()

    arrays = [np.load(path) for path in args.inputs]
    label_names = load_label_names(args.label_map)
    selection = None
    if args.validation_inputs:
        if args.weights:
            raise ValueError("--weights and --validation_inputs are mutually exclusive")
        if len(arrays) != 2 or args.method != "mean":
            raise ValueError("validation weight selection requires exactly two mean-fusion inputs")
        if args.weight_grid_size < 2:
            raise ValueError("--weight_grid_size must be at least 2")
        valid_arrays = [np.load(path) for path in args.validation_inputs]
        y_valid = valid_arrays[0]["y_true"].astype(np.int64)
        for path, data in zip(args.validation_inputs, valid_arrays):
            if not np.array_equal(y_valid, data["y_true"]):
                raise ValueError(f"validation label alignment mismatch: {path}")
        valid_probabilities = [data["probabilities"].astype(np.float32) for data in valid_arrays]
        best = None
        for first_weight in np.linspace(0.0, 1.0, args.weight_grid_size):
            candidate = first_weight * valid_probabilities[0] + (1.0 - first_weight) * valid_probabilities[1]
            candidate_metrics = packet_classification_metrics(
                y_valid,
                candidate.argmax(axis=1),
                len(label_names),
                label_names,
            )
            score = (
                float(candidate_metrics[args.select_metric]),
                float(candidate_metrics["macro_f1"]),
                float(candidate_metrics["accuracy"]),
            )
            if best is None or score > best[0]:
                best = (score, float(first_weight), candidate_metrics, candidate)
        assert best is not None
        weights = np.asarray([best[1], 1.0 - best[1]], dtype=np.float64)
        selection = {
            "scope": "validation_only",
            "validation_inputs": args.validation_inputs,
            "select_metric": args.select_metric,
            "weight_grid_size": args.weight_grid_size,
            "validation_metrics": best[2],
        }
    elif args.weights:
        weights = np.asarray([float(value) for value in args.weights.split(",")], dtype=np.float64)
        if len(weights) != len(arrays):
            raise ValueError("--weights must contain one value per --inputs path")
        if np.any(weights < 0) or not np.isfinite(weights).all() or float(weights.sum()) <= 0:
            raise ValueError("--weights must be finite, non-negative, and have a positive sum")
        weights /= weights.sum()
    else:
        weights = np.full(len(arrays), 1.0 / len(arrays), dtype=np.float64)
    y_true = arrays[0]["y_true"].astype(np.int64)
    flow_ids = arrays[0]["flow_ids"] if "flow_ids" in arrays[0] else None
    fused = None
    for path, data, weight in zip(args.inputs, arrays, weights):
        if not np.array_equal(y_true, data["y_true"]):
            raise ValueError(f"label alignment mismatch: {path}")
        if ("flow_ids" in data) != (flow_ids is not None):
            raise ValueError(f"flow-id availability mismatch: {path}")
        if flow_ids is not None and not np.array_equal(flow_ids, data["flow_ids"]):
            raise ValueError(f"flow-id alignment mismatch: {path}")
        probabilities = data["probabilities"].astype(np.float32)
        contribution = np.log(np.clip(probabilities, 1e-12, 1.0)) if args.method == "log_mean" else probabilities
        contribution = float(weight) * contribution
        fused = contribution if fused is None else fused + contribution
    assert fused is not None
    if args.method == "log_mean":
        fused = np.exp(fused)
        fused /= fused.sum(axis=1, keepdims=True)
    metrics = packet_classification_metrics(y_true, fused.argmax(axis=1), len(label_names), label_names)
    output = {
        "task": "packet-level-classification",
        "sample_unit": "one_packet",
        "method": args.method,
        "inputs": args.inputs,
        "weights": weights.tolist(),
        "weight_selection": selection,
        "metrics": metrics,
    }
    path = Path(args.output_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    if args.output_npz:
        output_npz = Path(args.output_npz)
        output_npz.parent.mkdir(parents=True, exist_ok=True)
        output_arrays = {
            "y_true": y_true,
            "probabilities": fused.astype(np.float32),
        }
        if flow_ids is not None:
            output_arrays["flow_ids"] = flow_ids
        np.savez_compressed(output_npz, **output_arrays)
    if args.output_validation_npz:
        if not args.validation_inputs:
            raise ValueError("--output_validation_npz requires --validation_inputs")
        output_validation_npz = Path(args.output_validation_npz)
        output_validation_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_validation_npz,
            y_true=y_valid,
            probabilities=np.asarray(best[3], dtype=np.float32),
        )
    print(f"accuracy={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f}")
    print(f"saved {path}" + (f" and {args.output_npz}" if args.output_npz else ""))


if __name__ == "__main__":
    main()
