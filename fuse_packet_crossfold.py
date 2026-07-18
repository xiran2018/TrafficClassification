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
    ap.add_argument("--method", choices=["mean", "log_mean"], default="mean")
    args = ap.parse_args()

    arrays = [np.load(path) for path in args.inputs]
    y_true = arrays[0]["y_true"].astype(np.int64)
    fused = None
    for path, data in zip(args.inputs, arrays):
        if not np.array_equal(y_true, data["y_true"]):
            raise ValueError(f"label alignment mismatch: {path}")
        probabilities = data["probabilities"].astype(np.float32)
        contribution = np.log(np.clip(probabilities, 1e-12, 1.0)) if args.method == "log_mean" else probabilities
        fused = contribution if fused is None else fused + contribution
    assert fused is not None
    fused /= float(len(arrays))
    if args.method == "log_mean":
        fused = np.exp(fused)
        fused /= fused.sum(axis=1, keepdims=True)
    label_names = load_label_names(args.label_map)
    metrics = packet_classification_metrics(y_true, fused.argmax(axis=1), len(label_names), label_names)
    output = {
        "task": "packet-level-classification",
        "sample_unit": "one_packet",
        "method": args.method,
        "inputs": args.inputs,
        "metrics": metrics,
    }
    path = Path(args.output_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    if args.output_npz:
        output_npz = Path(args.output_npz)
        output_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_npz, y_true=y_true, probabilities=fused.astype(np.float32))
    print(f"accuracy={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f}")
    print(f"saved {path}" + (f" and {args.output_npz}" if args.output_npz else ""))


if __name__ == "__main__":
    main()
