#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support

def load_label_names(path: str):
    if not path:
        return None, None
    with open(path, "r", encoding="utf-8") as f:
        label_map = json.load(f)
    if not label_map:
        return None, label_map
    max_id = max(int(v) for v in label_map.values())
    label_names = [str(i) for i in range(max_id + 1)]
    for name, idx in label_map.items():
        label_names[int(idx)] = name
    return label_names, label_map


def compute_metrics(y_true, y_pred):
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    p_weight, r_weight, f_weight, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    return {
        "accuracy": accuracy_score(y_true, y_pred) if len(y_true) else 0.0,
        "macro_precision": p_macro,
        "macro_recall": r_macro,
        "macro_f1": f_macro,
        "weighted_precision": p_weight,
        "weighted_recall": r_weight,
        "weighted_f1": f_weight,
    }


def load_payload(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    required = ["valid_flow_ids", "valid_y_true", "valid_prob", "flow_ids", "flow_y_true", "flow_prob"]
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"{path} is missing required probability fields: {missing}")
    return data


def align_prob(data: Dict[str, Any], split: str, fids: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    if split == "valid":
        ids, labels, probs = data["valid_flow_ids"], data["valid_y_true"], data["valid_prob"]
    elif split == "test":
        ids, labels, probs = data["flow_ids"], data["flow_y_true"], data["flow_prob"]
    else:
        raise ValueError(split)
    idx = {str(fid): i for i, fid in enumerate(ids)}
    y = np.asarray([labels[idx[fid]] for fid in fids], dtype=np.int64)
    p = np.asarray([probs[idx[fid]] for fid in fids], dtype=np.float32)
    p = p / np.maximum(p.sum(axis=1, keepdims=True), 1e-12)
    return y, p


def simplex_weights(n: int, step: float):
    units = int(round(1.0 / step))
    if units <= 0 or abs(units * step - 1.0) > 1e-6:
        raise ValueError("--simplex_step must divide 1.0")
    for cuts in product(range(units + 1), repeat=n - 1):
        used = sum(cuts)
        if used > units:
            continue
        weights = list(cuts) + [units - used]
        yield np.asarray(weights, dtype=np.float32) / float(units)


def parse_min_weights(raw_items: List[List[str]] | None) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for raw in raw_items or []:
        if len(raw) != 2:
            raise ValueError("--min_weight expects NAME VALUE")
        name, value = raw
        out[name] = float(value)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", nargs=2, action="append", metavar=("NAME", "JSON"), required=True)
    ap.add_argument("--label_map", default="")
    ap.add_argument("--simplex_step", type=float, default=0.05)
    ap.add_argument("--select_metric", choices=["accuracy", "macro_f1"], default="accuracy")
    ap.add_argument("--min_weight", nargs=2, action="append", metavar=("NAME", "VALUE"), help="Constrain a named input to receive at least VALUE fusion weight.")
    ap.add_argument("--output_json", default="")
    args = ap.parse_args()

    named_payloads = [(name, load_payload(path), path) for name, path in args.input]
    min_weights = parse_min_weights(args.min_weight)
    input_names = [name for name, _, _ in named_payloads]
    unknown = sorted(set(min_weights) - set(input_names))
    if unknown:
        raise ValueError(f"--min_weight references unknown inputs: {unknown}")
    if sum(min_weights.values()) > 1.0 + 1e-6:
        raise ValueError("--min_weight constraints sum to more than 1.0")
    valid_common = sorted(set.intersection(*(set(map(str, data["valid_flow_ids"])) for _, data, _ in named_payloads)))
    test_common = sorted(set.intersection(*(set(map(str, data["flow_ids"])) for _, data, _ in named_payloads)))
    if not valid_common or not test_common:
        raise ValueError("No common flow ids across prediction JSONs.")

    valid_probs = []
    test_probs = []
    y_valid_ref = None
    y_test_ref = None
    for name, data, _ in named_payloads:
        y_valid, p_valid = align_prob(data, "valid", valid_common)
        y_test, p_test = align_prob(data, "test", test_common)
        if y_valid_ref is None:
            y_valid_ref, y_test_ref = y_valid, y_test
        elif not np.array_equal(y_valid_ref, y_valid) or not np.array_equal(y_test_ref, y_test):
            raise ValueError(f"Labels do not align for {name}.")
        valid_probs.append(p_valid)
        test_probs.append(p_test)

    reports = []
    best = None
    best_valid_prob = None
    for weights in simplex_weights(len(named_payloads), args.simplex_step):
        if any(float(weights[input_names.index(name)]) + 1e-8 < min_value for name, min_value in min_weights.items()):
            continue
        valid_prob = sum(float(w) * p for w, p in zip(weights, valid_probs))
        pred = valid_prob.argmax(axis=1)
        metrics = compute_metrics(y_valid_ref.tolist(), pred.tolist())
        row = {
            "weights": {name: float(w) for w, (name, _, _) in zip(weights, named_payloads)},
            "metrics": metrics,
        }
        reports.append(row)
        key = (metrics[args.select_metric], metrics["macro_f1"], metrics["accuracy"])
        if best is None or key > best[0]:
            best = (key, row)
            best_valid_prob = valid_prob
    if best is None or best_valid_prob is None:
        raise ValueError("No fusion weights satisfy the provided constraints.")

    selected_weights = np.asarray([best[1]["weights"][name] for name, _, _ in named_payloads], dtype=np.float32)
    test_prob = sum(float(w) * p for w, p in zip(selected_weights, test_probs))
    y_pred = test_prob.argmax(axis=1).astype(np.int64)
    metrics = compute_metrics(y_test_ref.tolist(), y_pred.tolist())
    print("selected_fusion", json.dumps(best[1], sort_keys=True))
    print("test_fusion", json.dumps(metrics, indent=2, sort_keys=True))
    label_names, label_map = load_label_names(args.label_map)
    if label_names:
        print(classification_report(y_test_ref, y_pred, labels=list(range(len(label_names))), target_names=label_names, zero_division=0))
    else:
        print(classification_report(y_test_ref, y_pred, zero_division=0))

    if args.output_json:
        payload = {
            "metrics": {"flow_level": metrics},
            "selected_weights": best[1]["weights"],
            "min_weights": min_weights,
            "fusion_reports": reports,
            "inputs": [{"name": name, "path": path} for name, _, path in named_payloads],
            "label_map": label_map,
            "flow_ids": test_common,
            "flow_y_true": y_test_ref.tolist(),
            "flow_y_pred": y_pred.tolist(),
            "flow_prob": test_prob.tolist(),
            "valid_flow_ids": valid_common,
            "valid_y_true": y_valid_ref.tolist(),
            "valid_y_pred": best_valid_prob.argmax(axis=1).astype(np.int64).tolist(),
            "valid_prob": best_valid_prob.tolist(),
        }
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()
