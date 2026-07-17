#!/usr/bin/env python3
"""Fuse same-dataset predictions from multiple ready-made folds.

The SWEET split folders expose a shared test set with several train/valid
partitions. This script treats fold-specific models as a stability ensemble:
all inputs must predict the same shared test flow ids, and the output is a
label-free consensus over those fold models.
"""
from __future__ import annotations

import argparse
import json
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


def normalize_prob(prob: np.ndarray) -> np.ndarray:
    prob = np.asarray(prob, dtype=np.float64)
    prob = np.clip(prob, 1e-12, None)
    return prob / np.maximum(prob.sum(axis=1, keepdims=True), 1e-12)


def parse_input(raw: List[str]) -> Tuple[str, str]:
    if len(raw) != 2:
        raise ValueError("--input expects NAME JSON")
    return raw[0], raw[1]


def load_payload(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    required = ["flow_ids", "flow_y_true", "flow_prob"]
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"{path} is missing required fields: {missing}")
    return data


def align_test(data: Dict[str, Any], fids: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    idx = {str(fid): i for i, fid in enumerate(data["flow_ids"])}
    y = np.asarray([data["flow_y_true"][idx[fid]] for fid in fids], dtype=np.int64)
    p = normalize_prob(np.asarray([data["flow_prob"][idx[fid]] for fid in fids], dtype=np.float64))
    return y, p


def fuse_probs(probs: List[np.ndarray], mode: str) -> np.ndarray:
    if mode == "mean":
        return normalize_prob(np.mean(probs, axis=0))
    if mode == "log_mean":
        logp = np.mean([np.log(np.clip(p, 1e-12, 1.0)) for p in probs], axis=0)
        logp = logp - logp.max(axis=1, keepdims=True)
        return normalize_prob(np.exp(logp))
    if mode in {"vote", "vote_priority"}:
        out = np.zeros_like(probs[0], dtype=np.float64)
        for p in probs:
            pred = p.argmax(axis=1)
            out[np.arange(len(pred)), pred] += 1.0
        if mode == "vote_priority":
            priority_pred = probs[0].argmax(axis=1)
            max_votes = out.max(axis=1)
            tied = out[np.arange(out.shape[0]), priority_pred] == max_votes
            out[tied] += 0.0
            out[np.where(tied)[0], priority_pred[tied]] += 1e-3
        out += 1e-6
        return normalize_prob(out)
    raise ValueError(mode)


def selected_mode(probs: List[np.ndarray], requested: str, confidence_threshold: float) -> str:
    if requested != "auto_confidence":
        return requested
    mean_conf = float(np.mean([p.max(axis=1).mean() for p in probs]))
    return "vote_priority" if mean_conf >= confidence_threshold else "log_mean"


def write_payload(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", nargs=2, action="append", metavar=("NAME", "JSON"), required=True)
    ap.add_argument("--mode", choices=["mean", "log_mean", "vote", "vote_priority", "auto_confidence"], default="auto_confidence")
    ap.add_argument("--confidence_threshold", type=float, default=0.9)
    ap.add_argument("--label_map", default="")
    ap.add_argument("--output_json", required=True)
    ap.add_argument("--fold_alias", action="append", default=[], help="Also write a copy with _foldN before .json for summary scanners.")
    ap.add_argument("--no_report", action="store_true")
    args = ap.parse_args()

    named = [parse_input(raw) for raw in args.input]
    payloads = [(name, load_payload(path), path) for name, path in named]
    common = sorted(set.intersection(*(set(map(str, data["flow_ids"])) for _, data, _ in payloads)))
    if not common:
        raise ValueError("No common flow ids across inputs.")

    probs = []
    y_ref = None
    input_reports = []
    for name, data, path in payloads:
        y, p = align_test(data, common)
        if y_ref is None:
            y_ref = y
        elif not np.array_equal(y_ref, y):
            raise ValueError(f"Labels do not align for input {name}.")
        pred = p.argmax(axis=1).astype(np.int64)
        probs.append(p)
        input_reports.append({
            "name": name,
            "path": path,
            "metrics": compute_metrics(y.tolist(), pred.tolist()),
            "mean_confidence": float(p.max(axis=1).mean()),
        })

    assert y_ref is not None
    mode = selected_mode(probs, args.mode, args.confidence_threshold)
    fused = fuse_probs(probs, mode)
    pred = fused.argmax(axis=1).astype(np.int64)
    metrics = compute_metrics(y_ref.tolist(), pred.tolist())
    label_names, label_map = load_label_names(args.label_map)
    config = {
        "requested_mode": args.mode,
        "selected_mode": mode,
        "confidence_threshold": args.confidence_threshold,
        "mean_input_confidence": float(np.mean([r["mean_confidence"] for r in input_reports])),
        "num_inputs": len(input_reports),
        "num_flows": len(common),
    }
    print("cross_fold_consensus", json.dumps({"metrics": metrics, "config": config}, indent=2, sort_keys=True))
    if not args.no_report:
        if label_names:
            print(classification_report(y_ref, pred, labels=list(range(len(label_names))), target_names=label_names, zero_division=0))
        else:
            print(classification_report(y_ref, pred, zero_division=0))

    payload = {
        "metrics": {"flow_level": metrics},
        "label_map": label_map,
        "flow_ids": common,
        "flow_y_true": y_ref.tolist(),
        "flow_y_pred": pred.tolist(),
        "flow_prob": fused.tolist(),
        "inputs": input_reports,
        "config": config,
    }
    output = Path(args.output_json)
    write_payload(output, payload)
    stem = output.with_suffix("")
    for alias in args.fold_alias:
        alias = str(alias).strip()
        if not alias:
            continue
        write_payload(Path(f"{stem}_fold{alias}{output.suffix}"), payload)


if __name__ == "__main__":
    main()
