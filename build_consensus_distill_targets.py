#!/usr/bin/env python3
"""Build flow-level soft targets for consensus distillation.

This script fuses several prediction JSON files into one `flow_ids` +
`flow_prob` teacher file that `train_tower2.py --distill_targets_json` can
consume. Use `--split valid` for paper-safe supervised distillation targets
created from validation predictions. Use `--split test` only for explicitly
declared transductive/unlabeled-target ablations.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from probability_metrics import calibration_metrics


def normalize_prob(prob: np.ndarray) -> np.ndarray:
    prob = np.asarray(prob, dtype=np.float64)
    prob = np.clip(prob, 1e-12, None)
    return prob / np.maximum(prob.sum(axis=1, keepdims=True), 1e-12)


def compute_metrics(y_true: np.ndarray, prob: np.ndarray) -> Dict[str, Any]:
    pred = prob.argmax(axis=1).astype(np.int64)
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(y_true, pred, average="macro", zero_division=0)
    p_weight, r_weight, f_weight, _ = precision_recall_fscore_support(y_true, pred, average="weighted", zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, pred)) if len(y_true) else 0.0,
        "macro_precision": float(p_macro),
        "macro_recall": float(r_macro),
        "macro_f1": float(f_macro),
        "weighted_precision": float(p_weight),
        "weighted_recall": float(r_weight),
        "weighted_f1": float(f_weight),
        "calibration": calibration_metrics(y_true, prob),
    }


def split_keys(split: str) -> Tuple[str, str, str]:
    if split == "valid":
        return "valid_flow_ids", "valid_y_true", "valid_prob"
    if split == "test":
        return "flow_ids", "flow_y_true", "flow_prob"
    raise ValueError(split)


def parse_input(raw: List[str]) -> Tuple[str, str]:
    if len(raw) != 2:
        raise ValueError("--input expects NAME JSON")
    return raw[0], raw[1]


def load_payload(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_split(data: Dict[str, Any], split: str) -> Tuple[List[str], np.ndarray | None, np.ndarray]:
    id_key, y_key, prob_key = split_keys(split)
    if id_key not in data or prob_key not in data:
        raise ValueError(f"Missing {id_key}/{prob_key}; choose another --split or build a fusion payload first.")
    ids = [str(fid) for fid in data[id_key]]
    prob = normalize_prob(np.asarray(data[prob_key], dtype=np.float64))
    if len(ids) != prob.shape[0]:
        raise ValueError(f"Mismatched ids/prob lengths for split={split}")
    y = None
    if y_key in data and data[y_key] is not None:
        y = np.asarray(data[y_key], dtype=np.int64)
        if len(y) != len(ids):
            raise ValueError(f"Mismatched ids/labels lengths for split={split}")
    return ids, y, prob


def align_payload(data: Dict[str, Any], split: str, flow_ids: List[str]) -> Tuple[np.ndarray | None, np.ndarray]:
    ids, y, prob = extract_split(data, split)
    pos = {fid: i for i, fid in enumerate(ids)}
    aligned_prob = normalize_prob(np.asarray([prob[pos[fid]] for fid in flow_ids], dtype=np.float64))
    aligned_y = None if y is None else np.asarray([y[pos[fid]] for fid in flow_ids], dtype=np.int64)
    return aligned_y, aligned_prob


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
            out[np.where(tied)[0], priority_pred[tied]] += 1e-3
        out += 1e-6
        return normalize_prob(out)
    raise ValueError(mode)


def select_mode(probs: List[np.ndarray], requested: str, threshold: float) -> str:
    if requested != "auto_confidence":
        return requested
    mean_conf = float(np.mean([p.max(axis=1).mean() for p in probs]))
    return "vote_priority" if mean_conf >= threshold else "log_mean"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", nargs=2, action="append", metavar=("NAME", "JSON"), required=True)
    ap.add_argument("--split", choices=["valid", "test"], default="valid")
    ap.add_argument("--mode", choices=["mean", "log_mean", "vote", "vote_priority", "auto_confidence"], default="auto_confidence")
    ap.add_argument("--confidence_threshold", type=float, default=0.9)
    ap.add_argument("--min_teacher_confidence", type=float, default=0.0, help="Drop fused targets below this max probability.")
    ap.add_argument("--min_input_accuracy", type=float, default=0.0, help="Reject labeled teacher inputs below this split accuracy.")
    ap.add_argument("--min_input_macro_f1", type=float, default=0.0, help="Reject labeled teacher inputs below this split macro-F1.")
    ap.add_argument("--output_json", required=True)
    ap.add_argument("--allow_test_split", action="store_true", help="Required for --split test to make transductive use explicit.")
    args = ap.parse_args()

    if args.split == "test" and not args.allow_test_split:
        raise ValueError("--split test requires --allow_test_split; prefer --split valid for paper-safe distillation.")

    named = [parse_input(raw) for raw in args.input]
    payloads = [(name, load_payload(path), path) for name, path in named]
    id_sets = []
    for _, data, _ in payloads:
        ids, _, _ = extract_split(data, args.split)
        id_sets.append(set(ids))
    common = sorted(set.intersection(*id_sets))
    if not common:
        raise ValueError("No common flow ids across inputs.")

    probs = []
    y_ref = None
    inputs = []
    for name, data, path in payloads:
        y, prob = align_payload(data, args.split, common)
        if y_ref is None:
            y_ref = y
        elif y is not None and y_ref is not None and not np.array_equal(y_ref, y):
            raise ValueError(f"Labels do not align for input {name}.")
        probs.append(prob)
        row = {
            "name": name,
            "path": path,
            "mean_confidence": float(prob.max(axis=1).mean()),
        }
        if y is not None:
            row["metrics"] = compute_metrics(y, prob)
            if row["metrics"]["accuracy"] < args.min_input_accuracy or row["metrics"]["macro_f1"] < args.min_input_macro_f1:
                raise ValueError(
                    f"Input {name} is below teacher-quality floor: "
                    f"acc={row['metrics']['accuracy']:.4f}, macro_f1={row['metrics']['macro_f1']:.4f}, "
                    f"required acc>={args.min_input_accuracy:.4f}, macro_f1>={args.min_input_macro_f1:.4f}"
                )
        inputs.append(row)

    mode = select_mode(probs, args.mode, args.confidence_threshold)
    fused = fuse_probs(probs, mode)
    keep = np.ones(fused.shape[0], dtype=bool)
    if args.min_teacher_confidence > 0:
        keep = fused.max(axis=1) >= float(args.min_teacher_confidence)
    kept_ids = [fid for fid, ok in zip(common, keep) if ok]
    fused = fused[keep]
    y_out = None if y_ref is None else y_ref[keep]

    metrics = compute_metrics(y_out, fused) if y_out is not None else {}
    config = {
        "split": args.split,
        "requested_mode": args.mode,
        "selected_mode": mode,
        "confidence_threshold": args.confidence_threshold,
        "min_teacher_confidence": args.min_teacher_confidence,
        "min_input_accuracy": args.min_input_accuracy,
        "min_input_macro_f1": args.min_input_macro_f1,
        "num_inputs": len(inputs),
        "num_common_flows": len(common),
        "num_output_flows": len(kept_ids),
        "mean_input_confidence": float(np.mean([row["mean_confidence"] for row in inputs])),
        "paper_safe_note": "valid split is suitable for supervised paper-safe distillation; test split is transductive only.",
    }
    payload: Dict[str, Any] = {
        "flow_ids": kept_ids,
        "flow_prob": fused.tolist(),
        "inputs": inputs,
        "metrics": {"flow_level": metrics} if metrics else {},
        "config": config,
    }
    if y_out is not None:
        payload["flow_y_true"] = y_out.astype(int).tolist()
        payload["flow_y_pred"] = fused.argmax(axis=1).astype(int).tolist()

    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print("consensus_distill_targets", json.dumps({"metrics": metrics, "config": config}, indent=2, sort_keys=True))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
