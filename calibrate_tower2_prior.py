#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_recall_fscore_support

from ensemble_tower2 import flow_logits_for_checkpoint
from test_tower2 import load_label_names


def simplex_project(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    if v.size == 0:
        return v
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, len(u) + 1) > (cssv - 1))[0]
    if len(rho) == 0:
        return np.ones_like(v) / len(v)
    rho = rho[-1]
    theta = (cssv[rho] - 1) / (rho + 1)
    w = np.maximum(v - theta, 0)
    return w / max(w.sum(), 1e-12)


def compute_metrics(y_true, y_pred):
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    p_weight, r_weight, f_weight, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    return {
        "accuracy": accuracy_score(y_true, y_pred) if y_true else 0.0,
        "macro_precision": p_macro,
        "macro_recall": r_macro,
        "macro_f1": f_macro,
        "weighted_precision": p_weight,
        "weighted_recall": r_weight,
        "weighted_f1": f_weight,
    }


def estimate_target_prior(y_cal, pred_cal, pred_target, num_classes: int, ridge: float) -> np.ndarray:
    cm = confusion_matrix(y_cal, pred_cal, labels=list(range(num_classes))).astype("float64")
    cm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1.0)
    q = np.bincount(pred_target, minlength=num_classes).astype("float64")
    q = q / max(q.sum(), 1.0)
    a = cm.T
    if ridge > 0:
        lhs = a.T @ a + ridge * np.eye(num_classes)
        rhs = a.T @ q
        prior = np.linalg.solve(lhs, rhs)
    else:
        prior = np.linalg.lstsq(a, q, rcond=None)[0]
    return simplex_project(prior)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--calibration_dataset", required=True, help="Labeled validation dataset used to estimate P(pred|true).")
    ap.add_argument("--target_dataset", required=True, help="Target dataset. Labels are used only for reporting metrics.")
    ap.add_argument("--label_map", default="")
    ap.add_argument("--strengths", default="0,0.25,0.5,0.75,1.0", help="Comma-separated prior correction strengths.")
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--output_json", default="")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    label_names, label_map = load_label_names(args.label_map)
    cal_labels, cal_logits = flow_logits_for_checkpoint(args.checkpoint, args.calibration_dataset, args.device, args.batch_size)
    tgt_labels, tgt_logits = flow_logits_for_checkpoint(args.checkpoint, args.target_dataset, args.device, args.batch_size)
    common_cal = sorted(cal_logits)
    common_tgt = sorted(tgt_logits)
    num_classes = len(next(iter(tgt_logits.values())))

    y_cal = np.asarray([cal_labels[fid] for fid in common_cal], dtype=np.int64)
    pred_cal = np.asarray([int(cal_logits[fid].argmax()) for fid in common_cal], dtype=np.int64)
    raw_target_logits = np.stack([tgt_logits[fid] for fid in common_tgt], axis=0)
    y_target = np.asarray([tgt_labels[fid] for fid in common_tgt], dtype=np.int64)
    pred_target = raw_target_logits.argmax(axis=1)

    estimated_prior = estimate_target_prior(y_cal, pred_cal, pred_target, num_classes, args.ridge)
    calibration_prior = np.bincount(y_cal, minlength=num_classes).astype("float64")
    calibration_prior = calibration_prior / max(calibration_prior.sum(), 1.0)
    prior_logit = np.log(estimated_prior + 1e-12) - np.log(calibration_prior + 1e-12)

    reports = {}
    best_key = None
    best_macro = -1.0
    for raw_strength in args.strengths.split(","):
        strength = float(raw_strength)
        logits = raw_target_logits + strength * prior_logit[None, :]
        y_pred = logits.argmax(axis=1).astype("int64")
        metrics = compute_metrics(y_target.tolist(), y_pred.tolist())
        key = f"{strength:g}"
        reports[key] = metrics
        print(f"strength={key}", json.dumps(metrics, sort_keys=True))
        if metrics["macro_f1"] > best_macro:
            best_macro = metrics["macro_f1"]
            best_key = key

    best_strength = float(best_key)
    best_pred = (raw_target_logits + best_strength * prior_logit[None, :]).argmax(axis=1).astype("int64")
    if label_names:
        print(classification_report(y_target, best_pred, labels=list(range(len(label_names))), target_names=label_names, zero_division=0))
    else:
        print(classification_report(y_target, best_pred, zero_division=0))

    payload = {
        "metrics_by_strength": reports,
        "selected_strength_by_macro_f1": best_key,
        "estimated_target_prior": estimated_prior.tolist(),
        "calibration_prior": calibration_prior.tolist(),
        "label_map": label_map,
        "flow_ids": common_tgt,
        "flow_y_true": y_target.tolist(),
        "flow_y_pred": best_pred.tolist(),
    }
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()
