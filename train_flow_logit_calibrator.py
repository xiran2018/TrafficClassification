#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ensemble_tower2 import flow_logits_for_checkpoint
from test_tower2 import load_label_names


def softmax_np(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=-1, keepdims=True).clip(min=1e-12)


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


def parse_member(raw: List[str]) -> Tuple[str, str, str, str]:
    if len(raw) != 4:
        raise ValueError("--member expects NAME CHECKPOINT CAL_DATASET TARGET_DATASET")
    return raw[0], raw[1], raw[2], raw[3]


def build_feature_matrix(logit_maps: List[Dict[str, np.ndarray]], fids: List[str], include_probs: bool, include_stats: bool) -> np.ndarray:
    parts = []
    for logits_by_fid in logit_maps:
        logits = np.stack([logits_by_fid[fid] for fid in fids], axis=0).astype("float32")
        parts.append(logits)
        probs = softmax_np(logits)
        if include_probs:
            parts.append(probs.astype("float32"))
        if include_stats:
            sorted_probs = np.sort(probs, axis=1)
            conf = sorted_probs[:, -1:]
            margin = (sorted_probs[:, -1:] - sorted_probs[:, -2:-1]) if probs.shape[1] > 1 else conf
            entropy = -np.sum(probs * np.log(probs.clip(min=1e-12)), axis=1, keepdims=True)
            parts.append(np.concatenate([conf, margin, entropy], axis=1).astype("float32"))
    return np.concatenate(parts, axis=1)


def cv_score(x: np.ndarray, y: np.ndarray, c: float, metric: str, n_splits: int, seed: int) -> Tuple[float, float]:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    preds = np.zeros_like(y)
    for train_idx, val_idx in skf.split(x, y):
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=c, max_iter=5000, solver="lbfgs"),
        )
        clf.fit(x[train_idx], y[train_idx])
        preds[val_idx] = clf.predict(x[val_idx])
    metrics = compute_metrics(y.tolist(), preds.tolist())
    return float(metrics[metric]), float(metrics["accuracy"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--member", nargs=4, action="append", metavar=("NAME", "CHECKPOINT", "CAL_DATASET", "TARGET_DATASET"), required=True)
    ap.add_argument("--label_map", default="")
    ap.add_argument("--c_grid", default="0.01,0.03,0.1,0.3,1,3,10")
    ap.add_argument("--select_metric", choices=["accuracy", "macro_f1"], default="macro_f1")
    ap.add_argument("--include_probs", action="store_true")
    ap.add_argument("--include_stats", action="store_true")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_json", default="")
    args = ap.parse_args()

    members = [parse_member(raw) for raw in args.member]
    cal_labels_all = []
    cal_logits_all = []
    tgt_labels_all = []
    tgt_logits_all = []
    member_reports = []
    for name, checkpoint, cal_dataset, target_dataset in members:
        cal_labels, cal_logits = flow_logits_for_checkpoint(checkpoint, cal_dataset, args.device, args.batch_size)
        tgt_labels, tgt_logits = flow_logits_for_checkpoint(checkpoint, target_dataset, args.device, args.batch_size)
        cal_fids = sorted(cal_logits)
        tgt_fids = sorted(tgt_logits)
        cal_pred = [int(cal_logits[fid].argmax()) for fid in cal_fids]
        tgt_pred = [int(tgt_logits[fid].argmax()) for fid in tgt_fids]
        member_reports.append({
            "name": name,
            "calibration_metrics": compute_metrics([cal_labels[fid] for fid in cal_fids], cal_pred),
            "target_metrics": compute_metrics([tgt_labels[fid] for fid in tgt_fids], tgt_pred),
        })
        print(name, json.dumps(member_reports[-1], sort_keys=True))
        cal_labels_all.append(cal_labels)
        cal_logits_all.append(cal_logits)
        tgt_labels_all.append(tgt_labels)
        tgt_logits_all.append(tgt_logits)

    cal_common = sorted(set.intersection(*(set(x) for x in cal_logits_all)))
    tgt_common = sorted(set.intersection(*(set(x) for x in tgt_logits_all)))
    if not cal_common or not tgt_common:
        raise ValueError("No common flow_ids across members.")

    y_cal = np.asarray([cal_labels_all[0][fid] for fid in cal_common], dtype=np.int64)
    y_tgt = np.asarray([tgt_labels_all[0][fid] for fid in tgt_common], dtype=np.int64)
    x_cal = build_feature_matrix(cal_logits_all, cal_common, args.include_probs, args.include_stats)
    x_tgt = build_feature_matrix(tgt_logits_all, tgt_common, args.include_probs, args.include_stats)

    min_class_count = int(np.bincount(y_cal).min())
    n_splits = max(2, min(5, min_class_count))
    c_values = [float(x) for x in args.c_grid.split(",") if x.strip()]
    cv_reports = []
    best = None
    for c in c_values:
        score, acc = cv_score(x_cal, y_cal, c, args.select_metric, n_splits, args.seed)
        row = {"C": c, args.select_metric: score, "accuracy": acc}
        cv_reports.append(row)
        print("cv", json.dumps(row, sort_keys=True))
        key = (score, acc)
        if best is None or key > best[0]:
            best = (key, c)
    best_c = float(best[1])

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=best_c, max_iter=5000, solver="lbfgs"),
    )
    clf.fit(x_cal, y_cal)
    y_pred = clf.predict(x_tgt).astype("int64")
    metrics = compute_metrics(y_tgt.tolist(), y_pred.tolist())
    label_names, label_map = load_label_names(args.label_map)
    print("calibrator", json.dumps(metrics, indent=2, sort_keys=True))
    if label_names:
        print(classification_report(y_tgt, y_pred, labels=list(range(len(label_names))), target_names=label_names, zero_division=0))
    else:
        print(classification_report(y_tgt, y_pred, zero_division=0))

    if args.output_json:
        payload = {
            "metrics": {"flow_level": metrics},
            "selected_C": best_c,
            "select_metric": args.select_metric,
            "cv_reports": cv_reports,
            "member_reports": member_reports,
            "label_map": label_map,
            "flow_ids": tgt_common,
            "flow_y_true": y_tgt.tolist(),
            "flow_y_pred": y_pred.tolist(),
            "feature_config": {
                "include_probs": args.include_probs,
                "include_stats": args.include_stats,
            },
        }
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()
