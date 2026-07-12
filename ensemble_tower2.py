#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support

from test_tower2 import load_label_names, load_model, predict_graph, predict_seq
from train_tower2 import apply_hierarchical_logits, build_class_to_coarse


def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred) if y_true else 0.0
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    p_weight, r_weight, f_weight, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    return {
        "accuracy": acc,
        "macro_precision": p_macro,
        "macro_recall": r_macro,
        "macro_f1": f_macro,
        "weighted_precision": p_weight,
        "weighted_recall": r_weight,
        "weighted_f1": f_weight,
    }


def flow_logits_for_checkpoint(checkpoint: str, dataset: str, device: str, batch_size: int):
    model, ckpt, flow_head = load_model(checkpoint, device)
    class_to_coarse = None
    if ckpt.get("num_coarse_classes", 0) > 0:
        class_to_coarse, _ = build_class_to_coarse(ckpt.get("coarse_groups", "vpn_app"), ckpt["num_classes"], device)

    if ckpt["model_type"] == "seq":
        y_true, _, flow_ids, logits_all, emb_all = predict_seq(model, dataset, device, batch_size)
    else:
        y_true, _, flow_ids, logits_all, emb_all = predict_graph(model, dataset, device)

    logit_buckets: Dict[str, List[np.ndarray]] = defaultdict(list)
    emb_buckets: Dict[str, List[np.ndarray]] = defaultdict(list)
    labels: Dict[str, int] = {}
    for i, (label, flow_id, logits) in enumerate(zip(y_true, flow_ids, logits_all)):
        logit_buckets[flow_id].append(logits)
        emb_buckets[flow_id].append(emb_all[i])
        labels[flow_id] = int(label)

    flow_logits: Dict[str, np.ndarray] = {}
    for flow_id, logits_list in logit_buckets.items():
        if flow_head is None:
            logits = np.mean(np.stack(logits_list, axis=0), axis=0)
        else:
            emb = torch.tensor(np.stack(emb_buckets[flow_id], axis=0), dtype=torch.float32, device=device)
            window_logits = torch.tensor(np.stack(logits_list, axis=0), dtype=torch.float32, device=device)
            pooled = flow_head(emb, window_logits=window_logits)
            logits_t = pooled["logits"]
            if ckpt.get("hierarchical_mode", "logit") != "expert":
                logits_t = apply_hierarchical_logits(
                    logits_t,
                    pooled.get("coarse_logits"),
                    class_to_coarse,
                    ckpt.get("hierarchical_logit_weight", 0.0),
                )
            logits = logits_t.detach().cpu().numpy()
        flow_logits[flow_id] = logits
    return labels, flow_logits


def parse_member(raw: List[str]) -> Tuple[str, str, str]:
    if len(raw) != 3:
        raise ValueError("--member expects NAME CHECKPOINT DATASET")
    return raw[0], raw[1], raw[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--member", nargs=3, action="append", metavar=("NAME", "CHECKPOINT", "DATASET"), required=True)
    ap.add_argument("--weights", default="", help="Optional comma-separated ensemble weights, one per --member.")
    ap.add_argument("--label_map", default="")
    ap.add_argument("--output_json", default="")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    members = [parse_member(raw) for raw in args.member]
    if args.weights:
        weights = [float(x) for x in args.weights.split(",")]
        if len(weights) != len(members):
            raise ValueError("--weights length must match --member count")
    else:
        weights = [1.0] * len(members)
    weights = np.asarray(weights, dtype="float32")
    weights = weights / max(float(weights.sum()), 1e-12)

    all_labels = []
    all_logits = []
    for name, checkpoint, dataset in members:
        labels, logits = flow_logits_for_checkpoint(checkpoint, dataset, args.device, args.batch_size)
        all_labels.append(labels)
        all_logits.append(logits)
        fids = sorted(logits)
        y_true = [labels[fid] for fid in fids]
        y_pred = [int(logits[fid].argmax()) for fid in fids]
        print(name, json.dumps(compute_metrics(y_true, y_pred), sort_keys=True))

    common_fids = sorted(set.intersection(*(set(logits) for logits in all_logits)))
    if not common_fids:
        raise ValueError("No overlapping flow_ids across ensemble members.")
    base_labels = all_labels[0]
    y_true = [base_labels[fid] for fid in common_fids]
    y_pred = []
    for fid in common_fids:
        combined = sum(float(w) * logits[fid] for w, logits in zip(weights, all_logits))
        y_pred.append(int(combined.argmax()))

    metrics = compute_metrics(y_true, y_pred)
    label_names, label_map = load_label_names(args.label_map)
    print("ensemble", json.dumps(metrics, indent=2, sort_keys=True))
    if label_names:
        print(classification_report(y_true, y_pred, labels=list(range(len(label_names))), target_names=label_names, zero_division=0))
    else:
        print(classification_report(y_true, y_pred, zero_division=0))

    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "metrics": {"flow_level": metrics},
                    "label_map": label_map,
                    "members": [{"name": m[0], "checkpoint": m[1], "dataset": m[2], "weight": float(w)} for m, w in zip(members, weights)],
                    "flow_y_true": y_true,
                    "flow_y_pred": y_pred,
                    "flow_ids": common_fids,
                },
                f,
                indent=2,
            )


if __name__ == "__main__":
    main()
