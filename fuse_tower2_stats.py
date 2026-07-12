#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support

from ensemble_tower2 import flow_logits_for_checkpoint
from test_tower2 import load_label_names
from train_flow_stats_classifier import iter_model_candidates, load_split, make_model


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


def select_stats_model(x_train, y_train, x_valid, y_valid, args):
    reports = []
    best = None
    for kind, max_depth, min_samples_leaf, class_weight in iter_model_candidates(args.model_kinds):
        model = make_model(kind, args.n_estimators, max_depth, min_samples_leaf, class_weight, args.seed)
        model.fit(x_train, y_train)
        pred = model.predict(x_valid)
        metrics = compute_metrics(y_valid.tolist(), pred.tolist())
        row = {
            "kind": kind,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "class_weight": class_weight,
            "metrics": metrics,
        }
        reports.append(row)
        key = (metrics[args.select_metric], metrics["accuracy"])
        if best is None or key > best[0]:
            best = (key, row)
    return best[1], reports


def proba_for_fids(model, x, fids, selected_fids, num_classes: int) -> np.ndarray:
    idx = {fid: i for i, fid in enumerate(fids)}
    probs = model.predict_proba(x)
    aligned = np.zeros((len(selected_fids), num_classes), dtype=np.float32)
    classes = getattr(model, "classes_", None)
    if classes is None and hasattr(model, "named_steps"):
        classes = model.named_steps[list(model.named_steps.keys())[-1]].classes_
    for col, cls in enumerate(classes):
        aligned[:, int(cls)] = probs[[idx[fid] for fid in selected_fids], col]
    return aligned


def tower_probs(checkpoint: str, dataset: str, fids, device: str, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    labels, logits = flow_logits_for_checkpoint(checkpoint, dataset, device, batch_size)
    y = np.asarray([labels[fid] for fid in fids], dtype=np.int64)
    logit_arr = np.stack([logits[fid] for fid in fids], axis=0)
    return y, softmax_np(logit_arr)


def parse_tower_members(args):
    if args.tower_member:
        return [(raw[0], raw[1], raw[2], raw[3]) for raw in args.tower_member]
    if not args.tower_checkpoint or not args.valid_tower2_dataset or not args.test_tower2_dataset:
        raise ValueError("Provide either --tower_member or the legacy --tower_checkpoint/--valid_tower2_dataset/--test_tower2_dataset trio.")
    return [("tower", args.tower_checkpoint, args.valid_tower2_dataset, args.test_tower2_dataset)]


def simplex_weights(n: int, step: float):
    if n == 2:
        # Preserve exact legacy grid behavior for stats + one tower.
        return None
    units = int(round(1.0 / step))
    if units <= 0 or abs(units * step - 1.0) > 1e-6:
        raise ValueError("--simplex_step must divide 1.0, e.g. 0.1 or 0.05")
    for cuts in product(range(units + 1), repeat=n - 1):
        used = sum(cuts)
        if used > units:
            continue
        weights = list(cuts) + [units - used]
        yield np.asarray(weights, dtype=np.float32) / float(units)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tower_checkpoint", default="", help="Legacy single Tower2 checkpoint input.")
    ap.add_argument("--valid_tower2_dataset", default="", help="Legacy single Tower2 validation dataset input.")
    ap.add_argument("--test_tower2_dataset", default="", help="Legacy single Tower2 test dataset input.")
    ap.add_argument(
        "--tower_member",
        nargs=4,
        action="append",
        metavar=("NAME", "CHECKPOINT", "VALID_DATASET", "TEST_DATASET"),
        help="Add one Tower2 member for multi-model stats fusion.",
    )
    ap.add_argument("--train_index", required=True)
    ap.add_argument("--valid_index", required=True)
    ap.add_argument("--test_index", required=True)
    ap.add_argument("--label_map", default="")
    ap.add_argument("--model_kinds", default="extra_trees")
    ap.add_argument("--n_estimators", type=int, default=500)
    ap.add_argument("--max_packets", type=int, default=64)
    ap.add_argument("--prefix_len", type=int, default=32)
    ap.add_argument("--use_ports", action="store_true")
    ap.add_argument("--feature_version", choices=["basic", "message", "message_header", "message_header_endpoint", "message_header_fullbytes"], default="basic")
    ap.add_argument("--select_metric", choices=["accuracy", "macro_f1"], default="macro_f1")
    ap.add_argument("--weight_grid", default="0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1")
    ap.add_argument("--simplex_step", type=float, default=0.05, help="Weight grid step for stats + multiple Tower2 members.")
    ap.add_argument("--print_all_fusion", action="store_true", help="Print every validation fusion candidate.")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_json", default="")
    args = ap.parse_args()
    tower_members = parse_tower_members(args)

    x_train, y_train, _ = load_split(args.train_index, args.max_packets, args.prefix_len, args.use_ports, args.feature_version)
    x_valid, y_valid, valid_fids = load_split(args.valid_index, args.max_packets, args.prefix_len, args.use_ports, args.feature_version)
    x_test, y_test, test_fids = load_split(args.test_index, args.max_packets, args.prefix_len, args.use_ports, args.feature_version)
    num_classes = int(max(y_train.max(), y_valid.max(), y_test.max()) + 1)

    selected, stats_reports = select_stats_model(x_train, y_train, x_valid, y_valid, args)
    print("selected_stats", json.dumps(selected, sort_keys=True))
    stats_model = make_model(
        selected["kind"],
        args.n_estimators,
        selected["max_depth"],
        selected["min_samples_leaf"],
        selected["class_weight"],
        args.seed,
    )
    stats_model.fit(x_train, y_train)

    valid_labels_by_fid = {fid: int(label) for fid, label in zip(valid_fids, y_valid)}
    valid_common = sorted(set(valid_fids))
    valid_tower_probs = []
    y_valid_ref = None
    for name, checkpoint, valid_dataset, _ in tower_members:
        y_member, prob_member = tower_probs(checkpoint, valid_dataset, valid_common, args.device, args.batch_size)
        valid_tower_probs.append(prob_member)
        if y_valid_ref is None:
            y_valid_ref = y_member
        elif not np.array_equal(y_valid_ref, y_member):
            raise ValueError(f"Validation labels do not align for Tower2 member {name}.")
    valid_stats_prob = proba_for_fids(stats_model, x_valid, valid_fids, valid_common, num_classes)
    if not np.array_equal(y_valid_ref, np.asarray([valid_labels_by_fid[fid] for fid in valid_common], dtype=np.int64)):
        raise ValueError("Validation labels do not align between Tower2 and stats features.")

    fusion_reports = []
    best = None
    weight_candidates = []
    if len(tower_members) == 1:
        for raw_w in args.weight_grid.split(","):
            tower_weight = float(raw_w)
            weight_candidates.append(np.asarray([1.0 - tower_weight, tower_weight], dtype=np.float32))
    else:
        weight_candidates.extend(simplex_weights(len(tower_members) + 1, args.simplex_step))

    valid_components = [valid_stats_prob] + valid_tower_probs
    selected_valid_prob = None
    for weights in weight_candidates:
        prob = sum(float(w) * component for w, component in zip(weights, valid_components))
        pred = prob.argmax(axis=1)
        metrics = compute_metrics(y_valid_ref.tolist(), pred.tolist())
        row = {
            "weights": {"stats": float(weights[0]), **{name: float(weights[i + 1]) for i, (name, *_rest) in enumerate(tower_members)}},
            "metrics": metrics,
        }
        fusion_reports.append(row)
        if args.print_all_fusion:
            print("valid_fusion", json.dumps(row, sort_keys=True))
        key = (metrics[args.select_metric], metrics["accuracy"])
        if best is None or key > best[0]:
            best = (key, row)
            selected_valid_prob = prob
    selected_weights = best[1]["weights"]
    print("selected_fusion", json.dumps(best[1], sort_keys=True))

    final_stats_model = make_model(
        selected["kind"],
        max(args.n_estimators, 800),
        selected["max_depth"],
        selected["min_samples_leaf"],
        selected["class_weight"],
        args.seed,
    )
    final_stats_model.fit(np.concatenate([x_train, x_valid], axis=0), np.concatenate([y_train, y_valid], axis=0))
    test_common = sorted(set(test_fids))
    test_tower_probs = []
    y_test_ref = None
    for name, checkpoint, _, test_dataset in tower_members:
        y_member, prob_member = tower_probs(checkpoint, test_dataset, test_common, args.device, args.batch_size)
        test_tower_probs.append(prob_member)
        if y_test_ref is None:
            y_test_ref = y_member
        elif not np.array_equal(y_test_ref, y_member):
            raise ValueError(f"Test labels do not align for Tower2 member {name}.")
    test_stats_prob = proba_for_fids(final_stats_model, x_test, test_fids, test_common, num_classes)
    test_components = [test_stats_prob] + test_tower_probs
    weights_arr = np.asarray([selected_weights["stats"]] + [selected_weights[name] for name, *_rest in tower_members], dtype=np.float32)
    prob = sum(float(w) * component for w, component in zip(weights_arr, test_components))
    y_pred = prob.argmax(axis=1).astype("int64")
    metrics = compute_metrics(y_test_ref.tolist(), y_pred.tolist())
    print("test_fusion", json.dumps(metrics, indent=2, sort_keys=True))

    label_names, label_map = load_label_names(args.label_map)
    if label_names:
        print(classification_report(y_test_ref, y_pred, labels=list(range(len(label_names))), target_names=label_names, zero_division=0))
    else:
        print(classification_report(y_test_ref, y_pred, zero_division=0))

    if args.output_json:
        payload = {
            "metrics": {"flow_level": metrics},
            "selected_stats": selected,
            "selected_weights": selected_weights,
            "tower_members": [{"name": name, "checkpoint": checkpoint, "valid_dataset": valid_dataset, "test_dataset": test_dataset} for name, checkpoint, valid_dataset, test_dataset in tower_members],
            "stats_reports": stats_reports,
            "fusion_reports": fusion_reports,
            "label_map": label_map,
            "flow_ids": test_common,
            "flow_y_true": y_test_ref.tolist(),
            "flow_y_pred": y_pred.tolist(),
            "flow_prob": prob.tolist(),
            "valid_flow_ids": valid_common,
            "valid_y_true": y_valid_ref.tolist(),
            "valid_prob": selected_valid_prob.tolist(),
            "feature_config": {
                "max_packets": args.max_packets,
                "prefix_len": args.prefix_len,
                "use_ports": args.use_ports,
                "feature_version": args.feature_version,
            },
        }
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()
