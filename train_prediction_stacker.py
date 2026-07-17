#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from probability_metrics import calibration_metrics
from validation_gated_selector import apply_unified_expert_slots, parse_name_list, target_shift_guard


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


def load_prob_payload(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    required = ["valid_flow_ids", "valid_y_true", "valid_prob", "flow_ids", "flow_y_true", "flow_prob"]
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"{path} is missing required probability fields: {missing}")
    return data


def normalize_prob(prob: np.ndarray) -> np.ndarray:
    prob = np.asarray(prob, dtype=np.float32)
    prob = np.clip(prob, 1e-12, None)
    return prob / prob.sum(axis=1, keepdims=True)


def align_prob(data: Dict[str, Any], split: str, fids: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    if split == "valid":
        ids = data["valid_flow_ids"]
        labels = data["valid_y_true"]
        probs = data["valid_prob"]
    elif split == "test":
        ids = data["flow_ids"]
        labels = data["flow_y_true"]
        probs = data["flow_prob"]
    else:
        raise ValueError(split)
    idx = {str(fid): i for i, fid in enumerate(ids)}
    y = np.asarray([labels[idx[fid]] for fid in fids], dtype=np.int64)
    p = normalize_prob(np.asarray([probs[idx[fid]] for fid in fids], dtype=np.float32))
    return y, p


def prob_features(prob_list: List[np.ndarray], include_logits: bool, include_confidence: bool) -> np.ndarray:
    parts = []
    for probs in prob_list:
        probs = np.asarray(probs, dtype=np.float32)
        probs = probs / probs.sum(axis=1, keepdims=True).clip(min=1e-12)
        parts.append(probs)
        if include_logits:
            clipped = probs.clip(min=1e-6, max=1.0)
            parts.append(np.log(clipped).astype(np.float32))
        if include_confidence:
            sorted_probs = np.sort(probs, axis=1)
            conf = sorted_probs[:, -1:]
            margin = (sorted_probs[:, -1:] - sorted_probs[:, -2:-1]) if probs.shape[1] > 1 else conf
            entropy = -np.sum(probs * np.log(probs.clip(min=1e-12)), axis=1, keepdims=True)
            parts.append(np.concatenate([conf, margin, entropy], axis=1).astype(np.float32))
    return np.concatenate(parts, axis=1)


def make_clf(c: float, class_weight: str | None, seed: int):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=c, max_iter=5000, solver="lbfgs", class_weight=class_weight, random_state=seed),
    )


def oof_predict_proba(x: np.ndarray, y: np.ndarray, c: float, class_weight: str | None, n_splits: int, seed: int, num_classes: int) -> np.ndarray:
    out = np.zeros((len(y), num_classes), dtype=np.float32)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_idx, val_idx in skf.split(x, y):
        clf = make_clf(c, class_weight, seed)
        clf.fit(x[train_idx], y[train_idx])
        fold_prob = clf.predict_proba(x[val_idx])
        classes = clf.named_steps["logisticregression"].classes_
        for col, cls in enumerate(classes):
            out[val_idx, int(cls)] = fold_prob[:, col]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", nargs=2, action="append", metavar=("NAME", "JSON"), required=True)
    ap.add_argument("--label_map", default="")
    ap.add_argument("--c_grid", default="0.01,0.03,0.1,0.3,1,3,10")
    ap.add_argument("--class_weight_grid", default="none,balanced")
    ap.add_argument("--select_metric", choices=["accuracy", "macro_f1"], default="accuracy")
    ap.add_argument("--include_logits", action="store_true")
    ap.add_argument("--include_confidence", action="store_true")
    ap.add_argument("--base_input", default="", help="Input name used as the fallback/base expert. Defaults to the first input after slot alignment.")
    ap.add_argument("--min_valid_gain_over_base", type=float, default=0.0, help="Use stacker only if selected validation metric improves over base by at least this margin.")
    ap.add_argument("--max_prediction_change_rate", type=float, default=1.0, help="Use stacker only if test prediction change rate versus base is no larger than this value.")
    ap.add_argument("--max_prediction_js_divergence", type=float, default=1.0, help="Use stacker only if test prediction-distribution JS divergence versus base is no larger than this value.")
    ap.add_argument(
        "--unified_expert_slots",
        default="",
        help="Comma-separated expert slots for a fixed cross-dataset stacker input. Missing slots are base identity experts.",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_json", default="")
    args = ap.parse_args()

    named_payloads = [(name, load_prob_payload(path), path) for name, path in args.input]
    unified_expert_slots = parse_name_list(args.unified_expert_slots)
    named_payloads, input_slot_status = apply_unified_expert_slots(named_payloads, unified_expert_slots)
    input_names = [name for name, _, _ in named_payloads]
    if args.base_input:
        if args.base_input not in input_names:
            raise ValueError(f"--base_input={args.base_input} is not one of stacker inputs after slot alignment: {input_names}")
        base_index = input_names.index(args.base_input)
    else:
        base_index = 0
    valid_common = sorted(set.intersection(*(set(map(str, data["valid_flow_ids"])) for _, data, _ in named_payloads)))
    test_common = sorted(set.intersection(*(set(map(str, data["flow_ids"])) for _, data, _ in named_payloads)))
    if not valid_common or not test_common:
        raise ValueError("No common flow ids across stacker inputs.")

    valid_probs = []
    test_probs = []
    y_valid_ref = None
    y_test_ref = None
    for name, data, _ in named_payloads:
        y_valid, p_valid = align_prob(data, "valid", valid_common)
        y_test, p_test = align_prob(data, "test", test_common)
        if y_valid_ref is None:
            y_valid_ref = y_valid
            y_test_ref = y_test
        elif not np.array_equal(y_valid_ref, y_valid) or not np.array_equal(y_test_ref, y_test):
            raise ValueError(f"Labels do not align for input {name}.")
        valid_probs.append(p_valid)
        test_probs.append(p_test)

    x_valid = prob_features(valid_probs, args.include_logits, args.include_confidence)
    x_test = prob_features(test_probs, args.include_logits, args.include_confidence)
    base_valid_prob = valid_probs[base_index]
    base_test_prob = test_probs[base_index]
    base_valid_pred = base_valid_prob.argmax(axis=1).astype(np.int64)
    base_test_pred = base_test_prob.argmax(axis=1).astype(np.int64)
    base_valid_metrics = compute_metrics(y_valid_ref.tolist(), base_valid_pred.tolist())
    base_valid_metrics["calibration"] = calibration_metrics(y_valid_ref, base_valid_prob)
    base_test_metrics = compute_metrics(y_test_ref.tolist(), base_test_pred.tolist())
    base_test_metrics["calibration"] = calibration_metrics(y_test_ref, base_test_prob)
    num_classes = int(max(y_valid_ref.max(), y_test_ref.max()) + 1)
    min_class_count = int(np.bincount(y_valid_ref, minlength=num_classes).min())
    n_splits = max(2, min(5, min_class_count))

    c_values = [float(x) for x in args.c_grid.split(",") if x.strip()]
    class_weights = [None if x.strip() == "none" else x.strip() for x in args.class_weight_grid.split(",") if x.strip()]
    reports = []
    best = None
    for c in c_values:
        for class_weight in class_weights:
            valid_prob = oof_predict_proba(x_valid, y_valid_ref, c, class_weight, n_splits, args.seed, num_classes)
            valid_pred = valid_prob.argmax(axis=1)
            metrics = compute_metrics(y_valid_ref.tolist(), valid_pred.tolist())
            metrics["calibration"] = calibration_metrics(y_valid_ref, valid_prob)
            row = {"C": c, "class_weight": class_weight, "metrics": metrics}
            reports.append(row)
            print("valid_stacker", json.dumps(row, sort_keys=True))
            key = (metrics[args.select_metric], metrics["macro_f1"], metrics["accuracy"])
            if best is None or key > best[0]:
                best = (key, row, valid_prob)

    selected = best[1]
    selected_valid_prob = best[2]
    clf = make_clf(selected["C"], selected["class_weight"], args.seed)
    clf.fit(x_valid, y_valid_ref)
    test_prob_raw = clf.predict_proba(x_test)
    test_prob = np.zeros((len(y_test_ref), num_classes), dtype=np.float32)
    classes = clf.named_steps["logisticregression"].classes_
    for col, cls in enumerate(classes):
        test_prob[:, int(cls)] = test_prob_raw[:, col]
    y_pred = test_prob.argmax(axis=1).astype(np.int64)
    candidate_metrics = compute_metrics(y_test_ref.tolist(), y_pred.tolist())
    candidate_metrics["calibration"] = calibration_metrics(y_test_ref, test_prob)
    selected_valid_pred = selected_valid_prob.argmax(axis=1).astype(np.int64)
    selected_valid_metric = float(selected["metrics"][args.select_metric])
    base_valid_metric = float(base_valid_metrics[args.select_metric])
    valid_gain_over_base = selected_valid_metric - base_valid_metric
    valid_shift = target_shift_guard(selected_valid_prob, base_valid_prob)
    test_shift = target_shift_guard(test_prob, base_test_prob)
    safety_reasons = []
    if valid_gain_over_base < args.min_valid_gain_over_base:
        safety_reasons.append(
            f"valid_gain_over_base={valid_gain_over_base:.6f}<min_valid_gain_over_base={args.min_valid_gain_over_base:.6f}"
        )
    if test_shift["prediction_change_rate"] > args.max_prediction_change_rate:
        safety_reasons.append(
            f"test_prediction_change_rate={test_shift['prediction_change_rate']:.6f}>max_prediction_change_rate={args.max_prediction_change_rate:.6f}"
        )
    if test_shift["prediction_js_divergence"] > args.max_prediction_js_divergence:
        safety_reasons.append(
            f"test_prediction_js_divergence={test_shift['prediction_js_divergence']:.6f}>max_prediction_js_divergence={args.max_prediction_js_divergence:.6f}"
        )
    use_stacker = not safety_reasons
    if use_stacker:
        final_prob = test_prob
        final_valid_prob = selected_valid_prob
        final_pred = y_pred
        final_valid_pred = selected_valid_pred
        metrics = candidate_metrics
    else:
        final_prob = base_test_prob
        final_valid_prob = base_valid_prob
        final_pred = base_test_pred
        final_valid_pred = base_valid_pred
        metrics = base_test_metrics
    safety = {
        "enabled": True,
        "use_stacker": bool(use_stacker),
        "fallback_input": input_names[base_index],
        "reasons": safety_reasons,
        "select_metric": args.select_metric,
        "selected_valid_metric": selected_valid_metric,
        "base_valid_metric": base_valid_metric,
        "valid_gain_over_base": valid_gain_over_base,
        "valid_shift_vs_base": valid_shift,
        "test_shift_vs_base": test_shift,
        "min_valid_gain_over_base": args.min_valid_gain_over_base,
        "max_prediction_change_rate": args.max_prediction_change_rate,
        "max_prediction_js_divergence": args.max_prediction_js_divergence,
    }
    print("selected_stacker", json.dumps(selected, sort_keys=True))
    print("base_stacker", json.dumps({"name": input_names[base_index], "valid_metrics": base_valid_metrics, "test_metrics": base_test_metrics}, sort_keys=True))
    print("candidate_stacker", json.dumps(candidate_metrics, indent=2, sort_keys=True))
    print("stacker_safety", json.dumps(safety, indent=2, sort_keys=True))
    print("test_stacker", json.dumps(metrics, indent=2, sort_keys=True))

    label_names, label_map = load_label_names(args.label_map)
    if label_names:
        print(classification_report(y_test_ref, final_pred, labels=list(range(len(label_names))), target_names=label_names, zero_division=0))
    else:
        print(classification_report(y_test_ref, final_pred, zero_division=0))

    if args.output_json:
        payload = {
            "metrics": {"flow_level": metrics},
            "selected": selected,
            "safety": safety,
            "candidate_metrics": {"flow_level": candidate_metrics},
            "base_metrics": {"valid": base_valid_metrics, "flow_level": base_test_metrics},
            "valid_reports": reports,
            "inputs": [{"name": name, "path": path} for name, _, path in named_payloads],
            "label_map": label_map,
            "flow_ids": test_common,
            "flow_y_true": y_test_ref.tolist(),
            "flow_y_pred": final_pred.tolist(),
            "flow_prob": final_prob.tolist(),
            "valid_flow_ids": valid_common,
            "valid_y_true": y_valid_ref.tolist(),
            "valid_y_pred": final_valid_pred.tolist(),
            "valid_prob": final_valid_prob.tolist(),
            "candidate_flow_y_pred": y_pred.tolist(),
            "candidate_flow_prob": test_prob.tolist(),
            "candidate_valid_y_pred": selected_valid_pred.tolist(),
            "candidate_valid_prob": selected_valid_prob.tolist(),
            "feature_config": {
                "include_logits": args.include_logits,
                "include_confidence": args.include_confidence,
                "select_metric": args.select_metric,
                "unified_expert_slots": unified_expert_slots,
                "input_slot_status": input_slot_status,
                "base_input": input_names[base_index],
            },
        }
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()
