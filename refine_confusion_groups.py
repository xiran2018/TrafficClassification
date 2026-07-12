#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support

from train_flow_stats_classifier import iter_model_candidates, load_split, make_model


DEFAULT_GROUPS = [
    "social:aim,email,facebook,gmail,hangout,icq,skype,voipbuster",
    "media:netflix,spotify,vimeo,youtube",
]


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


def load_base(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    required = ["valid_flow_ids", "valid_y_true", "valid_prob", "flow_ids", "flow_y_true", "flow_prob"]
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"{path} is missing required probability fields: {missing}")
    return data


def parse_groups(raw_groups: Sequence[str], label_map: Dict[str, int]) -> List[Tuple[str, List[int]]]:
    groups = []
    for raw in raw_groups:
        if ":" not in raw:
            raise ValueError(f"Group must be NAME:label,label,... got {raw}")
        name, labels_raw = raw.split(":", 1)
        ids = []
        for label in labels_raw.split(","):
            label = label.strip()
            if not label:
                continue
            if label not in label_map:
                raise ValueError(f"Unknown label {label} in group {name}")
            ids.append(int(label_map[label]))
        if len(ids) >= 2:
            groups.append((name, sorted(set(ids))))
    return groups


def align_features(x: np.ndarray, y: np.ndarray, fids: List[str], selected_fids: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    idx = {fid: i for i, fid in enumerate(fids)}
    missing = [fid for fid in selected_fids if fid not in idx]
    if missing:
        raise ValueError(f"Missing feature rows for {len(missing)} flow ids, e.g. {missing[:3]}")
    rows = [idx[fid] for fid in selected_fids]
    return x[rows], y[rows]


def select_group_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    group_ids: List[int],
    args,
):
    train_mask = np.isin(y_train, group_ids)
    valid_mask = np.isin(y_valid, group_ids)
    if train_mask.sum() == 0 or valid_mask.sum() == 0:
        raise ValueError(f"Cannot train specialist for group ids={group_ids}; empty train/valid split.")
    best = None
    reports = []
    for kind, max_depth, min_samples_leaf, class_weight in iter_model_candidates(args.model_kinds):
        model = make_model(kind, args.n_estimators, max_depth, min_samples_leaf, class_weight, args.seed)
        model.fit(x_train[train_mask], y_train[train_mask])
        pred = model.predict(x_valid[valid_mask])
        metrics = compute_metrics(y_valid[valid_mask].tolist(), pred.tolist())
        row = {
            "kind": kind,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "class_weight": class_weight,
            "metrics": metrics,
        }
        reports.append(row)
        key = (metrics[args.select_metric], metrics["macro_f1"], metrics["accuracy"])
        if best is None or key > best[0]:
            best = (key, row)
    selected = best[1]
    model = make_model(
        selected["kind"],
        args.n_estimators,
        selected["max_depth"],
        selected["min_samples_leaf"],
        selected["class_weight"],
        args.seed,
    )
    model.fit(x_train[train_mask], y_train[train_mask])
    return model, selected, reports


def model_group_prob(model, x: np.ndarray, num_classes: int) -> np.ndarray:
    pred_prob = model.predict_proba(x)
    out = np.zeros((x.shape[0], num_classes), dtype=np.float32)
    classes = getattr(model, "classes_", None)
    if classes is None and hasattr(model, "named_steps"):
        classes = model.named_steps[list(model.named_steps.keys())[-1]].classes_
    for col, cls in enumerate(classes):
        out[:, int(cls)] = pred_prob[:, col]
    return out


def refine_probs(
    base_prob: np.ndarray,
    group_probs: List[np.ndarray],
    groups: List[Tuple[str, List[int]]],
    thresholds: Sequence[float],
    alpha: float,
) -> np.ndarray:
    out = np.asarray(base_prob, dtype=np.float64).copy()
    out = out / np.maximum(out.sum(axis=1, keepdims=True), 1e-12)
    for (_, group_ids), specialist_prob, threshold in zip(groups, group_probs, thresholds):
        group_ids_arr = np.asarray(group_ids, dtype=np.int64)
        mass = out[:, group_ids_arr].sum(axis=1)
        mask = mass >= threshold
        if not np.any(mask):
            continue
        base_group = out[:, group_ids_arr] / np.maximum(mass[:, None], 1e-12)
        spec_group = specialist_prob[:, group_ids_arr]
        spec_group = spec_group / np.maximum(spec_group.sum(axis=1, keepdims=True), 1e-12)
        blended = (1.0 - alpha) * base_group + alpha * spec_group
        out[np.ix_(mask, group_ids_arr)] = mass[mask, None] * blended[mask]
    out = out / np.maximum(out.sum(axis=1, keepdims=True), 1e-12)
    return out.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_json", required=True)
    ap.add_argument("--train_index", required=True)
    ap.add_argument("--valid_index", required=True)
    ap.add_argument("--test_index", required=True)
    ap.add_argument("--label_map", required=True)
    ap.add_argument("--group", action="append", default=None, help="NAME:label,label,... Defaults to social and media groups.")
    ap.add_argument("--model_kinds", default="extra_trees")
    ap.add_argument("--n_estimators", type=int, default=500)
    ap.add_argument("--max_packets", type=int, default=64)
    ap.add_argument("--prefix_len", type=int, default=32)
    ap.add_argument("--feature_version", choices=["basic", "message", "message_header", "message_header_endpoint", "message_header_fullbytes"], default="message")
    ap.add_argument("--use_ports", action="store_true")
    ap.add_argument("--threshold_grid", default="0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    ap.add_argument("--alpha_grid", default="0.25,0.5,0.75,1.0")
    ap.add_argument("--select_metric", choices=["accuracy", "macro_f1"], default="accuracy")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_json", default="")
    args = ap.parse_args()

    label_names, label_map = load_label_names(args.label_map)
    if not label_map:
        raise ValueError("--label_map is required and must be readable.")
    groups = parse_groups(args.group or DEFAULT_GROUPS, label_map)
    if not groups:
        raise ValueError("No valid confusion groups were provided.")

    base = load_base(args.base_json)
    valid_fids = [str(fid) for fid in base["valid_flow_ids"]]
    test_fids = [str(fid) for fid in base["flow_ids"]]
    y_valid = np.asarray(base["valid_y_true"], dtype=np.int64)
    y_test = np.asarray(base["flow_y_true"], dtype=np.int64)
    valid_prob = np.asarray(base["valid_prob"], dtype=np.float32)
    test_prob = np.asarray(base["flow_prob"], dtype=np.float32)
    num_classes = test_prob.shape[1]

    x_train, y_train, _ = load_split(args.train_index, args.max_packets, args.prefix_len, args.use_ports, args.feature_version)
    x_valid_all, y_valid_all, valid_feature_fids = load_split(args.valid_index, args.max_packets, args.prefix_len, args.use_ports, args.feature_version)
    x_test_all, y_test_all, test_feature_fids = load_split(args.test_index, args.max_packets, args.prefix_len, args.use_ports, args.feature_version)
    x_valid, y_valid_aligned = align_features(x_valid_all, y_valid_all, valid_feature_fids, valid_fids)
    x_test, y_test_aligned = align_features(x_test_all, y_test_all, test_feature_fids, test_fids)
    if not np.array_equal(y_valid, y_valid_aligned) or not np.array_equal(y_test, y_test_aligned):
        raise ValueError("Base probabilities and flow feature labels are not aligned.")

    valid_group_probs = []
    test_group_probs = []
    group_reports = []
    for name, group_ids in groups:
        train_model, selected, reports = select_group_model(x_train, y_train, x_valid, y_valid, group_ids, args)
        valid_group_probs.append(model_group_prob(train_model, x_valid, num_classes))

        final_model = make_model(
            selected["kind"],
            max(args.n_estimators, 800),
            selected["max_depth"],
            selected["min_samples_leaf"],
            selected["class_weight"],
            args.seed,
        )
        final_mask = np.isin(np.concatenate([y_train, y_valid]), group_ids)
        final_model.fit(
            np.concatenate([x_train, x_valid], axis=0)[final_mask],
            np.concatenate([y_train, y_valid], axis=0)[final_mask],
        )
        test_group_probs.append(model_group_prob(final_model, x_test, num_classes))
        group_reports.append({"name": name, "group_ids": group_ids, "selected": selected, "reports": reports})
        print("specialist", json.dumps({"name": name, "selected": selected}, sort_keys=True))

    thresholds_values = [float(x) for x in args.threshold_grid.split(",") if x.strip()]
    alpha_values = [float(x) for x in args.alpha_grid.split(",") if x.strip()]
    threshold_products = list(product(thresholds_values, repeat=len(groups)))
    best = None
    reports = []
    for alpha in alpha_values:
        for thresholds in threshold_products:
            refined = refine_probs(valid_prob, valid_group_probs, groups, thresholds, alpha)
            pred = refined.argmax(axis=1).astype(np.int64)
            metrics = compute_metrics(y_valid.tolist(), pred.tolist())
            row = {
                "alpha": alpha,
                "thresholds": {name: float(t) for (name, _), t in zip(groups, thresholds)},
                "metrics": metrics,
            }
            reports.append(row)
            key = (metrics[args.select_metric], metrics["macro_f1"], metrics["accuracy"])
            if best is None or key > best[0]:
                best = (key, row, refined)
    selected = best[1]
    selected_thresholds = [selected["thresholds"][name] for name, _ in groups]
    refined_test = refine_probs(test_prob, test_group_probs, groups, selected_thresholds, selected["alpha"])
    y_pred = refined_test.argmax(axis=1).astype(np.int64)
    metrics = compute_metrics(y_test.tolist(), y_pred.tolist())
    print("selected_refinement", json.dumps(selected, sort_keys=True))
    print("test_refinement", json.dumps(metrics, indent=2, sort_keys=True))
    if label_names:
        print(classification_report(y_test, y_pred, labels=list(range(len(label_names))), target_names=label_names, zero_division=0))
    else:
        print(classification_report(y_test, y_pred, zero_division=0))

    if args.output_json:
        payload = {
            "metrics": {"flow_level": metrics},
            "selected_refinement": selected,
            "refinement_reports": reports,
            "group_reports": group_reports,
            "base_json": args.base_json,
            "label_map": label_map,
            "flow_ids": test_fids,
            "flow_y_true": y_test.tolist(),
            "flow_y_pred": y_pred.tolist(),
            "flow_prob": refined_test.tolist(),
            "valid_flow_ids": valid_fids,
            "valid_y_true": y_valid.tolist(),
            "valid_y_pred": best[2].argmax(axis=1).astype(np.int64).tolist(),
            "valid_prob": best[2].tolist(),
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
