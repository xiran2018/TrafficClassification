#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support

from train_flow_stats_classifier import iter_model_candidates, load_split, make_model


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


def align_features(x: np.ndarray, y: np.ndarray, fids: List[str], selected_fids: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    idx = {str(fid): i for i, fid in enumerate(fids)}
    rows = [idx[str(fid)] for fid in selected_fids]
    return x[rows], y[rows]


def pair_key(a: int, b: int) -> Tuple[int, int]:
    return (a, b) if a < b else (b, a)


def top2_info(prob: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(prob, axis=1)
    top1 = order[:, -1]
    top2 = order[:, -2]
    margin = prob[np.arange(prob.shape[0]), top1] - prob[np.arange(prob.shape[0]), top2]
    return top1.astype(np.int64), top2.astype(np.int64), margin.astype(np.float64)


def select_pair_model(x_train, y_train, x_valid, y_valid, pair: Tuple[int, int], args):
    train_mask = np.isin(y_train, pair)
    valid_mask = np.isin(y_valid, pair)
    if train_mask.sum() < args.min_pair_train or valid_mask.sum() < args.min_pair_valid:
        return None, None, []
    if len(set(y_train[train_mask].tolist())) < 2 or len(set(y_valid[valid_mask].tolist())) < 2:
        return None, None, []
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
        key = (metrics[args.pair_select_metric], metrics["macro_f1"], metrics["accuracy"])
        if best is None or key > best[0]:
            best = (key, row)
    selected = best[1]
    model = make_model(selected["kind"], args.n_estimators, selected["max_depth"], selected["min_samples_leaf"], selected["class_weight"], args.seed)
    model.fit(x_train[train_mask], y_train[train_mask])
    return model, selected, reports


def aligned_pair_prob(model, x: np.ndarray, pair: Tuple[int, int]) -> np.ndarray:
    raw = model.predict_proba(x)
    classes = getattr(model, "classes_", None)
    if classes is None and hasattr(model, "named_steps"):
        classes = model.named_steps[list(model.named_steps.keys())[-1]].classes_
    out = np.zeros((x.shape[0], 2), dtype=np.float64)
    for col, cls in enumerate(classes):
        if int(cls) == pair[0]:
            out[:, 0] = raw[:, col]
        elif int(cls) == pair[1]:
            out[:, 1] = raw[:, col]
    out = out / np.maximum(out.sum(axis=1, keepdims=True), 1e-12)
    return out


def refine(
    base_prob: np.ndarray,
    x: np.ndarray,
    pair_models: Dict[Tuple[int, int], Any],
    margin_threshold: float,
    alpha: float,
    min_pair_mass: float,
    apply_mode: str = "top2",
) -> Tuple[np.ndarray, int]:
    out = np.asarray(base_prob, dtype=np.float64).copy()
    out = out / np.maximum(out.sum(axis=1, keepdims=True), 1e-12)
    top1, top2, margin = top2_info(out)
    changed = 0
    for pair, model in pair_models.items():
        pair_mass = out[:, pair[0]] + out[:, pair[1]]
        if apply_mode == "top2":
            mask = np.array([pair_key(int(a), int(b)) == pair for a, b in zip(top1, top2)], dtype=bool)
            mask &= margin <= margin_threshold
            mask &= pair_mass >= min_pair_mass
        elif apply_mode == "pair_mass":
            pair_margin = np.abs(out[:, pair[0]] - out[:, pair[1]]) / np.maximum(pair_mass, 1e-12)
            mask = pair_mass >= min_pair_mass
            mask &= pair_margin <= margin_threshold
        else:
            raise ValueError(f"Unknown apply_mode: {apply_mode}")
        if not np.any(mask):
            continue
        spec = aligned_pair_prob(model, x[mask], pair)
        base_pair = out[np.ix_(mask, pair)]
        base_pair = base_pair / np.maximum(base_pair.sum(axis=1, keepdims=True), 1e-12)
        blended = (1.0 - alpha) * base_pair + alpha * spec
        out[np.ix_(mask, pair)] = pair_mass[mask, None] * blended
        changed += int(mask.sum())
    out = out / np.maximum(out.sum(axis=1, keepdims=True), 1e-12)
    return out.astype(np.float32), changed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_json", required=True)
    ap.add_argument("--train_index", required=True)
    ap.add_argument("--valid_index", required=True)
    ap.add_argument("--test_index", required=True)
    ap.add_argument("--label_map", required=True)
    ap.add_argument("--model_kinds", default="extra_trees")
    ap.add_argument("--n_estimators", type=int, default=500)
    ap.add_argument("--max_packets", type=int, default=64)
    ap.add_argument("--prefix_len", type=int, default=32)
    ap.add_argument("--feature_version", choices=["basic", "message", "message_header", "message_header_endpoint", "message_header_fullbytes"], default="message_header")
    ap.add_argument("--use_ports", action="store_true")
    ap.add_argument("--margin_grid", default="0.02,0.04,0.06,0.08,0.1,0.15,0.2,0.3,0.4,0.5")
    ap.add_argument("--alpha_grid", default="0.25,0.5,0.75,1.0")
    ap.add_argument("--min_pair_mass_grid", default="0.4,0.5,0.6,0.7,0.8")
    ap.add_argument("--pair_source", choices=["top2", "valid_errors", "top2_valid_errors"], default="top2")
    ap.add_argument("--pair", action="append", default=None, help="Explicit class pair A-B to include, e.g. 2-5.")
    ap.add_argument("--apply_mode", choices=["top2", "pair_mass"], default="top2")
    ap.add_argument("--select_metric", choices=["accuracy", "macro_f1"], default="accuracy")
    ap.add_argument("--pair_select_metric", choices=["accuracy", "macro_f1"], default="accuracy")
    ap.add_argument("--min_pair_train", type=int, default=16)
    ap.add_argument("--min_pair_valid", type=int, default=6)
    ap.add_argument("--max_pairs", type=int, default=20, help="Train only the most frequent validation top-2 pairs. 0 keeps all candidate pairs.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_json", default="")
    args = ap.parse_args()

    label_names, label_map = load_label_names(args.label_map)
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

    valid_top1, valid_top2, _ = top2_info(valid_prob)
    pair_counts: Dict[Tuple[int, int], int] = {}
    if args.pair_source in {"top2", "top2_valid_errors"}:
        for a, b in zip(valid_top1, valid_top2):
            if int(a) == int(b):
                continue
            key = pair_key(int(a), int(b))
            pair_counts[key] = pair_counts.get(key, 0) + 1
    if args.pair_source in {"valid_errors", "top2_valid_errors"}:
        for y, pred in zip(y_valid, valid_top1):
            if int(y) == int(pred):
                continue
            key = pair_key(int(y), int(pred))
            pair_counts[key] = pair_counts.get(key, 0) + 1
    for raw_pair in args.pair or []:
        left, right = raw_pair.replace(":", "-").split("-", 1)
        key = pair_key(int(left), int(right))
        pair_counts[key] = pair_counts.get(key, 0) + 1
    candidate_pairs = [pair for pair, _ in sorted(pair_counts.items(), key=lambda item: (-item[1], item[0]))]
    if args.max_pairs > 0:
        candidate_pairs = candidate_pairs[: args.max_pairs]
    print("candidate_pairs", json.dumps({"count": len(candidate_pairs), "pairs": [f"{a}-{b}:{pair_counts[(a, b)]}" for a, b in candidate_pairs]}, sort_keys=True))
    pair_models = {}
    pair_reports = {}
    for pair in candidate_pairs:
        model, selected, reports = select_pair_model(x_train, y_train, x_valid, y_valid, pair, args)
        if model is None:
            continue
        pair_models[pair] = model
        pair_reports[f"{pair[0]}-{pair[1]}"] = {"selected": selected, "reports": reports}
    print("trained_pairs", json.dumps({"count": len(pair_models), "pairs": list(pair_reports.keys())}, sort_keys=True))

    best = None
    reports = []
    for alpha in [float(x) for x in args.alpha_grid.split(",") if x.strip()]:
        for margin_threshold in [float(x) for x in args.margin_grid.split(",") if x.strip()]:
            for min_pair_mass in [float(x) for x in args.min_pair_mass_grid.split(",") if x.strip()]:
                refined_valid, changed = refine(valid_prob, x_valid, pair_models, margin_threshold, alpha, min_pair_mass, args.apply_mode)
                pred = refined_valid.argmax(axis=1).astype(np.int64)
                metrics = compute_metrics(y_valid.tolist(), pred.tolist())
                row = {
                    "alpha": alpha,
                    "margin_threshold": margin_threshold,
                    "min_pair_mass": min_pair_mass,
                    "changed_valid": changed,
                    "metrics": metrics,
                }
                reports.append(row)
                key = (metrics[args.select_metric], metrics["macro_f1"], metrics["accuracy"])
                if best is None or key > best[0]:
                    best = (key, row, refined_valid)

    selected = best[1]
    final_pair_models = {}
    x_final = np.concatenate([x_train, x_valid], axis=0)
    y_final = np.concatenate([y_train, y_valid], axis=0)
    for pair in pair_models:
        train_mask = np.isin(y_final, pair)
        if train_mask.sum() < args.min_pair_train:
            continue
        selected_pair = pair_reports[f"{pair[0]}-{pair[1]}"]["selected"]
        model = make_model(
            selected_pair["kind"],
            max(args.n_estimators, 800),
            selected_pair["max_depth"],
            selected_pair["min_samples_leaf"],
            selected_pair["class_weight"],
            args.seed,
        )
        model.fit(x_final[train_mask], y_final[train_mask])
        final_pair_models[pair] = model
    refined_test, changed_test = refine(
        test_prob,
        x_test,
        final_pair_models,
        selected["margin_threshold"],
        selected["alpha"],
        selected["min_pair_mass"],
        args.apply_mode,
    )
    y_pred = refined_test.argmax(axis=1).astype(np.int64)
    metrics = compute_metrics(y_test.tolist(), y_pred.tolist())
    print("selected_pairwise", json.dumps(selected, sort_keys=True))
    print("test_pairwise", json.dumps({"metrics": metrics, "changed_test": changed_test}, indent=2, sort_keys=True))
    if label_names:
        print(classification_report(y_test, y_pred, labels=list(range(len(label_names))), target_names=label_names, zero_division=0))
    else:
        print(classification_report(y_test, y_pred, zero_division=0))

    if args.output_json:
        payload = {
            "metrics": {"flow_level": metrics},
            "selected_pairwise": selected,
            "pair_reports": pair_reports,
            "refinement_reports": reports,
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
                "pair_source": args.pair_source,
                "apply_mode": args.apply_mode,
            },
            "changed_test": changed_test,
        }
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()
