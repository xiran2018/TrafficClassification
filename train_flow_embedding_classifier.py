#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from train_flow_stats_classifier import flow_features, load_jsonl


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


def embedding_summary(row: Dict[str, Any], max_packets: int) -> np.ndarray:
    path = row.get("embedding_path", "")
    emb = np.load(path).astype(np.float32)
    if max_packets > 0:
        emb = emb[:max_packets]
    dim = int(row.get("embedding_dim", emb.shape[1] if emb.ndim == 2 else 0))
    if emb.ndim != 2 or emb.shape[0] == 0:
        return np.zeros(dim * 8, dtype=np.float32)
    metas = list(row.get("packet_metas", []))[: emb.shape[0]]
    dirs = np.asarray([1 if m.get("direction") == "C2S" else -1 for m in metas], dtype=np.int64)
    if len(dirs) != emb.shape[0]:
        dirs = np.ones((emb.shape[0],), dtype=np.int64)

    chunks = [
        emb.mean(axis=0),
        emb.std(axis=0),
        emb.min(axis=0),
        emb.max(axis=0),
        emb[0],
        emb[-1],
    ]
    for direction in [1, -1]:
        mask = dirs == direction
        if np.any(mask):
            chunks.append(emb[mask].mean(axis=0))
        else:
            chunks.append(np.zeros((emb.shape[1],), dtype=np.float32))
    return np.concatenate(chunks, axis=0).astype(np.float32)


def load_split(
    path: str,
    max_packets: int,
    include_meta: bool,
    meta_prefix_len: int,
    use_ports: bool,
    meta_feature_version: str,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    xs: List[np.ndarray] = []
    ys: List[int] = []
    fids: List[str] = []
    for row in load_jsonl(path):
        parts = [embedding_summary(row, max_packets)]
        if include_meta:
            parts.append(np.asarray(flow_features(row, max_packets, meta_prefix_len, use_ports, meta_feature_version), dtype=np.float32))
        xs.append(np.concatenate(parts, axis=0).astype(np.float32))
        ys.append(int(row["label_id"]))
        fids.append(str(row.get("flow_id", "")))
    width = max(len(x) for x in xs)
    arr = np.zeros((len(xs), width), dtype=np.float32)
    for i, x in enumerate(xs):
        arr[i, : len(x)] = x
    return arr, np.asarray(ys, dtype=np.int64), fids


def make_model(kind: str, n_components: int, c: float, class_weight: str | None, seed: int):
    steps = [
        StandardScaler(),
        PCA(n_components=n_components, svd_solver="randomized", random_state=seed),
    ]
    if kind == "logreg":
        clf = LogisticRegression(C=c, max_iter=5000, solver="lbfgs", class_weight=class_weight, random_state=seed)
    elif kind == "extra_trees":
        clf = ExtraTreesClassifier(
            n_estimators=600,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight=class_weight,
            random_state=seed,
            n_jobs=-1,
        )
    elif kind == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=500,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight=class_weight,
            random_state=seed,
            n_jobs=-1,
        )
    elif kind == "hist_gbdt":
        clf = HistGradientBoostingClassifier(
            max_iter=250,
            learning_rate=0.05,
            l2_regularization=0.1,
            random_state=seed,
        )
    elif kind == "svc":
        clf = SVC(C=c, kernel="rbf", gamma="scale", class_weight=class_weight, probability=True, random_state=seed)
    else:
        raise ValueError(f"Unknown --model_kinds entry: {kind}")
    return make_pipeline(*steps, clf)


def aligned_proba(model, x: np.ndarray, num_classes: int) -> np.ndarray:
    raw = model.predict_proba(x)
    out = np.zeros((x.shape[0], num_classes), dtype=np.float32)
    clf = model.steps[-1][1]
    classes = clf.classes_
    for col, cls in enumerate(classes):
        out[:, int(cls)] = raw[:, col]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_index", required=True)
    ap.add_argument("--valid_index", required=True)
    ap.add_argument("--test_index", required=True)
    ap.add_argument("--label_map", default="")
    ap.add_argument("--max_packets", type=int, default=64)
    ap.add_argument("--include_meta", action="store_true")
    ap.add_argument("--meta_prefix_len", type=int, default=32)
    ap.add_argument("--meta_feature_version", choices=["basic", "message", "message_header", "message_header_endpoint", "message_header_fullbytes"], default="message")
    ap.add_argument("--use_ports", action="store_true")
    ap.add_argument("--components_grid", default="64,128,256")
    ap.add_argument("--c_grid", default="0.1,0.3,1,3")
    ap.add_argument("--class_weight_grid", default="none,balanced")
    ap.add_argument("--model_kinds", default="logreg")
    ap.add_argument("--select_metric", choices=["accuracy", "macro_f1"], default="macro_f1")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_json", default="")
    args = ap.parse_args()

    x_train, y_train, _ = load_split(args.train_index, args.max_packets, args.include_meta, args.meta_prefix_len, args.use_ports, args.meta_feature_version)
    x_valid, y_valid, valid_fids = load_split(args.valid_index, args.max_packets, args.include_meta, args.meta_prefix_len, args.use_ports, args.meta_feature_version)
    x_test, y_test, test_fids = load_split(args.test_index, args.max_packets, args.include_meta, args.meta_prefix_len, args.use_ports, args.meta_feature_version)
    num_classes = int(max(y_train.max(), y_valid.max(), y_test.max()) + 1)
    max_components = min(x_train.shape[0] - 1, x_train.shape[1])
    print(f"features train={x_train.shape} valid={x_valid.shape} test={x_test.shape} max_components={max_components}")

    components = [min(int(x), max_components) for x in args.components_grid.split(",") if x.strip()]
    components = sorted(set(x for x in components if x > 0))
    c_values = [float(x) for x in args.c_grid.split(",") if x.strip()]
    class_weights = [None if x.strip() == "none" else x.strip() for x in args.class_weight_grid.split(",") if x.strip()]
    model_kinds = [x.strip() for x in args.model_kinds.split(",") if x.strip()]

    reports = []
    best = None
    for kind in model_kinds:
        for n_components in components:
            for c in c_values:
                for class_weight in class_weights:
                    model = make_model(kind, n_components, c, class_weight, args.seed)
                    model.fit(x_train, y_train)
                    pred = model.predict(x_valid)
                    metrics = compute_metrics(y_valid.tolist(), pred.tolist())
                    row = {
                        "kind": kind,
                        "n_components": n_components,
                        "C": c,
                        "class_weight": class_weight,
                        "metrics": metrics,
                    }
                    reports.append(row)
                    print("valid", json.dumps(row, sort_keys=True), flush=True)
                    key = (metrics[args.select_metric], metrics["accuracy"])
                    if best is None or key > best[0]:
                        best = (key, row)

    selected = best[1]
    print("selected", json.dumps(selected, sort_keys=True))
    final_model = make_model(selected["kind"], selected["n_components"], selected["C"], selected["class_weight"], args.seed)
    final_model.fit(np.concatenate([x_train, x_valid], axis=0), np.concatenate([y_train, y_valid], axis=0))
    test_prob = aligned_proba(final_model, x_test, num_classes)
    y_pred = test_prob.argmax(axis=1).astype(np.int64)
    metrics = compute_metrics(y_test.tolist(), y_pred.tolist())
    print("test", json.dumps(metrics, indent=2, sort_keys=True))

    valid_model = make_model(selected["kind"], selected["n_components"], selected["C"], selected["class_weight"], args.seed)
    valid_model.fit(x_train, y_train)
    valid_prob = aligned_proba(valid_model, x_valid, num_classes)

    label_names, label_map = load_label_names(args.label_map)
    if label_names:
        print(classification_report(y_test, y_pred, labels=list(range(len(label_names))), target_names=label_names, zero_division=0))
    else:
        print(classification_report(y_test, y_pred, zero_division=0))

    if args.output_json:
        payload = {
            "metrics": {"flow_level": metrics},
            "selected": selected,
            "valid_reports": reports,
            "label_map": label_map,
            "flow_ids": test_fids,
            "flow_y_true": y_test.tolist(),
            "flow_y_pred": y_pred.tolist(),
            "flow_prob": test_prob.tolist(),
            "valid_flow_ids": valid_fids,
            "valid_y_true": y_valid.tolist(),
            "valid_y_pred": valid_prob.argmax(axis=1).astype(np.int64).tolist(),
            "valid_prob": valid_prob.tolist(),
            "feature_config": {
                "max_packets": args.max_packets,
                "include_meta": args.include_meta,
                "meta_prefix_len": args.meta_prefix_len,
                "meta_feature_version": args.meta_feature_version,
                "use_ports": args.use_ports,
            },
        }
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()
