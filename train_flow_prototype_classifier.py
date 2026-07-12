#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, normalize

from train_flow_embedding_classifier import load_label_names, load_split


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


def full_prob(raw: np.ndarray, classes: np.ndarray, num_classes: int) -> np.ndarray:
    out = np.zeros((raw.shape[0], num_classes), dtype=np.float32)
    for col, cls in enumerate(classes):
        out[:, int(cls)] = raw[:, col]
    return out / np.maximum(out.sum(axis=1, keepdims=True), 1e-12)


def centroid_prob(x_train: np.ndarray, y_train: np.ndarray, x: np.ndarray, num_classes: int, temperature: float) -> np.ndarray:
    centroids = np.zeros((num_classes, x_train.shape[1]), dtype=np.float32)
    valid = np.zeros((num_classes,), dtype=bool)
    for cls in range(num_classes):
        mask = y_train == cls
        if mask.any():
            centroids[cls] = x_train[mask].mean(axis=0)
            valid[cls] = True
    dist = ((x[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=-1)
    dist[:, ~valid] = 1e9
    logits = -dist / max(float(temperature), 1e-6)
    logits -= logits.max(axis=1, keepdims=True)
    prob = np.exp(logits)
    return prob / np.maximum(prob.sum(axis=1, keepdims=True), 1e-12)


def transform_features(x_train: np.ndarray, x_valid: np.ndarray, x_test: np.ndarray, n_components: int, l2_normalize: bool, seed: int):
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_valid = scaler.transform(x_valid)
    x_test = scaler.transform(x_test)
    n_components = min(int(n_components), x_train.shape[0] - 1, x_train.shape[1])
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=seed)
    x_train = pca.fit_transform(x_train)
    x_valid = pca.transform(x_valid)
    x_test = pca.transform(x_test)
    if l2_normalize:
        x_train = normalize(x_train)
        x_valid = normalize(x_valid)
        x_test = normalize(x_test)
    return x_train.astype(np.float32), x_valid.astype(np.float32), x_test.astype(np.float32), n_components


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
    ap.add_argument("--components_grid", default="32,64,128,256")
    ap.add_argument("--k_grid", default="1,3,5,7,9,11,15,21")
    ap.add_argument("--prototype_modes", default="knn,centroid")
    ap.add_argument("--temperature_grid", default="0.1,0.3,1,3,10")
    ap.add_argument("--select_metric", choices=["accuracy", "macro_f1"], default="accuracy")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_json", default="")
    args = ap.parse_args()

    x_train, y_train, _ = load_split(args.train_index, args.max_packets, args.include_meta, args.meta_prefix_len, args.use_ports, args.meta_feature_version)
    x_valid, y_valid, valid_fids = load_split(args.valid_index, args.max_packets, args.include_meta, args.meta_prefix_len, args.use_ports, args.meta_feature_version)
    x_test, y_test, test_fids = load_split(args.test_index, args.max_packets, args.include_meta, args.meta_prefix_len, args.use_ports, args.meta_feature_version)
    num_classes = int(max(y_train.max(), y_valid.max(), y_test.max()) + 1)
    print(f"features train={x_train.shape} valid={x_valid.shape} test={x_test.shape}")

    reports: List[Dict[str, Any]] = []
    best = None
    selected_valid_prob = None
    selected_test_prob = None
    prototype_modes = [x.strip() for x in args.prototype_modes.split(",") if x.strip()]
    components = [int(x) for x in args.components_grid.split(",") if x.strip()]
    k_values = [int(x) for x in args.k_grid.split(",") if x.strip()]
    temperatures = [float(x) for x in args.temperature_grid.split(",") if x.strip()]

    for n_components in components:
        for l2_normalize in [False, True]:
            xtr, xv, xt, actual_components = transform_features(x_train, x_valid, x_test, n_components, l2_normalize, args.seed)
            if "knn" in prototype_modes:
                for k in k_values:
                    if k > len(y_train):
                        continue
                    clf = KNeighborsClassifier(n_neighbors=k, weights="distance", metric="euclidean")
                    clf.fit(xtr, y_train)
                    valid_prob = full_prob(clf.predict_proba(xv), clf.classes_, num_classes)
                    test_prob = full_prob(clf.predict_proba(xt), clf.classes_, num_classes)
                    valid_pred = valid_prob.argmax(axis=1)
                    metrics = compute_metrics(y_valid.tolist(), valid_pred.tolist())
                    row = {
                        "mode": "knn",
                        "n_components": actual_components,
                        "l2_normalize": l2_normalize,
                        "k": k,
                        "temperature": None,
                        "metrics": metrics,
                    }
                    reports.append(row)
                    print("valid", json.dumps(row, sort_keys=True), flush=True)
                    key = (metrics[args.select_metric], metrics["macro_f1"], metrics["accuracy"])
                    if best is None or key > best[0]:
                        best = (key, row)
                        selected_valid_prob = valid_prob
                        selected_test_prob = test_prob
            if "centroid" in prototype_modes:
                for temperature in temperatures:
                    valid_prob = centroid_prob(xtr, y_train, xv, num_classes, temperature)
                    test_prob = centroid_prob(xtr, y_train, xt, num_classes, temperature)
                    valid_pred = valid_prob.argmax(axis=1)
                    metrics = compute_metrics(y_valid.tolist(), valid_pred.tolist())
                    row = {
                        "mode": "centroid",
                        "n_components": actual_components,
                        "l2_normalize": l2_normalize,
                        "k": None,
                        "temperature": temperature,
                        "metrics": metrics,
                    }
                    reports.append(row)
                    print("valid", json.dumps(row, sort_keys=True), flush=True)
                    key = (metrics[args.select_metric], metrics["macro_f1"], metrics["accuracy"])
                    if best is None or key > best[0]:
                        best = (key, row)
                        selected_valid_prob = valid_prob
                        selected_test_prob = test_prob

    if best is None or selected_valid_prob is None or selected_test_prob is None:
        raise SystemExit("No prototype candidates were evaluated.")
    selected = best[1]
    y_pred = selected_test_prob.argmax(axis=1).astype(np.int64)
    metrics = compute_metrics(y_test.tolist(), y_pred.tolist())
    print("selected", json.dumps(selected, sort_keys=True))
    print("test", json.dumps(metrics, indent=2, sort_keys=True))

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
            "flow_prob": selected_test_prob.tolist(),
            "valid_flow_ids": valid_fids,
            "valid_y_true": y_valid.tolist(),
            "valid_y_pred": selected_valid_prob.argmax(axis=1).astype(np.int64).tolist(),
            "valid_prob": selected_valid_prob.tolist(),
            "feature_config": {
                "max_packets": args.max_packets,
                "include_meta": args.include_meta,
                "meta_prefix_len": args.meta_prefix_len,
                "meta_feature_version": args.meta_feature_version,
                "use_ports": args.use_ports,
                "select_metric": args.select_metric,
            },
        }
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()
