#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


COMMON_PORTS = [20, 21, 22, 25, 53, 80, 110, 143, 443, 465, 587, 993, 995, 1194, 6881]
TCP_FLAGS = ["F", "S", "R", "P", "A", "U", "E", "C"]


def load_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


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


def parse_hex_prefix(value: Any, byte_prefix_len: int, zero_ip_bytes: bool) -> List[float]:
    vals: List[int] = []
    for part in str(value or "").replace(",", " ").split():
        try:
            vals.append(int(part, 16))
        except ValueError:
            continue
    vals = vals[:byte_prefix_len]
    if zero_ip_bytes:
        for i in range(12, min(20, len(vals))):
            vals[i] = 0
    vals += [0] * max(0, byte_prefix_len - len(vals))
    return [v / 255.0 for v in vals[:byte_prefix_len]]


def packet_features(m: Dict[str, Any], byte_prefix_len: int, use_ports: bool, zero_ip_bytes: bool) -> List[float]:
    direction = 1.0 if m.get("direction") == "C2S" else -1.0
    packet_len = float(m.get("packet_len", 0) or 0)
    payload_len = float(m.get("payload_len", 0) or 0)
    iat = float(m.get("iat", 0) or 0)
    sport = int(m.get("sport", -1) or -1)
    dport = int(m.get("dport", -1) or -1)
    ttl = float(m.get("ip_ttl", 0) or 0)
    window = float(m.get("tcp_window", 0) or 0)
    flags = str(m.get("tcp_flags", "") or "")
    l4 = str(m.get("l4", "") or "")
    feats = [
        direction,
        math.log1p(packet_len) / math.log1p(1514.0),
        math.log1p(payload_len) / math.log1p(1460.0),
        min(math.log1p(max(iat, 0.0)) / math.log1p(300.0), 1.0),
        float(m.get("payload_entropy", 0.0) or 0.0) / 8.0,
        ttl / 255.0,
        math.log1p(max(window, 0.0)) / math.log1p(65535.0),
        1.0 if l4 == "TCP" else 0.0,
        1.0 if l4 == "UDP" else 0.0,
        1.0 if l4 == "ICMP" else 0.0,
    ]
    feats.extend([1.0 if flag in flags else 0.0 for flag in TCP_FLAGS])
    if use_ports:
        valid_ports = [p for p in [sport, dport] if p >= 0]
        feats.extend([
            math.log1p(max(sport, 0)) / math.log1p(65535.0),
            math.log1p(max(dport, 0)) / math.log1p(65535.0),
            1.0 if 443 in valid_ports else 0.0,
            1.0 if 80 in valid_ports else 0.0,
            1.0 if any(p in COMMON_PORTS for p in valid_ports) else 0.0,
        ])
        feats.extend([1.0 if p in valid_ports else 0.0 for p in COMMON_PORTS])
    if byte_prefix_len > 0:
        feats.extend(parse_hex_prefix(m.get("l3_hex_prefix", ""), byte_prefix_len, zero_ip_bytes))
    return feats


def flow_sequence_features(row: Dict[str, Any], max_packets: int, byte_prefix_len: int, use_ports: bool, zero_ip_bytes: bool) -> np.ndarray:
    metas = list(row.get("packet_metas", []))[:max_packets]
    if metas:
        per_packet = [packet_features(m, byte_prefix_len, use_ports, zero_ip_bytes) for m in metas]
        width = len(per_packet[0])
    else:
        width = len(packet_features({}, byte_prefix_len, use_ports, zero_ip_bytes))
        per_packet = []
    arr = np.zeros((max_packets, width), dtype=np.float32)
    for i, feats in enumerate(per_packet[:max_packets]):
        arr[i, : len(feats)] = np.asarray(feats, dtype=np.float32)

    lengths = [float(m.get("packet_len", 0) or 0) for m in metas]
    payloads = [float(m.get("payload_len", 0) or 0) for m in metas]
    iats = [float(m.get("iat", 0) or 0) for m in metas]
    dirs = [1.0 if m.get("direction") == "C2S" else -1.0 for m in metas]
    summary = np.asarray([
        len(metas) / max(float(max_packets), 1.0),
        sum(1 for d in dirs if d > 0) / max(float(len(dirs)), 1.0),
        math.log1p(sum(lengths)) / math.log1p(max(1514.0 * max_packets, 1.0)),
        math.log1p(sum(payloads)) / math.log1p(max(1460.0 * max_packets, 1.0)),
        float(np.mean(lengths)) / 1514.0 if lengths else 0.0,
        float(np.std(lengths)) / 1514.0 if lengths else 0.0,
        float(np.mean(payloads)) / 1460.0 if payloads else 0.0,
        float(np.std(payloads)) / 1460.0 if payloads else 0.0,
        float(np.mean(np.log1p(iats))) / math.log1p(300.0) if iats else 0.0,
        float(np.std(np.log1p(iats))) / math.log1p(300.0) if iats else 0.0,
        sum(1 for i in range(1, len(dirs)) if dirs[i] != dirs[i - 1]) / max(float(len(dirs) - 1), 1.0),
    ], dtype=np.float32)
    return np.concatenate([arr.reshape(-1), summary], axis=0).astype(np.float32)


def load_split(path: str, max_packets: int, byte_prefix_len: int, use_ports: bool, zero_ip_bytes: bool) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    xs: List[np.ndarray] = []
    ys: List[int] = []
    fids: List[str] = []
    for row in load_jsonl(path):
        xs.append(flow_sequence_features(row, max_packets, byte_prefix_len, use_ports, zero_ip_bytes))
        ys.append(int(row["label_id"]))
        fids.append(str(row.get("flow_id", "")))
    return np.stack(xs).astype(np.float32), np.asarray(ys, dtype=np.int64), fids


def make_model(kind: str, seed: int):
    if kind == "extra_trees":
        return ExtraTreesClassifier(n_estimators=900, max_features="sqrt", min_samples_leaf=1, random_state=seed, n_jobs=-1)
    if kind == "extra_trees_leaf2":
        return ExtraTreesClassifier(n_estimators=900, max_features="sqrt", min_samples_leaf=2, random_state=seed, n_jobs=-1)
    if kind == "random_forest":
        return RandomForestClassifier(n_estimators=700, max_features="sqrt", min_samples_leaf=1, random_state=seed, n_jobs=-1)
    if kind == "hist_gbdt":
        return HistGradientBoostingClassifier(max_iter=350, learning_rate=0.04, l2_regularization=0.05, random_state=seed)
    if kind == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=5000, solver="lbfgs", random_state=seed))
    raise ValueError(f"Unknown model kind: {kind}")


def aligned_proba(model, x: np.ndarray, num_classes: int) -> np.ndarray:
    raw = model.predict_proba(x)
    out = np.zeros((x.shape[0], num_classes), dtype=np.float32)
    clf = model.steps[-1][1] if hasattr(model, "steps") else model
    for col, cls in enumerate(clf.classes_):
        out[:, int(cls)] = raw[:, col]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_index", required=True)
    ap.add_argument("--valid_index", required=True)
    ap.add_argument("--test_index", required=True)
    ap.add_argument("--label_map", default="")
    ap.add_argument("--max_packets", type=int, default=64)
    ap.add_argument("--byte_prefix_len", type=int, default=32)
    ap.add_argument("--use_ports", action="store_true")
    ap.add_argument("--zero_ip_bytes", action="store_true")
    ap.add_argument("--model_kinds", default="extra_trees,extra_trees_leaf2,hist_gbdt,logreg")
    ap.add_argument("--select_metric", choices=["accuracy", "macro_f1"], default="accuracy")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_json", default="")
    args = ap.parse_args()

    x_train, y_train, _ = load_split(args.train_index, args.max_packets, args.byte_prefix_len, args.use_ports, args.zero_ip_bytes)
    x_valid, y_valid, valid_fids = load_split(args.valid_index, args.max_packets, args.byte_prefix_len, args.use_ports, args.zero_ip_bytes)
    x_test, y_test, test_fids = load_split(args.test_index, args.max_packets, args.byte_prefix_len, args.use_ports, args.zero_ip_bytes)
    num_classes = int(max(y_train.max(), y_valid.max(), y_test.max()) + 1)
    print(f"sequence_features train={x_train.shape} valid={x_valid.shape} test={x_test.shape}")

    best = None
    reports = []
    for kind in [x.strip() for x in args.model_kinds.split(",") if x.strip()]:
        model = make_model(kind, args.seed)
        model.fit(x_train, y_train)
        pred = model.predict(x_valid)
        metrics = compute_metrics(y_valid.tolist(), pred.tolist())
        row = {"kind": kind, "metrics": metrics}
        reports.append(row)
        print("valid", json.dumps(row, sort_keys=True), flush=True)
        key = (metrics[args.select_metric], metrics["macro_f1"], metrics["accuracy"])
        if best is None or key > best[0]:
            best = (key, row)

    selected = best[1]
    print("selected", json.dumps(selected, sort_keys=True))
    final_model = make_model(selected["kind"], args.seed)
    final_model.fit(np.concatenate([x_train, x_valid], axis=0), np.concatenate([y_train, y_valid], axis=0))
    test_prob = aligned_proba(final_model, x_test, num_classes)
    y_pred = test_prob.argmax(axis=1).astype(np.int64)
    metrics = compute_metrics(y_test.tolist(), y_pred.tolist())
    print("test", json.dumps(metrics, indent=2, sort_keys=True))

    valid_model = make_model(selected["kind"], args.seed)
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
                "byte_prefix_len": args.byte_prefix_len,
                "use_ports": args.use_ports,
                "zero_ip_bytes": args.zero_ip_bytes,
                "select_metric": args.select_metric,
            },
        }
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()
