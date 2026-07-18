#!/usr/bin/env python3
"""Train a strict one-packet structural expert under a Per-flow Split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

from packet_eval_utils import packet_classification_metrics
from train_tower1_multitask import load_label_names


FLAG_BITS = {name: idx for idx, name in enumerate("FSRPAUEC")}


def _zero_slice(values: np.ndarray, start: int, end: int) -> None:
    if start < len(values):
        values[start:min(len(values), end)] = 0.0


def packet_features(
    row: dict,
    byte_prefix_len: int,
    zero_ip_bytes: bool,
    zero_port_bytes: bool = False,
    mask_session_fields: bool = False,
) -> np.ndarray:
    meta = row["meta"]
    raw = bytes.fromhex(str(meta.get("l3_hex_prefix", "")).replace(" ", ""))
    byte_values = np.zeros(byte_prefix_len, dtype=np.float32)
    n = min(len(raw), byte_prefix_len)
    if n:
        byte_values[:n] = np.frombuffer(raw[:n], dtype=np.uint8).astype(np.float32) / 255.0
    if (zero_ip_bytes or mask_session_fields) and raw:
        if raw[0] >> 4 == 4 and byte_prefix_len > 12:
            byte_values[12:min(byte_prefix_len, 20)] = 0.0
        elif raw[0] >> 4 == 6 and byte_prefix_len > 8:
            byte_values[8:min(byte_prefix_len, 40)] = 0.0
    if (zero_port_bytes or mask_session_fields) and raw:
        version = raw[0] >> 4
        l4_offset = (raw[0] & 0x0F) * 4 if version == 4 else 40 if version == 6 else -1
        if l4_offset >= 0 and byte_prefix_len > l4_offset:
            byte_values[l4_offset:min(byte_prefix_len, l4_offset + 4)] = 0.0
    if mask_session_fields and raw:
        version = raw[0] >> 4
        l4_offset = (raw[0] & 0x0F) * 4 if version == 4 else 40 if version == 6 else -1
        if version == 4:
            # Remove capture- and route-specific fields while retaining packet
            # lengths, transport protocol, TCP flags, and payload bytes.
            _zero_slice(byte_values, 4, 6)   # IPv4 identification
            _zero_slice(byte_values, 8, 9)   # TTL
            _zero_slice(byte_values, 10, 12) # IPv4 checksum
        elif version == 6:
            if len(byte_values):
                byte_values[0] = float(6 << 4) / 255.0
            _zero_slice(byte_values, 1, 4)   # traffic class and flow label
            _zero_slice(byte_values, 7, 8)   # hop limit
        if l4_offset >= 0 and len(raw) > l4_offset:
            protocol = int(meta.get("ip_proto", raw[9] if version == 4 and len(raw) > 9 else raw[6] if version == 6 and len(raw) > 6 else -1))
            if protocol == 6:
                _zero_slice(byte_values, l4_offset + 4, l4_offset + 12)   # TCP seq/ack
                _zero_slice(byte_values, l4_offset + 16, l4_offset + 18) # TCP checksum
            elif protocol == 17:
                _zero_slice(byte_values, l4_offset + 6, l4_offset + 8)   # UDP checksum

    flags = np.zeros(8, dtype=np.float32)
    for flag in str(meta.get("tcp_flags", "")):
        if flag in FLAG_BITS:
            flags[FLAG_BITS[flag]] = 1.0
    l4 = str(meta.get("l4", "OTHER"))
    structured = np.asarray(
        [
            float(meta.get("packet_len", 0)) / 1514.0,
            float(meta.get("payload_len", 0)) / 1460.0,
            float(meta.get("payload_entropy", 0)) / 8.0,
            0.0 if mask_session_fields else float(meta.get("ip_ttl", 0)) / 255.0,
            float(meta.get("ip_total_len", 0)) / 65535.0,
            float(meta.get("ip_header_len", 0)) / 60.0,
            0.0 if (zero_port_bytes or mask_session_fields) else float(meta.get("sport", -1)) / 65535.0,
            0.0 if (zero_port_bytes or mask_session_fields) else float(meta.get("dport", -1)) / 65535.0,
            0.0 if (zero_port_bytes or mask_session_fields) else (
                float(min(x for x in (meta.get("sport", -1), meta.get("dport", -1)) if x >= 0)) / 65535.0
                if max(meta.get("sport", -1), meta.get("dport", -1)) >= 0 else 0.0
            ),
            0.0 if (zero_port_bytes or mask_session_fields) else float(max(meta.get("sport", -1), meta.get("dport", -1), 0)) / 65535.0,
            float(meta.get("tcp_window", 0)) / 65535.0,
            float(meta.get("tcp_data_offset", 0)) / 60.0,
            float(meta.get("udp_len", 0)) / 65535.0,
            0.0 if mask_session_fields else float(meta.get("seq", 0) & 0xFFFF) / 65535.0,
            0.0 if mask_session_fields else float(meta.get("ack", 0) & 0xFFFF) / 65535.0,
            1.0 if l4 == "TCP" else 0.0,
            1.0 if l4 == "UDP" else 0.0,
            1.0 if l4 not in {"TCP", "UDP"} else 0.0,
            0.0 if mask_session_fields else (1.0 if str(meta.get("direction", "UNK")) == "C2S" else 0.0),
            0.0 if mask_session_fields else (1.0 if str(meta.get("direction", "UNK")) == "S2C" else 0.0),
        ],
        dtype=np.float32,
    )
    return np.concatenate([byte_values, structured, flags])


def load_split(
    path: str,
    byte_prefix_len: int,
    zero_ip_bytes: bool,
    zero_port_bytes: bool = False,
    mask_session_fields: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            xs.append(packet_features(row, byte_prefix_len, zero_ip_bytes, zero_port_bytes, mask_session_fields))
            ys.append(int(row["label_id"]))
    return np.stack(xs), np.asarray(ys, dtype=np.int64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_index", required=True)
    ap.add_argument("--valid_index", required=True)
    ap.add_argument("--test_index", required=True)
    ap.add_argument("--label_map", required=True)
    ap.add_argument("--output_json", required=True)
    ap.add_argument("--model_out", required=True)
    ap.add_argument("--byte_prefix_len", type=int, nargs="+", default=[96], help="Validation-selected current-packet byte-prefix candidates.")
    ap.add_argument("--min_samples_leaf", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--n_estimators", type=int, default=300)
    ap.add_argument(
        "--estimator_types",
        nargs="+",
        choices=["extra_trees", "random_forest"],
        default=["extra_trees"],
        help="Validation-selected structural estimator family.",
    )
    ap.add_argument("--zero_ip_bytes", action="store_true")
    ap.add_argument("--zero_port_bytes", action="store_true")
    ap.add_argument(
        "--mask_session_fields",
        action="store_true",
        help="Mask endpoint, route, checksum, flow-label, and TCP sequence identifiers while retaining current-packet behavior and payload.",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_jobs", type=int, default=-1)
    args = ap.parse_args()

    label_names = load_label_names(args.label_map)
    num_classes = len(label_names)
    trials = []
    best = None
    for prefix_len in args.byte_prefix_len:
        x_train, y_train = load_split(
            args.train_index, prefix_len, args.zero_ip_bytes, args.zero_port_bytes, args.mask_session_fields
        )
        x_valid, y_valid = load_split(
            args.valid_index, prefix_len, args.zero_ip_bytes, args.zero_port_bytes, args.mask_session_fields
        )
        print(f"prefix={prefix_len} train={x_train.shape} valid={x_valid.shape}", flush=True)
        for estimator_type in args.estimator_types:
            for leaf in args.min_samples_leaf:
                estimator_class = ExtraTreesClassifier if estimator_type == "extra_trees" else RandomForestClassifier
                model = estimator_class(
                    n_estimators=args.n_estimators,
                    min_samples_leaf=leaf,
                    max_features="sqrt",
                    class_weight="balanced",
                    n_jobs=args.n_jobs,
                    random_state=args.seed,
                )
                model.fit(x_train, y_train)
                pred = model.predict(x_valid)
                metrics = packet_classification_metrics(y_valid, pred, num_classes, label_names)
                trial = {
                    "estimator_type": estimator_type,
                    "byte_prefix_len": prefix_len,
                    "min_samples_leaf": leaf,
                    "validation_metrics": metrics,
                }
                trials.append(trial)
                key = (metrics["macro_f1"], metrics["accuracy"])
                print(
                    f"estimator={estimator_type} prefix={prefix_len} leaf={leaf} "
                    f"valid_acc={metrics['accuracy']:.4f} valid_macro_f1={metrics['macro_f1']:.4f}",
                    flush=True,
                )
                if best is None or key > best[0]:
                    best = (key, model, trial)

    assert best is not None
    _, model, selected = best
    selected_prefix = int(selected["byte_prefix_len"])
    x_test, y_test = load_split(
        args.test_index, selected_prefix, args.zero_ip_bytes, args.zero_port_bytes, args.mask_session_fields
    )
    print(
        f"selected estimator={selected.get('estimator_type', 'extra_trees')} "
        f"prefix={selected_prefix} leaf={selected['min_samples_leaf']} test={x_test.shape}",
        flush=True,
    )
    test_pred = model.predict(x_test)
    test_metrics = packet_classification_metrics(y_test, test_pred, num_classes, label_names)
    output = {
        "task": "packet-level-classification",
        "split_protocol": "per-flow-split",
        "sample_unit": "one_packet",
        "feature_scope": "current_packet_only",
        "config": vars(args),
        "trials": trials,
        "selected": selected,
        "test_metrics": test_metrics,
    }
    output_path = Path(args.output_json)
    model_path = Path(args.model_out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    joblib.dump(model, model_path, compress=3)
    print(f"test_acc={test_metrics['accuracy']:.4f} test_macro_f1={test_metrics['macro_f1']:.4f}")
    print(f"saved {output_path} and {model_path}")


if __name__ == "__main__":
    main()
