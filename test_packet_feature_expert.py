#!/usr/bin/env python3
"""Evaluate a saved one-packet feature expert and export aligned probabilities."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from packet_eval_utils import packet_classification_metrics
from test_packet_byte_transformer import load_packet_uids
from train_packet_feature_expert import load_split
from train_tower1_multitask import load_label_names


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--training_result", required=True, help="JSON containing the validation-selected byte prefix.")
    ap.add_argument("--test_index", required=True)
    ap.add_argument("--label_map", required=True)
    ap.add_argument("--output_json", required=True)
    ap.add_argument("--output_npz", required=True)
    args = ap.parse_args()

    with open(args.training_result, "r", encoding="utf-8") as f:
        training_result = json.load(f)
    prefix = int(training_result["selected"]["byte_prefix_len"])
    zero_ip = bool(training_result["config"].get("zero_ip_bytes", False))
    zero_ports = bool(training_result["config"].get("zero_port_bytes", False))
    mask_session_fields = bool(training_result["config"].get("mask_session_fields", False))
    label_names = load_label_names(args.label_map)
    x_test, y_true = load_split(args.test_index, prefix, zero_ip, zero_ports, mask_session_fields)
    packet_uids = load_packet_uids(args.test_index)
    if len(packet_uids) != len(y_true):
        raise ValueError(
            f"packet index/prediction length mismatch: {len(packet_uids)} != {len(y_true)}"
        )
    model = joblib.load(args.model)
    raw_probs = model.predict_proba(x_test)
    probabilities = np.zeros((len(y_true), len(label_names)), dtype=np.float32)
    probabilities[:, np.asarray(model.classes_, dtype=np.int64)] = raw_probs
    y_pred = probabilities.argmax(axis=1)
    metrics = packet_classification_metrics(y_true, y_pred, len(label_names), label_names)
    output = {
        "task": "packet-level-classification",
        "sample_unit": "one_packet",
        "model": args.model,
        "selected_byte_prefix_len": prefix,
        "mask_session_fields": mask_session_fields,
        "metrics": metrics,
    }
    output_json = Path(args.output_json)
    output_npz = Path(args.output_npz)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    np.savez_compressed(
        output_npz,
        y_true=y_true,
        probabilities=probabilities,
        packet_uids=packet_uids,
    )
    print(f"accuracy={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f}")
    print(f"saved {output_json} and {output_npz}")


if __name__ == "__main__":
    main()
