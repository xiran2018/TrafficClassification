#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from probability_metrics import calibration_metrics


def load_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_prob(prob: np.ndarray) -> np.ndarray:
    prob = np.asarray(prob, dtype=np.float64)
    prob = np.clip(prob, 1e-12, None)
    return prob / np.maximum(prob.sum(axis=1, keepdims=True), 1e-12)


def canonical_pcap_key(row: dict[str, Any]) -> str:
    label = str(row.get("label", ""))
    pcap_path = str(row.get("pcap_path", ""))
    if not label or not pcap_path:
        raise ValueError("flow embedding index rows must contain label and pcap_path")
    return f"{label}/{Path(pcap_path).name}"


def read_index(path: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            flow_id = str(row["flow_id"])
            row = dict(row)
            row["canonical_key"] = canonical_pcap_key(row)
            out[flow_id] = row
    return out


def load_label_map(path: str) -> dict[str, int]:
    if not path:
        return {}
    data = load_json(path)
    return {str(k): int(v) for k, v in data.items()}


def compute_metrics(y_true: list[int], prob: np.ndarray) -> dict[str, Any]:
    if not y_true:
        return {}
    y = np.asarray(y_true, dtype=np.int64)
    pred = prob.argmax(axis=1).astype(np.int64)
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(y, pred, average="macro", zero_division=0)
    p_weight, r_weight, f_weight, _ = precision_recall_fscore_support(y, pred, average="weighted", zero_division=0)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_precision": float(p_macro),
        "macro_recall": float(r_macro),
        "macro_f1": float(f_macro),
        "weighted_precision": float(p_weight),
        "weighted_recall": float(r_weight),
        "weighted_f1": float(f_weight),
        "calibration": calibration_metrics(y, prob),
    }


def build_source_teacher(
    teacher: dict[str, Any],
    source_indexes: list[str],
) -> tuple[dict[str, list[np.ndarray]], dict[str, list[int]]]:
    flow_ids = [str(fid) for fid in teacher.get("flow_ids", [])]
    probs = normalize_prob(np.asarray(teacher.get("flow_prob", []), dtype=np.float64))
    y_true_raw = teacher.get("flow_y_true", [])
    if len(flow_ids) != probs.shape[0]:
        raise ValueError("teacher flow_ids and flow_prob lengths do not match")
    prob_by_flow = {fid: probs[idx] for idx, fid in enumerate(flow_ids)}
    y_by_flow = {fid: int(y_true_raw[idx]) for idx, fid in enumerate(flow_ids)} if len(y_true_raw) == len(flow_ids) else {}
    key_to_probs: dict[str, list[np.ndarray]] = {}
    key_to_y: dict[str, list[int]] = {}
    indexed = 0
    matched = 0
    for path in source_indexes:
        for flow_id, row in read_index(path).items():
            indexed += 1
            prob = prob_by_flow.get(flow_id)
            if prob is None:
                continue
            matched += 1
            key = row["canonical_key"]
            key_to_probs.setdefault(key, []).append(prob)
            if flow_id in y_by_flow:
                key_to_y.setdefault(key, []).append(y_by_flow[flow_id])
    if matched == 0:
        raise ValueError("No teacher flow_ids matched the provided source indexes")
    return key_to_probs, key_to_y


def main() -> None:
    ap = argparse.ArgumentParser(description="Remap flow-id teacher targets through canonical label/basename pcap keys.")
    ap.add_argument("--teacher_json", required=True)
    ap.add_argument("--source_index", action="append", required=True, help="Flow embedding index JSONL for teacher flow_id namespace.")
    ap.add_argument("--target_index", required=True, help="Flow embedding index JSONL whose flow_ids should receive teacher targets.")
    ap.add_argument("--label_map", default="")
    ap.add_argument("--min_teacher_confidence", type=float, default=0.0)
    ap.add_argument("--output_json", required=True)
    args = ap.parse_args()

    teacher = load_json(args.teacher_json)
    label_map = load_label_map(args.label_map)
    key_to_probs, key_to_y = build_source_teacher(teacher, args.source_index)
    target_rows = list(read_index(args.target_index).values())
    out_ids: list[str] = []
    out_prob: list[np.ndarray] = []
    out_y: list[int] = []
    duplicate_source_keys = 0
    for row in target_rows:
        probs = key_to_probs.get(row["canonical_key"])
        if not probs:
            continue
        if len(probs) > 1:
            duplicate_source_keys += 1
        prob = normalize_prob(np.mean(np.stack(probs, axis=0), axis=0, keepdims=True)).squeeze(0)
        if float(prob.max()) < args.min_teacher_confidence:
            continue
        out_ids.append(str(row["flow_id"]))
        out_prob.append(prob)
        if label_map:
            out_y.append(label_map[str(row["label"])])
    prob_arr = normalize_prob(np.stack(out_prob, axis=0)) if out_prob else np.zeros((0, 0), dtype=np.float64)
    metrics = compute_metrics(out_y, prob_arr) if out_y else {}
    payload: dict[str, Any] = {
        "flow_ids": out_ids,
        "flow_prob": prob_arr.tolist(),
        "config": {
            "teacher_json": args.teacher_json,
            "source_index": args.source_index,
            "target_index": args.target_index,
            "label_map": args.label_map,
            "canonical_key": "label/basename(pcap_path)",
            "min_teacher_confidence": args.min_teacher_confidence,
            "target_rows": len(target_rows),
            "output_flows": len(out_ids),
            "coverage": len(out_ids) / max(1, len(target_rows)),
            "duplicate_source_keys_averaged": duplicate_source_keys,
        },
        "source_teacher_metrics": (teacher.get("metrics") or {}).get("flow_level", {}),
        "metrics": {"flow_level": metrics} if metrics else {},
    }
    if out_y:
        payload["flow_y_true"] = out_y
        payload["flow_y_pred"] = prob_arr.argmax(axis=1).astype(int).tolist()

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"config": payload["config"], "metrics": payload["metrics"]}, indent=2, ensure_ascii=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
