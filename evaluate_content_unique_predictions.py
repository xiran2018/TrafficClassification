#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def load_jsonl(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def sha256_file(path: str, cache: Dict[str, str]) -> str:
    if path not in cache:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        cache[path] = digest.hexdigest()
    return cache[path]


def metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> dict:
    precision, recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
    }


def flow_content_map(index_path: str) -> tuple[Dict[str, str], Dict[str, str]]:
    cache: Dict[str, str] = {}
    flow_to_hash = {}
    flow_to_path = {}
    for row in load_jsonl(index_path):
        flow_id = str(row.get("flow_id", ""))
        pcap_path = str(row.get("pcap_path", ""))
        if flow_id and pcap_path:
            flow_to_hash[flow_id] = sha256_file(pcap_path, cache)
            flow_to_path[flow_id] = pcap_path
    return flow_to_hash, flow_to_path


def aggregate_unique_content(
    flow_ids: Sequence[str],
    y_true: Sequence[int],
    prob: np.ndarray,
    flow_to_hash: Dict[str, str],
) -> tuple[list[str], np.ndarray, np.ndarray, dict]:
    if len(flow_ids) != len(y_true) or len(flow_ids) != len(prob):
        raise ValueError("flow_ids, labels, and probabilities must have equal lengths")
    buckets = defaultdict(list)
    missing = []
    for index, flow_id in enumerate(flow_ids):
        content_hash = flow_to_hash.get(str(flow_id))
        if content_hash is None:
            missing.append(str(flow_id))
            continue
        buckets[content_hash].append(index)
    if missing:
        raise ValueError(f"Embedding index is missing {len(missing)} prediction flow IDs; first={missing[0]}")

    hashes = []
    labels = []
    probabilities = []
    duplicate_groups = 0
    duplicate_rows = 0
    for content_hash, indices in buckets.items():
        group_labels = {int(y_true[index]) for index in indices}
        if len(group_labels) != 1:
            raise ValueError(f"Content hash {content_hash} has conflicting labels: {sorted(group_labels)}")
        hashes.append(content_hash)
        labels.append(next(iter(group_labels)))
        probabilities.append(np.asarray(prob[indices], dtype=np.float64).mean(axis=0))
        if len(indices) > 1:
            duplicate_groups += 1
            duplicate_rows += len(indices) - 1
    probability_array = np.stack(probabilities, axis=0)
    probability_array /= np.maximum(probability_array.sum(axis=1, keepdims=True), 1e-12)
    audit = {
        "input_flows": len(flow_ids),
        "unique_content_flows": len(hashes),
        "duplicate_content_groups": duplicate_groups,
        "duplicate_rows_removed": duplicate_rows,
    }
    return hashes, np.asarray(labels, dtype=np.int64), probability_array, audit


def bootstrap_ci(y_true: np.ndarray, prob: np.ndarray, samples: int, seed: int) -> dict:
    if samples <= 0:
        return {}
    rng = np.random.default_rng(seed)
    accuracy = []
    macro_f1 = []
    for _ in range(samples):
        indices = rng.integers(0, len(y_true), size=len(y_true))
        result = metrics(y_true[indices], prob[indices].argmax(axis=1))
        accuracy.append(result["accuracy"])
        macro_f1.append(result["macro_f1"])
    return {
        "samples": samples,
        "seed": seed,
        "accuracy_95_ci": np.percentile(accuracy, [2.5, 97.5]).astype(float).tolist(),
        "macro_f1_95_ci": np.percentile(macro_f1, [2.5, 97.5]).astype(float).tolist(),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Recompute prediction metrics with exact-PCAP duplicate groups counted once.")
    ap.add_argument("--prediction_json", required=True)
    ap.add_argument("--flow_embedding_index", required=True)
    ap.add_argument("--output_json", required=True)
    ap.add_argument("--bootstrap_samples", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--summary_only", action="store_true", help="Omit per-content probabilities from the output JSON.")
    args = ap.parse_args()

    with open(args.prediction_json, "r", encoding="utf-8") as handle:
        prediction = json.load(handle)
    flow_ids = [str(value) for value in prediction["flow_ids"]]
    y_true = np.asarray(prediction["flow_y_true"], dtype=np.int64)
    prob = np.asarray(prediction["flow_prob"], dtype=np.float64)
    flow_to_hash, _ = flow_content_map(args.flow_embedding_index)
    hashes, unique_y, unique_prob, audit = aggregate_unique_content(
        flow_ids, y_true, prob, flow_to_hash
    )
    raw_metrics = metrics(y_true, prob.argmax(axis=1))
    unique_metrics = metrics(unique_y, unique_prob.argmax(axis=1))
    ci = bootstrap_ci(unique_y, unique_prob, args.bootstrap_samples, args.seed)
    payload = {
        "method": "exact_pcap_sha256_unique_content_evaluation",
        "prediction_json": args.prediction_json,
        "flow_embedding_index": args.flow_embedding_index,
        "metrics": {
            "original_flow_level": raw_metrics,
            "content_unique_flow_level": unique_metrics,
        },
        "audit": audit,
        "content_unique_bootstrap": ci,
    }
    if not args.summary_only:
        payload.update({
            "content_hashes": hashes,
            "content_y_true": unique_y.astype(int).tolist(),
            "content_y_pred": unique_prob.argmax(axis=1).astype(int).tolist(),
            "content_prob": unique_prob.tolist(),
        })
    print(json.dumps({"metrics": payload["metrics"], "audit": audit, "bootstrap": ci}, indent=2))
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


if __name__ == "__main__":
    main()
