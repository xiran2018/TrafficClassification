#!/usr/bin/env python3
"""Audit what a strict one-packet classifier can identify under a Per-flow Split."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np


def load_rows(paths: Iterable[str]) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return rows


def packet_bytes(row: dict) -> bytearray:
    value = str(row.get("meta", {}).get("l3_hex_prefix", "")).replace(" ", "")
    return bytearray.fromhex(value) if value else bytearray()


def _zero(values: bytearray, start: int, end: int) -> None:
    values[max(start, 0) : min(end, len(values))] = b"\x00" * max(
        0, min(end, len(values)) - max(start, 0)
    )


def normalized_packet_bytes(row: dict, level: str) -> bytes:
    """Normalize only fields present in the current packet."""
    values = packet_bytes(row)
    if not values or level == "raw":
        return bytes(values)
    version = values[0] >> 4
    if version == 4:
        l4_offset = (values[0] & 0x0F) * 4
        _zero(values, 12, 20)
    elif version == 6:
        l4_offset = 40
        _zero(values, 8, 40)
    else:
        l4_offset = -1
    if l4_offset >= 0:
        _zero(values, l4_offset, l4_offset + 4)
    if level == "endpoint":
        return bytes(values)

    if version == 4:
        _zero(values, 4, 6)
        _zero(values, 8, 9)
        _zero(values, 10, 12)
        protocol = values[9] if len(values) > 9 else -1
    elif version == 6:
        if values:
            values[0] = 6 << 4
        _zero(values, 1, 4)
        _zero(values, 7, 8)
        protocol = values[6] if len(values) > 6 else -1
    else:
        protocol = -1
    if l4_offset >= 0 and protocol == 6:
        _zero(values, l4_offset + 4, l4_offset + 12)
        _zero(values, l4_offset + 16, l4_offset + 18)
        # TCP timestamps and other options can encode a session clock. Retain
        # only the fixed header and payload boundary for this diagnostic view.
        data_offset = ((values[l4_offset + 12] >> 4) * 4) if len(values) > l4_offset + 12 else 20
        _zero(values, l4_offset + 20, l4_offset + data_offset)
    elif l4_offset >= 0 and protocol == 17:
        _zero(values, l4_offset + 6, l4_offset + 8)
    return bytes(values)


def semantic_signature(row: dict) -> tuple:
    meta = row.get("meta", {})
    payload_len = int(meta.get("payload_len", 0))
    payload_prefix = normalized_packet_bytes(row, "session")
    if str(meta.get("l4", "")) == "TCP":
        header_len = int(meta.get("ip_header_len", 0)) + int(meta.get("tcp_data_offset", 0))
    elif str(meta.get("l4", "")) == "UDP":
        header_len = int(meta.get("ip_header_len", 0)) + 8
    else:
        header_len = int(meta.get("ip_header_len", 0))
    payload = payload_prefix[header_len:] if header_len < len(payload_prefix) else b""
    payload_digest = hashlib.blake2b(payload[:32], digest_size=8).hexdigest() if payload else ""
    return (
        str(meta.get("l3", "OTHER")),
        str(meta.get("l4", "OTHER")),
        str(meta.get("tcp_flags", "")),
        int(meta.get("packet_len", 0)),
        int(meta.get("ip_header_len", 0)),
        int(meta.get("tcp_data_offset", 0)),
        payload_len,
        round(float(meta.get("payload_entropy", 0.0)), 1),
        payload_digest,
    )


def packet_signature(row: dict, level: str) -> str:
    if level == "semantic":
        value = json.dumps(semantic_signature(row), separators=(",", ":")).encode()
    else:
        value = normalized_packet_bytes(row, level)
    return hashlib.blake2b(value, digest_size=16).hexdigest()


def signature_index(rows: list[dict], level: str) -> dict[str, Counter]:
    index: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        index[packet_signature(row, level)][int(row["label_id"])] += 1
    return dict(index)


def _counter_summary(index: dict[str, Counter]) -> dict:
    conflicting = {key: value for key, value in index.items() if len(value) > 1}
    conflicting_samples = sum(sum(value.values()) for value in conflicting.values())
    total_samples = sum(sum(value.values()) for value in index.values())
    return {
        "unique_signatures": len(index),
        "conflicting_signatures": len(conflicting),
        "samples_in_conflicting_signatures": conflicting_samples,
        "conflicting_sample_rate": conflicting_samples / total_samples if total_samples else 0.0,
    }


def _per_class_conflict_summary(
    rows: list[dict],
    level: str,
    index: dict[str, Counter],
) -> dict[str, dict[str, float | int]]:
    support: Counter = Counter()
    conflicting: Counter = Counter()
    for row in rows:
        label = int(row["label_id"])
        support[label] += 1
        counts = index[packet_signature(row, level)]
        if len(counts) > 1:
            conflicting[label] += 1
    return {
        str(label): {
            "support": int(count),
            "samples_in_conflicting_signatures": int(conflicting[label]),
            "conflicting_sample_rate": float(conflicting[label] / count),
        }
        for label, count in sorted(support.items())
    }


def audit_level(
    train_rows: list[dict],
    test_rows: list[dict],
    level: str,
    train_index: dict[str, Counter] | None = None,
) -> dict:
    train_index = train_index if train_index is not None else signature_index(train_rows, level)
    test_index = signature_index(test_rows, level)
    seen = ambiguous = train_lookup_correct = train_lookup_count = 0
    for row in test_rows:
        counts = train_index.get(packet_signature(row, level))
        if not counts:
            continue
        seen += 1
        if len(counts) > 1:
            ambiguous += 1
        prediction = counts.most_common(1)[0][0]
        train_lookup_correct += int(prediction == int(row["label_id"]))
        train_lookup_count += 1
    oracle_correct = sum(max(counts.values()) for counts in test_index.values())
    train_summary = _counter_summary(train_index)
    train_summary["per_class"] = _per_class_conflict_summary(
        train_rows, level, train_index
    )
    test_summary = _counter_summary(test_index)
    test_summary["per_class"] = _per_class_conflict_summary(
        test_rows, level, test_index
    )
    return {
        "train": train_summary,
        "test": test_summary,
        "test_seen_in_train": seen,
        "test_seen_rate": seen / len(test_rows) if test_rows else 0.0,
        "test_seen_with_conflicting_train_labels": ambiguous,
        "train_lookup_accuracy_on_seen_test": (
            train_lookup_correct / train_lookup_count if train_lookup_count else None
        ),
        "test_signature_oracle_accuracy": oracle_correct / len(test_rows) if test_rows else 0.0,
        "oracle_scope": "post_hoc_test_identifiability_audit_only",
    }


def error_strata(
    rows: list[dict],
    probabilities: np.ndarray,
    train_indexes: dict[str, dict[str, Counter]] | None = None,
) -> dict:
    y_true = np.asarray([int(row["label_id"]) for row in rows], dtype=np.int64)
    y_pred = probabilities.argmax(axis=1)
    errors = y_true != y_pred
    categories = {
        "all": np.ones(len(rows), dtype=bool),
        "payload_free": np.asarray([int(row.get("meta", {}).get("payload_len", 0)) == 0 for row in rows]),
        "payload_bearing": np.asarray([int(row.get("meta", {}).get("payload_len", 0)) > 0 for row in rows]),
        "tcp_control": np.asarray([
            str(row.get("meta", {}).get("l4", "")) == "TCP"
            and int(row.get("meta", {}).get("payload_len", 0)) == 0
            for row in rows
        ]),
    }
    output = {}
    for name, mask in categories.items():
        count = int(mask.sum())
        error_count = int((errors & mask).sum())
        output[name] = {
            "samples": count,
            "errors": error_count,
            "error_rate": error_count / count if count else 0.0,
        }
    examples = []
    for index in np.flatnonzero(errors)[:100]:
        row = rows[index]
        example = {
            "row_index": int(index),
            "packet_uid": row.get("packet_uid", ""),
            "flow_id": row.get("flow_id", ""),
            "label_id": int(y_true[index]),
            "predicted_id": int(y_pred[index]),
            "confidence": float(probabilities[index, y_pred[index]]),
            "l4": row.get("meta", {}).get("l4", ""),
            "tcp_flags": row.get("meta", {}).get("tcp_flags", ""),
            "packet_len": int(row.get("meta", {}).get("packet_len", 0)),
            "payload_len": int(row.get("meta", {}).get("payload_len", 0)),
        }
        if train_indexes is not None:
            example["train_signature_label_counts"] = {
                level: {
                    str(label): int(count)
                    for label, count in level_index.get(packet_signature(row, level), {}).items()
                }
                for level, level_index in train_indexes.items()
            }
        examples.append(example)
    output["error_examples"] = examples
    return output


def build_report(
    train_rows: list[dict],
    test_rows: list[dict],
    probabilities: np.ndarray | None = None,
) -> dict:
    for split_name, rows in (("train", train_rows), ("test", test_rows)):
        missing = sum(
            "l3_hex_prefix" not in (row.get("meta") or {}) for row in rows
        )
        if missing:
            raise ValueError(
                f"{split_name} rows missing meta.l3_hex_prefix: {missing}/{len(rows)}; "
                "use packet_index.jsonl rather than packet_auxiliary.jsonl"
            )
    levels = ("raw", "endpoint", "session", "semantic")
    train_indexes = {level: signature_index(train_rows, level) for level in levels}
    report = {
        "task": "packet-level-classification",
        "split_protocol": "per-flow-split",
        "sample_unit": "one_packet",
        "feature_scope": "current_packet_only",
        "purpose": "post_hoc_identifiability_audit_not_model_selection",
        "num_train_rows": len(train_rows),
        "num_test_rows": len(test_rows),
        "levels": {
            level: audit_level(train_rows, test_rows, level, train_indexes[level])
            for level in levels
        },
    }
    if probabilities is not None:
        if probabilities.ndim != 2 or len(probabilities) != len(test_rows):
            raise ValueError("prediction probabilities must align with test rows")
        report["model_error_strata"] = error_strata(test_rows, probabilities, train_indexes)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_index", nargs="+", required=True)
    parser.add_argument("--test_index", required=True)
    parser.add_argument("--prediction_npz", default="")
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    train_rows = load_rows(args.train_index)
    test_rows = load_rows([args.test_index])
    probabilities = None
    if args.prediction_npz:
        prediction = np.load(args.prediction_npz)
        expected = np.asarray([int(row["label_id"]) for row in test_rows], dtype=np.int64)
        if not np.array_equal(expected, prediction["y_true"].astype(np.int64)):
            raise ValueError("prediction labels do not align with test index")
        probabilities = prediction["probabilities"].astype(np.float64)
    report = build_report(train_rows, test_rows, probabilities)
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    print(
        f"train={len(train_rows)} test={len(test_rows)} "
        f"session_seen={report['levels']['session']['test_seen_rate']:.4f} "
        f"semantic_oracle={report['levels']['semantic']['test_signature_oracle_accuracy']:.4f}"
    )
    if probabilities is not None:
        errors = report["model_error_strata"]["all"]["errors"]
        print(f"model_errors={errors}")
    print(f"saved {output}")


if __name__ == "__main__":
    main()
