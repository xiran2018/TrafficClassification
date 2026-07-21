#!/usr/bin/env python3
"""Quantify train-only packet/flow class-prior divergence."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize(values: list[float]) -> list[float]:
    total = sum(values)
    if total <= 0:
        raise ValueError("cannot normalize an empty count distribution")
    return [value / total for value in values]


def total_variation(left: list[float], right: list[float]) -> float:
    return 0.5 * sum(abs(a - b) for a, b in zip(left, right))


def kl_divergence(left: list[float], right: list[float]) -> float:
    return sum(a * math.log(a / b) for a, b in zip(left, right) if a > 0 and b > 0)


def jensen_shannon(left: list[float], right: list[float]) -> float:
    midpoint = [(a + b) / 2.0 for a, b in zip(left, right)]
    return 0.5 * kl_divergence(left, midpoint) + 0.5 * kl_divergence(right, midpoint)


def effective_weights(counts: list[int], beta: float) -> list[float]:
    raw = [
        (1.0 - beta) / max(1.0 - beta ** int(count), 1e-12)
        for count in counts
    ]
    mean = sum(raw) / len(raw)
    return [value / mean for value in raw]


def pearson(left: list[float], right: list[float]) -> float | None:
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum(
        (a - left_mean) * (b - right_mean) for a, b in zip(left, right)
    )
    left_scale = math.sqrt(sum((a - left_mean) ** 2 for a in left))
    right_scale = math.sqrt(sum((b - right_mean) ** 2 for b in right))
    denominator = left_scale * right_scale
    if left_scale <= 1e-12 or right_scale <= 1e-12:
        return None
    return numerator / denominator


def analyze(
    packet_aux_jsonl: str | Path,
    label_map_path: str | Path,
    *,
    beta: float = 0.9999,
) -> dict[str, Any]:
    packet_aux_jsonl = Path(packet_aux_jsonl)
    label_map_path = Path(label_map_path)
    label_map = {
        str(name): int(index)
        for name, index in json.loads(label_map_path.read_text(encoding="utf-8")).items()
    }
    names = [""] * len(label_map)
    for name, index in label_map.items():
        names[index] = name
    packet_counts = [0] * len(names)
    flow_labels: dict[str, int] = {}
    num_rows = 0
    with packet_aux_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            label = int(row["label_id"])
            flow_id = str(row.get("flow_id") or "")
            if not flow_id:
                raise ValueError("every training packet must have flow_id")
            if not 0 <= label < len(names):
                raise ValueError(f"label_id outside label map: {label}")
            previous = flow_labels.setdefault(flow_id, label)
            if previous != label:
                raise ValueError(f"conflicting labels for flow_id={flow_id}")
            packet_counts[label] += 1
            num_rows += 1
    flow_counts = [0] * len(names)
    for label in flow_labels.values():
        flow_counts[label] += 1
    missing = [names[index] for index, count in enumerate(flow_counts) if count == 0]
    if missing:
        raise ValueError(f"classes without training flows: {missing}")

    packet_prior = normalize([float(value) for value in packet_counts])
    flow_prior = normalize([float(value) for value in flow_counts])
    packet_weights = effective_weights(packet_counts, beta)
    flow_weights = effective_weights(flow_counts, beta)
    duplication = [
        packet_counts[index] / flow_counts[index] for index in range(len(names))
    ]
    rows = []
    for index, name in enumerate(names):
        rows.append(
            {
                "label_id": index,
                "label": name,
                "packet_count": packet_counts[index],
                "flow_count": flow_counts[index],
                "packets_per_flow": duplication[index],
                "packet_prior": packet_prior[index],
                "flow_prior": flow_prior[index],
                "packet_effective_weight": packet_weights[index],
                "flow_effective_weight": flow_weights[index],
                "flow_to_packet_weight_ratio": (
                    flow_weights[index] / packet_weights[index]
                ),
            }
        )
    duplication_mean = sum(duplication) / len(duplication)
    duplication_std = math.sqrt(
        sum((value - duplication_mean) ** 2 for value in duplication)
        / len(duplication)
    )
    return {
        "schema": "class_sampling_hierarchy_analysis_v1",
        "selection_role": "train_only_reporting_not_model_selection",
        "test_labels_used": False,
        "inputs": {
            "packet_aux_jsonl": {
                "path": str(packet_aux_jsonl.resolve()),
                "sha256": file_sha256(packet_aux_jsonl),
            },
            "label_map": {
                "path": str(label_map_path.resolve()),
                "sha256": file_sha256(label_map_path),
            },
        },
        "summary": {
            "num_packets": num_rows,
            "num_flows": len(flow_labels),
            "num_classes": len(names),
            "minimum_packet_class_count": min(packet_counts),
            "maximum_packet_class_count": max(packet_counts),
            "minimum_flow_class_count": min(flow_counts),
            "maximum_flow_class_count": max(flow_counts),
            "packet_flow_prior_total_variation": total_variation(
                packet_prior, flow_prior
            ),
            "packet_flow_prior_jensen_shannon": jensen_shannon(
                packet_prior, flow_prior
            ),
            "log_count_pearson": pearson(
                [math.log(value) for value in packet_counts],
                [math.log(value) for value in flow_counts],
            ),
            "effective_weight_log_pearson": pearson(
                [math.log(value) for value in packet_weights],
                [math.log(value) for value in flow_weights],
            ),
            "packets_per_flow_class_mean": duplication_mean,
            "packets_per_flow_class_coefficient_of_variation": (
                duplication_std / duplication_mean
            ),
            "minimum_packets_per_flow_class": min(duplication),
            "maximum_packets_per_flow_class": max(duplication),
            "effective_number_beta": beta,
        },
        "classes": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet_aux_jsonl", required=True)
    parser.add_argument("--label_map", required=True)
    parser.add_argument("--beta", type=float, default=0.9999)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()
    if not 0.0 < args.beta < 1.0:
        parser.error("--beta must be in (0,1)")
    payload = analyze(args.packet_aux_jsonl, args.label_map, beta=args.beta)
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
