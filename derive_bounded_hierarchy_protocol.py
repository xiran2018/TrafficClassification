#!/usr/bin/env python3
"""Derive risk-bounded hierarchy strengths from training counts only."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from hierarchy_class_weights import (
    bounded_flow_risk_strength,
    hierarchy_class_weights,
)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_named_paths(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"expected DATASET=PATH, got {value}")
        dataset, raw_path = value.split("=", 1)
        if not dataset or dataset in result:
            raise ValueError(f"invalid or duplicate dataset: {dataset}")
        result[dataset] = Path(raw_path)
    if not result:
        raise ValueError("at least one training hierarchy report is required")
    return result


def derive(
    reports: dict[str, Path],
    *,
    max_weight_ratio: float,
    beta: float,
) -> dict[str, Any]:
    datasets: dict[str, Any] = {}
    for dataset, path in sorted(reports.items()):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not (
            payload.get("schema") == "class_sampling_hierarchy_analysis_v1"
            and payload.get("selection_role")
            == "train_only_reporting_not_model_selection"
            and payload.get("test_labels_used") is False
        ):
            raise ValueError(f"{dataset} input is not a train-only hierarchy report")
        rows = payload.get("classes") or []
        packet_counts = {
            int(row["label_id"]): int(row["packet_count"]) for row in rows
        }
        flow_counts = {
            int(row["label_id"]): int(row["flow_count"]) for row in rows
        }
        if (
            not rows
            or len(packet_counts) != len(rows)
            or set(packet_counts) != set(flow_counts)
        ):
            raise ValueError(f"{dataset} hierarchy class rows are incomplete")
        observed_beta = float((payload.get("summary") or {})["effective_number_beta"])
        if abs(observed_beta - beta) > 1e-12:
            raise ValueError(f"{dataset} hierarchy beta disagrees with the protocol")
        eta = bounded_flow_risk_strength(
            flow_counts,
            max_weight_ratio=max_weight_ratio,
            beta=beta,
        )
        weights = hierarchy_class_weights(
            packet_counts,
            flow_counts,
            alpha=1.0,
            gamma=eta,
            beta=beta,
        )
        realized_ratio = max(weights.values()) / min(weights.values())
        datasets[dataset] = {
            "class_weight_basis": "flow",
            "class_weight_strength": eta,
            "unbounded_effective_weight_ratio": (
                max(float(row["flow_effective_weight"]) for row in rows)
                / min(float(row["flow_effective_weight"]) for row in rows)
            ),
            "bounded_effective_weight_ratio": realized_ratio,
            "num_classes": len(rows),
            "input": {
                "path": str(path.resolve()),
                "sha256": file_sha256(path),
            },
        }
    return {
        "schema": "bounded_hierarchy_risk_protocol_v1",
        "status": "derived_from_training_counts_only",
        "selection_role": "candidate_numeric_derivation_not_model_selection",
        "test_labels_used": False,
        "shared_algorithm": "largest_flow_risk_power_subject_to_max_min_ratio",
        "formula": "eta=min(1, log(R)/log(max(w_flow)/min(w_flow)))",
        "max_weight_ratio": max_weight_ratio,
        "effective_number_beta": beta,
        "datasets": datasets,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hierarchy_report", action="append", required=True)
    parser.add_argument("--max_weight_ratio", type=float, default=4.0)
    parser.add_argument("--beta", type=float, default=0.9999)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()
    payload = derive(
        parse_named_paths(args.hierarchy_report),
        max_weight_ratio=args.max_weight_ratio,
        beta=args.beta,
    )
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
