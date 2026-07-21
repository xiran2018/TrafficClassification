#!/usr/bin/env python3
"""Apply the preregistered cross-dataset Flow validation non-inferiority gate."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from freeze_shared_core_v2_config import file_sha256


MAX_DROP = 0.003


def parse_named_paths(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"expected DATASET=PATH, got {value}")
        dataset, raw_path = value.split("=", 1)
        if not dataset or dataset in result:
            raise ValueError(f"invalid or duplicate dataset: {dataset}")
        result[dataset] = Path(raw_path)
    return result


def load_valid_metrics(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("evaluation_split") != "valid":
        raise ValueError(f"flow selection input is not explicitly valid-only: {path}")
    metrics = payload.get("metrics") or {}
    result = {
        "accuracy": float(metrics["accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
    }
    if any(not math.isfinite(value) for value in result.values()):
        raise ValueError(f"non-finite flow validation metric: {path}")
    return {
        **result,
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
    }


def select(
    baseline_paths: dict[str, Path], candidate_paths: dict[str, Path]
) -> dict[str, Any]:
    if set(baseline_paths) != {"vpn-app", "tls-120"} or set(candidate_paths) != set(
        baseline_paths
    ):
        raise ValueError("flow gate requires exactly VPN and TLS-120 in both arms")
    datasets: dict[str, Any] = {}
    for dataset in sorted(baseline_paths):
        baseline = load_valid_metrics(baseline_paths[dataset])
        candidate = load_valid_metrics(candidate_paths[dataset])
        delta_accuracy = candidate["accuracy"] - baseline["accuracy"]
        delta_macro_f1 = candidate["macro_f1"] - baseline["macro_f1"]
        accuracy_passes = delta_accuracy >= -MAX_DROP
        macro_f1_passes = delta_macro_f1 >= -MAX_DROP
        datasets[dataset] = {
            "baseline": baseline,
            "candidate": candidate,
            "delta_accuracy": delta_accuracy,
            "delta_macro_f1": delta_macro_f1,
            "accuracy_passes": accuracy_passes,
            "macro_f1_passes": macro_f1_passes,
            "passes": accuracy_passes and macro_f1_passes,
        }
    promoted = all(row["passes"] for row in datasets.values())
    return {
        "schema": "flow_noninferiority_selection_v1",
        "selection_scope": "heldout_validation_only",
        "metric": "accuracy_and_macro_f1_noninferiority",
        "promotion_scope": "same_candidate_must_pass_every_dataset",
        "thresholds": {
            "accuracy_max_drop": MAX_DROP,
            "macro_f1_max_drop": MAX_DROP,
        },
        "datasets": datasets,
        "candidate_promoted_for_all_datasets": promoted,
        "selected": "candidate" if promoted else "baseline",
        "test_labels_used": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", action="append", required=True)
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()
    payload = select(
        parse_named_paths(args.baseline), parse_named_paths(args.candidate)
    )
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
