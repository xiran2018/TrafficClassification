#!/usr/bin/env python3
"""Freeze dataset-specific numeric hierarchy strengths from validation only."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from select_class_weight_protocol import _contract_config
from select_unified_tower1_candidate import (
    file_sha256,
    load_validation_metrics,
    parse_named_paths,
    training_completion_evidence,
)


ETA_TO_ARM = {0.0: "packet_full", 0.5: "flow_sqrt", 1.0: "flow_full"}


def select_eta(
    metrics: dict[float, dict[str, float]],
    *,
    min_delta: float,
    max_accuracy_drop: float,
) -> dict[str, Any]:
    if 0.0 not in metrics:
        raise ValueError("eta=0 reference metrics are required")
    reference = metrics[0.0]
    arms: dict[str, Any] = {}
    for eta in sorted(metrics):
        row = metrics[eta]
        delta_f1 = float(row["macro_f1"]) - float(reference["macro_f1"])
        delta_accuracy = float(row["accuracy"]) - float(reference["accuracy"])
        eligible = eta == 0.0 or (
            delta_f1 >= min_delta and delta_accuracy >= -max_accuracy_drop
        )
        arms[str(eta)] = {
            "eta": eta,
            "accuracy": float(row["accuracy"]),
            "macro_f1": float(row["macro_f1"]),
            "delta_accuracy": delta_accuracy,
            "delta_macro_f1": delta_f1,
            "eligible": eligible,
        }
    eligible = [row for row in arms.values() if row["eligible"]]
    selected = max(
        eligible,
        key=lambda row: (row["macro_f1"], row["accuracy"], -row["eta"]),
    )
    return {"selected_eta": selected["eta"], "arms": arms}


def validate_gate_and_prereg(
    gate: dict[str, Any], prereg: dict[str, Any], *, gate_path: Path, prereg_path: Path
) -> None:
    if gate.get("schema") != "hierarchy_adaptive_class_weight_gate_v1":
        raise ValueError("unexpected hierarchy gate schema")
    if gate.get("selection_scope") != "heldout_validation_only":
        raise ValueError("hierarchy gate is not validation-only")
    if gate.get("test_labels_used") is not False:
        raise ValueError("hierarchy gate used test labels")
    if prereg.get("schema") != "hierarchy_adaptive_class_weight_preregistration_v1":
        raise ValueError("unexpected hierarchy preregistration schema")
    if prereg.get("status") != "preregistered_before_complete_validation_histories":
        raise ValueError("hierarchy method was not preregistered in time")
    gate_prereg = (gate.get("inputs") or {}).get("preregistration") or {}
    if not (
        Path(str(gate_prereg.get("path") or "")).resolve() == prereg_path.resolve()
        and gate_prereg.get("sha256") == file_sha256(prereg_path)
    ):
        raise ValueError("hierarchy gate is not bound to this preregistration")
    if gate.get("status") not in {"launch", "do_not_launch"}:
        raise ValueError(f"invalid hierarchy gate status in {gate_path}")


def existing_metric_paths(class_selection: dict[str, Any]) -> dict[float, dict[str, Path]]:
    completions = class_selection["multi_arm_selection"][
        "all_arm_training_completion_evidence"
    ]
    return {
        eta: {
            dataset: Path(row["metric_path"])
            for dataset, row in completions[arm]["datasets"].items()
        }
        for eta, arm in ETA_TO_ARM.items()
    }


def validate_eta025_completion(
    paths: dict[str, Path],
    class_selection: dict[str, Any],
    *,
    required_points: int,
    scheduler: str,
) -> dict[str, Any]:
    completion = training_completion_evidence(paths, required_points, scheduler)
    if completion.get("status") != "pass":
        raise ValueError("eta=0.25 lacks complete verified validation evidence")
    expected_sha = class_selection["multi_arm_selection"][
        "all_arm_training_implementation_consistency"
    ]["trainer_source_sha256"]
    reference_rows = class_selection["multi_arm_selection"][
        "all_arm_training_completion_evidence"
    ]["packet_full"]["datasets"]
    excluded = {"class_weight_basis", "class_weight_strength"}
    for dataset, row in completion["datasets"].items():
        config = _contract_config(row)
        if not (
            config.get("class_weight_basis") == "flow"
            and math.isclose(
                float(config.get("class_weight_strength", -1.0)),
                0.25,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            and row.get("trainer_source_stable_through_completion") is True
            and row.get("trainer_source_sha256") == expected_sha
            and row.get("completion_trainer_source_sha256") == expected_sha
        ):
            raise ValueError(f"eta=0.25 contract mismatch for {dataset}")
        reference = _contract_config(reference_rows[dataset])
        keys = (set(reference) | set(config)) - excluded
        differences = sorted(key for key in keys if reference.get(key) != config.get(key))
        if differences:
            raise ValueError(
                f"eta=0.25 {dataset} differs outside class hierarchy fields: {differences}"
            )
    return completion


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", required=True)
    parser.add_argument("--preregistration", required=True)
    parser.add_argument("--class_weight_selection", required=True)
    parser.add_argument("--eta025", action="append", default=[], help="DATASET=PATH")
    parser.add_argument("--required_validation_points", type=int, default=8)
    parser.add_argument(
        "--required_packet_batch_scheduler",
        default="epoch_resampled_dataloader_v1",
    )
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    gate_path = Path(args.gate)
    prereg_path = Path(args.preregistration)
    class_path = Path(args.class_weight_selection)
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    class_selection = json.loads(class_path.read_text(encoding="utf-8"))
    validate_gate_and_prereg(
        gate, prereg, gate_path=gate_path, prereg_path=prereg_path
    )
    gate_class = (gate.get("inputs") or {}).get("class_weight_selection") or {}
    if not (
        Path(str(gate_class.get("path") or "")).resolve() == class_path.resolve()
        and gate_class.get("sha256") == file_sha256(class_path)
    ):
        raise ValueError("hierarchy gate is not bound to this class selection")

    paths = existing_metric_paths(class_selection)
    completions_by_eta = {
        eta: class_selection["multi_arm_selection"][
            "all_arm_training_completion_evidence"
        ][arm]
        for eta, arm in ETA_TO_ARM.items()
    }
    datasets = sorted(paths[0.0])
    if datasets != sorted((prereg["launch_gate"])["required_datasets"]):
        raise ValueError("hierarchy datasets disagree with preregistration")
    eta025_completion = None
    if gate["launch"] is True:
        eta025 = parse_named_paths(args.eta025)
        if sorted(eta025) != datasets:
            raise ValueError("launched hierarchy screen requires eta=0.25 for every dataset")
        eta025_completion = validate_eta025_completion(
            eta025,
            class_selection,
            required_points=args.required_validation_points,
            scheduler=args.required_packet_batch_scheduler,
        )
        paths[0.25] = eta025
        completions_by_eta[0.25] = eta025_completion
    elif args.eta025:
        raise ValueError("eta=0.25 evidence is forbidden when launch gate is false")

    eligibility = prereg["launch_gate"]["eligibility"]
    min_delta = float(eligibility["minimum_macro_f1_gain_over_packet_full"])
    max_drop = float(eligibility["maximum_accuracy_drop_from_packet_full"])
    selections: dict[str, Any] = {}
    for dataset in datasets:
        metrics = {
            eta: load_validation_metrics(dataset_paths[dataset])
            for eta, dataset_paths in paths.items()
        }
        selections[dataset] = select_eta(
            metrics, min_delta=min_delta, max_accuracy_drop=max_drop
        )
        selected_eta = float(selections[dataset]["selected_eta"])
        selected_metric_path = paths[selected_eta][dataset]
        selections[dataset]["selected_validation_metric"] = {
            "path": str(selected_metric_path.resolve()),
            "sha256": file_sha256(selected_metric_path),
        }
        selections[dataset]["selected_training_evidence"] = completions_by_eta[
            selected_eta
        ]["datasets"][dataset]
        selections[dataset]["training_hyperparameters"] = {
            "class_weight_basis": "flow",
            "class_weight_strength": selected_eta,
        }

    payload = {
        "schema": "hierarchy_adaptive_class_weight_selection_v1",
        "status": "frozen_numeric_hyperparameters_from_validation",
        "selection_scope": "heldout_validation_only",
        "test_labels_used": False,
        "shared_algorithm": "normalized_effective_flow_class_risk_power_eta",
        "datasets": selections,
        "thresholds": {
            "minimum_macro_f1_gain": min_delta,
            "maximum_accuracy_drop": max_drop,
        },
        "eta025_training_completion_evidence": eta025_completion,
        "inputs": {
            "gate": {"path": str(gate_path.resolve()), "sha256": file_sha256(gate_path)},
            "preregistration": {
                "path": str(prereg_path.resolve()),
                "sha256": file_sha256(prereg_path),
            },
            "class_weight_selection": {
                "path": str(class_path.resolve()),
                "sha256": file_sha256(class_path),
            },
        },
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
