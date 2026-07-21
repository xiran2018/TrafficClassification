#!/usr/bin/env python3
"""Promote a train-derived bounded hierarchy rule by validation non-inferiority."""
from __future__ import annotations

import argparse
import copy
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


MAX_DROP = 0.003
SCHEDULER = "epoch_resampled_dataloader_v1"


def noninferiority_decision(
    reference: dict[str, dict[str, float]],
    candidate: dict[str, dict[str, float]],
    *,
    max_drop: float = MAX_DROP,
) -> dict[str, Any]:
    if set(reference) != set(candidate) or not reference:
        raise ValueError("reference and candidate datasets must match")
    datasets: dict[str, Any] = {}
    for dataset in sorted(reference):
        delta_accuracy = candidate[dataset]["accuracy"] - reference[dataset]["accuracy"]
        delta_macro_f1 = candidate[dataset]["macro_f1"] - reference[dataset]["macro_f1"]
        datasets[dataset] = {
            "reference": reference[dataset],
            "candidate": candidate[dataset],
            "delta_accuracy": delta_accuracy,
            "delta_macro_f1": delta_macro_f1,
            "accuracy_passes": delta_accuracy >= -max_drop,
            "macro_f1_passes": delta_macro_f1 >= -max_drop,
            "passes": delta_accuracy >= -max_drop and delta_macro_f1 >= -max_drop,
        }
    promoted = all(row["passes"] for row in datasets.values())
    return {
        "datasets": datasets,
        "candidate_promoted_for_all_datasets": promoted,
        "selected": "candidate" if promoted else "reference",
    }


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_derivation(
    derivation: dict[str, Any],
    preregistration: dict[str, Any],
    *,
    derivation_path: Path,
) -> None:
    if not (
        derivation.get("schema") == "bounded_hierarchy_risk_protocol_v1"
        and derivation.get("status") == "derived_from_training_counts_only"
        and derivation.get("test_labels_used") is False
        and derivation.get("shared_algorithm")
        == "largest_flow_risk_power_subject_to_max_min_ratio"
    ):
        raise ValueError("invalid bounded hierarchy derivation")
    amendment = preregistration.get("bounded_risk_geometry_amendment") or {}
    binding = amendment.get("numeric_derivation") or {}
    if not (
        amendment.get("status") == "preregistered_before_complete_validation_histories"
        and amendment.get("test_labels_used") is False
        and Path(str(binding.get("path") or "")).resolve() == derivation_path.resolve()
        and binding.get("sha256") == file_sha256(derivation_path)
        and math.isclose(
            float(amendment.get("fixed_max_weight_ratio", -1.0)),
            float(derivation.get("max_weight_ratio", -2.0)),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise ValueError("bounded hierarchy derivation is not preregistered")
    protocol = amendment.get("validation_protocol") or {}
    if not (
        protocol.get("test_labels_used") is False
        and protocol.get("promotion_scope")
        == "same_bounded_risk_algorithm_must_pass_every_dataset"
        and math.isclose(
            float(protocol.get("maximum_accuracy_drop", -1.0)),
            MAX_DROP,
            abs_tol=1e-12,
        )
        and math.isclose(
            float(protocol.get("maximum_macro_f1_drop", -1.0)),
            MAX_DROP,
            abs_tol=1e-12,
        )
    ):
        raise ValueError("bounded hierarchy non-inferiority rule changed")


def validate_reference(reference: dict[str, Any]) -> None:
    if not (
        reference.get("schema") == "hierarchy_adaptive_class_weight_selection_v1"
        and reference.get("status") == "frozen_numeric_hyperparameters_from_validation"
        and reference.get("selection_scope") == "heldout_validation_only"
        and reference.get("test_labels_used") is False
    ):
        raise ValueError("invalid hierarchy reference selection")


def select(
    reference: dict[str, Any],
    derivation: dict[str, Any],
    candidate_paths: dict[str, Path],
    *,
    required_validation_points: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    validate_reference(reference)
    reference_rows = reference.get("datasets") or {}
    derivation_rows = derivation.get("datasets") or {}
    if not (
        set(reference_rows) == set(derivation_rows) == set(candidate_paths)
        and set(reference_rows) == {"vpn-app", "tls-120"}
    ):
        raise ValueError("bounded hierarchy datasets do not match")
    completion = training_completion_evidence(
        candidate_paths,
        required_validation_points,
        SCHEDULER,
    )
    if completion.get("status") != "pass":
        raise ValueError("bounded hierarchy candidate lacks complete validation evidence")

    reference_metrics: dict[str, dict[str, float]] = {}
    candidate_metrics: dict[str, dict[str, float]] = {}
    expected_trainer_hashes = set()
    candidate_trainer_hashes = set()
    for dataset in sorted(reference_rows):
        reference_row = reference_rows[dataset]
        reference_metric = Path(reference_row["selected_validation_metric"]["path"])
        if file_sha256(reference_metric) != reference_row["selected_validation_metric"]["sha256"]:
            raise ValueError(f"{dataset} hierarchy reference metric hash mismatch")
        reference_metrics[dataset] = load_validation_metrics(reference_metric)
        candidate_metrics[dataset] = load_validation_metrics(candidate_paths[dataset])

        derived_eta = float(derivation_rows[dataset]["class_weight_strength"])
        candidate_evidence = completion["datasets"][dataset]
        candidate_config = _contract_config(candidate_evidence)
        if not (
            candidate_config.get("class_weight_basis") == "flow"
            and math.isclose(
                float(candidate_config.get("class_weight_strength", -1.0)),
                derived_eta,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError(f"{dataset} bounded hierarchy training contract mismatch")
        reference_evidence = reference_row["selected_training_evidence"]
        reference_config = _contract_config(reference_evidence)
        excluded = {"class_weight_basis", "class_weight_strength"}
        keys = (set(reference_config) | set(candidate_config)) - excluded
        mismatches = sorted(
            key for key in keys if reference_config.get(key) != candidate_config.get(key)
        )
        if mismatches:
            raise ValueError(
                f"{dataset} bounded candidate differs outside hierarchy fields: {mismatches}"
            )
        expected_trainer_hashes.add(reference_evidence.get("trainer_source_sha256"))
        candidate_trainer_hashes.add(candidate_evidence.get("trainer_source_sha256"))
    if not (
        len(expected_trainer_hashes) == 1
        and candidate_trainer_hashes == expected_trainer_hashes
        and None not in expected_trainer_hashes
    ):
        raise ValueError("bounded hierarchy candidates do not share the reference trainer")
    return (
        noninferiority_decision(reference_metrics, candidate_metrics),
        completion,
    )


def build_output(
    reference: dict[str, Any],
    derivation: dict[str, Any],
    decision: dict[str, Any],
    completion: dict[str, Any],
    *,
    reference_path: Path,
    derivation_path: Path,
    preregistration_path: Path,
) -> dict[str, Any]:
    promoted = decision["candidate_promoted_for_all_datasets"]
    datasets: dict[str, Any] = {}
    for dataset, reference_row in sorted(reference["datasets"].items()):
        if not promoted:
            datasets[dataset] = copy.deepcopy(reference_row)
            continue
        eta = float(derivation["datasets"][dataset]["class_weight_strength"])
        evidence = completion["datasets"][dataset]
        metrics = decision["datasets"][dataset]["candidate"]
        datasets[dataset] = {
            "selected_eta": eta,
            "training_hyperparameters": {
                "class_weight_basis": "flow",
                "class_weight_strength": eta,
            },
            "arms": {
                "reference": {
                    **decision["datasets"][dataset]["reference"],
                    "eta": float(reference_row["selected_eta"]),
                    "eligible": True,
                },
                "bounded_risk": {
                    **metrics,
                    "eta": eta,
                    "eligible": True,
                    "noninferiority_passed": True,
                },
            },
            "selected_validation_metric": {
                "path": evidence["metric_path"],
                "sha256": evidence["metric_sha256"],
            },
            "selected_training_evidence": evidence,
        }
    inputs = copy.deepcopy(reference.get("inputs") or {})
    inputs.update(
        {
            "reference_hierarchy_selection": {
                "path": str(reference_path.resolve()),
                "sha256": file_sha256(reference_path),
            },
            "bounded_risk_derivation": {
                "path": str(derivation_path.resolve()),
                "sha256": file_sha256(derivation_path),
            },
            "preregistration": {
                "path": str(preregistration_path.resolve()),
                "sha256": file_sha256(preregistration_path),
            },
        }
    )
    return {
        "schema": "hierarchy_adaptive_class_weight_selection_v1",
        "status": "frozen_numeric_hyperparameters_from_validation",
        "selection_scope": "heldout_validation_only",
        "test_labels_used": False,
        "shared_algorithm": (
            "bounded_effective_flow_class_risk_power_eta"
            if promoted
            else reference["shared_algorithm"]
        ),
        "numeric_protocol_selection": {
            "schema": "bounded_hierarchy_risk_selection_v1",
            "metric": "accuracy_and_macro_f1_noninferiority",
            "maximum_drop": MAX_DROP,
            "promotion_scope": "same_bounded_risk_algorithm_must_pass_every_dataset",
            **decision,
            "candidate_training_completion_evidence": completion,
        },
        "datasets": datasets,
        "inputs": inputs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference_selection", required=True)
    parser.add_argument("--derivation", required=True)
    parser.add_argument("--preregistration", required=True)
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--required_validation_points", type=int, default=8)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()
    reference_path = Path(args.reference_selection)
    derivation_path = Path(args.derivation)
    preregistration_path = Path(args.preregistration)
    reference = load_json(reference_path)
    derivation = load_json(derivation_path)
    preregistration = load_json(preregistration_path)
    validate_derivation(
        derivation,
        preregistration,
        derivation_path=derivation_path,
    )
    decision, completion = select(
        reference,
        derivation,
        parse_named_paths(args.candidate),
        required_validation_points=args.required_validation_points,
    )
    payload = build_output(
        reference,
        derivation,
        decision,
        completion,
        reference_path=reference_path,
        derivation_path=derivation_path,
        preregistration_path=preregistration_path,
    )
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
