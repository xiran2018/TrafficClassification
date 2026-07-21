#!/usr/bin/env python3
"""Select one class-weight protocol jointly on VPN and TLS validation."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

from select_unified_tower1_candidate import (
    file_sha256,
    load_validation_metrics,
    parse_named_paths,
    training_completion_evidence,
    training_dynamics_evidence,
)


ARM_CONFIGS = {
    "packet_full": {"class_weight_basis": "packet", "class_weight_strength": 1.0},
    "flow_sqrt": {"class_weight_basis": "flow", "class_weight_strength": 0.5},
    "flow_full": {"class_weight_basis": "flow", "class_weight_strength": 1.0},
}

DECLARED_FACTORIAL_FIELDS = {"class_weight_basis", "class_weight_strength"}


def _contract_config(row: dict[str, Any]) -> dict[str, Any]:
    path = Path(str(row.get("provenance_path") or ""))
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "tower1_training_contract_v1":
        raise ValueError(f"not a Tower1 training contract: {path}")
    return payload["training_config"]


def _validate_arm_contracts(arm: str, completion: dict[str, Any]) -> None:
    expected = ARM_CONFIGS[arm]
    for dataset, row in completion["datasets"].items():
        config = _contract_config(row)
        if config.get("class_weight_basis") != expected["class_weight_basis"]:
            raise ValueError(f"{arm} {dataset} has wrong class_weight_basis")
        if not math.isclose(
            float(config.get("class_weight_strength", -1.0)),
            float(expected["class_weight_strength"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"{arm} {dataset} has wrong class_weight_strength")


def _implementation_consistency(completions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows = [
        row
        for completion in completions.values()
        for row in completion["datasets"].values()
    ]
    hashes = sorted(
        {
            str(row["trainer_source_sha256"])
            for row in rows
            if row.get("trainer_source_sha256")
        }
    )
    stable = bool(rows) and all(
        row.get("trainer_source_stable_through_completion") is True for row in rows
    )
    passed = bool(stable and len(hashes) == 1)
    return {
        "required": True,
        "status": "pass" if passed else "fail",
        "num_runs": len(rows),
        "trainer_source_sha256": hashes[0] if len(hashes) == 1 else None,
        "observed_trainer_source_sha256": hashes,
        "all_runs_stable_through_completion": stable,
    }


def _factorial_config_integrity(
    completions: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Prove that each dataset's arms differ only in declared factors."""
    datasets = sorted(completions["packet_full"]["datasets"])
    evidence: dict[str, Any] = {}
    passed = True
    for dataset in datasets:
        configs = {
            arm: _contract_config(completion["datasets"][dataset])
            for arm, completion in completions.items()
        }
        controlled = {
            arm: {
                key: value
                for key, value in config.items()
                if key not in DECLARED_FACTORIAL_FIELDS
            }
            for arm, config in configs.items()
        }
        baseline = controlled["packet_full"]
        mismatched_fields: dict[str, list[str]] = {}
        all_fields = sorted(set().union(*(set(config) for config in controlled.values())))
        for arm in ("flow_sqrt", "flow_full"):
            differences = [
                key
                for key in all_fields
                if baseline.get(key) != controlled[arm].get(key)
            ]
            if differences:
                mismatched_fields[arm] = differences
        dataset_passed = not mismatched_fields
        passed = passed and dataset_passed
        evidence[dataset] = {
            "status": "pass" if dataset_passed else "fail",
            "reference_arm": "packet_full",
            "mismatched_fields": mismatched_fields,
        }
    return {
        "required": True,
        "status": "pass" if passed else "fail",
        "declared_factorial_fields": sorted(DECLARED_FACTORIAL_FIELDS),
        "datasets": evidence,
    }


def _evaluate_candidate(
    baseline_paths: dict[str, Path],
    candidate_paths: dict[str, Path],
    min_delta: float,
    max_accuracy_drop: float,
) -> dict[str, Any]:
    datasets: dict[str, Any] = {}
    for dataset in sorted(baseline_paths):
        baseline = load_validation_metrics(baseline_paths[dataset])
        candidate = load_validation_metrics(candidate_paths[dataset])
        delta_f1 = candidate["macro_f1"] - baseline["macro_f1"]
        delta_accuracy = candidate["accuracy"] - baseline["accuracy"]
        f1_pass = delta_f1 >= min_delta
        accuracy_pass = delta_accuracy >= -max_accuracy_drop
        datasets[dataset] = {
            "baseline_accuracy": baseline["accuracy"],
            "candidate_accuracy": candidate["accuracy"],
            "delta_accuracy": delta_accuracy,
            "baseline_macro_f1": baseline["macro_f1"],
            "candidate_macro_f1": candidate["macro_f1"],
            "delta_macro_f1": delta_f1,
            "macro_f1_passes": f1_pass,
            "accuracy_guard_passes": accuracy_pass,
            "passes": f1_pass and accuracy_pass,
        }
    f1_deltas = [row["delta_macro_f1"] for row in datasets.values()]
    accuracy_deltas = [row["delta_accuracy"] for row in datasets.values()]
    return {
        "passes_all_datasets": all(row["passes"] for row in datasets.values()),
        "datasets": datasets,
        "ranking_key": {
            "minimum_macro_f1_delta": min(f1_deltas),
            "mean_macro_f1_delta": statistics.fmean(f1_deltas),
            "minimum_accuracy_delta": min(accuracy_deltas),
        },
    }


def _rank_key(row: dict[str, Any], arm: str) -> tuple[float, float, float, int]:
    key = row["ranking_key"]
    # Prefer sqrt only after all declared performance tie-breaks are exactly tied.
    simplicity = 1 if arm == "flow_sqrt" else 0
    return (
        float(key["minimum_macro_f1_delta"]),
        float(key["mean_macro_f1_delta"]),
        float(key["minimum_accuracy_delta"]),
        simplicity,
    )


def choose_protocol(candidates: dict[str, dict[str, Any]]) -> tuple[str, str, list[str]]:
    expected = {"flow_sqrt", "flow_full"}
    if set(candidates) != expected:
        raise ValueError(f"expected candidate arms {sorted(expected)}")
    eligible = [arm for arm, row in candidates.items() if row["passes_all_datasets"]]
    ranked = sorted(
        candidates,
        key=lambda arm: _rank_key(candidates[arm], arm),
        reverse=True,
    )
    comparison_arm = ranked[0]
    selected_arm = (
        max(eligible, key=lambda arm: _rank_key(candidates[arm], arm))
        if eligible
        else "packet_full"
    )
    return selected_arm, comparison_arm, ranked


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", action="append", required=True, help="DATASET=PATH")
    parser.add_argument("--flow_sqrt", action="append", required=True, help="DATASET=PATH")
    parser.add_argument("--flow_full", action="append", required=True, help="DATASET=PATH")
    parser.add_argument("--min_delta", type=float, default=0.005)
    parser.add_argument("--max_accuracy_drop", type=float, default=0.005)
    parser.add_argument("--required_validation_points", type=int, default=8)
    parser.add_argument(
        "--required_packet_batch_scheduler",
        default="epoch_resampled_dataloader_v1",
    )
    parser.add_argument("--preregistration", required=True)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    paths = {
        "packet_full": parse_named_paths(args.packet),
        "flow_sqrt": parse_named_paths(args.flow_sqrt),
        "flow_full": parse_named_paths(args.flow_full),
    }
    datasets = set(paths["packet_full"])
    if len(datasets) < 2 or any(set(value) != datasets for value in paths.values()):
        raise ValueError("every arm must contain the same two or more datasets")

    completions = {
        arm: training_completion_evidence(
            arm_paths,
            args.required_validation_points,
            args.required_packet_batch_scheduler,
        )
        for arm, arm_paths in paths.items()
    }
    for arm, completion in completions.items():
        if completion["status"] != "pass":
            raise ValueError(f"{arm} does not have complete verified training evidence")
        _validate_arm_contracts(arm, completion)
    implementation = _implementation_consistency(completions)
    if implementation["status"] != "pass":
        raise ValueError("all class-weight arms must use one stable trainer source")
    factorial_integrity = _factorial_config_integrity(completions)
    if factorial_integrity["status"] != "pass":
        raise ValueError(
            "class-weight arms differ outside the declared factorial fields"
        )

    preregistration = Path(args.preregistration)
    prereg = json.loads(preregistration.read_text(encoding="utf-8"))
    if prereg.get("schema") != "cross_dataset_class_weight_protocol_preregistration_v1":
        raise ValueError("unexpected preregistration schema")
    gate = prereg.get("promotion_gate") or {}
    fixed = prereg.get("fixed_factors") or {}
    if not (
        prereg.get("scope") == "heldout_fold0_validation_only"
        and prereg.get("test_access") == "forbidden"
        and set(prereg.get("datasets") or []) == datasets
        and math.isclose(
            float(gate.get("minimum_macro_f1_delta_per_dataset", -1.0)),
            args.min_delta,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and math.isclose(
            float(gate.get("maximum_accuracy_drop_per_dataset", -1.0)),
            args.max_accuracy_drop,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and gate.get("same_arm_must_pass_every_dataset") is True
        and int(fixed.get("required_validation_points", 0))
        == args.required_validation_points
        and fixed.get("packet_batch_scheduler")
        == args.required_packet_batch_scheduler
        and fixed.get("trainer_source_sha256")
        == implementation["trainer_source_sha256"]
    ):
        raise ValueError("runtime selection arguments disagree with preregistration")

    candidates = {
        arm: _evaluate_candidate(
            paths["packet_full"], paths[arm], args.min_delta, args.max_accuracy_drop
        )
        for arm in ("flow_sqrt", "flow_full")
    }
    selected_arm, comparison_arm, ranked_flow_arms = choose_protocol(candidates)
    eligible = [arm for arm, row in candidates.items() if row["passes_all_datasets"]]
    selected_flow = selected_arm != "packet_full"
    comparison = candidates[comparison_arm]

    selected_candidate_arm = selected_arm if selected_flow else comparison_arm
    selected_candidate_completion = completions[selected_candidate_arm]
    selected_implementation = _implementation_consistency(
        {
            "baseline": completions["packet_full"],
            "candidate": selected_candidate_completion,
        }
    )
    payload = {
        "selection_scope": "heldout_validation_only",
        "metric": "macro_f1_with_accuracy_guard",
        "promotion_scope": "same_candidate_must_pass_every_dataset",
        "num_datasets": len(datasets),
        "min_delta": args.min_delta,
        "max_accuracy_drop": args.max_accuracy_drop,
        "candidate_promoted_for_all_datasets": selected_flow,
        "selected": "candidate" if selected_flow else "baseline",
        "datasets": candidates[selected_candidate_arm]["datasets"],
        "training_completion_evidence": {
            "baseline": completions["packet_full"],
            "candidate": selected_candidate_completion,
        },
        "training_implementation_consistency": selected_implementation,
        "training_dynamics_evidence": training_dynamics_evidence(
            paths["packet_full"], paths[selected_candidate_arm]
        ),
        "multi_arm_selection": {
            "schema": "cross_dataset_class_weight_protocol_selection_v1",
            "preregistration_path": str(preregistration.resolve()),
            "preregistration_sha256": file_sha256(preregistration),
            "selected_protocol": selected_arm,
            "selected_config": ARM_CONFIGS[selected_arm],
            "comparison_arm_when_baseline_selected": comparison_arm,
            "eligible_flow_arms": eligible,
            "ranked_flow_arms": ranked_flow_arms,
            "candidates": candidates,
            "all_arm_training_completion_evidence": completions,
            "all_arm_training_implementation_consistency": implementation,
            "factorial_config_integrity": factorial_integrity,
        },
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
