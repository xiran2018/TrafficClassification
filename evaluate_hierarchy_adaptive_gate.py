#!/usr/bin/env python3
"""Evaluate the preregistered trigger for continuous packet/flow class weighting."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ARM_PARAMETERS = {
    "packet_full": {"alpha": 0.0, "gamma": 1.0},
    "flow_sqrt": {"alpha": 1.0, "gamma": 0.5},
    "flow_full": {"alpha": 1.0, "gamma": 1.0},
}


def followup_search_space(prereg: dict[str, Any]) -> dict[str, Any] | None:
    amendment = prereg.get("train_only_identifiability_amendment")
    if amendment is None:
        return None
    if amendment.get("status") != "preregistered_before_complete_validation_histories":
        raise ValueError("identifiability amendment was not preregistered in time")
    if amendment.get("test_labels_used") is not False:
        raise ValueError("identifiability amendment does not forbid test labels")
    rule = amendment.get("execution_rule") or {}
    eta_grid = [float(value) for value in rule.get("unique_effective_eta_grid") or []]
    if eta_grid != sorted(set(eta_grid)) or not eta_grid:
        raise ValueError("effective eta grid must be sorted and unique")
    canonical = rule.get("canonical_parameterization") or {}
    if len(canonical) != len(eta_grid):
        raise ValueError("canonical eta parameterizations are incomplete")
    for eta in eta_grid:
        parameters = canonical.get(str(eta))
        if parameters is None:
            raise ValueError(f"missing canonical parameterization for eta={eta}")
        product = float(parameters["alpha"]) * float(parameters["gamma"])
        if abs(product - eta) > 1e-12:
            raise ValueError(f"canonical parameterization disagrees for eta={eta}")
    duplicate = rule.get("duplicate_grid_point") or {}
    duplicate_of = rule.get("duplicate_of") or {}
    duplicate_eta = float(duplicate["alpha"]) * float(duplicate["gamma"])
    canonical_eta = float(duplicate_of["alpha"]) * float(duplicate_of["gamma"])
    if abs(duplicate_eta - canonical_eta) > 1e-12:
        raise ValueError("declared duplicate parameterizations are not equivalent")
    if duplicate_eta not in eta_grid:
        raise ValueError("duplicate parameterization is outside the eta grid")
    return {
        "identifiable_parameter": "eta=alpha*gamma",
        "unique_effective_eta_grid": eta_grid,
        "canonical_parameterization": canonical,
        "omitted_redundant_parameterization": duplicate,
        "equivalent_to": duplicate_of,
    }


def verify_train_only_amendment_evidence(prereg: dict[str, Any]) -> dict[str, Any] | None:
    amendment = prereg.get("train_only_identifiability_amendment")
    if amendment is None:
        return None
    search_space = followup_search_space(prereg)
    verified: dict[str, Any] = {}
    for dataset, expected in sorted((amendment.get("evidence") or {}).items()):
        path = Path(str(expected.get("path") or ""))
        expected_hash = str(expected.get("sha256") or "")
        if not path.is_file() or file_sha256(path) != expected_hash:
            raise ValueError(f"{dataset} identifiability evidence hash mismatch")
        report = load_json(path)
        if report.get("test_labels_used") is not False:
            raise ValueError(f"{dataset} identifiability evidence used test labels")
        summary = report.get("summary") or {}
        minimum = int(summary["minimum_packet_class_count"])
        maximum = int(summary["maximum_packet_class_count"])
        if minimum != maximum:
            raise ValueError(f"{dataset} packet class counts are not constant")
        for field in (
            "minimum_packet_class_count",
            "maximum_packet_class_count",
            "minimum_flow_class_count",
            "maximum_flow_class_count",
        ):
            if int(summary[field]) != int(expected[field]):
                raise ValueError(f"{dataset} evidence summary mismatch for {field}")
        verified[dataset] = {
            "path": str(path.resolve()),
            "sha256": expected_hash,
            "packet_class_count": minimum,
            "flow_class_count_range": [
                int(summary["minimum_flow_class_count"]),
                int(summary["maximum_flow_class_count"]),
            ],
        }
    if not verified:
        raise ValueError("identifiability amendment has no evidence")
    return {"search_space": search_space, "verified_train_only_evidence": verified}


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_inputs(report: dict[str, Any], prereg: dict[str, Any]) -> None:
    multi = report.get("multi_arm_selection") or {}
    if multi.get("schema") != "cross_dataset_class_weight_protocol_selection_v1":
        raise ValueError("class-weight report is not a completed multi-arm selection")
    if report.get("selection_scope") != "heldout_validation_only":
        raise ValueError("class-weight report is not validation-only")
    if prereg.get("schema") != "hierarchy_adaptive_class_weight_preregistration_v1":
        raise ValueError("unexpected hierarchy-adaptive preregistration schema")
    if prereg.get("status") != "preregistered_before_complete_validation_histories":
        raise ValueError("hierarchy-adaptive followup was not preregistered in time")
    if (prereg.get("method") or {}).get("test_labels_used") is not False:
        raise ValueError("hierarchy-adaptive method does not forbid test labels")
    if (prereg.get("fixed_followup_grid") or {}).get("test_labels_used") is not False:
        raise ValueError("hierarchy-adaptive selection does not forbid test labels")

    required_points = int(
        (prereg.get("launch_gate") or {}).get("required_validation_points_per_arm", 0)
    )
    completions = multi.get("all_arm_training_completion_evidence") or {}
    if set(completions) != set(ARM_PARAMETERS):
        raise ValueError("class-weight report does not contain all preregistered arms")
    for arm, completion in completions.items():
        if not (
            completion.get("status") == "pass"
            and int(completion.get("required_validation_points", 0)) == required_points
            and all(
                row.get("passed") is True
                and int(row.get("validation_points", 0)) == required_points
                for row in (completion.get("datasets") or {}).values()
            )
        ):
            raise ValueError(f"{arm} lacks complete validation evidence")
    factorial = multi.get("factorial_config_integrity") or {}
    if factorial.get("status") != "pass":
        raise ValueError("class-weight factorial integrity did not pass")


def choose_dataset_arm(
    dataset: str,
    candidates: dict[str, Any],
    *,
    min_delta: float,
    max_accuracy_drop: float,
) -> tuple[str, dict[str, Any]]:
    reference = candidates["flow_sqrt"]["datasets"][dataset]
    arms: dict[str, dict[str, Any]] = {
        "packet_full": {
            "accuracy": float(reference["baseline_accuracy"]),
            "macro_f1": float(reference["baseline_macro_f1"]),
            "delta_accuracy": 0.0,
            "delta_macro_f1": 0.0,
            "eligible": True,
        }
    }
    for arm in ("flow_sqrt", "flow_full"):
        row = candidates[arm]["datasets"][dataset]
        delta_f1 = float(row["delta_macro_f1"])
        delta_accuracy = float(row["delta_accuracy"])
        arms[arm] = {
            "accuracy": float(row["candidate_accuracy"]),
            "macro_f1": float(row["candidate_macro_f1"]),
            "delta_accuracy": delta_accuracy,
            "delta_macro_f1": delta_f1,
            "eligible": bool(
                delta_f1 >= min_delta and delta_accuracy >= -max_accuracy_drop
            ),
        }
    eligible = [arm for arm, row in arms.items() if row["eligible"]]

    def ranking_key(arm: str) -> tuple[float, float, float, float]:
        params = ARM_PARAMETERS[arm]
        return (
            arms[arm]["macro_f1"],
            arms[arm]["accuracy"],
            -params["alpha"],
            -params["gamma"],
        )

    selected = max(eligible, key=ranking_key)
    return selected, {"selected": selected, "arms": arms}


def evaluate(report: dict[str, Any], prereg: dict[str, Any]) -> dict[str, Any]:
    validate_inputs(report, prereg)
    multi = report["multi_arm_selection"]
    candidates = multi["candidates"]
    datasets = sorted(candidates["flow_sqrt"]["datasets"])
    expected_datasets = sorted((prereg.get("launch_gate") or {})["required_datasets"])
    if datasets != expected_datasets:
        raise ValueError("selection datasets disagree with preregistration")
    eligibility = (prereg.get("launch_gate") or {})["eligibility"]
    min_delta = float(eligibility["minimum_macro_f1_gain_over_packet_full"])
    max_drop = float(eligibility["maximum_accuracy_drop_from_packet_full"])
    selections = {
        dataset: choose_dataset_arm(
            dataset,
            candidates,
            min_delta=min_delta,
            max_accuracy_drop=max_drop,
        )[1]
        for dataset in datasets
    }
    selected_arms = {row["selected"] for row in selections.values()}
    launch = len(selected_arms) > 1
    payload = {
        "schema": "hierarchy_adaptive_class_weight_gate_v1",
        "status": "launch" if launch else "do_not_launch",
        "launch": launch,
        "reason": (
            "validation_best_eligible_arms_diverge_across_datasets"
            if launch
            else "one_common_validation_best_eligible_arm"
        ),
        "selection_scope": "heldout_validation_only",
        "test_labels_used": False,
        "thresholds": {
            "minimum_macro_f1_gain": min_delta,
            "maximum_accuracy_drop": max_drop,
        },
        "datasets": selections,
        "selected_numeric_corners": {
            dataset: ARM_PARAMETERS[row["selected"]]
            for dataset, row in selections.items()
        },
    }
    search_space = followup_search_space(prereg)
    if search_space is not None:
        payload["conditional_followup_search_space"] = search_space
    return payload


def matched_trajectory_diagnostics(
    report: dict[str, Any], prereg: dict[str, Any]
) -> dict[str, Any]:
    """Report matched-step stability without changing the preregistered gate."""
    multi = report["multi_arm_selection"]
    completions = multi["all_arm_training_completion_evidence"]
    eligibility = prereg["launch_gate"]["eligibility"]
    min_delta = float(eligibility["minimum_macro_f1_gain_over_packet_full"])
    max_drop = float(eligibility["maximum_accuracy_drop_from_packet_full"])
    datasets = sorted(completions["packet_full"]["datasets"])
    output: dict[str, Any] = {}
    for dataset in datasets:
        histories: dict[str, dict[int, dict[str, Any]]] = {}
        for arm in ARM_PARAMETERS:
            evidence = completions[arm]["datasets"][dataset]
            path = Path(str(evidence.get("validation_history_path") or ""))
            expected_hash = str(evidence.get("validation_history_sha256") or "")
            if not path.is_file() or file_sha256(path) != expected_hash:
                raise ValueError(
                    f"{dataset} {arm} validation trajectory hash mismatch"
                )
            rows = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            histories[arm] = {int(row["step"]): row["metrics"] for row in rows}
        common_steps = sorted(set.intersection(*(set(rows) for rows in histories.values())))
        required_points = int(prereg["launch_gate"]["required_validation_points_per_arm"])
        if len(common_steps) != required_points:
            raise ValueError(
                f"{dataset} has {len(common_steps)} matched steps, expected {required_points}"
            )

        trajectory = []
        preference_counts = {arm: 0 for arm in ARM_PARAMETERS}
        eligible_counts = {arm: 0 for arm in ARM_PARAMETERS}
        delta_sums = {
            arm: {"accuracy": 0.0, "macro_f1": 0.0}
            for arm in ("flow_sqrt", "flow_full")
        }
        for step in common_steps:
            baseline = histories["packet_full"][step]
            arms = {
                "packet_full": {
                    "accuracy": float(baseline["accuracy"]),
                    "macro_f1": float(baseline["macro_f1"]),
                    "delta_accuracy": 0.0,
                    "delta_macro_f1": 0.0,
                    "eligible": True,
                }
            }
            for arm in ("flow_sqrt", "flow_full"):
                metrics = histories[arm][step]
                delta_accuracy = float(metrics["accuracy"]) - float(
                    baseline["accuracy"]
                )
                delta_f1 = float(metrics["macro_f1"]) - float(
                    baseline["macro_f1"]
                )
                arms[arm] = {
                    "accuracy": float(metrics["accuracy"]),
                    "macro_f1": float(metrics["macro_f1"]),
                    "delta_accuracy": delta_accuracy,
                    "delta_macro_f1": delta_f1,
                    "eligible": bool(
                        delta_f1 >= min_delta and delta_accuracy >= -max_drop
                    ),
                }
                delta_sums[arm]["accuracy"] += delta_accuracy
                delta_sums[arm]["macro_f1"] += delta_f1
            for arm, row in arms.items():
                eligible_counts[arm] += int(row["eligible"])
            eligible_arms = [arm for arm, row in arms.items() if row["eligible"]]

            def key(arm: str) -> tuple[float, float, float, float]:
                params = ARM_PARAMETERS[arm]
                return (
                    arms[arm]["macro_f1"],
                    arms[arm]["accuracy"],
                    -params["alpha"],
                    -params["gamma"],
                )

            preferred = max(eligible_arms, key=key)
            preference_counts[preferred] += 1
            trajectory.append(
                {"step": step, "preferred_eligible_arm": preferred, "arms": arms}
            )
        denominator = float(len(common_steps))
        output[dataset] = {
            "matched_steps": common_steps,
            "trajectory": trajectory,
            "eligible_step_counts": eligible_counts,
            "preferred_step_counts": preference_counts,
            "last_three_preferred_arms": [
                row["preferred_eligible_arm"] for row in trajectory[-3:]
            ],
            "mean_candidate_deltas": {
                arm: {
                    metric: value / denominator
                    for metric, value in metrics.items()
                }
                for arm, metrics in delta_sums.items()
            },
        }
    return {
        "selection_role": "reporting_only_not_launch_gate",
        "test_labels_used": False,
        "datasets": output,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--class_weight_selection", required=True)
    parser.add_argument("--preregistration", required=True)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()
    report_path = Path(args.class_weight_selection)
    prereg_path = Path(args.preregistration)
    report = load_json(report_path)
    prereg = load_json(prereg_path)
    amendment_evidence = verify_train_only_amendment_evidence(prereg)
    payload = evaluate(report, prereg)
    if amendment_evidence is not None:
        payload["identifiability_amendment_evidence"] = amendment_evidence
    payload["matched_trajectory_diagnostics"] = matched_trajectory_diagnostics(
        report, prereg
    )
    payload["inputs"] = {
        "class_weight_selection": {
            "path": str(report_path.resolve()),
            "sha256": file_sha256(report_path),
        },
        "preregistration": {
            "path": str(prereg_path.resolve()),
            "sha256": file_sha256(prereg_path),
        },
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
