import hashlib
import json

import pytest

from evaluate_hierarchy_adaptive_gate import (
    evaluate,
    followup_search_space,
    matched_trajectory_diagnostics,
    verify_train_only_amendment_evidence,
)


def preregistration():
    return {
        "schema": "hierarchy_adaptive_class_weight_preregistration_v1",
        "status": "preregistered_before_complete_validation_histories",
        "method": {"test_labels_used": False},
        "fixed_followup_grid": {"test_labels_used": False},
        "launch_gate": {
            "required_validation_points_per_arm": 8,
            "required_datasets": ["vpn-app", "tls-120"],
            "eligibility": {
                "minimum_macro_f1_gain_over_packet_full": 0.005,
                "maximum_accuracy_drop_from_packet_full": 0.005,
            },
        },
    }


def candidate_rows(vpn, tls):
    rows = {}
    for dataset, (base_acc, base_f1, acc, f1) in {
        "vpn-app": vpn,
        "tls-120": tls,
    }.items():
        rows[dataset] = {
            "baseline_accuracy": base_acc,
            "baseline_macro_f1": base_f1,
            "candidate_accuracy": acc,
            "candidate_macro_f1": f1,
            "delta_accuracy": acc - base_acc,
            "delta_macro_f1": f1 - base_f1,
        }
    return {"datasets": rows}


def report(*, divergent=True):
    completion = {
        "status": "pass",
        "required_validation_points": 8,
        "datasets": {
            dataset: {"passed": True, "validation_points": 8}
            for dataset in ("vpn-app", "tls-120")
        },
    }
    tls_sqrt = (0.83, 0.81, 0.85, 0.84) if divergent else (0.83, 0.81, 0.82, 0.82)
    return {
        "selection_scope": "heldout_validation_only",
        "multi_arm_selection": {
            "schema": "cross_dataset_class_weight_protocol_selection_v1",
            "all_arm_training_completion_evidence": {
                arm: {
                    **completion,
                    "datasets": {
                        key: dict(value) for key, value in completion["datasets"].items()
                    },
                }
                for arm in ("packet_full", "flow_sqrt", "flow_full")
            },
            "factorial_config_integrity": {"status": "pass"},
            "candidates": {
                "flow_sqrt": candidate_rows(
                    (0.80, 0.75, 0.78, 0.73),
                    tls_sqrt,
                ),
                "flow_full": candidate_rows(
                    (0.80, 0.75, 0.77, 0.72),
                    (0.83, 0.81, 0.82, 0.80),
                ),
            },
        },
    }


def test_launches_only_when_validation_best_eligible_corners_diverge():
    payload = evaluate(report(divergent=True), preregistration())
    assert payload["launch"] is True
    assert payload["datasets"]["vpn-app"]["selected"] == "packet_full"
    assert payload["datasets"]["tls-120"]["selected"] == "flow_sqrt"
    assert payload["selected_numeric_corners"]["vpn-app"] == {
        "alpha": 0.0,
        "gamma": 1.0,
    }
    assert payload["test_labels_used"] is False


def test_does_not_launch_when_one_common_corner_wins():
    payload = evaluate(report(divergent=False), preregistration())
    assert payload["launch"] is False
    assert {row["selected"] for row in payload["datasets"].values()} == {
        "packet_full"
    }


def test_rejects_incomplete_validation_history():
    incomplete = report(divergent=True)
    incomplete["multi_arm_selection"]["all_arm_training_completion_evidence"][
        "flow_full"
    ]["datasets"]["tls-120"]["validation_points"] = 7
    with pytest.raises(ValueError, match="complete validation evidence"):
        evaluate(incomplete, preregistration())


def amendment_preregistration(tmp_path):
    evidence_path = tmp_path / "hierarchy.json"
    evidence = {
        "test_labels_used": False,
        "summary": {
            "minimum_packet_class_count": 3,
            "maximum_packet_class_count": 3,
            "minimum_flow_class_count": 1,
            "maximum_flow_class_count": 2,
        },
    }
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    payload = preregistration()
    payload["train_only_identifiability_amendment"] = {
        "status": "preregistered_before_complete_validation_histories",
        "test_labels_used": False,
        "evidence": {
            "vpn-app": {
                "path": str(evidence_path),
                "sha256": hashlib.sha256(evidence_path.read_bytes()).hexdigest(),
                **evidence["summary"],
            }
        },
        "execution_rule": {
            "unique_effective_eta_grid": [0.0, 0.25, 0.5, 1.0],
            "canonical_parameterization": {
                "0.0": {"alpha": 0.0, "gamma": 1.0},
                "0.25": {"alpha": 0.5, "gamma": 0.5},
                "0.5": {"alpha": 1.0, "gamma": 0.5},
                "1.0": {"alpha": 1.0, "gamma": 1.0},
            },
            "duplicate_grid_point": {"alpha": 0.5, "gamma": 1.0},
            "duplicate_of": {"alpha": 1.0, "gamma": 0.5},
        },
    }
    return payload, evidence_path


def test_deduplicates_unidentifiable_alpha_gamma_grid(tmp_path):
    prereg, _ = amendment_preregistration(tmp_path)

    search = followup_search_space(prereg)

    assert search["identifiable_parameter"] == "eta=alpha*gamma"
    assert search["unique_effective_eta_grid"] == [0.0, 0.25, 0.5, 1.0]
    assert search["omitted_redundant_parameterization"] == {
        "alpha": 0.5,
        "gamma": 1.0,
    }
    evaluated = evaluate(report(divergent=True), prereg)
    assert evaluated["conditional_followup_search_space"] == search


def test_identifiability_evidence_is_train_only_and_hash_bound(tmp_path):
    prereg, evidence_path = amendment_preregistration(tmp_path)

    verified = verify_train_only_amendment_evidence(prereg)

    assert verified["verified_train_only_evidence"]["vpn-app"][
        "packet_class_count"
    ] == 3
    evidence_path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="evidence hash mismatch"):
        verify_train_only_amendment_evidence(prereg)


def test_matched_trajectory_is_hash_bound_and_reporting_only(tmp_path):
    payload = report(divergent=True)
    completions = payload["multi_arm_selection"][
        "all_arm_training_completion_evidence"
    ]
    for arm in ("packet_full", "flow_sqrt", "flow_full"):
        for dataset in ("vpn-app", "tls-120"):
            rows = []
            for step in range(1, 9):
                accuracy = 0.80
                macro_f1 = 0.75
                if arm == "flow_sqrt" and dataset == "tls-120":
                    accuracy, macro_f1 = 0.81, 0.77
                rows.append(
                    {
                        "step": step,
                        "metrics": {
                            "accuracy": accuracy,
                            "macro_f1": macro_f1,
                        },
                    }
                )
            path = tmp_path / f"{dataset}_{arm}.jsonl"
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            completions[arm]["datasets"][dataset].update(
                {
                    "validation_history_path": str(path),
                    "validation_history_sha256": hashlib.sha256(
                        path.read_bytes()
                    ).hexdigest(),
                }
            )

    diagnostics = matched_trajectory_diagnostics(payload, preregistration())
    assert diagnostics["selection_role"] == "reporting_only_not_launch_gate"
    assert diagnostics["datasets"]["tls-120"]["preferred_step_counts"][
        "flow_sqrt"
    ] == 8
    assert diagnostics["datasets"]["vpn-app"]["preferred_step_counts"][
        "packet_full"
    ] == 8

    path = tmp_path / "tls-120_flow_sqrt.jsonl"
    path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="trajectory hash mismatch"):
        matched_trajectory_diagnostics(payload, preregistration())
