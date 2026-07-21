import hashlib
import json

import pytest

import select_bounded_hierarchy_protocol as bounded
from select_bounded_hierarchy_protocol import (
    build_output,
    noninferiority_decision,
    select,
    validate_derivation,
)


def test_promotes_only_when_every_dataset_is_noninferior():
    reference = {
        "vpn-app": {"accuracy": 0.80, "macro_f1": 0.75},
        "tls-120": {"accuracy": 0.85, "macro_f1": 0.83},
    }
    candidate = {
        "vpn-app": {"accuracy": 0.799, "macro_f1": 0.748},
        "tls-120": {"accuracy": 0.86, "macro_f1": 0.84},
    }

    result = noninferiority_decision(reference, candidate)

    assert result["selected"] == "candidate"
    assert result["candidate_promoted_for_all_datasets"] is True


@pytest.mark.parametrize("metric", ["accuracy", "macro_f1"])
def test_rejects_when_either_metric_exceeds_drop_on_one_dataset(metric):
    reference = {
        "vpn-app": {"accuracy": 0.80, "macro_f1": 0.75},
        "tls-120": {"accuracy": 0.85, "macro_f1": 0.83},
    }
    candidate = {dataset: dict(metrics) for dataset, metrics in reference.items()}
    candidate["tls-120"][metric] -= 0.0031

    result = noninferiority_decision(reference, candidate)

    assert result["selected"] == "reference"
    assert result["datasets"]["tls-120"][f"{metric}_passes"] is False


def test_requires_matching_dataset_sets():
    with pytest.raises(ValueError, match="datasets must match"):
        noninferiority_decision(
            {"vpn-app": {"accuracy": 1.0, "macro_f1": 1.0}},
            {"tls-120": {"accuracy": 1.0, "macro_f1": 1.0}},
        )


def write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selection_inputs(tmp_path):
    common_config = {
        "packet_batch_scheduler": "epoch_resampled_dataloader_v1",
        "class_weighting": "effective",
        "class_weight_basis": "flow",
        "class_weight_strength": 0.5,
        "lr": 1e-5,
    }
    datasets = {}
    candidates = {}
    completion_rows = {}
    derivation_rows = {}
    for index, dataset in enumerate(("vpn-app", "tls-120")):
        reference_metric = write_json(
            tmp_path / f"{dataset}_reference.json",
            {"metrics": {"accuracy": 0.8 + index * 0.05, "macro_f1": 0.75 + index * 0.08}},
        )
        candidate_metric = write_json(
            tmp_path / f"{dataset}_candidate.json",
            {"metrics": {"accuracy": 0.799 + index * 0.06, "macro_f1": 0.749 + index * 0.09}},
        )
        eta = 0.4 - index * 0.05
        reference_evidence = {
            "trainer_source_sha256": "a" * 64,
            "config": dict(common_config),
        }
        candidate_evidence = {
            "metric_path": str(candidate_metric),
            "metric_sha256": sha256(candidate_metric),
            "trainer_source_sha256": "a" * 64,
            "config": {
                **common_config,
                "class_weight_strength": eta,
            },
        }
        datasets[dataset] = {
            "selected_eta": 0.5,
            "selected_validation_metric": {
                "path": str(reference_metric),
                "sha256": sha256(reference_metric),
            },
            "selected_training_evidence": reference_evidence,
        }
        candidates[dataset] = candidate_metric
        completion_rows[dataset] = candidate_evidence
        derivation_rows[dataset] = {"class_weight_strength": eta}
    reference = {
        "schema": "hierarchy_adaptive_class_weight_selection_v1",
        "status": "frozen_numeric_hyperparameters_from_validation",
        "selection_scope": "heldout_validation_only",
        "test_labels_used": False,
        "shared_algorithm": "normalized_effective_flow_class_risk_power_eta",
        "inputs": {},
        "datasets": datasets,
    }
    derivation = {
        "schema": "bounded_hierarchy_risk_protocol_v1",
        "status": "derived_from_training_counts_only",
        "test_labels_used": False,
        "shared_algorithm": "largest_flow_risk_power_subject_to_max_min_ratio",
        "max_weight_ratio": 4.0,
        "datasets": derivation_rows,
    }
    completion = {"status": "pass", "datasets": completion_rows}
    return reference, derivation, candidates, completion


def test_select_verifies_factorial_contract_and_builds_frozen_output(
    tmp_path, monkeypatch
):
    reference, derivation, candidates, completion = selection_inputs(tmp_path)
    monkeypatch.setattr(
        bounded,
        "training_completion_evidence",
        lambda *args, **kwargs: completion,
    )
    monkeypatch.setattr(bounded, "_contract_config", lambda row: row["config"])

    decision, observed_completion = select(
        reference,
        derivation,
        candidates,
        required_validation_points=8,
    )
    reference_path = write_json(tmp_path / "reference.json", reference)
    derivation_path = write_json(tmp_path / "derivation.json", derivation)
    preregistration_path = write_json(tmp_path / "preregistration.json", {})
    output = build_output(
        reference,
        derivation,
        decision,
        observed_completion,
        reference_path=reference_path,
        derivation_path=derivation_path,
        preregistration_path=preregistration_path,
    )

    assert decision["selected"] == "candidate"
    assert output["shared_algorithm"] == (
        "bounded_effective_flow_class_risk_power_eta"
    )
    assert output["datasets"]["vpn-app"]["selected_eta"] == pytest.approx(0.4)
    assert output["datasets"]["tls-120"]["selected_eta"] == pytest.approx(0.35)
    assert output["inputs"]["bounded_risk_derivation"]["sha256"] == sha256(
        derivation_path
    )


def test_select_rejects_hidden_nonhierarchy_training_change(tmp_path, monkeypatch):
    reference, derivation, candidates, completion = selection_inputs(tmp_path)
    completion["datasets"]["tls-120"]["config"]["lr"] = 2e-5
    monkeypatch.setattr(
        bounded,
        "training_completion_evidence",
        lambda *args, **kwargs: completion,
    )
    monkeypatch.setattr(bounded, "_contract_config", lambda row: row["config"])

    with pytest.raises(ValueError, match="outside hierarchy fields"):
        select(
            reference,
            derivation,
            candidates,
            required_validation_points=8,
        )


def test_derivation_must_match_preregistered_hash_and_threshold(tmp_path):
    derivation = {
        "schema": "bounded_hierarchy_risk_protocol_v1",
        "status": "derived_from_training_counts_only",
        "test_labels_used": False,
        "shared_algorithm": "largest_flow_risk_power_subject_to_max_min_ratio",
        "max_weight_ratio": 4.0,
    }
    derivation_path = write_json(tmp_path / "derivation.json", derivation)
    preregistration = {
        "bounded_risk_geometry_amendment": {
            "status": "preregistered_before_complete_validation_histories",
            "test_labels_used": False,
            "fixed_max_weight_ratio": 4.0,
            "numeric_derivation": {
                "path": str(derivation_path),
                "sha256": sha256(derivation_path),
            },
            "validation_protocol": {
                "test_labels_used": False,
                "promotion_scope": (
                    "same_bounded_risk_algorithm_must_pass_every_dataset"
                ),
                "maximum_accuracy_drop": 0.003,
                "maximum_macro_f1_drop": 0.003,
            },
        }
    }

    validate_derivation(
        derivation,
        preregistration,
        derivation_path=derivation_path,
    )
    preregistration["bounded_risk_geometry_amendment"]["validation_protocol"][
        "maximum_accuracy_drop"
    ] = 0.01
    with pytest.raises(ValueError, match="rule changed"):
        validate_derivation(
            derivation,
            preregistration,
            derivation_path=derivation_path,
        )
