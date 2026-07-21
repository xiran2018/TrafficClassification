import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from freeze_shared_core_v2_config import canonical_sha256, freeze_config


def report(
    tmp_path,
    prefix,
    selected="baseline",
    datasets=("vpn-app", "tls-120"),
    *,
    screen="balance",
    paired_basis="packet",
    candidate_strength=0.5,
    paired_strength=None,
):
    promoted = selected == "candidate"
    payload = {
        "selection_scope": "heldout_validation_only",
        "metric": "macro_f1_with_accuracy_guard",
        "promotion_scope": "same_candidate_must_pass_every_dataset",
        "min_delta": 0.005,
        "max_accuracy_drop": 0.005,
        "candidate_promoted_for_all_datasets": promoted,
        "selected": selected,
        "datasets": {
            name: {
                "baseline_macro_f1": 0.6,
                "candidate_macro_f1": 0.61 if promoted else 0.602,
                "delta_macro_f1": 0.01 if promoted else 0.002,
                "delta_accuracy": 0.0,
                "macro_f1_passes": promoted,
                "accuracy_guard_passes": True,
                "passes": promoted,
            }
            for name in datasets
        },
    }
    trainer_sha256 = "e" * 64
    completion = {}
    for role in ("baseline", "candidate"):
        rows = {}
        for name in datasets:
            artifacts = {}
            for kind in ("metric", "final_checkpoint", "validation_history", "provenance"):
                path = tmp_path / f"{prefix}_{role}_{name}_{kind}"
                if kind == "provenance":
                    if screen == "balance":
                        basis = "flow" if role == "candidate" else "packet"
                        strength = candidate_strength if role == "candidate" else 1.0
                        paired_enabled = False
                    else:
                        basis = paired_basis
                        strength = (
                            paired_strength
                            if paired_strength is not None
                            else (0.5 if basis == "flow" else 1.0)
                        )
                        paired_enabled = role == "candidate"
                    config = {
                        "packet_batch_scheduler": "epoch_resampled_dataloader_v1",
                        "class_weighting": "effective",
                        "class_weight_basis": basis,
                        "class_weight_strength": strength,
                        "disable_packet_information_weights": True,
                        "flow_balanced_packet_batches": True,
                        "packets_per_flow": 2,
                        "no_sft": True,
                        "paired_packet_aux_jsonl": (
                            f"{name}_paired.jsonl" if paired_enabled else ""
                        ),
                        "paired_consistency_weight": 0.05 if paired_enabled else 0.0,
                        "paired_cls_weight": 0.2 if paired_enabled else 0.0,
                        "paired_logit_kl_weight": 0.5,
                        "paired_raw_consistency_weight": 1.0,
                    }
                    path.write_text(
                        json.dumps(
                            {
                                "schema": "tower1_training_contract_v1",
                                "training_config": config,
                            }
                        ),
                        encoding="utf-8",
                    )
                else:
                    path.write_text(f"{prefix}:{role}:{name}:{kind}", encoding="utf-8")
                artifacts[kind] = path
            rows[name] = {
                "validation_points": 8,
                "passed": True,
                "best_metric_matches_history": True,
                "provenance_verified": True,
                "packet_batch_scheduler": "epoch_resampled_dataloader_v1",
                "trainer_source_sha256": trainer_sha256,
                "completion_trainer_source_sha256": trainer_sha256,
                "trainer_source_stable_through_completion": True,
                **{
                    f"{kind}_path": str(path)
                    for kind, path in artifacts.items()
                },
                **{
                    f"{kind}_sha256": hashlib.sha256(path.read_bytes()).hexdigest()
                    for kind, path in artifacts.items()
                },
            }
        completion[role] = {
                "required": True,
                "required_validation_points": 8,
                "required_packet_batch_scheduler": "epoch_resampled_dataloader_v1",
                "status": "pass",
                "datasets": rows,
        }
    payload["training_completion_evidence"] = completion
    payload["training_implementation_consistency"] = {
        "required": True,
        "status": "pass",
        "num_runs": len(datasets) * 2,
        "trainer_source_sha256": trainer_sha256,
        "observed_trainer_source_sha256": [trainer_sha256],
        "all_runs_stable_through_completion": True,
    }
    return payload


def bind_paired_baseline(paired, balance):
    selected = balance["selected"]
    paired["training_completion_evidence"]["baseline"] = deepcopy(
        balance["training_completion_evidence"][selected]
    )
    return paired


def write_report(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_freeze_uses_one_cross_dataset_selection_for_both_tasks(tmp_path):
    balance = report(tmp_path, "balance", "candidate")
    paired = bind_paired_baseline(
        report(
            tmp_path,
            "paired",
            "baseline",
            screen="paired",
            paired_basis="flow",
        ),
        balance,
    )
    balance_path = write_report(tmp_path / "balance.json", balance)
    paired_path = write_report(tmp_path / "paired.json", paired)

    frozen = freeze_config(
        balance,
        paired,
        balance_path=balance_path,
        paired_path=paired_path,
    )

    assert frozen["datasets"] == [
        "tls-120",
        "ustc-app",
        "ustc-binary",
        "vpn-app",
        "vpn-binary",
        "vpn-service",
    ]
    assert frozen["task_datasets"] == {
        "packet-level-classification": frozen["datasets"],
        "flow-level-classification": ["tls-120", "vpn-app"],
    }
    assert frozen["tower1"]["class_weight_basis"] == "flow"
    assert frozen["tower1"]["class_weight_strength"] == 0.5
    assert frozen["tower1"]["paired_consistency_weight"] == 0.0
    assert frozen["packet_core"]["num_layers"] == 2
    assert frozen["packet_core"]["semantic_packet_context_policy"] == "single_packet"
    assert frozen["empirical_risk"]["content_group_loss_reduction"] == "group_mean"
    assert frozen["tower1"]["packet_batch_scheduler"] == (
        "epoch_resampled_dataloader_v1"
    )
    assert frozen["selection_protocol"]["metric"] == (
        "macro_f1_with_accuracy_guard"
    )
    assert frozen["selection_protocol"]["max_accuracy_drop"] == 0.005
    assert frozen["embedding_extraction"] == {
        "scheduler": "cross_flow_length_bucketed_v1",
        "embedding_mode": "concat",
        "batch_size": 8,
        "flow_batch_packets": 128,
    }
    assert frozen["task_contract"]["shared_packet_module_reuse"] == (
        "architecture_and_representation_contract_only"
    )
    assert frozen["task_contract"]["flow_task_training_source"] == (
        "flow_task_train_split_packets"
    )
    assert frozen["task_contract"]["cross_task_supervised_weights_reused"] is False
    fingerprint = frozen.pop("config_sha256")
    assert fingerprint == canonical_sha256(frozen)


def test_freeze_binds_hierarchy_numeric_overrides_before_paired_screen(tmp_path):
    balance = report(tmp_path, "hierarchy_balance", "candidate")
    trainer_sha = balance["training_implementation_consistency"][
        "trainer_source_sha256"
    ]
    balance["multi_arm_selection"] = {
        "selected_config": {
            "class_weight_basis": "flow",
            "class_weight_strength": 0.5,
        },
        "factorial_config_integrity": {"required": True, "status": "pass"},
        "all_arm_training_implementation_consistency": {
            "trainer_source_sha256": trainer_sha,
        },
    }
    paired = report(
        tmp_path,
        "hierarchy_paired",
        "candidate",
        screen="paired",
        paired_basis="flow",
        paired_strength=0.5,
    )
    balance_path = write_report(tmp_path / "hierarchy_balance.json", balance)
    hierarchy = {
        "schema": "hierarchy_adaptive_class_weight_selection_v1",
        "status": "frozen_numeric_hyperparameters_from_validation",
        "selection_scope": "heldout_validation_only",
        "test_labels_used": False,
        "shared_algorithm": "normalized_effective_flow_class_risk_power_eta",
        "inputs": {
            "class_weight_selection": {
                "path": str(balance_path.resolve()),
                "sha256": hashlib.sha256(balance_path.read_bytes()).hexdigest(),
            }
        },
        "datasets": {},
    }
    for dataset in ("vpn-app", "tls-120"):
        evidence = deepcopy(
            balance["training_completion_evidence"]["candidate"]["datasets"][
                dataset
            ]
        )
        hierarchy["datasets"][dataset] = {
            "selected_eta": 0.5,
            "training_hyperparameters": {
                "class_weight_basis": "flow",
                "class_weight_strength": 0.5,
            },
            "arms": {
                "0.0": {
                    "eta": 0.0,
                    "accuracy": 0.8,
                    "macro_f1": 0.6,
                    "eligible": True,
                },
                "0.5": {
                    "eta": 0.5,
                    "accuracy": 0.8,
                    "macro_f1": 0.61,
                    "eligible": True,
                },
            },
            "selected_validation_metric": {
                "path": evidence["metric_path"],
                "sha256": evidence["metric_sha256"],
            },
            "selected_training_evidence": evidence,
        }
    hierarchy_path = write_report(tmp_path / "hierarchy.json", hierarchy)
    paired_path = write_report(tmp_path / "hierarchy_paired.json", paired)

    frozen = freeze_config(
        balance,
        paired,
        balance_path=balance_path,
        paired_path=paired_path,
        hierarchy_report=hierarchy,
        hierarchy_path=hierarchy_path,
    )

    assert frozen["tower1"]["class_weight_basis"] == "flow"
    assert frozen["dataset_numeric_hyperparameter_overrides"][
        "packet-level-classification"
    ]["vpn-app"]["class_weight_strength"] == 0.5
    assert frozen["selection_evidence"]["hierarchy_class_weight"][
        "sha256"
    ] == hashlib.sha256(hierarchy_path.read_bytes()).hexdigest()


def test_freeze_accepts_globally_noninferior_bounded_hierarchy_rule(tmp_path):
    balance = report(tmp_path, "bounded_balance", "candidate")
    trainer_sha = balance["training_implementation_consistency"][
        "trainer_source_sha256"
    ]
    balance["multi_arm_selection"] = {
        "selected_config": {
            "class_weight_basis": "flow",
            "class_weight_strength": 0.5,
        },
        "factorial_config_integrity": {"required": True, "status": "pass"},
        "all_arm_training_implementation_consistency": {
            "trainer_source_sha256": trainer_sha,
        },
    }
    bounded_runs = report(
        tmp_path,
        "bounded_runs",
        "candidate",
        candidate_strength=0.4,
    )
    paired = report(
        tmp_path,
        "bounded_paired",
        "candidate",
        screen="paired",
        paired_basis="flow",
        paired_strength=0.4,
    )
    balance_path = write_report(tmp_path / "bounded_balance.json", balance)
    hierarchy = {
        "schema": "hierarchy_adaptive_class_weight_selection_v1",
        "status": "frozen_numeric_hyperparameters_from_validation",
        "selection_scope": "heldout_validation_only",
        "test_labels_used": False,
        "shared_algorithm": "bounded_effective_flow_class_risk_power_eta",
        "inputs": {
            "class_weight_selection": {
                "path": str(balance_path.resolve()),
                "sha256": hashlib.sha256(balance_path.read_bytes()).hexdigest(),
            }
        },
        "numeric_protocol_selection": {
            "schema": "bounded_hierarchy_risk_selection_v1",
            "selected": "candidate",
            "candidate_promoted_for_all_datasets": True,
            "promotion_scope": (
                "same_bounded_risk_algorithm_must_pass_every_dataset"
            ),
            "maximum_drop": 0.003,
            "datasets": {
                dataset: {"passes": True}
                for dataset in ("vpn-app", "tls-120")
            },
        },
        "datasets": {},
    }
    for dataset in ("vpn-app", "tls-120"):
        evidence = deepcopy(
            bounded_runs["training_completion_evidence"]["candidate"]["datasets"][
                dataset
            ]
        )
        hierarchy["datasets"][dataset] = {
            "selected_eta": 0.4,
            "training_hyperparameters": {
                "class_weight_basis": "flow",
                "class_weight_strength": 0.4,
            },
            "arms": {
                "reference": {
                    "eta": 0.5,
                    "accuracy": 0.8,
                    "macro_f1": 0.61,
                    "eligible": True,
                },
                "bounded_risk": {
                    "eta": 0.4,
                    "accuracy": 0.8,
                    "macro_f1": 0.61,
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
    hierarchy_path = write_report(tmp_path / "bounded_hierarchy.json", hierarchy)
    paired_path = write_report(tmp_path / "bounded_paired.json", paired)
    flow_derivation = {
        "schema": "bounded_hierarchy_risk_protocol_v1",
        "status": "derived_from_training_counts_only",
        "selection_role": "candidate_numeric_derivation_not_model_selection",
        "test_labels_used": False,
        "shared_algorithm": "largest_flow_risk_power_subject_to_max_min_ratio",
        "max_weight_ratio": 4.0,
        "effective_number_beta": 0.9999,
        "datasets": {},
    }
    flow_strengths = {"vpn-app": 1.0, "tls-120": 0.7}
    for dataset, strength in flow_strengths.items():
        source = write_report(
            tmp_path / f"{dataset}_flow_train_hierarchy.json",
            {
                "schema": "class_sampling_hierarchy_analysis_v1",
                "selection_role": "train_only_reporting_not_model_selection",
                "test_labels_used": False,
            },
        )
        flow_derivation["datasets"][dataset] = {
            "class_weight_basis": "flow",
            "class_weight_strength": strength,
            "bounded_effective_weight_ratio": 1.0 if dataset == "vpn-app" else 4.0,
            "input": {
                "path": str(source.resolve()),
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            },
        }
    flow_derivation_path = write_report(
        tmp_path / "flow_hierarchy_derivation.json", flow_derivation
    )
    packet_strengths = {
        "vpn-app": 0.4,
        "vpn-binary": 1.0,
        "vpn-service": 0.6,
        "tls-120": 0.4,
        "ustc-app": 0.8,
        "ustc-binary": 1.0,
    }
    packet_derivation = deepcopy(flow_derivation)
    packet_derivation["datasets"] = {}
    for dataset, strength in packet_strengths.items():
        source = write_report(
            tmp_path / f"{dataset}_packet_train_hierarchy.json",
            {
                "schema": "class_sampling_hierarchy_analysis_v1",
                "selection_role": "train_only_reporting_not_model_selection",
                "test_labels_used": False,
            },
        )
        packet_derivation["datasets"][dataset] = {
            "class_weight_basis": "flow",
            "class_weight_strength": strength,
            "bounded_effective_weight_ratio": 4.0,
            "input": {
                "path": str(source.resolve()),
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            },
        }
    packet_derivation_path = write_report(
        tmp_path / "packet_hierarchy_derivation.json", packet_derivation
    )

    frozen = freeze_config(
        balance,
        paired,
        balance_path=balance_path,
        paired_path=paired_path,
        hierarchy_report=hierarchy,
        hierarchy_path=hierarchy_path,
        packet_hierarchy_derivation=packet_derivation,
        packet_hierarchy_derivation_path=packet_derivation_path,
        flow_hierarchy_derivation=flow_derivation,
        flow_hierarchy_derivation_path=flow_derivation_path,
    )

    for dataset in packet_strengths:
        assert frozen["dataset_numeric_hyperparameter_overrides"][
            "packet-level-classification"
        ][dataset]["class_weight_strength"] == pytest.approx(
            packet_strengths[dataset]
        )
    for dataset in ("vpn-app", "tls-120"):
        assert frozen["dataset_numeric_hyperparameter_overrides"][
            "flow-level-classification"
        ][dataset]["class_weight_strength"] == pytest.approx(
            flow_strengths[dataset]
        )
    assert frozen["selection_evidence"]["flow_task_hierarchy_derivation"][
        "sha256"
    ] == hashlib.sha256(flow_derivation_path.read_bytes()).hexdigest()
    assert set(frozen["task_datasets"]["packet-level-classification"]) == set(
        packet_strengths
    )
    assert frozen["selection_evidence"]["hierarchy_class_weight"][
        "shared_algorithm"
    ] == "bounded_effective_flow_class_risk_power_eta"


def test_freeze_promotes_paired_invariance_only_when_both_datasets_pass(tmp_path):
    balance = report(tmp_path, "balance", "baseline")
    paired = bind_paired_baseline(
        report(
            tmp_path,
            "paired",
            "candidate",
            screen="paired",
            paired_basis="packet",
        ),
        balance,
    )
    frozen = freeze_config(
        balance,
        paired,
        balance_path=write_report(tmp_path / "balance.json", balance),
        paired_path=write_report(tmp_path / "paired.json", paired),
    )
    assert frozen["tower1"]["class_weight_basis"] == "packet"
    assert frozen["tower1"]["paired_consistency_weight"] == 0.05
    assert frozen["tower1"]["paired_cls_weight"] == 0.2
    assert frozen["selection_evidence"]["paired_invariance"][
        "factorial_config_integrity"
    ]["status"] == "pass"


def test_freeze_rejects_hidden_paired_training_factor(tmp_path):
    balance = report(tmp_path, "balance_hidden_paired", "baseline")
    paired = bind_paired_baseline(
        report(
            tmp_path,
            "paired_hidden_factor",
            "candidate",
            screen="paired",
            paired_basis="packet",
        ),
        balance,
    )
    row = paired["training_completion_evidence"]["candidate"]["datasets"][
        "tls-120"
    ]
    provenance = Path(row["provenance_path"])
    contract = json.loads(provenance.read_text(encoding="utf-8"))
    contract["training_config"]["lr"] = 2e-5
    provenance.write_text(json.dumps(contract), encoding="utf-8")
    row["provenance_sha256"] = hashlib.sha256(provenance.read_bytes()).hexdigest()

    with pytest.raises(ValueError, match="declared intervention factors"):
        freeze_config(
            balance,
            paired,
            balance_path=write_report(tmp_path / "balance_hidden.json", balance),
            paired_path=write_report(tmp_path / "paired_hidden.json", paired),
        )


def test_freeze_supports_preregistered_full_flow_correction(tmp_path):
    balance = report(
        tmp_path,
        "balance_full",
        "candidate",
        candidate_strength=1.0,
    )
    balance["multi_arm_selection"] = {
        "selected_config": {
            "class_weight_basis": "flow",
            "class_weight_strength": 1.0,
        },
        "factorial_config_integrity": {"required": True, "status": "pass"},
    }
    paired = bind_paired_baseline(
        report(
            tmp_path,
            "paired_full",
            "candidate",
            screen="paired",
            paired_basis="flow",
            paired_strength=1.0,
        ),
        balance,
    )
    frozen = freeze_config(
        balance,
        paired,
        balance_path=write_report(tmp_path / "balance_full.json", balance),
        paired_path=write_report(tmp_path / "paired_full.json", paired),
    )
    assert frozen["tower1"]["class_weight_basis"] == "flow"
    assert frozen["tower1"]["class_weight_strength"] == 1.0
    assert frozen["tower1"]["paired_consistency_weight"] == 0.05


def test_freeze_rejects_multi_arm_selection_without_factorial_integrity(tmp_path):
    balance = report(tmp_path, "balance_uncontrolled", "baseline")
    balance["multi_arm_selection"] = {
        "selected_config": {
            "class_weight_basis": "packet",
            "class_weight_strength": 1.0,
        },
        "factorial_config_integrity": {"required": True, "status": "fail"},
    }
    paired = bind_paired_baseline(
        report(tmp_path, "paired_uncontrolled", "baseline", screen="paired"),
        balance,
    )
    with pytest.raises(ValueError, match="factorial integrity"):
        freeze_config(
            balance,
            paired,
            balance_path=write_report(tmp_path / "balance_uncontrolled.json", balance),
            paired_path=write_report(tmp_path / "paired_uncontrolled.json", paired),
        )


def test_freeze_rejects_single_dataset_or_non_validation_selection(tmp_path):
    invalid = report(tmp_path, "single", datasets=("vpn-app",))
    valid = report(tmp_path, "valid")
    with pytest.raises(ValueError, match="exactly"):
        freeze_config(
            invalid,
            valid,
            balance_path=write_report(tmp_path / "invalid.json", invalid),
            paired_path=write_report(tmp_path / "valid.json", valid),
        )

    invalid_scope = report(tmp_path, "invalid_scope")
    invalid_scope["selection_scope"] = "test"
    with pytest.raises(ValueError, match="heldout_validation_only"):
        freeze_config(
            valid,
            invalid_scope,
            balance_path=write_report(tmp_path / "valid2.json", valid),
            paired_path=write_report(tmp_path / "invalid_scope.json", invalid_scope),
        )


def test_freeze_rejects_intermediate_training_selection(tmp_path):
    invalid = report(tmp_path, "invalid")
    invalid["training_completion_evidence"]["candidate"]["status"] = "fail"
    valid = report(tmp_path, "valid")
    with pytest.raises(ValueError, match="completion evidence did not pass"):
        freeze_config(
            invalid,
            valid,
            balance_path=write_report(tmp_path / "invalid_completion.json", invalid),
            paired_path=write_report(tmp_path / "valid_completion.json", valid),
        )


def test_freeze_rejects_best_metric_that_does_not_match_history(tmp_path):
    invalid = report(tmp_path, "invalid")
    invalid["training_completion_evidence"]["candidate"]["datasets"]["vpn-app"][
        "best_metric_matches_history"
    ] = False
    valid = report(tmp_path, "valid")
    with pytest.raises(ValueError, match="history-consistent"):
        freeze_config(
            invalid,
            valid,
            balance_path=write_report(tmp_path / "stale_best.json", invalid),
            paired_path=write_report(tmp_path / "valid_history.json", valid),
        )


def test_freeze_rehashes_completion_artifacts(tmp_path):
    invalid = report(tmp_path, "invalid")
    valid = report(tmp_path, "valid")
    metric_path = invalid["training_completion_evidence"]["baseline"]["datasets"][
        "vpn-app"
    ]["metric_path"]
    with open(metric_path, "a", encoding="utf-8") as handle:
        handle.write("mutated")

    with pytest.raises(ValueError, match="no longer matches"):
        freeze_config(
            invalid,
            valid,
            balance_path=write_report(tmp_path / "mutated.json", invalid),
            paired_path=write_report(tmp_path / "valid_artifacts.json", valid),
        )


def test_freeze_requires_one_stable_trainer_source(tmp_path):
    invalid = report(tmp_path, "invalid")
    valid = report(tmp_path, "valid")
    invalid["training_implementation_consistency"]["status"] = "fail"

    with pytest.raises(ValueError, match="one stable trainer source"):
        freeze_config(
            invalid,
            valid,
            balance_path=write_report(tmp_path / "trainer_drift.json", invalid),
            paired_path=write_report(tmp_path / "valid_trainer.json", valid),
        )


def test_freeze_recomputes_dual_metric_selection(tmp_path):
    invalid = report(tmp_path, "invalid", "candidate")
    valid = report(tmp_path, "valid", "baseline")
    invalid["datasets"]["vpn-app"]["delta_accuracy"] = -0.01

    with pytest.raises(ValueError, match="inconsistent promotion flags"):
        freeze_config(
            invalid,
            valid,
            balance_path=write_report(tmp_path / "invalid_gate.json", invalid),
            paired_path=write_report(tmp_path / "valid_gate.json", valid),
        )


def test_freeze_rejects_unbound_paired_baseline(tmp_path):
    balance = report(tmp_path, "balance", "candidate")
    paired = report(
        tmp_path,
        "paired",
        "baseline",
        screen="paired",
        paired_basis="flow",
    )

    with pytest.raises(ValueError, match="artifact-identical"):
        freeze_config(
            balance,
            paired,
            balance_path=write_report(tmp_path / "balance.json", balance),
            paired_path=write_report(tmp_path / "unbound_paired.json", paired),
        )


def test_freeze_rejects_wrong_paired_candidate_config(tmp_path):
    balance = report(tmp_path, "balance", "candidate")
    paired = bind_paired_baseline(
        report(
            tmp_path,
            "paired",
            "candidate",
            screen="paired",
            paired_basis="flow",
        ),
        balance,
    )
    row = paired["training_completion_evidence"]["candidate"]["datasets"][
        "vpn-app"
    ]
    contract_path = Path(row["provenance_path"])
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["training_config"]["paired_cls_weight"] = 0.0
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    row["provenance_sha256"] = hashlib.sha256(contract_path.read_bytes()).hexdigest()

    with pytest.raises(ValueError, match="paired_cls_weight"):
        freeze_config(
            balance,
            paired,
            balance_path=write_report(tmp_path / "balance_valid.json", balance),
            paired_path=write_report(tmp_path / "paired_wrong_config.json", paired),
        )
