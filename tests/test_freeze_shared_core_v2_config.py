import hashlib
import json

import pytest

from freeze_shared_core_v2_config import canonical_sha256, freeze_config


def report(tmp_path, prefix, selected="baseline", datasets=("vpn-app", "tls-120")):
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


def write_report(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_freeze_uses_one_cross_dataset_selection_for_both_tasks(tmp_path):
    balance = report(tmp_path, "balance", "candidate")
    paired = report(tmp_path, "paired", "baseline")
    balance_path = write_report(tmp_path / "balance.json", balance)
    paired_path = write_report(tmp_path / "paired.json", paired)

    frozen = freeze_config(
        balance,
        paired,
        balance_path=balance_path,
        paired_path=paired_path,
    )

    assert frozen["datasets"] == ["tls-120", "vpn-app"]
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


def test_freeze_promotes_paired_invariance_only_when_both_datasets_pass(tmp_path):
    balance = report(tmp_path, "balance", "baseline")
    paired = report(tmp_path, "paired", "candidate")
    frozen = freeze_config(
        balance,
        paired,
        balance_path=write_report(tmp_path / "balance.json", balance),
        paired_path=write_report(tmp_path / "paired.json", paired),
    )
    assert frozen["tower1"]["class_weight_basis"] == "packet"
    assert frozen["tower1"]["paired_consistency_weight"] == 0.05
    assert frozen["tower1"]["paired_cls_weight"] == 0.2


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
