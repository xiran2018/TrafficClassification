import json

import pytest

from freeze_shared_core_v2_config import canonical_sha256, freeze_config


def report(selected="baseline", datasets=("vpn-app", "tls-120")):
    payload = {
        "selection_scope": "heldout_validation_only",
        "metric": "macro_f1",
        "promotion_scope": "same_candidate_must_pass_every_dataset",
        "selected": selected,
        "datasets": {
            name: {
                "baseline_macro_f1": 0.6,
                "candidate_macro_f1": 0.61,
                "delta_macro_f1": 0.01,
                "passes": True,
            }
            for name in datasets
        },
    }
    payload["training_completion_evidence"] = {
        role: {
                "required": True,
                "required_validation_points": 8,
                "required_packet_batch_scheduler": "epoch_resampled_dataloader_v1",
                "status": "pass",
                "datasets": {
                name: {
                    "validation_points": 8,
                    "passed": True,
                    "best_metric_matches_history": True,
                    "provenance_verified": True,
                    "packet_batch_scheduler": "epoch_resampled_dataloader_v1",
                    "metric_sha256": "a" * 64,
                    "final_checkpoint_sha256": "b" * 64,
                    "validation_history_sha256": "c" * 64,
                    "provenance_sha256": "d" * 64,
                }
                for name in datasets
            },
        }
        for role in ("baseline", "candidate")
    }
    return payload


def write_report(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_freeze_uses_one_cross_dataset_selection_for_both_tasks(tmp_path):
    balance = report("candidate")
    paired = report("baseline")
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
    balance = report("baseline")
    paired = report("candidate")
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
    invalid = report(datasets=("vpn-app",))
    valid = report()
    with pytest.raises(ValueError, match="exactly"):
        freeze_config(
            invalid,
            valid,
            balance_path=write_report(tmp_path / "invalid.json", invalid),
            paired_path=write_report(tmp_path / "valid.json", valid),
        )

    invalid_scope = report()
    invalid_scope["selection_scope"] = "test"
    with pytest.raises(ValueError, match="heldout_validation_only"):
        freeze_config(
            valid,
            invalid_scope,
            balance_path=write_report(tmp_path / "valid2.json", valid),
            paired_path=write_report(tmp_path / "invalid_scope.json", invalid_scope),
        )


def test_freeze_rejects_intermediate_training_selection(tmp_path):
    invalid = report()
    invalid["training_completion_evidence"]["candidate"]["status"] = "fail"
    valid = report()
    with pytest.raises(ValueError, match="completion evidence did not pass"):
        freeze_config(
            invalid,
            valid,
            balance_path=write_report(tmp_path / "invalid_completion.json", invalid),
            paired_path=write_report(tmp_path / "valid_completion.json", valid),
        )


def test_freeze_rejects_best_metric_that_does_not_match_history(tmp_path):
    invalid = report()
    invalid["training_completion_evidence"]["candidate"]["datasets"]["vpn-app"][
        "best_metric_matches_history"
    ] = False
    valid = report()
    with pytest.raises(ValueError, match="history-consistent"):
        freeze_config(
            invalid,
            valid,
            balance_path=write_report(tmp_path / "stale_best.json", invalid),
            paired_path=write_report(tmp_path / "valid_history.json", valid),
        )
