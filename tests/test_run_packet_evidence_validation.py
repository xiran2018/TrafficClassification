import json
from pathlib import Path

import pytest

import run_packet_evidence_validation as runner
from run_packet_evidence_validation import experiment_suffix, stage8_command


def manifest_payload(fingerprint: str = "a" * 64) -> dict:
    return {
        "dataset": "vpn-app",
        "fold": 0,
        "num_classes": 16,
        "framework_profile": "paper_unified",
        "shared_core_config_sha256": fingerprint,
        "exact_shared_packet_encoder": True,
        "source_suffix": "change_weight",
        "tower1_data_suffix": "strict_shared_core_v2_fold0",
        "embedding_suffix": "rawproj_strict_shared_core_v2_fold0",
        "native_structural_suffix": "shared_content_strict_shared_core_v2_fold0",
        "framework": {
            "notes": {
                "packet_module_training_source": "flow_task_train_split_packets",
                "cross_task_trained_weights_reused": False,
                "tower2_data_suffix": "strict_primary_data",
                "paired_embedding_suffix": "strict_intervention_data",
            }
        },
    }


def test_candidate_suffix_is_dataset_independent_in_shape():
    assert experiment_suffix(manifest_payload(), 0.4, "candidate") == (
        "strict_primary_data_stage8_flowaware_shared_packet_evidence_bound_0p4_fold0"
    )
    assert experiment_suffix(manifest_payload(), 0.4, "control") == (
        "strict_primary_data_stage8_flowaware_shared_packet_evidence_control_fold0"
    )


def test_baseline_manifest_rejects_cross_task_weight_reuse(tmp_path):
    payload = manifest_payload()
    payload["framework"]["notes"]["cross_task_trained_weights_reused"] = True
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="cross-task"):
        from run_packet_evidence_validation import load_manifest

        load_manifest(manifest_path)


def test_validation_command_reuses_flow_data_and_never_requests_test(tmp_path, monkeypatch):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest_payload()))
    monkeypatch.setattr(
        runner,
        "load_frozen_shared_core",
        lambda _path: {"config_sha256": "a" * 64},
    )

    command = stage8_command(
        manifest_path,
        tmp_path / "frozen.json",
        stage="eval",
        bound=0.4,
    )

    assert command[command.index("--tower2_suffix") + 1] == "strict_primary_data"
    assert command[command.index("--paired_embedding_suffix") + 1] == (
        "strict_intervention_data"
    )
    assert command[command.index("--eval_splits") + 1] == "valid"
    assert "test" not in command

    control = stage8_command(
        manifest_path,
        tmp_path / "frozen.json",
        stage="tower2_train",
        bound=0.4,
        arm="control",
    )
    assert control[control.index("--packet_evidence_max_weight") + 1] == "0.0"
    assert "--packet_evidence_ablation_control" in control
