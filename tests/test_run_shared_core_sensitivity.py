from run_shared_core_sensitivity import (
    DIAGNOSTICS,
    flow_command,
    packet_command,
    validate_pair,
)


def manifests():
    fingerprint = "a" * 64
    packet = {
        "dataset": "vpn-app",
        "fold": 1,
        "paths": {"artifact_dir": "/tmp/packet/vpn-app/fold1"},
        "framework": {
            "notes": {
                "shared_core_config_sha256": fingerprint,
                "packet_module_training_source": "packet_task_train_split_packets",
                "cross_task_trained_weights_reused": False,
                "packet_model_checkpoint": "/tmp/checkpoints/packet/best.pt",
            }
        },
    }
    flow = {
        "dataset": "vpn-app",
        "fold": 1,
        "tower1_data_suffix": "strict_fold1",
        "shared_core_config_sha256": fingerprint,
        "framework": {
            "notes": {
                "packet_module_training_source": "flow_task_train_split_packets",
                "cross_task_trained_weights_reused": False,
                "tower2_data_suffix": "primary_fold1",
                "paired_embedding_suffix": "intervention_fold1",
                "tower2_checkpoints": {"seq": "/tmp/checkpoints/flow/best.pt"},
            }
        },
    }
    return packet, flow


def test_sensitivity_pair_requires_same_task_local_core():
    packet, flow = manifests()
    assert validate_pair(packet, flow) == ("vpn-app", 1, "a" * 64)


def test_sensitivity_builds_identical_diagnostic_names_for_both_tasks(tmp_path):
    packet, flow = manifests()
    diagnostic = DIAGNOSTICS[1]
    packet_cmd = packet_command(packet, "test", diagnostic, tmp_path / "packet.json")
    flow_cmd = flow_command(flow, "test", diagnostic, tmp_path / "flow.json")

    assert packet_cmd[packet_cmd.index("--ablate_input_channel") + 1] == "content"
    assert flow_cmd[flow_cmd.index("--ablate_input_channel") + 1] == "content"
    assert packet_cmd[packet_cmd.index("--output_npz") + 1] == str(
        tmp_path / "packet.npz"
    )
    assert packet_cmd[packet_cmd.index("--required_semantic_header_policy") + 1] == "full"
    assert "test_tower2_primary_fold1" in flow_cmd[flow_cmd.index("--dataset") + 1]
    assert "test_tower2_intervention_fold1" in flow_cmd[
        flow_cmd.index("--paired_view_dataset") + 1
    ]
