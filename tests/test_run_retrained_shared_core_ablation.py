from pathlib import Path

from run_retrained_shared_core_ablation import (
    ABLATIONS,
    flow_eval_command,
    packet_command,
    replace_option,
)


def packet_manifest():
    return {
        "dataset": "vpn-app",
        "fold": 1,
        "paths": {
            "artifact_dir": "reasoningDataset/packet-level/vpn-app/fold1"
        },
        "framework": {
            "notes": {
                "resolved_protocol_content_checkpoint": "/tmp/native/best.pt",
                "shared_core_config": "/tmp/frozen.json",
            }
        },
    }


def flow_manifest():
    return {
        "dataset": "vpn-app",
        "fold": 1,
        "tower1_data_suffix": "strict_fold1",
        "framework": {
            "notes": {
                "tower2_data_suffix": "strict_data_fold1",
                "paired_embedding_suffix": "strict_intervention_fold1",
            }
        },
    }


def test_packet_matched_ablation_reuses_inputs_and_isolates_outputs(tmp_path):
    command, result_dir = packet_command(
        packet_manifest(), "no_content", tmp_path, dry_run=True
    )
    assert command[command.index("--artifact_root") + 1] == "reasoningDataset/packet-level"
    assert command[command.index("--train_ablate_input_channel") + 1] == "content"
    assert command[command.index("--train_ablate_intervention_view") + 1] == "none"
    assert command[command.index("--protocol_content_checkpoint") + 1] == "/tmp/native/best.pt"
    assert command[command.index("--ablation_output_dir") + 1] == str(result_dir)
    assert result_dir.is_relative_to(tmp_path)
    assert command[-1] == "--dry_run"


def test_flow_matched_ablation_evaluation_has_no_inference_only_toggle(tmp_path):
    output = tmp_path / "valid.json"
    command = flow_eval_command(
        flow_manifest(), Path("/tmp/checkpoints/best.pt"), "valid", output
    )
    assert command[command.index("--dataset") + 1].endswith(
        "valid_tower2_strict_data_fold1/seq_dataset.pt"
    )
    assert command[command.index("--paired_view_dataset") + 1].endswith(
        "valid_tower2_strict_intervention_fold1/seq_dataset.pt"
    )
    assert "--ablate_input_channel" not in command
    assert "--ablate_intervention_view" not in command


def test_ablation_registry_and_option_replacement_are_fixed():
    assert set(ABLATIONS) == {
        "no_semantic",
        "no_content",
        "no_structural",
        "factual_only",
        "intervened_only",
        "fixed_fusion",
        "row_risk",
    }
    command = ["python", "train.py", "--output_dir", "old"]
    replace_option(command, "--output_dir", "new")
    assert command[-1] == "new"
