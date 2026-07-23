import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from run_packet_level_pipeline import verify_paper_unified_evaluation_source


ROOT = Path(__file__).resolve().parents[1]


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "run_packet_level_pipeline.py", *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_protocol_pretrain_stage_inherits_exact_packet_core_architecture(tmp_path):
    artifact_root = tmp_path / "artifacts"
    checkpoint_root = tmp_path / "checkpoints"
    result = run_cli(
        "--dataset",
        "vpn-app",
        "--fold",
        "0",
        "--stage",
        "protocol_pretrain",
        "--framework_profile",
        "legacy",
        "--dry_run",
        "--artifact_root",
        str(artifact_root),
        "--checkpoint_root",
        str(checkpoint_root),
        "--byte_max_bytes",
        "96",
        "--byte_hidden_dim",
        "64",
        "--byte_num_layers",
        "2",
        "--byte_num_heads",
        "4",
        "--byte_dropout",
        "0.1",
    )

    assert result.returncode == 0, result.stderr
    command = result.stdout
    assert "pretrain_native_flow_encoder.py" in command
    assert "--max_bytes 96" in command
    assert "--hidden_dim 64" in command
    assert "--byte_layers 2" in command
    assert "--num_heads 4" in command
    assert "--dropout 0.1" in command
    assert "train_packet_byte_transformer.py" not in command

    manifest_path = artifact_root / "vpn-app" / "fold0" / "packet_framework_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    notes = manifest["framework"]["notes"]
    assert notes["completed"] is False
    assert notes["pretrain_protocol_content"] is True
    assert notes["resolved_protocol_content_checkpoint"] == str(
        checkpoint_root / "vpn-app_fold0" / "shared_content_pretraining" / "best.pt"
    )


def test_auto_pretraining_rejects_an_explicit_initialization_checkpoint(tmp_path):
    result = run_cli(
        "--dataset",
        "vpn-app",
        "--fold",
        "0",
        "--stage",
        "byte",
        "--dry_run",
        "--artifact_root",
        str(tmp_path / "artifacts"),
        "--checkpoint_root",
        str(tmp_path / "checkpoints"),
        "--pretrain_protocol_content",
        "--protocol_content_checkpoint",
        "existing.pt",
    )

    assert result.returncode != 0
    assert "mutually exclusive" in result.stderr


def test_byte_stage_wires_generated_checkpoint_into_packet_training(tmp_path):
    result = run_cli(
        "--dataset",
        "vpn-app",
        "--fold",
        "1",
        "--stage",
        "byte",
        "--dry_run",
        "--artifact_root",
        str(tmp_path / "artifacts"),
        "--checkpoint_root",
        str(tmp_path / "checkpoints"),
        "--pretrain_protocol_content",
        "--byte_num_layers",
        "2",
        "--byte_dropout",
        "0.1",
        "--byte_content_group_loss_reduction",
        "group_mean",
    )

    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    pretrain_line = next(line for line in lines if "pretrain_native_flow_encoder.py" in line)
    packet_line = next(line for line in lines if "train_packet_byte_transformer.py" in line)
    generated = (
        tmp_path
        / "checkpoints"
        / "vpn-app_fold1"
        / "shared_content_pretraining"
        / "best.pt"
    )
    assert lines.index(pretrain_line) < lines.index(packet_line)
    assert f"--protocol_content_checkpoint {generated}" in packet_line
    assert "--content_group_loss_reduction group_mean" in packet_line


def test_paper_unified_eval_only_builds_requested_split_and_never_trains(tmp_path):
    artifact_root = tmp_path / "artifacts"
    checkpoint_root = tmp_path / "checkpoints"
    result = run_cli(
        "--dataset",
        "vpn-app",
        "--fold",
        "0",
        "--stage",
        "paper_unified_eval",
        "--framework_profile",
        "paper_unified",
        "--prepared_splits",
        "test",
        "--eval_splits",
        "test",
        "--dry_run",
        "--artifact_root",
        str(artifact_root),
        "--checkpoint_root",
        str(checkpoint_root),
    )

    assert result.returncode == 0, result.stderr
    commands = result.stdout
    assert "preprocess_tower1.py" in commands
    assert "--embedding_header_policy full" in commands
    assert "--embedding_header_policy protocol_closed_mixture" in commands
    assert commands.count("extract_packet_embeddings_qwen.py") == 2
    assert "test_packet_byte_transformer.py" in commands
    assert "train_tower1_multitask.py" not in commands
    assert "pretrain_native_flow_encoder.py" not in commands
    assert "train_packet_byte_transformer.py" not in commands
    assert "/train/packet_index.jsonl --output_dir" not in commands
    assert "/valid/packet_index.jsonl --output_dir" not in commands

    artifacts = artifact_root / "vpn-app" / "fold0"
    assert not (artifacts / "packet_framework_manifest.json").exists()
    eval_manifest = json.loads(
        (artifacts / "packet_framework_manifest_evaluation.json").read_text(
            encoding="utf-8"
        )
    )
    notes = eval_manifest["framework"]["notes"]
    assert eval_manifest["stage"] == "paper_unified_eval"
    assert notes["prepared_splits"] == ["test"]
    assert notes["eval_splits"] == ["test"]
    assert notes["evaluation_source"]["status"] == "pending_dry_run_verification"
    assert notes["completed"] is False


def test_paper_unified_eval_requires_paper_profile(tmp_path):
    result = run_cli(
        "--dataset",
        "vpn-app",
        "--fold",
        "0",
        "--stage",
        "paper_unified_eval",
        "--framework_profile",
        "legacy",
        "--prepared_splits",
        "test",
        "--eval_splits",
        "test",
        "--dry_run",
        "--artifact_root",
        str(tmp_path / "artifacts"),
        "--checkpoint_root",
        str(tmp_path / "checkpoints"),
    )

    assert result.returncode != 0
    assert "requires --framework_profile paper_unified" in result.stderr


def test_paper_unified_eval_verifies_frozen_training_provenance(tmp_path):
    artifacts = tmp_path / "artifacts"
    checkpoint = tmp_path / "checkpoints" / "vpn-app_fold0"
    manifest = {
        "dataset": "vpn-app",
        "fold": 0,
        "framework": {
            "notes": {
                "completed": True,
                "framework_profile": "paper_unified",
                "shared_core_method_sha256": "m" * 64,
                "shared_core_config_sha256": "c" * 64,
            }
        },
    }
    artifacts.mkdir(parents=True)
    (artifacts / "packet_framework_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    required = (
        checkpoint / "best" / "adapter" / "adapter_config.json",
        checkpoint / "best" / "tower1_heads.pt",
        checkpoint / "byte_transformer" / "best.pt",
    )
    for path in required:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"frozen")
    args = SimpleNamespace(
        dataset="vpn-app",
        fold=0,
        shared_core_method_sha256="m" * 64,
        shared_core_config_sha256="c" * 64,
    )

    evidence = verify_paper_unified_evaluation_source(args, artifacts, checkpoint)
    assert evidence["status"] == "pass"
    assert evidence["training_manifest_sha256"]
    assert set(evidence["checkpoints"]) == {
        "tower1_adapter_config",
        "tower1_heads",
        "packet_classifier",
    }

    args.shared_core_config_sha256 = "different"
    with pytest.raises(ValueError, match="matching completed run"):
        verify_paper_unified_evaluation_source(args, artifacts, checkpoint)
