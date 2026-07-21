import json
import subprocess
import sys
from pathlib import Path


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
