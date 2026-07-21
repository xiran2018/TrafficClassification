#!/usr/bin/env python3
"""Run one matched retrained shared-core ablation for Packet and Flow."""
from __future__ import annotations

import argparse
import json
import subprocess
from argparse import Namespace
from pathlib import Path
from typing import Any

from run_shared_core_sensitivity import load_json, notes, validate_pair
from run_stage8_flowaware_pipeline import tower2_train_cmd


ABLATIONS = {
    "no_semantic": ("semantic", "none", False, False),
    "no_content": ("content", "none", False, False),
    "no_structural": ("structural", "none", False, False),
    "factual_only": ("none", "factual_only", False, False),
    "intervened_only": ("none", "intervened_only", False, False),
    "fixed_fusion": ("none", "none", True, False),
    "row_risk": ("none", "none", False, True),
}


def replace_option(command: list[str], option: str, value: str) -> None:
    index = command.index(option)
    command[index + 1] = value


def run(command: list[str], dry_run: bool) -> None:
    print("$ " + " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True)


def packet_command(
    manifest: dict[str, Any],
    diagnostic: str,
    output_root: Path,
    dry_run: bool,
) -> tuple[list[str], Path]:
    manifest_notes = notes(manifest)
    artifact_dir = Path(manifest["paths"]["artifact_dir"])
    artifact_root = artifact_dir.parents[1]
    protocol_checkpoint = str(
        manifest_notes.get("resolved_protocol_content_checkpoint") or ""
    )
    shared_core_config = str(manifest_notes.get("shared_core_config") or "")
    if not protocol_checkpoint:
        raise ValueError("Packet reference lacks its label-free content checkpoint")
    if not shared_core_config:
        raise ValueError("Packet reference lacks its frozen shared-core config")
    channel, view, fixed_fusion, row_risk = ABLATIONS[diagnostic]
    result_dir = output_root / "packet" / str(manifest["dataset"]) / f"fold{manifest['fold']}" / diagnostic
    checkpoint_root = output_root / "checkpoints" / "packet" / diagnostic
    command = [
        "python",
        "run_packet_level_pipeline.py",
        "--dataset", str(manifest["dataset"]),
        "--fold", str(manifest["fold"]),
        "--stage", "shared_core_ablation",
        "--framework_profile", "paper_unified",
        "--shared_core_config", shared_core_config,
        "--artifact_root", str(artifact_root),
        "--checkpoint_root", str(checkpoint_root),
        "--ablation_output_dir", str(result_dir),
        "--protocol_content_checkpoint", protocol_checkpoint,
        "--train_ablate_input_channel", channel,
        "--train_ablate_intervention_view", view,
    ]
    if dry_run:
        command.append("--dry_run")
    if fixed_fusion:
        command.append("--train_fixed_channel_fusion")
    if row_risk:
        command.append("--train_row_risk_ablation")
    return command, result_dir


def flow_train_command(
    manifest: dict[str, Any],
    diagnostic: str,
    output_root: Path,
) -> tuple[list[str], Path]:
    args = Namespace(**manifest)
    channel, view, fixed_fusion, row_risk = ABLATIONS[diagnostic]
    args.model_types = "seq"
    args.train_ablate_input_channel = channel
    args.train_ablate_intervention_view = view
    args.train_fixed_channel_fusion = fixed_fusion
    args.train_row_risk_ablation = row_risk
    if row_risk:
        args.content_group_loss_reduction = "none"
    args.run_tag = f"retrained_shared_core_{diagnostic}_fold{manifest['fold']}"
    command = tower2_train_cmd(args, "seq")
    checkpoint = (
        output_root
        / "checkpoints"
        / "flow"
        / str(manifest["dataset"])
        / f"fold{manifest['fold']}"
        / diagnostic
        / "best.pt"
    )
    replace_option(command, "--output_dir", str(checkpoint.parent))
    return command, checkpoint


def flow_eval_command(
    manifest: dict[str, Any], checkpoint: Path, split: str, output: Path
) -> list[str]:
    manifest_notes = notes(manifest)
    dataset = str(manifest["dataset"])
    data_suffix = str(manifest_notes["tower2_data_suffix"])
    paired_suffix = str(manifest_notes["paired_embedding_suffix"])
    label_map = (
        Path("reasoningDataset")
        / dataset
        / f"train_tower1_{manifest['tower1_data_suffix']}"
        / "label_map.json"
    )
    return [
        "python", "test_tower2.py",
        "--checkpoint", str(checkpoint),
        "--dataset", f"reasoningDataset/{dataset}/{split}_tower2_{data_suffix}/seq_dataset.pt",
        "--paired_view_dataset",
        f"reasoningDataset/{dataset}/{split}_tower2_{paired_suffix}/seq_dataset.pt",
        "--label_map", str(label_map),
        "--output_json", str(output),
        "--no_report",
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet_manifest", required=True)
    parser.add_argument("--flow_manifest", required=True)
    parser.add_argument("--diagnostic", choices=sorted(ABLATIONS), required=True)
    parser.add_argument("--task", choices=["packet", "flow", "both"], default="both")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    packet = load_json(args.packet_manifest)
    flow = load_json(args.flow_manifest)
    dataset, fold, fingerprint = validate_pair(packet, flow)
    if not notes(packet).get("completed") or not notes(flow).get("completed"):
        raise ValueError("retrained ablation requires completed strict reference manifests")

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Any] = {}
    if args.task in {"packet", "both"}:
        command, packet_result_dir = packet_command(
            packet, args.diagnostic, output_root, args.dry_run
        )
        run(command, args.dry_run)
        outputs["packet"] = {
            "result_dir": str(packet_result_dir),
            "manifest": str(
                packet_result_dir
                / (
                    "packet_framework_manifest_ablation_"
                    f"{ABLATIONS[args.diagnostic][0]}_"
                    f"{ABLATIONS[args.diagnostic][1]}.json"
                    if not ABLATIONS[args.diagnostic][2] and not ABLATIONS[args.diagnostic][3]
                    else (
                        "packet_framework_manifest_ablation_none_none_fixed.json"
                        if ABLATIONS[args.diagnostic][2]
                        else "packet_framework_manifest_ablation_none_none_row_risk.json"
                    )
                )
            ),
        }
    if args.task in {"flow", "both"}:
        command, checkpoint = flow_train_command(
            flow, args.diagnostic, output_root
        )
        run(command, args.dry_run)
        flow_dir = output_root / "flow" / dataset / f"fold{fold}" / args.diagnostic
        flow_dir.mkdir(parents=True, exist_ok=True)
        split_outputs = {}
        for split in ("valid", "test"):
            output = flow_dir / f"{split}_metrics.json"
            command = flow_eval_command(flow, checkpoint, split, output)
            command += ["--device", args.device]
            run(command, args.dry_run)
            split_outputs[split] = str(output)
        outputs["flow"] = {
            "checkpoint": str(checkpoint),
            "results": split_outputs,
        }

    summary = {
        "schema": "matched_retrained_shared_core_ablation_v1",
        "scope": "retrained_ablation",
        "dataset": dataset,
        "fold": fold,
        "diagnostic": args.diagnostic,
        "train_ablate_input_channel": ABLATIONS[args.diagnostic][0],
        "train_ablate_intervention_view": ABLATIONS[args.diagnostic][1],
        "train_fixed_channel_fusion": ABLATIONS[args.diagnostic][2],
        "train_row_risk_ablation": ABLATIONS[args.diagnostic][3],
        "shared_core_config_sha256": fingerprint,
        "packet_reference_manifest": str(args.packet_manifest),
        "flow_reference_manifest": str(args.flow_manifest),
        "packet_module_training_sources": {
            "packet": "packet_task_train_split_packets",
            "flow": "flow_task_train_split_packets",
        },
        "cross_task_supervised_weights_reused": False,
        "selection_split": "valid",
        "test_labels_used_for_model_selection": False,
        "outputs": outputs,
        "dry_run": args.dry_run,
    }
    summary_path = output_root / f"{dataset}_fold{fold}_{args.diagnostic}_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
