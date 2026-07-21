#!/usr/bin/env python3
"""Run identical inference-only shared-core diagnostics for Packet and Flow."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


DIAGNOSTICS = (
    ("no_semantic", "ablate_input_channel", "semantic"),
    ("no_content", "ablate_input_channel", "content"),
    ("no_structural", "ablate_input_channel", "structural"),
    ("factual_only", "ablate_intervention_view", "factual_only"),
    ("intervened_only", "ablate_intervention_view", "intervened_only"),
)


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def notes(manifest: dict[str, Any]) -> dict[str, Any]:
    return (manifest.get("framework") or {}).get("notes") or {}


def validate_pair(packet: dict[str, Any], flow: dict[str, Any]) -> tuple[str, int, str]:
    packet_notes = notes(packet)
    flow_notes = notes(flow)
    packet_dataset = packet.get("dataset")
    flow_dataset = flow.get("dataset")
    packet_fold = int(packet.get("fold", -1))
    flow_fold = int(flow.get("fold", -2))
    packet_sha = str(packet_notes.get("shared_core_config_sha256") or "")
    flow_sha = str(flow.get("shared_core_config_sha256") or flow_notes.get("shared_core_config_sha256") or "")
    if packet_dataset != flow_dataset or packet_fold != flow_fold:
        raise ValueError("Packet/Flow manifests do not describe the same dataset/fold")
    if len(packet_sha) != 64 or packet_sha != flow_sha:
        raise ValueError("Packet/Flow manifests do not share one frozen-core fingerprint")
    if packet_notes.get("packet_module_training_source") != "packet_task_train_split_packets":
        raise ValueError("Packet manifest has an invalid task-local training source")
    if flow_notes.get("packet_module_training_source") != "flow_task_train_split_packets":
        raise ValueError("Flow manifest has an invalid task-local training source")
    if packet_notes.get("cross_task_trained_weights_reused") is not False or flow_notes.get(
        "cross_task_trained_weights_reused"
    ) is not False:
        raise ValueError("cross-task supervised weights were reused")
    return str(packet_dataset), packet_fold, packet_sha


def packet_command(
    manifest: dict[str, Any], split: str, diagnostic: tuple[str, str, str], output: Path
) -> list[str]:
    manifest_notes = notes(manifest)
    artifacts = Path(manifest["paths"]["artifact_dir"])
    name, option, value = diagnostic
    del name
    return [
        "python",
        "test_packet_byte_transformer.py",
        "--checkpoint",
        str(manifest_notes["packet_model_checkpoint"]),
        "--packet_index",
        str(artifacts / split / "packet_index.jsonl"),
        "--label_map",
        str(artifacts / "train" / "label_map.json"),
        "--output_json",
        str(output),
        "--batch_size",
        "1024",
        "--semantic_embedding_cache",
        str(artifacts / f"{split}_semantic_embeddings_full.npy"),
        "--semantic_embedding_manifest",
        str(artifacts / f"{split}_semantic_embeddings_full_manifest.json"),
        "--required_semantic_header_policy",
        "full",
        "--intervened_semantic_embedding_cache",
        str(artifacts / f"{split}_semantic_embeddings_mask_ip_port.npy"),
        "--intervened_semantic_embedding_manifest",
        str(artifacts / f"{split}_semantic_embeddings_mask_ip_port_manifest.json"),
        "--required_intervened_semantic_header_policy",
        "mask_ip_port",
        f"--{option}",
        value,
    ]


def flow_command(
    manifest: dict[str, Any], split: str, diagnostic: tuple[str, str, str], output: Path
) -> list[str]:
    manifest_notes = notes(manifest)
    dataset = str(manifest["dataset"])
    name, option, value = diagnostic
    del name
    checkpoints = manifest_notes.get("tower2_checkpoints") or {}
    checkpoint = checkpoints.get("seq")
    if not checkpoint:
        raise ValueError("Flow manifest lacks the strict sequence checkpoint")
    tower2_suffix = str(manifest_notes["tower2_data_suffix"])
    paired_suffix = str(manifest_notes["paired_embedding_suffix"])
    label_map = (
        Path("reasoningDataset")
        / dataset
        / f"train_tower1_{manifest['tower1_data_suffix']}"
        / "label_map.json"
    )
    return [
        "python",
        "test_tower2.py",
        "--checkpoint",
        str(checkpoint),
        "--dataset",
        f"reasoningDataset/{dataset}/{split}_tower2_{tower2_suffix}/seq_dataset.pt",
        "--paired_view_dataset",
        f"reasoningDataset/{dataset}/{split}_tower2_{paired_suffix}/seq_dataset.pt",
        "--label_map",
        str(label_map),
        "--output_json",
        str(output),
        "--no_report",
        f"--{option}",
        value,
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet_manifest", required=True)
    parser.add_argument("--flow_manifest", required=True)
    parser.add_argument("--split", choices=["valid", "test"], default="test")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    packet = load_json(args.packet_manifest)
    flow = load_json(args.flow_manifest)
    dataset, fold, fingerprint = validate_pair(packet, flow)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, dict[str, str]] = {}
    for diagnostic in DIAGNOSTICS:
        name = diagnostic[0]
        outputs[name] = {}
        for task, builder, manifest in (
            ("packet", packet_command, packet),
            ("flow", flow_command, flow),
        ):
            output = output_dir / f"{dataset}_fold{fold}_{args.split}_{task}_{name}.json"
            command = builder(manifest, args.split, diagnostic, output)
            command += ["--device", args.device]
            print("$ " + " ".join(command), flush=True)
            if not args.dry_run:
                subprocess.run(command, check=True)
            outputs[name][task] = str(output)
    summary = {
        "schema": "exact_shared_core_inference_sensitivity_v1",
        "scope": "inference_only_not_retrained_ablation",
        "dataset": dataset,
        "fold": fold,
        "split": args.split,
        "shared_core_config_sha256": fingerprint,
        "packet_manifest": str(args.packet_manifest),
        "packet_manifest_sha256": file_sha256(args.packet_manifest),
        "flow_manifest": str(args.flow_manifest),
        "flow_manifest_sha256": file_sha256(args.flow_manifest),
        "diagnostics": outputs,
        "test_labels_used_for_selection": False,
        "dry_run": args.dry_run,
    }
    summary_path = output_dir / f"{dataset}_fold{fold}_{args.split}_sensitivity_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
