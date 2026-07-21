#!/usr/bin/env python3
"""Train and validate the shared Packet-to-Flow evidence candidate."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from freeze_shared_core_v2_config import file_sha256
from run_stage8_flowaware_pipeline import safe_name
from shared_core_v2 import load_frozen_shared_core


def load_manifest(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    notes = (payload.get("framework") or {}).get("notes") or {}
    if payload.get("framework_profile") != "paper_unified":
        raise ValueError("baseline manifest is not a paper_unified run")
    if payload.get("exact_shared_packet_encoder") is not True:
        raise ValueError("baseline manifest does not use the exact shared packet encoder")
    if notes.get("packet_module_training_source") != "flow_task_train_split_packets":
        raise ValueError("baseline packet module was not trained from the Flow train split")
    if notes.get("cross_task_trained_weights_reused") is not False:
        raise ValueError("baseline manifest reused cross-task supervised weights")
    if not notes.get("tower2_data_suffix") or not notes.get("paired_embedding_suffix"):
        raise ValueError("baseline manifest lacks reusable factual/intervention Tower2 data")
    if not payload.get("native_structural_suffix"):
        raise ValueError("baseline manifest lacks the native content channel")
    return payload


def experiment_suffix(manifest: dict[str, Any], bound: float, arm: str) -> str:
    if arm not in {"control", "candidate"}:
        raise ValueError("arm must be control or candidate")
    notes = manifest["framework"]["notes"]
    fold = int(manifest["fold"])
    if arm == "control":
        run_tag = f"shared_packet_evidence_control_fold{fold}"
    else:
        bound_tag = str(bound).replace(".", "p")
        run_tag = f"shared_packet_evidence_bound_{bound_tag}_fold{fold}"
    return f"{notes['tower2_data_suffix']}_stage8_flowaware_{run_tag}"


def stage8_command(
    manifest_path: str | Path,
    shared_core_config: str | Path,
    *,
    stage: str,
    bound: float,
    arm: str = "candidate",
) -> list[str]:
    if stage not in {"tower2_train", "eval"}:
        raise ValueError("stage must be tower2_train or eval")
    manifest = load_manifest(manifest_path)
    frozen = load_frozen_shared_core(shared_core_config)
    manifest_sha = str(manifest.get("shared_core_config_sha256") or "")
    if manifest_sha != frozen["config_sha256"]:
        raise ValueError("baseline manifest and frozen shared-core fingerprints differ")
    notes = manifest["framework"]["notes"]
    fold = int(manifest["fold"])
    if arm not in {"control", "candidate"}:
        raise ValueError("arm must be control or candidate")
    if arm == "control":
        effective_bound = 0.0
        run_tag = f"shared_packet_evidence_control_fold{fold}"
    else:
        effective_bound = float(bound)
        bound_tag = str(bound).replace(".", "p")
        run_tag = f"shared_packet_evidence_bound_{bound_tag}_fold{fold}"
    command = [
        "python",
        "run_stage8_flowaware_pipeline.py",
        "--dataset",
        str(manifest["dataset"]),
        "--fold",
        str(fold),
        "--stage",
        stage,
        "--framework_profile",
        "paper_unified",
        "--shared_core_config",
        str(shared_core_config),
        "--num_classes",
        str(manifest["num_classes"]),
        "--source_suffix",
        str(manifest["source_suffix"]),
        "--output_suffix",
        "shared_packet_evidence_validation",
        "--tower1_data_suffix",
        str(manifest["tower1_data_suffix"]),
        "--embedding_suffix",
        str(manifest["embedding_suffix"]),
        "--tower2_suffix",
        str(notes["tower2_data_suffix"]),
        "--native_structural_suffix",
        str(manifest["native_structural_suffix"]),
        "--paired_embedding_suffix",
        str(notes["paired_embedding_suffix"]),
        "--model_types",
        "seq",
        "--exact_shared_packet_encoder",
        "--packet_evidence_max_weight",
        str(effective_bound),
        "--run_tag",
        run_tag,
        "--local_files_only",
    ]
    if arm == "control":
        command.append("--packet_evidence_ablation_control")
    if stage == "eval":
        command += ["--eval_splits", "valid"]
    return command


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_manifest", required=True)
    parser.add_argument("--shared_core_config", required=True)
    parser.add_argument("--packet_evidence_max_weight", type=float, default=0.4)
    parser.add_argument(
        "--stages",
        default="tower2_train,eval",
        help="Comma-separated subset of tower2_train,eval.",
    )
    parser.add_argument("--summary_json", default="")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    if not 0.0 < args.packet_evidence_max_weight <= 1.0:
        parser.error("--packet_evidence_max_weight must be in (0, 1]")
    stages = [item.strip() for item in args.stages.split(",") if item.strip()]
    if not stages or any(item not in {"tower2_train", "eval"} for item in stages):
        parser.error("--stages must contain tower2_train and/or eval")
    manifest = load_manifest(args.baseline_manifest)
    control_suffix = experiment_suffix(manifest, args.packet_evidence_max_weight, "control")
    candidate_suffix = experiment_suffix(
        manifest, args.packet_evidence_max_weight, "candidate"
    )
    commands = [
        stage8_command(
            args.baseline_manifest,
            args.shared_core_config,
            stage=stage,
            bound=args.packet_evidence_max_weight,
            arm=arm,
        )
        for arm in ("control", "candidate")
        for stage in stages
    ]
    for command in commands:
        print("$ " + " ".join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True)
    dataset = str(manifest["dataset"])
    summary = {
        "schema": "shared_packet_evidence_validation_run_v2",
        "dataset": dataset,
        "fold": int(manifest["fold"]),
        "packet_evidence_max_weight": args.packet_evidence_max_weight,
        "baseline_manifest": str(args.baseline_manifest),
        "baseline_manifest_sha256": file_sha256(Path(args.baseline_manifest)),
        "shared_core_config": str(args.shared_core_config),
        "shared_core_config_sha256": load_frozen_shared_core(args.shared_core_config)[
            "config_sha256"
        ],
        "matched_control": {
            "packet_evidence_max_weight": 0.0,
            "flow_pooling": "late_fusion",
            "manifest": f"reasoningDataset/{dataset}/stage8_flowaware_manifest_{control_suffix}.json",
            "valid_result": f"reasoningDataset/{dataset}/valid_seq_metrics_flow_{control_suffix}_probs.json",
            "checkpoint": f"checkpoints/tower2_seq_flow_{safe_name(dataset)}_{control_suffix}/best.pt",
        },
        "candidate": {
            "packet_evidence_max_weight": args.packet_evidence_max_weight,
            "flow_pooling": "late_fusion",
            "manifest": f"reasoningDataset/{dataset}/stage8_flowaware_manifest_{candidate_suffix}.json",
            "valid_result": f"reasoningDataset/{dataset}/valid_seq_metrics_flow_{candidate_suffix}_probs.json",
            "checkpoint": f"checkpoints/tower2_seq_flow_{safe_name(dataset)}_{candidate_suffix}/best.pt",
        },
        "stages": stages,
        "test_evaluated": False,
        "dry_run": args.dry_run,
    }
    output = Path(args.summary_json) if args.summary_json else Path(
        f"reasoningDataset/{dataset}/shared_packet_evidence_validation_fold{manifest['fold']}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
