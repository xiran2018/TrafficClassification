#!/usr/bin/env python3
"""Run one fold of the unified Tower-1 Per-flow Split packet pipeline."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

from unified_framework_spec import (
    apply_framework_profile,
    build_framework_manifest,
    profile_shared_status,
    tower1_execution_evidence,
    tower1_training_contract,
)
from shared_core_v2 import (
    apply_frozen_shared_core,
    capture_training_hyperparameter_overrides,
    effective_shared_core_sha256,
    load_frozen_shared_core,
    restore_profile_training_hyperparameters,
)
from audit_flow_embeddings import audit_report_evidence
from embedding_shard_utils import (
    expected_embedding_shard_counts,
    jsonl_row_count,
    merge_embedding_shards,
)
from method_source_provenance import (
    complete_source_stability,
    source_tree_snapshot,
)


DATASET_DIRS = {
    "vpn-app": "vpn-app",
    "vpn-binary": "vpn-binary",
    "vpn-service": "vpn-service",
    "tls-120": "tls",
    "ustc-app": "ustc-app",
    "ustc-binary": "ustc-binary",
}

GPU_PROGRAMS = {
    "pretrain_native_flow_encoder.py",
    "train_tower1_multitask.py",
    "test_tower1_packet.py",
    "train_packet_byte_transformer.py",
    "test_packet_byte_transformer.py",
}


def command_program(command: list[str]) -> str:
    return Path(command[1]).name if len(command) > 1 else Path(command[0]).name


def physical_cuda_token() -> str:
    visible = [part.strip() for part in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if part.strip()]
    token = visible[0] if visible else "0"
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in token)


def query_free_gpu_mb(token: str) -> int:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            f"--id={token}",
            "--query-gpu=memory.free",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    return int(output.strip().splitlines()[0].strip())


@contextmanager
def gpu_command_guard(command: list[str], dry_run: bool):
    """Serialize packet GPU phases with Qwen embedding jobs on one physical GPU."""
    if dry_run or command_program(command) not in GPU_PROGRAMS:
        yield
        return

    token = physical_cuda_token()
    lock_dir = Path(os.environ.get("PACKET_GPU_LOCK_DIR", "/tmp/two_tower_embedding_gpu_locks"))
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_handle = open(lock_dir / f"qwen_embedding_gpu_{token}.lock", "a+", encoding="utf-8")
    min_free_mb = int(os.environ.get("PACKET_GPU_MIN_FREE_MB", "0"))
    poll_seconds = max(float(os.environ.get("PACKET_GPU_POLL_SECONDS", "30")), 1.0)
    print(f"waiting for packet GPU lock: physical={token} program={command_program(command)}", flush=True)
    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
    try:
        while min_free_mb > 0:
            free_mb = query_free_gpu_mb(token)
            if free_mb >= min_free_mb:
                break
            print(
                f"waiting for packet GPU capacity: physical={token} "
                f"free_mb={free_mb} required_mb={min_free_mb}",
                flush=True,
            )
            time.sleep(poll_seconds)
        print(f"acquired packet GPU lock: physical={token} program={command_program(command)}", flush=True)
        yield
    finally:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        lock_handle.close()


def run(command: list[str], dry_run: bool) -> None:
    print("+ " + " ".join(command), flush=True)
    if not dry_run:
        with gpu_command_guard(command, dry_run=False):
            subprocess.run(command, check=True)


def semantic_embedding_command(
    args,
    *,
    packet_index: Path,
    output_dir: Path,
    checkpoint: Path,
    shard_index: int | None = None,
    device: str | None = None,
) -> list[str]:
    command = [
        sys.executable,
        "extract_packet_embeddings_qwen.py",
        "--packet_index", str(packet_index),
        "--output_dir", str(output_dir),
        "--base_model", args.base_model,
        "--lora_path", str(checkpoint / "best" / "adapter"),
        "--tower1_heads", str(checkpoint / "best" / "tower1_heads.pt"),
        "--embedding_mode", args.semantic_embedding_mode,
        "--batch_size", str(args.semantic_embedding_batch_size),
        "--flow_batch_packets", str(args.semantic_embedding_flow_batch_packets),
        "--max_length", str(args.max_packet_length),
        "--device", device or args.semantic_embedding_device,
        "--resume_existing",
    ]
    if shard_index is not None:
        command.extend(
            [
                "--num_shards", str(args.semantic_embedding_num_shards),
                "--shard_index", str(shard_index),
            ]
        )
    if args.local_files_only:
        command.append("--local_files_only")
    return command


def semantic_embedding_audit_command(
    packet_index: Path,
    output_dir: Path,
    *,
    require_model_provenance: bool = False,
) -> list[str]:
    command = [
        sys.executable,
        "audit_flow_embeddings.py",
        "--packet_index", str(packet_index),
        "--flow_embedding_index", str(output_dir / "flow_embedding_index.jsonl"),
        "--output_json", str(output_dir / "embedding_audit.json"),
    ]
    if require_model_provenance:
        command.append("--require_model_provenance")
    return command


def run_semantic_embedding_stage(
    args,
    *,
    packet_index: Path,
    output_dir: Path,
    checkpoint: Path,
) -> None:
    """Extract one Packet semantic view, optionally over deterministic GPU shards."""
    if args.semantic_embedding_num_shards <= 1:
        run(
            semantic_embedding_command(
                args,
                packet_index=packet_index,
                output_dir=output_dir,
                checkpoint=checkpoint,
            ),
            args.dry_run,
        )
        run(
            semantic_embedding_audit_command(
                packet_index,
                output_dir,
                require_model_provenance=args.framework_profile == "paper_unified",
            ),
            args.dry_run,
        )
        return

    shard_root = output_dir / "_shards"
    devices = [
        value.strip()
        for value in args.semantic_embedding_cuda_devices.split(",")
        if value.strip()
    ]
    expected_counts = expected_embedding_shard_counts(
        packet_index, args.semantic_embedding_num_shards
    )
    processes: list[tuple[int, subprocess.Popen]] = []
    for shard_index, expected_count in enumerate(expected_counts):
        shard_dir = shard_root / f"shard_{shard_index}"
        actual_count = jsonl_row_count(shard_dir / "flow_embedding_index.jsonl")
        if args.semantic_embedding_resume_shards and actual_count == expected_count:
            print(
                f"skip completed semantic shard: shard={shard_index} flows={actual_count}",
                flush=True,
            )
            continue
        if actual_count:
            print(
                f"resume incomplete semantic shard: shard={shard_index} "
                f"flows={actual_count}/{expected_count}",
                flush=True,
            )
        raw_device = devices[shard_index % len(devices)] if devices else ""
        shard_device = (
            raw_device
            if raw_device.startswith("cuda:")
            else (f"cuda:{raw_device}" if raw_device else args.semantic_embedding_device)
        )
        command = semantic_embedding_command(
            args,
            packet_index=packet_index,
            output_dir=shard_dir,
            checkpoint=checkpoint,
            shard_index=shard_index,
            device=shard_device,
        )
        print("+ " + " ".join(command), flush=True)
        if not args.dry_run:
            processes.append((shard_index, subprocess.Popen(command, env=os.environ.copy())))

    if args.dry_run:
        run(
            semantic_embedding_audit_command(
                packet_index,
                output_dir,
                require_model_provenance=args.framework_profile == "paper_unified",
            ),
            True,
        )
        return
    failed = []
    for shard_index, process in processes:
        return_code = process.wait()
        if return_code != 0:
            failed.append((shard_index, return_code))
    if failed:
        raise subprocess.CalledProcessError(
            failed[0][1], f"semantic embedding shards failed: {failed}"
        )
    summary = merge_embedding_shards(
        packet_index=packet_index,
        output_dir=output_dir,
        num_shards=args.semantic_embedding_num_shards,
        shard_devices=devices,
    )
    print(
        f"merged semantic embedding shards: flows={summary['merged_flows']} "
        f"index={summary['merged_index']}",
        flush=True,
    )
    run(
        semantic_embedding_audit_command(
            packet_index,
            output_dir,
            require_model_provenance=args.framework_profile == "paper_unified",
        ),
        False,
    )


def semantic_cache_policy_evidence(args, artifacts: Path) -> dict:
    """Verify the policies recorded by the semantic caches actually consumed."""
    expected_context = str(args.packet_context_policy)
    expected_mode = str(getattr(args, "semantic_embedding_mode", "concat"))
    expected_batch_size = int(getattr(args, "semantic_embedding_batch_size", 8))
    expected_flow_batch_packets = int(
        getattr(args, "semantic_embedding_flow_batch_packets", 128)
    )
    expected_scheduler = (
        "cross_flow_length_bucketed_v1"
        if expected_flow_batch_packets > 0
        else "legacy_per_flow_v1"
    )
    splits = {}
    for split in ("train", "valid", "test"):
        views = {}
        for view, suffix, expected_header in (
            ("factual", "_full", "full"),
            ("intervened", "_mask_ip_port", "mask_ip_port"),
        ):
            path = artifacts / f"{split}_semantic_embeddings{suffix}_manifest.json"
            manifest = {}
            if path.is_file():
                try:
                    manifest = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, ValueError, TypeError):
                    manifest = {}
            actual_header = str(manifest.get("header_policy") or "missing")
            actual_context = str(manifest.get("packet_context_policy") or "missing")
            actual_mode = str(manifest.get("embedding_mode") or "missing")
            actual_scheduler = str(manifest.get("embedding_scheduler") or "missing")
            actual_batch_size = manifest.get("embedding_batch_size")
            actual_flow_batch_packets = manifest.get("embedding_flow_batch_packets")
            actual_num_shards = int(manifest.get("embedding_num_shards", 1) or 1)
            actual_merge_scheduler = str(
                manifest.get("embedding_merge_scheduler") or "none"
            )
            shard_execution_verified = bool(
                actual_num_shards == 1
                or actual_merge_scheduler == "deterministic_flow_sha1_v1"
            )
            embedding_policy = "full" if view == "factual" else "mask_ip_port"
            audit = audit_report_evidence(
                artifacts
                / f"{split}_semantic_flow_embeddings_{embedding_policy}"
                / "embedding_audit.json"
            )
            views[view] = {
                "manifest_path": str(path),
                "expected_header_policy": expected_header,
                "actual_header_policy": actual_header,
                "expected_packet_context_policy": expected_context,
                "actual_packet_context_policy": actual_context,
                "expected_embedding_mode": expected_mode,
                "actual_embedding_mode": actual_mode,
                "expected_embedding_scheduler": expected_scheduler,
                "actual_embedding_scheduler": actual_scheduler,
                "expected_embedding_batch_size": expected_batch_size,
                "actual_embedding_batch_size": actual_batch_size,
                "expected_embedding_flow_batch_packets": expected_flow_batch_packets,
                "actual_embedding_flow_batch_packets": actual_flow_batch_packets,
                "actual_embedding_num_shards": actual_num_shards,
                "actual_embedding_merge_scheduler": actual_merge_scheduler,
                "shard_execution_verified": shard_execution_verified,
                **audit,
                "verified": bool(
                    actual_header == expected_header
                    and actual_context == expected_context
                    and actual_mode == expected_mode
                    and actual_scheduler == expected_scheduler
                    and actual_batch_size == expected_batch_size
                    and actual_flow_batch_packets == expected_flow_batch_packets
                    and shard_execution_verified
                    and audit["audit_verified"]
                ),
            }
        splits[split] = views
    return {
        "expected_packet_context_policy": expected_context,
        "expected_embedding_mode": expected_mode,
        "expected_embedding_scheduler": expected_scheduler,
        "expected_embedding_batch_size": expected_batch_size,
        "expected_embedding_flow_batch_packets": expected_flow_batch_packets,
        "embedding_audits_verified": all(
            view["audit_verified"]
            for views in splits.values()
            for view in views.values()
        ),
        "verified": all(
            view["verified"]
            for views in splits.values()
            for view in views.values()
        ),
        "splits": splits,
    }


def write_framework_manifest(
    args,
    artifacts: Path,
    train_dir: Path,
    valid_dir: Path,
    test_dir: Path,
    profile_overrides: dict,
    completed: bool = False,
) -> None:
    artifacts.mkdir(parents=True, exist_ok=True)
    source_root = Path(__file__).resolve().parent
    if not hasattr(args, "_algorithm_source_launch"):
        args._algorithm_source_launch = source_tree_snapshot(source_root)
    if completed:
        source_evidence = complete_source_stability(
            args._algorithm_source_launch, source_root
        )
    else:
        source_evidence = {
            "schema": "algorithm_source_stability_evidence_v1",
            "status": "running",
            "scope": "all_non_test_python_sources",
            "launch_fingerprint": args._algorithm_source_launch["fingerprint"],
            "completion_fingerprint": None,
            "num_launch_files": args._algorithm_source_launch["num_files"],
            "num_completion_files": 0,
            "changed_paths": [],
            "launch_snapshot": args._algorithm_source_launch,
        }
    effective_completed = completed and source_evidence["status"] == "pass"
    tower1_checkpoint_dir = (
        Path(args.checkpoint_root) / f"{args.dataset}_fold{args.fold}"
    )
    semantic_policy_evidence = semantic_cache_policy_evidence(args, artifacts)
    shared_status = profile_shared_status(args.framework_profile)
    shared_status.update(
        {
            "field_aware_header_intervention": (
                "factual_full_plus_mask_ip_port_intervention"
                if args.framework_profile == "paper_unified"
                else args.embedding_header_policy
            ),
            "label_free_protocol_content_pretraining": "native_flow_multitask_v1",
            "current_packet_structural_encoder": "strict_current_packet_13d",
            "bounded_tri_channel_router": "semantic_anchor_residual_0_25",
            "content_group_empirical_risk": args.byte_content_group_loss_reduction,
            "fixed_cross_fold_consensus": "equal_log_mean",
        }
    )
    if args.framework_profile == "legacy":
        shared_status.update(
            {
                "semantic_tower1_channel": "available_ablation"
                if args.stage in {"train", "test", "all"}
                else "available_not_selected",
                "bounded_tri_channel_router": "legacy_optional",
            }
        )
    payload = {
        "dataset": args.dataset,
        "fold": args.fold,
        "stage": args.stage,
        "paths": {
            "train_dir": str(train_dir),
            "valid_dir": str(valid_dir),
            "test_dir": str(test_dir),
            "artifact_dir": str(artifacts),
        },
        "framework": build_framework_manifest(
            task="packet-level",
            dataset=args.dataset,
            input_unit="one_current_packet",
            stage=args.stage,
            shared_module_status=shared_status,
            task_module_status={
                "strict_current_packet_protocol": "enforced",
                "packet_level_classifier": (
                    "shared_representation_single_head"
                    if args.framework_profile == "paper_unified"
                    else "validation_selected_experts"
                ),
            },
            notes={
                "framework_profile": args.framework_profile,
                "profile_overrides": profile_overrides,
                "shared_core_config": args.shared_core_config,
                "shared_core_method_sha256": getattr(
                    args, "shared_core_method_sha256", ""
                ),
                "shared_core_config_sha256": getattr(args, "shared_core_config_sha256", ""),
                "shared_core_overrides": getattr(args, "shared_core_overrides", {}),
                "algorithm_source_evidence": source_evidence,
                "input_layout": "class_packet_pcaps",
                "semantic_packet_context_policy": args.packet_context_policy,
                "semantic_embedding_policy_evidence": semantic_policy_evidence,
                "max_packets_per_flow": args.max_packets_per_flow,
                "byte_max_bytes": args.byte_max_bytes,
                "byte_max_payload_bytes": args.byte_max_payload_bytes if args.byte_use_payload_channel else 0,
                "byte_hidden_dim": args.byte_hidden_dim,
                "byte_num_layers": args.byte_num_layers,
                "byte_num_heads": args.byte_num_heads,
                "byte_dropout": args.byte_dropout,
                "byte_content_group_loss_reduction": args.byte_content_group_loss_reduction,
                "protocol_content_checkpoint": args.protocol_content_checkpoint,
                "pretrain_protocol_content": (
                    (args.pretrain_protocol_content or args.stage == "protocol_pretrain")
                    and args.stage != "shared_core_ablation"
                ),
                "resolved_protocol_content_checkpoint": (
                    args.protocol_content_checkpoint
                    if args.stage == "shared_core_ablation"
                    else (
                        str(
                            Path(args.checkpoint_root)
                            / f"{args.dataset}_fold{args.fold}"
                            / "shared_content_pretraining"
                            / "best.pt"
                        )
                        if args.pretrain_protocol_content or args.stage == "protocol_pretrain"
                        else args.protocol_content_checkpoint
                    )
                ),
                "packet_model_checkpoint": str(
                    Path(args.checkpoint_root)
                    / f"{args.dataset}_fold{args.fold}"
                    / "byte_transformer"
                    / "best.pt"
                ),
                "result_paths": [
                    str(artifacts / "valid_unified_packet_single_head.json"),
                    str(artifacts / "test_unified_packet_single_head.json"),
                ] if args.stage == "paper_unified" else (
                    [
                        str(
                            artifacts
                            / (
                                "valid_shared_core_ablation_"
                                f"{args.train_ablate_input_channel}_"
                                f"{args.train_ablate_intervention_view}"
                                f"{'_fixed' if args.train_fixed_channel_fusion else ''}"
                                f"{'_row_risk' if args.train_row_risk_ablation else ''}.json"
                            )
                        ),
                        str(
                            artifacts
                            / (
                                "test_shared_core_ablation_"
                                f"{args.train_ablate_input_channel}_"
                                f"{args.train_ablate_intervention_view}"
                                f"{'_fixed' if args.train_fixed_channel_fusion else ''}"
                                f"{'_row_risk' if args.train_row_risk_ablation else ''}.json"
                            )
                        ),
                    ]
                    if args.stage == "shared_core_ablation"
                    else []
                ),
                "tower1_training_contract": tower1_training_contract(args, "packet-level"),
                "tower1_execution_evidence": tower1_execution_evidence(
                    tower1_checkpoint_dir,
                    tower1_training_contract(args, "packet-level"),
                ),
                "packet_module_training_source": "packet_task_train_split_packets",
                "cross_task_trained_weights_reused": False,
                "train_ablate_input_channel": args.train_ablate_input_channel,
                "train_ablate_intervention_view": args.train_ablate_intervention_view,
                "train_fixed_channel_fusion": args.train_fixed_channel_fusion,
                "train_row_risk_ablation": args.train_row_risk_ablation,
                "protocol_content_pretraining": {
                    "epochs": args.protocol_pretrain_epochs,
                    "max_packets": args.protocol_pretrain_max_packets,
                    "flow_layers": args.protocol_pretrain_flow_layers,
                    "projection_dim": args.protocol_pretrain_projection_dim,
                    "field_mask_probability": args.protocol_pretrain_field_mask_probability,
                    "payload_dropout_probability": args.protocol_pretrain_payload_dropout_probability,
                    "session_mask_probability": args.protocol_pretrain_session_mask_probability,
                    "masked_byte_weight": args.protocol_pretrain_masked_byte_weight,
                    "relative_order_weight": args.protocol_pretrain_relative_order_weight,
                    "same_flow_weight": args.protocol_pretrain_same_flow_weight,
                    "next_length_weight": args.protocol_pretrain_next_length_weight,
                    "next_iat_weight": args.protocol_pretrain_next_iat_weight,
                    "direction_weight": args.protocol_pretrain_direction_weight,
                    "packet_consistency_weight": args.protocol_pretrain_packet_consistency_weight,
                    "flow_contrastive_weight": args.protocol_pretrain_flow_contrastive_weight,
                },
                "paper_main_experts": [],
                "semantic_fusion_level": "representation",
                "factual_header_policy": (
                    "full" if args.framework_profile == "paper_unified" else args.embedding_header_policy
                ),
                "intervened_header_policy": (
                    "mask_ip_port" if args.framework_profile == "paper_unified" else ""
                ),
                "shared_test_dir": args.shared_test_dir,
                "dry_run": args.dry_run,
                "completed": effective_completed,
            },
        ),
    }
    manifest_name = "packet_framework_manifest.json"
    if args.stage == "shared_core_ablation":
        fixed_suffix = "_fixed" if args.train_fixed_channel_fusion else ""
        risk_suffix = "_row_risk" if args.train_row_risk_ablation else ""
        manifest_name = (
            "packet_framework_manifest_ablation_"
            f"{args.train_ablate_input_channel}_"
            f"{args.train_ablate_intervention_view}{fixed_suffix}{risk_suffix}.json"
        )
    with open(artifacts / manifest_name, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    if completed and source_evidence["status"] != "pass":
        raise RuntimeError(
            "executable Python sources changed during packet pipeline execution: "
            + ", ".join(source_evidence["changed_paths"])
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(DATASET_DIRS), required=True)
    ap.add_argument("--fold", type=int, choices=[0, 1, 2], required=True)
    ap.add_argument(
        "--stage",
        choices=[
            "preprocess",
            "audit",
            "protocol_pretrain",
            "feature",
            "byte",
            "fusion",
            "packet_best",
            "paper_unified",
            "shared_core_ablation",
            "train",
            "test",
            "all",
        ],
        default="all",
    )
    ap.add_argument("--source_root", default="/home/jing/download/sweet/packet-level-classification/per-flow-split")
    ap.add_argument("--artifact_root", default="reasoningDataset/packet-level")
    ap.add_argument("--checkpoint_root", default="checkpoints/packet-level")
    ap.add_argument(
        "--ablation_output_dir",
        default="",
        help=(
            "Independent result/manifest directory for --stage shared_core_ablation; "
            "the frozen fold artifacts remain read-only inputs."
        ),
    )
    ap.add_argument(
        "--shared_test_dir",
        default="",
        help="Reuse an already preprocessed shared test directory instead of writing a fold-local copy.",
    )
    ap.add_argument("--base_model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--init_checkpoint_dir", default="")
    ap.add_argument("--init_adapter_only", action="store_true")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--packet_batch_size", type=int, default=16)
    ap.add_argument("--eval_batch_size", type=int, default=16)
    ap.add_argument("--valid_packets_per_flow", type=int, default=2)
    ap.add_argument("--tower1_early_stop_patience", type=int, default=0)
    ap.add_argument("--semantic_embedding_batch_size", type=int, default=8)
    ap.add_argument("--semantic_embedding_flow_batch_packets", type=int, default=128)
    ap.add_argument(
        "--semantic_embedding_num_shards",
        type=int,
        default=1,
        help="Extract each semantic view as N deterministic flow-id shards.",
    )
    ap.add_argument(
        "--semantic_embedding_cuda_devices",
        default="",
        help="Comma-separated logical CUDA ids assigned round-robin to semantic shards.",
    )
    ap.add_argument(
        "--semantic_embedding_resume_shards",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip semantic shards with the exact expected number of flows.",
    )
    ap.add_argument(
        "--semantic_embedding_mode",
        default="concat",
        choices=["raw", "projected", "concat"],
    )
    ap.add_argument("--semantic_embedding_device", default="auto")
    ap.add_argument("--gradient_accumulation_steps", type=int, default=1)
    ap.add_argument("--max_packet_length", type=int, default=640)
    ap.add_argument("--max_packets_per_flow", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--head_lr", type=float, default=1e-4)
    ap.add_argument("--cls_weight", type=float, default=1.0)
    ap.add_argument("--contrastive_weight", type=float, default=0.1)
    ap.add_argument("--same_flow_positive_weight", type=float, default=1.0)
    ap.add_argument("--same_label_positive_weight", type=float, default=1.0)
    ap.add_argument("--temperature", type=float, default=0.07)
    ap.add_argument("--flow_proto_weight", type=float, default=0.0)
    ap.add_argument(
        "--flow_proto_positive",
        choices=["own_flow", "same_class"],
        default="same_class",
    )
    ap.add_argument(
        "--flow_proto_context",
        choices=["inclusive", "leave_one_out"],
        default="inclusive",
    )
    ap.add_argument(
        "--tower1_paired_consistency_weight",
        type=float,
        default=0.0,
        help="Full-header/mask-IP-port Tower-1 representation and logit consistency weight.",
    )
    ap.add_argument("--tower1_paired_cls_weight", type=float, default=0.0)
    ap.add_argument("--tower1_paired_logit_kl_weight", type=float, default=0.5)
    ap.add_argument("--tower1_paired_raw_consistency_weight", type=float, default=1.0)
    ap.add_argument("--class_weighting", choices=["none", "inverse", "effective"], default="effective")
    ap.add_argument("--class_weight_basis", choices=["packet", "flow"], default="packet")
    ap.add_argument("--class_weight_strength", type=float, default=1.0)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--byte_max_bytes", type=int, default=64)
    ap.add_argument(
        "--byte_use_payload_channel",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the shared current-packet payload branch in the byte expert.",
    )
    ap.add_argument("--byte_max_payload_bytes", type=int, default=128)
    ap.add_argument("--byte_hidden_dim", type=int, default=128)
    ap.add_argument("--byte_num_layers", type=int, default=3)
    ap.add_argument("--byte_num_heads", type=int, default=4)
    ap.add_argument("--byte_dropout", type=float, default=0.15)
    ap.add_argument("--byte_use_protocol_fields", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument(
        "--byte_exact_shared_representation",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    ap.add_argument(
        "--byte_mask_protocol_session_fields",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    ap.add_argument("--channel_fusion_base_mode", choices=["legacy", "semantic_anchor"], default="legacy")
    ap.add_argument("--channel_fusion_max_weight", type=float, default=0.25)
    ap.add_argument(
        "--train_ablate_input_channel",
        choices=["none", "semantic", "content", "structural"],
        default="none",
        help="Retrain the unified packet model without one shared input channel.",
    )
    ap.add_argument(
        "--train_ablate_intervention_view",
        choices=["none", "factual_only", "intervened_only"],
        default="none",
        help="Retrain the unified packet model with one intervention view only.",
    )
    ap.add_argument(
        "--train_fixed_channel_fusion",
        action="store_true",
        help="Retrain with fixed equal shared-channel fusion.",
    )
    ap.add_argument(
        "--train_row_risk_ablation",
        action="store_true",
        help="Retrain with ordinary row-mean risk instead of content-group mean risk.",
    )
    ap.add_argument("--intervention_max_residual_weight", type=float, default=0.25)
    ap.add_argument(
        "--intervention_view_base_mode",
        choices=["symmetric_mean", "factual_anchor"],
        default="symmetric_mean",
    )
    ap.add_argument("--byte_epochs", type=int, default=12)
    ap.add_argument("--byte_batch_size", type=int, default=512)
    ap.add_argument("--byte_eval_batch_size", type=int, default=2048)
    ap.add_argument(
        "--protocol_content_checkpoint",
        default="",
        help="Optional shared native pretraining checkpoint for the packet content encoder.",
    )
    ap.add_argument(
        "--pretrain_protocol_content",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Run label-free native flow-aware pretraining and strictly initialize the "
            "packet protocol-content encoder from its best checkpoint."
        ),
    )
    ap.add_argument("--protocol_pretrain_epochs", type=int, default=20)
    ap.add_argument("--protocol_pretrain_batch_size", type=int, default=8)
    ap.add_argument("--protocol_pretrain_eval_batch_size", type=int, default=16)
    ap.add_argument("--protocol_pretrain_num_workers", type=int, default=0)
    ap.add_argument("--protocol_pretrain_max_packets", type=int, default=64)
    ap.add_argument("--protocol_pretrain_flow_layers", type=int, default=2)
    ap.add_argument("--protocol_pretrain_projection_dim", type=int, default=128)
    ap.add_argument("--protocol_pretrain_learning_rate", type=float, default=3e-4)
    ap.add_argument("--protocol_pretrain_weight_decay", type=float, default=1e-2)
    ap.add_argument("--protocol_pretrain_field_mask_probability", type=float, default=0.2)
    ap.add_argument("--protocol_pretrain_payload_dropout_probability", type=float, default=0.5)
    ap.add_argument("--protocol_pretrain_session_mask_probability", type=float, default=1.0)
    ap.add_argument("--protocol_pretrain_masked_byte_weight", type=float, default=1.0)
    ap.add_argument("--protocol_pretrain_relative_order_weight", type=float, default=0.25)
    ap.add_argument("--protocol_pretrain_same_flow_weight", type=float, default=0.25)
    ap.add_argument("--protocol_pretrain_next_length_weight", type=float, default=0.2)
    ap.add_argument("--protocol_pretrain_next_iat_weight", type=float, default=0.2)
    ap.add_argument("--protocol_pretrain_direction_weight", type=float, default=0.1)
    ap.add_argument("--protocol_pretrain_packet_consistency_weight", type=float, default=0.25)
    ap.add_argument("--protocol_pretrain_flow_contrastive_weight", type=float, default=0.25)
    ap.add_argument("--protocol_pretrain_temperature", type=float, default=0.1)
    ap.add_argument("--protocol_pretrain_patience", type=int, default=4)
    ap.add_argument("--protocol_pretrain_seed", type=int, default=42)
    ap.add_argument("--protocol_pretrain_device", default="cuda")
    ap.add_argument(
        "--byte_content_group_loss_reduction",
        choices=["none", "group_mean"],
        default="none",
        help="Apply pcap-content group-mean reduction to the byte transformer's main packet CE.",
    )
    ap.add_argument(
        "--byte_ambiguity_aware_targets",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    ap.add_argument("--byte_mask_probability", type=float, default=1.0)
    ap.add_argument("--byte_masked_ce_weight", type=float, default=0.3)
    ap.add_argument("--byte_consistency_weight", type=float, default=0.1)
    ap.add_argument("--byte_ambiguity_gate_strength", type=float, default=1.0)
    ap.add_argument(
        "--byte_learned_identifiability_router",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    ap.add_argument("--byte_identifiability_loss_weight", type=float, default=0.1)
    ap.add_argument(
        "--byte_protect_main_gradient",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    ap.add_argument(
        "--byte_gradient_projection_scope",
        choices=["global", "layerwise"],
        default="layerwise",
    )
    ap.add_argument(
        "--byte_select_invariant_blend",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    ap.add_argument("--byte_invariant_blend_grid_size", type=int, default=21)
    ap.add_argument(
        "--embedding_header_policy",
        choices=["full", "randomize_ip_port", "mask_ip_port", "mask_session_fields"],
        default="full",
    )
    ap.add_argument(
        "--packet_context_policy",
        choices=["auto", "single_packet", "flow_context"],
        default="auto",
        help="Context available to each Tower1 semantic packet prompt.",
    )
    ap.add_argument(
        "--intervened_embedding_header_policy",
        choices=["randomize_ip_port", "mask_ip_port"],
        default="mask_ip_port",
    )
    ap.add_argument("--use_intervention_views", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument(
        "--framework_profile",
        choices=["legacy", "paper_unified"],
        default="paper_unified",
        help=(
            "Apply a shared module profile before building commands. The default "
            "is the paper-facing unified profile; use legacy only for historical "
            "ablations."
        ),
    )
    ap.add_argument(
        "--shared_core_config",
        default="",
        help="Frozen shared method/architecture defaults used by both task runners.",
    )
    ap.add_argument(
        "--training_hyperparameter_overrides",
        default="",
        help=(
            "Comma-separated runner argument names whose parsed values should be "
            "restored after the shared method profile. Only optimization, compute, "
            "regularization, and loss-magnitude fields are allowed."
        ),
    )
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--local_files_only", action="store_true")
    args = ap.parse_args()
    if args.semantic_embedding_num_shards <= 0:
        ap.error("--semantic_embedding_num_shards must be positive")
    if not 0.0 <= args.class_weight_strength <= 1.0:
        ap.error("--class_weight_strength must be in [0, 1]")
    if args.tower1_paired_consistency_weight < 0 or args.tower1_paired_cls_weight < 0:
        ap.error("Tower-1 paired loss weights must be non-negative")
    if args.tower1_paired_logit_kl_weight < 0:
        ap.error("--tower1_paired_logit_kl_weight must be non-negative")
    if args.tower1_paired_raw_consistency_weight < 0:
        ap.error("--tower1_paired_raw_consistency_weight must be non-negative")
    if args.tower1_paired_cls_weight > 0 and args.tower1_paired_consistency_weight <= 0:
        ap.error("--tower1_paired_cls_weight requires --tower1_paired_consistency_weight > 0")
    independent_training_hyperparameters = capture_training_hyperparameter_overrides(
        args, "packet-level", args.training_hyperparameter_overrides
    )
    profile_overrides = apply_framework_profile(args, "packet-level")
    args.shared_core_overrides = {}
    args.shared_core_config_sha256 = ""
    args.shared_core_method_sha256 = ""
    if args.shared_core_config:
        if args.framework_profile != "paper_unified":
            ap.error("--shared_core_config requires --framework_profile paper_unified")
        frozen = load_frozen_shared_core(args.shared_core_config)
        args.shared_core_overrides = apply_frozen_shared_core(
            args,
            "packet-level",
            frozen,
            training_hyperparameter_overrides=independent_training_hyperparameters,
        )
        args.shared_core_method_sha256 = frozen["config_sha256"]
        args.shared_core_config_sha256 = effective_shared_core_sha256(
            frozen,
            "packet-level",
            independent_training_hyperparameters,
        )
    elif independent_training_hyperparameters:
        args.shared_core_overrides = restore_profile_training_hyperparameters(
            args,
            "packet-level",
            independent_training_hyperparameters,
        )
    if args.train_row_risk_ablation:
        args.byte_content_group_loss_reduction = "none"
    if args.train_ablate_input_channel != "none" and not args.byte_exact_shared_representation:
        ap.error(
            "--train_ablate_input_channel requires --byte_exact_shared_representation"
        )
    if args.train_ablate_intervention_view != "none" and (
        not args.byte_exact_shared_representation or not args.use_intervention_views
    ):
        ap.error(
            "--train_ablate_intervention_view requires exact shared intervention views"
        )
    if args.train_fixed_channel_fusion and not args.byte_exact_shared_representation:
        ap.error("--train_fixed_channel_fusion requires exact shared representation")
    if args.stage == "shared_core_ablation":
        if (
            args.train_ablate_input_channel == "none"
            and args.train_ablate_intervention_view == "none"
            and not args.train_fixed_channel_fusion
            and not args.train_row_risk_ablation
        ):
            ap.error("--stage shared_core_ablation requires one training ablation")
        if not args.protocol_content_checkpoint:
            ap.error(
                "--stage shared_core_ablation requires the full model's label-free "
                "--protocol_content_checkpoint for a matched initialization"
            )
        if not args.ablation_output_dir:
            ap.error("--stage shared_core_ablation requires --ablation_output_dir")
    if (
        args.framework_profile == "paper_unified"
        and args.intervention_view_base_mode != "symmetric_mean"
    ):
        ap.error(
            "paper_unified fixes --intervention_view_base_mode symmetric_mean; "
            "use a direct ablation command for factual_anchor"
        )
    if (
        args.framework_profile == "paper_unified"
        and abs(args.channel_fusion_max_weight - 0.25) > 1e-12
    ):
        ap.error(
            "paper_unified fixes --channel_fusion_max_weight 0.25; "
            "use a direct ablation command for another residual bound"
        )
    if args.byte_learned_identifiability_router and not args.byte_ambiguity_aware_targets:
        ap.error(
            "--byte_learned_identifiability_router requires "
            "--byte_ambiguity_aware_targets"
        )
    if args.protocol_content_checkpoint and args.stage != "shared_core_ablation" and (
        args.pretrain_protocol_content or args.stage == "protocol_pretrain"
    ):
        ap.error(
            "--protocol_content_checkpoint is mutually exclusive with "
            "--pretrain_protocol_content/--stage protocol_pretrain"
        )
    if args.protocol_pretrain_epochs <= 0:
        ap.error("--protocol_pretrain_epochs must be positive")
    if args.protocol_pretrain_max_packets < 2:
        ap.error("--protocol_pretrain_max_packets must be at least 2")

    dataset_dir = DATASET_DIRS[args.dataset]
    source = Path(args.source_root) / dataset_dir
    run_name = f"{args.dataset}_fold{args.fold}"
    artifacts = Path(args.artifact_root) / args.dataset / f"fold{args.fold}"
    result_artifacts = (
        Path(args.ablation_output_dir)
        if args.stage == "shared_core_ablation"
        else artifacts
    )
    checkpoint = Path(args.checkpoint_root) / run_name
    train_source = source / f"train_val_split_{args.fold}" / "train"
    valid_source = source / f"train_val_split_{args.fold}" / "val"
    test_source = source / "test"
    train_dir, valid_dir = artifacts / "train", artifacts / "valid"
    test_dir = Path(args.shared_test_dir) if args.shared_test_dir else artifacts / "test"
    intervention_root = artifacts / "intervention_mask_ip_port"
    intervention_dirs = {
        "train": intervention_root / "train",
        "valid": intervention_root / "valid",
        "test": intervention_root / "test",
    }
    label_map = train_dir / "label_map.json"
    fusion_stem = "byte_structural_nested_one_se"
    protocol_pretrain_dir = checkpoint / "shared_content_pretraining"
    should_pretrain_protocol_content = args.stage == "protocol_pretrain" or (
        args.pretrain_protocol_content
        and args.stage in {"byte", "packet_best", "paper_unified", "all"}
    )
    resolved_protocol_content_checkpoint = (
        protocol_pretrain_dir / "best.pt"
        if should_pretrain_protocol_content
        else Path(args.protocol_content_checkpoint) if args.protocol_content_checkpoint else None
    )

    def tower1_packet_train_command() -> list[str]:
        command = [
            sys.executable,
            "train_tower1_multitask.py",
            "--base_model", args.base_model,
            "--label_map", str(label_map),
            "--packet_aux_jsonl", str(train_dir / "packet_auxiliary.jsonl"),
            "--valid_packet_aux_jsonl", str(valid_dir / "packet_auxiliary.jsonl"),
            "--valid_batch_size", str(args.eval_batch_size),
            "--valid_packets_per_flow", str(args.valid_packets_per_flow),
            "--select_metric", "macro_f1",
            "--early_stop_patience", str(args.tower1_early_stop_patience),
            "--output_dir", str(checkpoint),
            "--epochs", str(args.epochs),
            "--packet_batch_size", str(args.packet_batch_size),
            "--max_packet_length", str(args.max_packet_length),
            "--lr", str(args.lr),
            "--head_lr", str(args.head_lr),
            "--cls_weight", str(args.cls_weight),
            "--contrastive_weight", str(args.contrastive_weight),
            "--same_flow_positive_weight", str(args.same_flow_positive_weight),
            "--same_label_positive_weight", str(args.same_label_positive_weight),
            "--temperature", str(args.temperature),
            "--flow_proto_weight", str(args.flow_proto_weight),
            "--flow_proto_positive", args.flow_proto_positive,
            "--flow_proto_context", args.flow_proto_context,
            "--class_weighting", args.class_weighting,
            "--class_weight_basis", args.class_weight_basis,
            "--class_weight_strength", str(args.class_weight_strength),
            "--lora_r", str(args.lora_r),
            "--lora_alpha", str(args.lora_alpha),
            "--lora_dropout", str(args.lora_dropout),
            "--dtype", args.dtype,
            "--seed", str(args.seed),
            "--disable_packet_information_weights",
            "--gradient_accumulation_steps", str(args.gradient_accumulation_steps),
            "--gradient_checkpointing",
            "--flow_balanced_packet_batches",
            "--packets_per_flow", "2",
            "--no_sft",
        ]
        if args.init_checkpoint_dir:
            command.extend(["--init_checkpoint_dir", args.init_checkpoint_dir])
            if args.init_adapter_only:
                command.append("--init_adapter_only")
        if args.tower1_paired_consistency_weight > 0:
            paired_path = intervention_dirs["train"] / "packet_auxiliary.jsonl"
            if not args.dry_run and not paired_path.exists():
                raise FileNotFoundError(
                    "Tower-1 paired consistency requires the mask-IP-port training view: "
                    f"{paired_path}. Run --stage preprocess or --stage paper_unified first."
                )
            command.extend(
                [
                    "--paired_packet_aux_jsonl", str(paired_path),
                    "--paired_consistency_weight", str(args.tower1_paired_consistency_weight),
                    "--paired_cls_weight", str(args.tower1_paired_cls_weight),
                    "--paired_logit_kl_weight", str(args.tower1_paired_logit_kl_weight),
                    "--paired_raw_consistency_weight", str(args.tower1_paired_raw_consistency_weight),
                ]
            )
        if args.local_files_only:
            command.append("--local_files_only")
        return command

    def protocol_content_pretrain_command() -> list[str]:
        return [
            sys.executable,
            "pretrain_native_flow_encoder.py",
            "--train_index", str(train_dir / "packet_index.jsonl"),
            "--valid_index", str(valid_dir / "packet_index.jsonl"),
            "--output_dir", str(protocol_pretrain_dir),
            "--max_packets", str(args.protocol_pretrain_max_packets),
            "--max_bytes", str(args.byte_max_bytes),
            "--hidden_dim", str(args.byte_hidden_dim),
            "--projection_dim", str(args.protocol_pretrain_projection_dim),
            "--byte_layers", str(args.byte_num_layers),
            "--flow_layers", str(args.protocol_pretrain_flow_layers),
            "--num_heads", str(args.byte_num_heads),
            "--dropout", str(args.byte_dropout),
            "--epochs", str(args.protocol_pretrain_epochs),
            "--batch_size", str(args.protocol_pretrain_batch_size),
            "--eval_batch_size", str(args.protocol_pretrain_eval_batch_size),
            "--num_workers", str(args.protocol_pretrain_num_workers),
            "--learning_rate", str(args.protocol_pretrain_learning_rate),
            "--weight_decay", str(args.protocol_pretrain_weight_decay),
            "--field_mask_probability", str(args.protocol_pretrain_field_mask_probability),
            "--payload_dropout_probability", str(
                args.protocol_pretrain_payload_dropout_probability
            ),
            "--session_mask_probability", str(args.protocol_pretrain_session_mask_probability),
            "--masked_byte_weight", str(args.protocol_pretrain_masked_byte_weight),
            "--relative_order_weight", str(args.protocol_pretrain_relative_order_weight),
            "--same_flow_weight", str(args.protocol_pretrain_same_flow_weight),
            "--next_length_weight", str(args.protocol_pretrain_next_length_weight),
            "--next_iat_weight", str(args.protocol_pretrain_next_iat_weight),
            "--direction_weight", str(args.protocol_pretrain_direction_weight),
            "--packet_consistency_weight", str(
                args.protocol_pretrain_packet_consistency_weight
            ),
            "--flow_contrastive_weight", str(args.protocol_pretrain_flow_contrastive_weight),
            "--temperature", str(args.protocol_pretrain_temperature),
            "--patience", str(args.protocol_pretrain_patience),
            "--seed", str(args.protocol_pretrain_seed),
            "--device", args.protocol_pretrain_device,
        ]

    def semantic_cache_paths(split_name: str, intervened: bool = False) -> tuple[Path, Path]:
        suffix = "_mask_ip_port" if intervened else "_full"
        return (
            artifacts / f"{split_name}_semantic_embeddings{suffix}.npy",
            artifacts / f"{split_name}_semantic_embeddings{suffix}_manifest.json",
        )
    write_framework_manifest(
        args, result_artifacts, train_dir, valid_dir, test_dir, profile_overrides,
        completed=False,
    )

    if args.stage in {"preprocess", "paper_unified", "all"}:
        for split, input_dir, output_dir in (
            ("train", train_source, train_dir),
            ("valid", valid_source, valid_dir),
            ("test", test_source, test_dir),
        ):
            if split == "test" and args.shared_test_dir:
                required = output_dir / "packet_auxiliary.jsonl"
                if not required.exists():
                    raise FileNotFoundError(f"shared test artifact is incomplete: {required}")
                print(f"+ reuse shared test artifacts from {output_dir}", flush=True)
                continue
            factual_policy = "full" if args.framework_profile == "paper_unified" else args.embedding_header_policy
            command = [
                sys.executable,
                "preprocess_tower1.py",
                "--input_dir", str(input_dir),
                "--output_dir", str(output_dir),
                "--input_layout", "class_packet_pcaps",
                "--packet_context_policy", args.packet_context_policy,
                "--max_packets_per_flow", str(args.max_packets_per_flow),
                "--payload_prefix_len", "128",
                "--l3_prefix_len", "512",
                "--embedding_header_policy", factual_policy,
                "--classification_only",
            ]
            if split == "train":
                command.append("--write_label_map")
            else:
                command.extend(["--label_map_in", str(label_map)])
            run(command, args.dry_run)
            if args.stage == "paper_unified":
                intervention_command = [
                    sys.executable,
                    "preprocess_tower1.py",
                    "--input_dir", str(input_dir),
                    "--output_dir", str(intervention_dirs[split]),
                    "--input_layout", "class_packet_pcaps",
                    "--packet_context_policy", args.packet_context_policy,
                    "--max_packets_per_flow", str(args.max_packets_per_flow),
                    "--payload_prefix_len", "128",
                    "--l3_prefix_len", "512",
                    "--embedding_header_policy", "mask_ip_port",
                    "--classification_only",
                    "--label_map_in", str(label_map),
                ]
                run(intervention_command, args.dry_run)

    if args.stage in {"audit", "paper_unified", "all"}:
        run(
            [
                sys.executable,
                "audit_packet_flow_split.py",
                "--train", str(train_dir / "packet_auxiliary.jsonl"),
                "--valid", str(valid_dir / "packet_auxiliary.jsonl"),
                "--test", str(test_dir / "packet_auxiliary.jsonl"),
                "--output_json", str(artifacts / "flow_split_audit.json"),
                "--fail_on_overlap",
            ],
            args.dry_run,
        )

    if should_pretrain_protocol_content:
        run(protocol_content_pretrain_command(), args.dry_run)

    if args.stage == "protocol_pretrain":
        write_framework_manifest(
            args,
            result_artifacts,
            train_dir,
            valid_dir,
            test_dir,
            profile_overrides,
            completed=not args.dry_run,
        )
        return

    if args.stage == "paper_unified":
        run(tower1_packet_train_command(), args.dry_run)
        for split_name, split_dir in (("train", train_dir), ("valid", valid_dir), ("test", test_dir)):
            for intervened, prompt_dir, policy in (
                (False, split_dir, "full"),
                (True, intervention_dirs[split_name], "mask_ip_port"),
            ):
                embedding_dir = artifacts / f"{split_name}_semantic_flow_embeddings_{policy}"
                run_semantic_embedding_stage(
                    args,
                    packet_index=prompt_dir / "packet_index.jsonl",
                    output_dir=embedding_dir,
                    checkpoint=checkpoint,
                )
                cache_path, manifest_path = semantic_cache_paths(split_name, intervened)
                run(
                    [
                    sys.executable,
                    "build_packet_semantic_cache.py",
                    "--packet_index", str(split_dir / "packet_index.jsonl"),
                    "--flow_embedding_index", str(embedding_dir / "flow_embedding_index.jsonl"),
                    "--output_npy", str(cache_path),
                    "--output_json", str(manifest_path),
                    ],
                    args.dry_run,
                )

    # The tree feature expert is a legacy/ablation baseline.  The paper-facing
    # path must classify the shared learned packet representation directly.
    if args.stage in {"feature", "packet_best", "all"}:
        run(
            [
                sys.executable,
                "train_packet_feature_expert.py",
                "--train_index", str(train_dir / "packet_index.jsonl"),
                "--valid_index", str(valid_dir / "packet_index.jsonl"),
                "--test_index", str(test_dir / "packet_index.jsonl"),
                "--label_map", str(label_map),
                "--output_json", str(artifacts / "packet_feature_expert.json"),
                "--model_out", str(checkpoint / "feature_expert.joblib"),
                "--byte_prefix_len", "32", "64", "96", "128", "256",
                "--min_samples_leaf", "1", "2", "4",
                "--n_estimators", "300",
                "--estimator_types", "extra_trees", "random_forest",
            ],
            args.dry_run,
        )
        for split_name, split_dir in (("valid", valid_dir), ("test", test_dir)):
            run(
                [
                    sys.executable,
                    "test_packet_feature_expert.py",
                    "--model", str(checkpoint / "feature_expert.joblib"),
                    "--training_result", str(artifacts / "packet_feature_expert.json"),
                    "--test_index", str(split_dir / "packet_index.jsonl"),
                    "--label_map", str(label_map),
                    "--output_json", str(artifacts / f"{split_name}_feature_probs.json"),
                    "--output_npz", str(artifacts / f"{split_name}_feature_probs.npz"),
                ],
                args.dry_run,
            )

    if args.stage in {
        "byte", "packet_best", "paper_unified", "shared_core_ablation", "all"
    }:
        byte_checkpoint = checkpoint / "byte_transformer"
        ablation_tag = (
            f"ablation_{args.train_ablate_input_channel}_"
            f"{args.train_ablate_intervention_view}"
            f"{'_fixed' if args.train_fixed_channel_fusion else ''}"
            f"{'_row_risk' if args.train_row_risk_ablation else ''}"
        )
        validation_output = (
            result_artifacts / f"packet_byte_transformer_validation_{ablation_tag}.json"
            if args.stage == "shared_core_ablation"
            else artifacts / "packet_byte_transformer_validation.json"
        )
        byte_command = [
                sys.executable,
                "train_packet_byte_transformer.py",
                "--train_index", str(train_dir / "packet_index.jsonl"),
                "--valid_index", str(valid_dir / "packet_index.jsonl"),
                "--label_map", str(label_map),
                "--output_dir", str(byte_checkpoint),
                "--output_json", str(validation_output),
                "--max_bytes", str(args.byte_max_bytes),
                "--hidden_dim", str(args.byte_hidden_dim),
                "--num_layers", str(args.byte_num_layers),
                "--num_heads", str(args.byte_num_heads),
                "--dropout", str(args.byte_dropout),
                "--epochs", str(args.byte_epochs),
                "--batch_size", str(args.byte_batch_size),
                "--eval_batch_size", str(args.byte_eval_batch_size),
                "--packets_per_flow", "8",
                "--mask_probability", str(args.byte_mask_probability),
                "--masked_ce_weight", (
                    str(args.byte_masked_ce_weight)
                    if args.byte_ambiguity_aware_targets else "0"
                ),
                "--consistency_weight", (
                    str(args.byte_consistency_weight)
                    if args.byte_ambiguity_aware_targets else "0"
                ),
                "--contrastive_weight", "0",
                "--content_group_loss_reduction", args.byte_content_group_loss_reduction,
                "--channel_fusion_base_mode", args.channel_fusion_base_mode,
                "--channel_fusion_max_weight", str(args.channel_fusion_max_weight),
                "--train_ablate_input_channel", args.train_ablate_input_channel,
                "--train_ablate_intervention_view", args.train_ablate_intervention_view,
            ]
        if args.byte_use_payload_channel:
            byte_command.extend(
                ["--use_payload_channel", "--max_payload_bytes", str(args.byte_max_payload_bytes)]
            )
        if args.byte_use_protocol_fields:
            byte_command.append("--use_protocol_fields")
        if args.byte_exact_shared_representation:
            byte_command.append("--exact_shared_representation")
        if args.train_fixed_channel_fusion:
            byte_command.append("--train_fixed_channel_fusion")
        if args.byte_mask_protocol_session_fields:
            byte_command.append("--mask_protocol_session_fields")
        if resolved_protocol_content_checkpoint:
            byte_command.extend(
                ["--protocol_content_checkpoint", str(resolved_protocol_content_checkpoint)]
            )
        if args.stage in {"paper_unified", "shared_core_ablation"}:
            train_cache, train_manifest = semantic_cache_paths("train")
            valid_cache, valid_manifest = semantic_cache_paths("valid")
            train_intervened_cache, train_intervened_manifest = semantic_cache_paths("train", True)
            valid_intervened_cache, valid_intervened_manifest = semantic_cache_paths("valid", True)
            byte_command.extend(
                [
                    "--semantic_embedding_cache", str(train_cache),
                    "--semantic_embedding_manifest", str(train_manifest),
                    "--valid_semantic_embedding_cache", str(valid_cache),
                    "--valid_semantic_embedding_manifest", str(valid_manifest),
                    "--required_semantic_header_policy", "full",
                    "--required_semantic_packet_context_policy", args.packet_context_policy,
                    "--intervened_semantic_embedding_cache", str(train_intervened_cache),
                    "--intervened_semantic_embedding_manifest", str(train_intervened_manifest),
                    "--valid_intervened_semantic_embedding_cache", str(valid_intervened_cache),
                    "--valid_intervened_semantic_embedding_manifest", str(valid_intervened_manifest),
                    "--required_intervened_semantic_header_policy", "mask_ip_port",
                    "--intervention_max_residual_weight", str(args.intervention_max_residual_weight),
                    "--intervention_view_base_mode", args.intervention_view_base_mode,
                ]
            )
        if args.byte_ambiguity_aware_targets:
            byte_command.extend(
                [
                    "--ambiguity_aware_targets",
                    "--ambiguity_supervision",
                    "reliability_gate",
                    "--ambiguity_gate_strength",
                    str(args.byte_ambiguity_gate_strength),
                ]
            )
        if args.byte_learned_identifiability_router:
            byte_command.extend(
                [
                    "--learned_identifiability_router",
                    "--identifiability_loss_weight",
                    str(args.byte_identifiability_loss_weight),
                ]
            )
        if args.byte_protect_main_gradient:
            byte_command.extend(
                [
                    "--protect_main_gradient",
                    "--gradient_projection_scope",
                    args.byte_gradient_projection_scope,
                ]
            )
        if args.byte_select_invariant_blend:
            byte_command.extend(
                [
                    "--select_invariant_blend",
                    "--invariant_blend_grid_size",
                    str(args.byte_invariant_blend_grid_size),
                ]
            )
        run(byte_command, args.dry_run)
        for split_name, split_dir in (("valid", valid_dir), ("test", test_dir)):
            output_stem = (
                f"{split_name}_unified_packet_single_head"
                if args.stage == "paper_unified"
                else (
                    f"{split_name}_shared_core_{ablation_tag}"
                    if args.stage == "shared_core_ablation"
                    else f"{split_name}_byte_probs"
                )
            )
            test_command = [
                    sys.executable,
                    "test_packet_byte_transformer.py",
                    "--checkpoint", str(byte_checkpoint / "best.pt"),
                    "--packet_index", str(split_dir / "packet_index.jsonl"),
                    "--label_map", str(label_map),
                    "--output_json", str(
                        (result_artifacts if args.stage == "shared_core_ablation" else artifacts)
                        / f"{output_stem}.json"
                    ),
                    "--output_npz", str(
                        (result_artifacts if args.stage == "shared_core_ablation" else artifacts)
                        / f"{output_stem}.npz"
                    ),
                    "--batch_size", str(args.byte_eval_batch_size),
                ]
            if args.stage in {"paper_unified", "shared_core_ablation"}:
                cache_path, manifest_path = semantic_cache_paths(split_name)
                intervened_cache_path, intervened_manifest_path = semantic_cache_paths(split_name, True)
                test_command.extend(
                    [
                        "--semantic_embedding_cache", str(cache_path),
                        "--semantic_embedding_manifest", str(manifest_path),
                        "--required_semantic_header_policy", "full",
                        "--required_semantic_packet_context_policy", args.packet_context_policy,
                        "--intervened_semantic_embedding_cache", str(intervened_cache_path),
                        "--intervened_semantic_embedding_manifest", str(intervened_manifest_path),
                        "--required_intervened_semantic_header_policy", "mask_ip_port",
                    ]
                )
            run(test_command, args.dry_run)

    if args.stage in {"fusion", "packet_best", "all"}:
        run(
            [
                sys.executable,
                "fuse_packet_experts.py",
                "--valid_semantic", str(artifacts / "valid_byte_probs.npz"),
                "--valid_structural", str(artifacts / "valid_feature_probs.npz"),
                "--test_semantic", str(artifacts / "test_byte_probs.npz"),
                "--test_structural", str(artifacts / "test_feature_probs.npz"),
                "--label_map", str(label_map),
                "--output_json", str(artifacts / f"test_{fusion_stem}.json"),
                "--output_npz", str(artifacts / f"test_{fusion_stem}.npz"),
                "--output_valid_npz", str(artifacts / f"valid_{fusion_stem}.npz"),
                "--gate_out", str(checkpoint / f"{fusion_stem}_gate.joblib"),
            ],
            args.dry_run,
        )

    if args.stage in {"train", "all"}:
        run(tower1_packet_train_command(), args.dry_run)

    if args.stage in {"test", "all"}:
        for split_name, split_dir in (("valid", valid_dir), ("test", test_dir)):
            command = [
                sys.executable,
                "test_tower1_packet.py",
                "--checkpoint_dir", str(checkpoint / "best"),
                "--packet_aux_jsonl", str(split_dir / "packet_auxiliary.jsonl"),
                "--label_map", str(label_map),
                "--output_json", str(artifacts / f"{split_name}_semantic_probs.json"),
                "--output_npz", str(artifacts / f"{split_name}_semantic_probs.npz"),
                "--batch_size", str(args.eval_batch_size),
                "--max_packet_length", str(args.max_packet_length),
            ]
            if args.local_files_only:
                command.append("--local_files_only")
            run(command, args.dry_run)

    if args.stage == "all":
        run(
            [
                sys.executable,
                "fuse_packet_experts.py",
                "--valid_semantic", str(artifacts / "valid_byte_structural_nested_one_se.npz"),
                "--valid_structural", str(artifacts / "valid_semantic_probs.npz"),
                "--test_semantic", str(artifacts / "test_byte_structural_nested_one_se.npz"),
                "--test_structural", str(artifacts / "test_semantic_probs.npz"),
                "--label_map", str(label_map),
                "--output_json", str(artifacts / "test_unified_packet_nested_one_se.json"),
                "--output_npz", str(artifacts / "test_unified_packet_nested_one_se.npz"),
                "--gate_out", str(checkpoint / "unified_packet_nested_one_se_gate.joblib"),
            ],
            args.dry_run,
        )

    write_framework_manifest(
        args, result_artifacts, train_dir, valid_dir, test_dir, profile_overrides,
        completed=not args.dry_run,
    )


if __name__ == "__main__":
    main()
