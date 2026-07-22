#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import Iterable, List, Sequence

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
    merge_embedding_shards as merge_embedding_shard_outputs,
)
from method_source_provenance import (
    complete_source_stability,
    unified_method_source_snapshot,
)


def format_cmd(cmd: List[str]) -> str:
    return " ".join(shlex.quote(x) for x in cmd)


def run(cmd: List[str], dry_run: bool = False) -> None:
    print("$ " + format_cmd(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True)


def cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def default_split_paths(dataset: str, fold: int = 0) -> tuple[str, str, str]:
    root = "/home/jing/download/sweet/flow-level-classification"
    fold = max(int(fold), 0)
    if dataset == "vpn-app":
        base = f"{root}/vpn-app"
        return f"{base}/train_val_split_{fold}/train", f"{base}/train_val_split_{fold}/val", f"{base}/test"
    if dataset in {"tls", "tls-120"}:
        base = f"{root}/tls"
        return f"{base}/train_val_split_{fold}/train", f"{base}/train_val_split_{fold}/val", f"{base}/test"
    packet_root = "/home/jing/download/sweet/packet-level-classification/per-flow-split"
    if dataset in {"ustc-app", "ustc-binary"}:
        base = f"{packet_root}/{dataset}"
        return f"{base}/train_val_split_0/train", f"{base}/train_val_split_0/val", f"{base}/test"
    raise ValueError(f"No default paths for dataset={dataset}; pass --train_dir/--valid_dir/--test_dir.")


def selected_splits(raw: str | Sequence[str]) -> List[str]:
    out = []
    values = raw.split(",") if isinstance(raw, str) else raw
    for split in values:
        split = split.strip()
        if not split:
            continue
        if split not in {"train", "valid", "test"}:
            raise ValueError(f"Unknown split {split}; use train,valid,test")
        if split not in out:
            out.append(split)
    return out or ["train", "valid", "test"]


def selected_eval_splits(raw: str | Sequence[str]) -> List[str]:
    out: List[str] = []
    values = raw.split(",") if isinstance(raw, str) else raw
    for split in values:
        split = split.strip()
        if not split:
            continue
        if split not in {"valid", "test"}:
            raise ValueError(f"Unknown evaluation split {split}; use valid,test")
        if split not in out:
            out.append(split)
    if not out:
        raise ValueError("At least one evaluation split is required")
    return out


PAPER_UNIFIED_POST_STAGES = (
    "tower2_preprocess",
    "tower2_train",
    "eval",
    "fusion",
    "stacker",
    "prior",
    "selector",
)


def selected_paper_unified_stages(raw: str) -> List[str]:
    out: List[str] = []
    aliases = {
        "model": ("tower2_preprocess", "tower2_train", "eval"),
        "model_fusion": ("tower2_preprocess", "tower2_train", "eval", "fusion"),
        "all": PAPER_UNIFIED_POST_STAGES,
    }
    for stage in raw.split(","):
        stage = stage.strip()
        if not stage:
            continue
        expanded = aliases.get(stage, (stage,))
        for item in expanded:
            if item not in PAPER_UNIFIED_POST_STAGES:
                raise ValueError(
                    f"Unknown paper_unified stage {item}; use comma-separated values from "
                    + ",".join(PAPER_UNIFIED_POST_STAGES)
                )
            if item not in out:
                out.append(item)
    return out or list(PAPER_UNIFIED_POST_STAGES)


def merge_completed_paper_unified_stages(
    existing_payload: dict | None,
    requested_stages: List[str],
    completed: bool,
) -> List[str]:
    done: List[str] = []
    notes = (existing_payload or {}).get("framework", {}).get("notes", {})
    previous = notes.get("completed_paper_unified_stages")
    if previous is None and notes.get("completed"):
        previous = notes.get("paper_unified_stages")
    if isinstance(previous, list):
        for stage in previous:
            if stage in PAPER_UNIFIED_POST_STAGES and stage not in done:
                done.append(stage)
    if completed:
        for stage in requested_stages:
            if stage not in done:
                done.append(stage)
    return done


def py() -> str:
    return "python"


def safe_name(value: str) -> str:
    return value.replace("/", "_").replace("-", "_")


def train_tower1_dir(args) -> str:
    return f"reasoningDataset/{args.dataset}/train_tower1_{args.tower1_data_suffix}"


def valid_tower1_dir(args) -> str:
    return f"reasoningDataset/{args.dataset}/valid_tower1_{args.tower1_data_suffix}"


def train_label_map(args) -> str:
    return f"{train_tower1_dir(args)}/label_map.json"


def result_suffix(args) -> str:
    suffix = f"{tower2_data_suffix(args)}_stage8_flowaware"
    if args.run_tag:
        suffix += f"_{safe_name(args.run_tag)}"
    return suffix


def tower1_train_cmd(args) -> List[str]:
    cmd = [
        py(),
        "train_tower1_multitask.py",
        "--base_model",
        args.base_model,
        "--label_map",
        train_label_map(args),
        "--packet_aux_jsonl",
        f"{train_tower1_dir(args)}/packet_auxiliary.jsonl",
        "--valid_packet_aux_jsonl",
        f"{valid_tower1_dir(args)}/packet_auxiliary.jsonl",
        "--valid_batch_size",
        str(args.tower1_valid_batch_size),
        "--valid_packets_per_flow",
        str(args.tower1_valid_packets_per_flow),
        "--select_metric",
        "macro_f1",
        "--early_stop_patience",
        str(args.tower1_early_stop_patience),
        "--output_dir",
        args.tower1_output_dir,
        "--epochs",
        str(args.tower1_epochs),
        "--sft_batch_size",
        str(args.sft_batch_size),
        "--packet_batch_size",
        str(args.packet_batch_size),
        "--gradient_accumulation_steps",
        str(args.tower1_gradient_accumulation_steps),
        "--max_sft_length",
        str(args.max_sft_length),
        "--max_packet_length",
        str(args.max_packet_length),
        "--cls_weight",
        str(args.cls_weight),
        "--contrastive_weight",
        str(args.contrastive_weight),
        "--same_flow_positive_weight",
        str(args.same_flow_positive_weight),
        "--same_label_positive_weight",
        str(args.same_label_positive_weight),
        "--class_weighting",
        args.tower1_class_weighting,
        "--class_weight_basis",
        args.tower1_class_weight_basis,
        "--class_weight_strength",
        str(args.tower1_class_weight_strength),
        "--flow_proto_weight",
        str(args.flow_proto_weight),
        "--flow_proto_positive",
        args.flow_proto_positive,
        "--flow_proto_context",
        args.flow_proto_context,
        "--temperature",
        str(args.temperature),
        "--lr",
        str(args.tower1_lr),
        "--head_lr",
        str(args.tower1_head_lr),
        "--lora_r",
        str(args.lora_r),
        "--lora_alpha",
        str(args.lora_alpha),
        "--lora_dropout",
        str(args.lora_dropout),
        "--dtype",
        args.dtype,
        "--device",
        args.tower1_device,
        "--seed",
        str(args.seed),
    ]
    if getattr(args, "train_fixed_channel_fusion", False):
        cmd.append("--train_fixed_channel_fusion")
    if args.tower1_use_sft:
        cmd += [
            "--sft_jsonl",
            f"{train_tower1_dir(args)}/packet_instruction.jsonl",
            f"{train_tower1_dir(args)}/packet_validity.jsonl",
        ]
    else:
        cmd.append("--no_sft")
    if args.tower1_disable_packet_information_weights:
        cmd.append("--disable_packet_information_weights")
    if args.tower1_max_steps > 0:
        cmd += ["--max_steps", str(args.tower1_max_steps)]
    if args.tower1_save_steps > 0:
        cmd += ["--save_steps", str(args.tower1_save_steps)]
    if args.tower1_init_checkpoint_dir:
        cmd += ["--init_checkpoint_dir", args.tower1_init_checkpoint_dir]
    if args.tower1_paired_data_suffix:
        cmd += [
            "--paired_packet_aux_jsonl",
            f"reasoningDataset/{args.dataset}/train_tower1_{args.tower1_paired_data_suffix}/packet_auxiliary.jsonl",
            "--paired_consistency_weight",
            str(args.tower1_paired_consistency_weight),
            "--paired_cls_weight",
            str(args.tower1_paired_cls_weight),
            "--paired_logit_kl_weight",
            str(args.tower1_paired_logit_kl_weight),
            "--paired_raw_consistency_weight",
            str(args.tower1_paired_raw_consistency_weight),
        ]
    if args.flow_balanced_packet_batches:
        cmd += [
            "--flow_balanced_packet_batches",
            "--packets_per_flow",
            str(args.packets_per_flow),
            "--packet_batch_scheduler",
            args.packet_batch_scheduler,
        ]
    if args.local_files_only:
        cmd.append("--local_files_only")
    if args.gradient_checkpointing:
        cmd.append("--gradient_checkpointing")
    if args.no_progress:
        cmd.append("--no_load_progress")
    return cmd


def tower1_preprocess_cmd(args, split: str) -> List[str]:
    input_dir = getattr(args, f"{split}_dir")
    out_dir = f"reasoningDataset/{args.dataset}/{split}_tower1_{args.output_suffix}"
    source_label_map = f"reasoningDataset/{args.dataset}/train_tower1_{args.source_suffix}/label_map.json"
    cmd = [
        py(),
        "preprocess_tower1.py",
        "--input_dir",
        input_dir,
        "--output_dir",
        out_dir,
        "--max_packets_per_flow",
        str(args.max_packets_per_flow),
        "--payload_prefix_len",
        str(args.payload_prefix_len),
        "--l3_prefix_len",
        str(args.l3_prefix_len),
        "--packet_context_policy",
        args.packet_context_policy,
    ]
    if args.preprocess_max_flows > 0:
        cmd += ["--max_flows", str(args.preprocess_max_flows)]
    if split != "train" or not args.tower1_use_sft:
        cmd.append("--classification_only")
    if args.embedding_header_policy != "full":
        cmd += ["--embedding_header_policy", args.embedding_header_policy]
    if split == "train":
        cmd.append("--write_label_map")
        if Path(source_label_map).exists():
            cmd += ["--label_map_in", source_label_map]
    else:
        cmd += ["--label_map_in", train_label_map(args)]
    if args.no_progress:
        cmd.append("--no_progress")
    return cmd


def embedding_output_dir(args, split: str) -> str:
    return f"reasoningDataset/{args.dataset}/{split}_embeddings_{args.embedding_suffix}"


def flow_embedding_index_path(args, split: str) -> str:
    return f"{embedding_output_dir(args, split)}/flow_embedding_index.jsonl"


def tower2_data_suffix(args) -> str:
    if getattr(args, "tower2_suffix", ""):
        return args.tower2_suffix
    native_suffix = getattr(args, "native_structural_suffix", "")
    if native_suffix:
        return f"{args.embedding_suffix}_native_{native_suffix}"
    return args.embedding_suffix


def native_pretrain_output_dir(args) -> str:
    if getattr(args, "native_pretrain_output_dir", ""):
        return args.native_pretrain_output_dir
    suffix = getattr(args, "native_structural_suffix", "") or tower2_data_suffix(args)
    return f"checkpoints/native_flow_encoder_{safe_name(args.dataset)}_{suffix}"


def native_checkpoint_path(args) -> str:
    return getattr(args, "native_checkpoint", "") or f"{native_pretrain_output_dir(args)}/best.pt"


def native_structural_output_dir(args, split: str) -> str:
    suffix = getattr(args, "native_structural_suffix", "")
    if not suffix:
        raise ValueError("--native_structural_suffix is required for native structural extraction")
    return f"reasoningDataset/{args.dataset}/{split}_native_structural_{suffix}"


def native_structural_index_path(args, split: str) -> str:
    return f"{native_structural_output_dir(args, split)}/flow_structural_embedding_index.jsonl"


def effective_native_structural_dim(args) -> int:
    return int(getattr(args, "native_structural_dim", 0) or 0)


def embedding_cmd(
    args,
    split: str,
    output_dir: str | None = None,
    shard_index: int | None = None,
    device: str | None = None,
) -> List[str]:
    cmd = [
        py(),
        "extract_packet_embeddings_qwen.py",
        "--packet_index",
        f"reasoningDataset/{args.dataset}/{split}_tower1_{args.output_suffix}/packet_index.jsonl",
        "--output_dir",
        output_dir or embedding_output_dir(args, split),
        "--base_model",
        args.base_model,
        "--lora_path",
        f"{tower1_embedding_checkpoint_dir(args)}/adapter",
        "--tower1_heads",
        f"{tower1_embedding_checkpoint_dir(args)}/tower1_heads.pt",
        "--embedding_mode",
        args.embedding_mode,
        "--batch_size",
        str(args.embedding_batch_size),
        "--flow_batch_packets",
        str(getattr(args, "embedding_flow_batch_packets", 128)),
        "--max_length",
        str(args.embedding_max_length),
        "--device",
        device or args.embedding_device,
        "--local_files_only",
    ]
    if args.embedding_resume_existing:
        cmd.append("--resume_existing")
    if shard_index is not None:
        cmd += ["--num_shards", str(args.embedding_num_shards), "--shard_index", str(shard_index)]
    if args.no_progress:
        cmd.append("--no_progress")
    return cmd


def tower1_embedding_checkpoint_dir(args) -> str:
    if getattr(args, "framework_profile", "legacy") == "paper_unified":
        return str(Path(args.tower1_output_dir) / "best")
    return args.tower1_output_dir


def embedding_audit_cmd(args, split: str) -> List[str]:
    output_dir = Path(embedding_output_dir(args, split))
    cmd = [
        py(),
        "audit_flow_embeddings.py",
        "--packet_index",
        f"reasoningDataset/{args.dataset}/{split}_tower1_{args.output_suffix}/packet_index.jsonl",
        "--flow_embedding_index",
        str(output_dir / "flow_embedding_index.jsonl"),
        "--output_json",
        str(output_dir / "embedding_audit.json"),
    ]
    if getattr(args, "framework_profile", "legacy") == "paper_unified":
        cmd.append("--require_model_provenance")
    return cmd


def merge_embedding_shards(args, split: str) -> None:
    out_dir = Path(embedding_output_dir(args, split))
    packet_index = Path(
        f"reasoningDataset/{args.dataset}/{split}_tower1_{args.output_suffix}/packet_index.jsonl"
    )
    devices = [x.strip() for x in args.embedding_cuda_devices.split(",") if x.strip()]
    summary = merge_embedding_shard_outputs(
        packet_index=packet_index,
        output_dir=out_dir,
        num_shards=args.embedding_num_shards,
        shard_devices=devices,
    )
    print(
        f"merged embedding shards: split={split}, flows={summary['merged_flows']}, "
        f"index={summary['merged_index']}",
        flush=True,
    )


def run_embedding_stage(args, split: str) -> None:
    if args.embedding_num_shards <= 1:
        run(embedding_cmd(args, split), dry_run=args.dry_run)
        run(embedding_audit_cmd(args, split), dry_run=args.dry_run)
        return

    out_dir = Path(embedding_output_dir(args, split))
    shard_root = out_dir / "_shards"
    devices = [x.strip() for x in args.embedding_cuda_devices.split(",") if x.strip()]
    if args.dry_run:
        for shard_index in range(args.embedding_num_shards):
            device = devices[shard_index % len(devices)] if devices else ""
            device_arg = device if device.startswith("cuda:") else (f"cuda:{device}" if device else "")
            cmd = embedding_cmd(
                args,
                split,
                output_dir=str(shard_root / f"shard_{shard_index}"),
                shard_index=shard_index,
                device=device_arg or args.embedding_device,
            )
            print("$ " + format_cmd(cmd), flush=True)
        return
    packet_index = Path(f"reasoningDataset/{args.dataset}/{split}_tower1_{args.output_suffix}/packet_index.jsonl")
    expected_counts = expected_embedding_shard_counts(packet_index, args.embedding_num_shards)
    procs: list[tuple[int, subprocess.Popen]] = []
    for shard_index in range(args.embedding_num_shards):
        shard_dir = shard_root / f"shard_{shard_index}"
        shard_index_path = shard_dir / "flow_embedding_index.jsonl"
        actual_count = jsonl_row_count(shard_index_path)
        if args.embedding_resume_shards and actual_count == expected_counts[shard_index]:
            print(
                f"skip completed embedding shard: split={split}, shard={shard_index}, flows={actual_count}",
                flush=True,
            )
            continue
        if actual_count:
            print(
                f"rerun incomplete embedding shard: split={split}, shard={shard_index}, "
                f"flows={actual_count}/{expected_counts[shard_index]}",
                flush=True,
            )
        device_arg = ""
        device = devices[shard_index % len(devices)] if devices else ""
        if device:
            device_arg = device if device.startswith("cuda:") else f"cuda:{device}"
        cmd = embedding_cmd(
            args,
            split,
            output_dir=str(shard_dir),
            shard_index=shard_index,
            device=device_arg or args.embedding_device,
        )
        print("$ " + format_cmd(cmd), flush=True)
        if not args.dry_run:
            procs.append((shard_index, subprocess.Popen(cmd, env=os.environ.copy())))

    if args.dry_run:
        run(embedding_audit_cmd(args, split), dry_run=True)
        return
    failed: list[tuple[int, int]] = []
    for shard_index, proc in procs:
        code = proc.wait()
        if code != 0:
            failed.append((shard_index, code))
    if failed:
        raise subprocess.CalledProcessError(failed[0][1], f"embedding shards failed: {failed}")
    merge_embedding_shards(args, split)
    run(embedding_audit_cmd(args, split), dry_run=False)


def tower2_preprocess_cmd(args, split: str) -> List[str]:
    flow_index = flow_embedding_index_path(args, split)
    cmd = [
        py(),
        "preprocess_tower2.py",
        "--flow_embedding_index",
        flow_index,
        "--output_dir",
        f"reasoningDataset/{args.dataset}/{split}_tower2_{tower2_data_suffix(args)}",
        "--window_size",
        str(args.window_size),
        "--stride",
        str(args.stride),
    ]
    if getattr(args, "native_structural_suffix", ""):
        cmd += ["--structural_embedding_index", native_structural_index_path(args, split)]
        if getattr(args, "framework_profile", "legacy") == "paper_unified":
            cmd += [
                "--require_structural_scope", "strict_current_packet",
                "--strict_current_packet_features",
            ]
    if getattr(args, "content_group_guard", False):
        cmd += ["--content_group_index", flow_index]
    return cmd


def native_pretrain_cmd(args) -> List[str]:
    return [
        py(),
        "pretrain_native_flow_encoder.py",
        "--train_index",
        f"reasoningDataset/{args.dataset}/train_tower1_{args.output_suffix}/packet_index.jsonl",
        "--valid_index",
        f"reasoningDataset/{args.dataset}/valid_tower1_{args.output_suffix}/packet_index.jsonl",
        "--output_dir",
        native_pretrain_output_dir(args),
        "--max_packets",
        str(args.native_max_packets),
        "--max_bytes",
        str(args.native_max_bytes),
        "--epochs",
        str(args.native_epochs),
        "--batch_size",
        str(args.native_batch_size),
        "--eval_batch_size",
        str(args.native_eval_batch_size),
        "--num_workers",
        str(args.native_num_workers),
        "--learning_rate",
        str(args.native_learning_rate),
        "--weight_decay",
        str(args.native_weight_decay),
        "--hidden_dim",
        str(args.native_hidden_dim),
        "--projection_dim",
        str(args.native_projection_dim),
        "--byte_layers",
        str(args.native_byte_layers),
        "--flow_layers",
        str(args.native_flow_layers),
        "--num_heads",
        str(args.native_num_heads),
        "--dropout",
        str(args.native_dropout),
        "--field_mask_probability",
        str(args.native_field_mask_probability),
        "--payload_dropout_probability",
        str(args.native_payload_dropout_probability),
        "--session_mask_probability",
        str(args.native_session_mask_probability),
        "--masked_byte_weight",
        str(args.native_masked_byte_weight),
        "--relative_order_weight",
        str(args.native_relative_order_weight),
        "--same_flow_weight",
        str(args.native_same_flow_weight),
        "--next_length_weight",
        str(args.native_next_length_weight),
        "--next_iat_weight",
        str(args.native_next_iat_weight),
        "--direction_weight",
        str(args.native_direction_weight),
        "--flow_contrastive_weight",
        str(args.native_flow_contrastive_weight),
        "--packet_consistency_weight",
        str(args.native_packet_consistency_weight),
        "--temperature",
        str(args.native_temperature),
        "--patience",
        str(args.native_patience),
        "--seed",
        str(args.seed),
        "--device",
        args.native_device,
    ]


def native_extract_cmd(args, split: str) -> List[str]:
    return [
        py(),
        "extract_native_flow_embeddings.py",
        "--checkpoint",
        native_checkpoint_path(args),
        "--packet_index",
        f"reasoningDataset/{args.dataset}/{split}_tower1_{args.output_suffix}/packet_index.jsonl",
        "--output_dir",
        native_structural_output_dir(args, split),
        "--batch_size",
        str(args.native_extract_batch_size),
        "--session_mask_probability",
        str(args.native_session_mask_probability),
        "--device",
        args.native_device,
    ]


def tower2_train_cmd(args, model_type: str) -> List[str]:
    native_dim = effective_native_structural_dim(args)
    tower2_meta_dim = int(args.meta_feature_dim) + native_dim
    cmd = [
        py(),
        "train_tower2.py",
        "--model_type",
        model_type,
        "--dataset",
        f"reasoningDataset/{args.dataset}/train_tower2_{tower2_data_suffix(args)}/{model_type}_dataset.pt",
        "--valid_dataset",
        f"reasoningDataset/{args.dataset}/valid_tower2_{tower2_data_suffix(args)}/{model_type}_dataset.pt",
        "--output_dir",
        f"checkpoints/tower2_{model_type}_flow_{safe_name(args.dataset)}_{result_suffix(args)}",
        "--num_classes",
        str(args.num_classes),
        "--epochs",
        str(args.tower2_epochs),
        "--batch_size",
        str(args.tower2_batch_size),
        "--hidden_dim",
        str(args.hidden_dim),
        "--num_layers",
        str(args.num_layers),
        "--num_heads",
        str(args.num_heads),
        "--dropout",
        str(args.dropout),
        "--lr",
        str(args.tower2_lr),
        "--weight_decay",
        str(args.weight_decay),
        "--train_level",
        "flow",
        "--select_metric",
        args.tower2_select_metric,
        "--early_stop_patience",
        str(args.tower2_early_stop_patience),
        "--flow_pooling",
        args.flow_pooling,
        "--multi_view_gate_entropy_weight",
        str(args.multi_view_gate_entropy_weight),
        "--flow_stat_expert_weight",
        str(args.flow_stat_expert_weight),
        "--flow_stat_aux_weight",
        str(args.flow_stat_aux_weight),
        "--window_loss_weight",
        str(args.window_loss_weight),
        "--class_weighting",
        "effective",
        "--class_weight_beta",
        "0.9999",
        "--class_weight_strength",
        str(args.class_weight_strength),
        "--label_smoothing",
        str(args.label_smoothing),
        "--confidence_penalty_weight",
        str(args.confidence_penalty_weight),
        "--hierarchical_weight",
        str(args.hierarchical_weight),
        "--hierarchical_logit_weight",
        str(args.hierarchical_logit_weight),
        "--coarse_groups",
        args.coarse_groups,
        "--balanced_flow_batches",
        "--samples_per_class",
        str(args.samples_per_class),
        "--split_group_key",
        args.split_group_key,
        "--content_group_loss_reduction",
        args.content_group_loss_reduction,
        "--contrastive_mode",
        args.contrastive_mode,
        "--confusion_groups",
        args.confusion_groups,
        "--flow_contrastive_weight",
        str(args.flow_contrastive_weight),
        "--flow_temperature",
        str(args.flow_temperature),
        "--window_contrastive_weight",
        str(args.window_contrastive_weight),
        "--window_contrastive_temperature",
        str(args.window_contrastive_temperature),
        "--window_contrastive_positive",
        args.window_contrastive_positive,
        "--consistency_weight",
        str(args.consistency_weight),
        "--meta_dropout_prob",
        str(args.meta_dropout_prob),
        "--meta_feature_dim",
        str(tower2_meta_dim),
        "--dual_channel_mode",
        getattr(args, "dual_channel_mode", "concat"),
        "--dual_channel_gate_mode",
        getattr(args, "dual_channel_gate_mode", "global"),
        "--channel_fusion_base_mode",
        getattr(args, "channel_fusion_base_mode", "legacy"),
        "--dual_channel_max_weight",
        str(getattr(args, "dual_channel_max_weight", 0.25)),
        "--embedding_dropout_prob",
        str(args.embedding_dropout_prob),
        "--window_dropout_prob",
        str(args.window_dropout_prob),
        "--edge_attr_dropout_prob",
        str(args.edge_attr_dropout_prob),
        "--train_ablate_input_channel",
        getattr(args, "train_ablate_input_channel", "none"),
        "--train_ablate_intervention_view",
        getattr(args, "train_ablate_intervention_view", "none"),
        "--aux_weight",
        "0",
        "--coherence_weight",
        "0",
        "--seed",
        str(args.seed),
    ]
    if args.content_group_unique_batches:
        cmd.append("--content_group_unique_batches")
    if getattr(args, "exact_shared_packet_encoder", False):
        if model_type != "seq":
            raise ValueError(
                "the exact shared packet representation currently requires the "
                "sequence flow aggregator; graph is retained as a structural ablation"
            )
        cmd += [
            "--exact_shared_packet_encoder",
            "--shared_packet_hidden_dim",
            str(getattr(args, "shared_packet_hidden_dim", 128)),
            "--packet_evidence_max_weight",
            str(getattr(args, "packet_evidence_max_weight", 0.0)),
        ]
    if native_dim > 0:
        cmd += ["--native_structural_dim", str(native_dim)]
    if getattr(args, "paired_embedding_suffix", ""):
        cmd += [
            "--paired_view_dataset",
            f"reasoningDataset/{args.dataset}/train_tower2_{args.paired_embedding_suffix}/{model_type}_dataset.pt",
            "--paired_valid_dataset",
            f"reasoningDataset/{args.dataset}/valid_tower2_{args.paired_embedding_suffix}/{model_type}_dataset.pt",
            "--paired_view_weight",
            str(getattr(args, "paired_view_weight", 0.0)),
            "--paired_consistency_weight",
            str(getattr(args, "paired_consistency_weight", 0.0)),
            "--paired_alignment_weight",
            str(getattr(args, "paired_alignment_weight", 0.0)),
            "--paired_crossview_contrastive_weight",
            str(getattr(args, "paired_crossview_contrastive_weight", 0.0)),
            "--paired_crossview_temperature",
            str(getattr(args, "paired_crossview_temperature", 0.07)),
            "--paired_variance_weight",
            str(getattr(args, "paired_variance_weight", 0.0)),
            "--paired_variance_target",
            str(getattr(args, "paired_variance_target", 0.04)),
            "--paired_covariance_weight",
            str(getattr(args, "paired_covariance_weight", 0.0)),
            "--view_domain_adversarial_weight",
            str(getattr(args, "view_domain_adversarial_weight", 0.0)),
            "--domain_adversarial_lambda",
            str(getattr(args, "domain_adversarial_lambda", 1.0)),
        ]
        if getattr(args, "use_intervention_views", False):
            cmd += [
                "--use_intervention_views",
                "--intervention_max_residual_weight",
                str(getattr(args, "intervention_max_residual_weight", 0.25)),
                "--intervention_view_base_mode",
                getattr(args, "intervention_view_base_mode", "symmetric_mean"),
            ]
    if args.environment_map_json:
        cmd += [
            "--environment_map_json",
            args.environment_map_json,
            "--environment_risk_weight",
            str(args.environment_risk_weight),
            "--environment_alignment_weight",
            str(args.environment_alignment_weight),
        ]
    if args.distill_targets_json:
        cmd += [
            "--distill_targets_json",
            args.distill_targets_json,
            "--distill_weight",
            str(args.distill_weight),
            "--distill_class_prior_weight",
            str(args.distill_class_prior_weight),
            "--distill_temperature",
            str(args.distill_temperature),
            "--distill_min_confidence",
            str(args.distill_min_confidence),
            "--distill_max_confidence",
            str(args.distill_max_confidence),
            "--distill_confidence_power",
            str(args.distill_confidence_power),
            "--distill_min_teachers_per_flow",
            str(args.distill_min_teachers_per_flow),
            "--distill_min_coverage",
            str(args.distill_min_coverage),
            "--distill_low_coverage_action",
            args.distill_low_coverage_action,
        ]
        if args.distill_require_oof_exclusion_proof:
            cmd.append("--distill_require_oof_exclusion_proof")
    return cmd


def tower2_eval_cmd(args, model_type: str, split: str) -> List[str]:
    prefix = "valid" if split == "valid" else "test"
    cmd = [
        py(),
        "test_tower2.py",
        "--checkpoint",
        f"checkpoints/tower2_{model_type}_flow_{safe_name(args.dataset)}_{result_suffix(args)}/best.pt",
        "--dataset",
        f"reasoningDataset/{args.dataset}/{split}_tower2_{tower2_data_suffix(args)}/{model_type}_dataset.pt",
        "--label_map",
        train_label_map(args),
        "--output_json",
        f"reasoningDataset/{args.dataset}/{prefix}_{model_type}_metrics_flow_{result_suffix(args)}_probs.json",
        "--no_report",
    ]
    if args.use_intervention_views:
        cmd += [
            "--paired_view_dataset",
            f"reasoningDataset/{args.dataset}/{split}_tower2_{args.paired_embedding_suffix}/{model_type}_dataset.pt",
        ]
    return cmd


def model_prob_path(args, model_type: str, split: str) -> str:
    prefix = "valid" if split == "valid" else "test"
    return f"reasoningDataset/{args.dataset}/{prefix}_{model_type}_metrics_flow_{result_suffix(args)}_probs.json"


def fusion_payload_path(args, model_type: str) -> str:
    return f"reasoningDataset/{args.dataset}/{model_type}_metrics_flow_{result_suffix(args)}_fusion_payload.json"


def make_fusion_payload_cmd(args, model_type: str) -> List[str]:
    return [
        py(),
        "make_fusion_payload.py",
        "--valid_json",
        model_prob_path(args, model_type, "valid"),
        "--test_json",
        model_prob_path(args, model_type, "test"),
        "--output_json",
        fusion_payload_path(args, model_type),
    ]


def fusion_cmd(args) -> List[str]:
    cmd = [
        py(),
        "fuse_prediction_jsons.py",
    ]
    for model_type in selected_model_types(args):
        cmd += [
            "--input",
            model_type,
            fusion_payload_path(args, model_type),
        ]
    cmd += [
        "--label_map",
        train_label_map(args),
        "--simplex_step",
        "0.05",
        "--select_metric",
        "accuracy",
        "--output_json",
        fusion_output_path(args),
    ]
    return cmd


def fusion_output_path(args) -> str:
    return f"reasoningDataset/{args.dataset}/test_fusion_{'_'.join(selected_model_types(args))}_{result_suffix(args)}_valid_acc.json"


def stacker_output_path(args) -> str:
    return f"reasoningDataset/{args.dataset}/test_stacker_{'_'.join(selected_model_types(args))}_{result_suffix(args)}_{safe_name(args.stacker_select_metric)}.json"


def stacker_cmd(args) -> List[str]:
    cmd = [
        py(),
        "train_prediction_stacker.py",
    ]
    for model_type in selected_model_types(args):
        cmd += ["--input", model_type, fusion_payload_path(args, model_type)]
    cmd += [
        "--label_map",
        train_label_map(args),
        "--select_metric",
        args.stacker_select_metric,
        "--c_grid",
        args.stacker_c_grid,
        "--class_weight_grid",
        args.stacker_class_weight_grid,
        "--base_input",
        args.stacker_base_input or selected_model_types(args)[0],
        "--unified_expert_slots",
        args.stacker_unified_expert_slots or ",".join(selected_model_types(args)),
        "--min_valid_gain_over_base",
        str(args.stacker_min_valid_gain_over_base),
        "--max_prediction_change_rate",
        str(args.stacker_max_prediction_change_rate),
        "--max_prediction_js_divergence",
        str(args.stacker_max_prediction_js_divergence),
        "--seed",
        str(args.seed),
        "--output_json",
        stacker_output_path(args),
    ]
    if args.stacker_include_logits:
        cmd.append("--include_logits")
    if args.stacker_include_confidence:
        cmd.append("--include_confidence")
    return cmd


def prior_candidate_output_path(args) -> str:
    return f"reasoningDataset/{args.dataset}/test_fusion_{'_'.join(selected_model_types(args))}_{result_suffix(args)}_safe_prior_candidate.json"


def safe_prior_output_path(args) -> str:
    return f"reasoningDataset/{args.dataset}/test_fusion_{'_'.join(selected_model_types(args))}_{result_suffix(args)}_safe_prior_residual.json"


def selector_output_path(args) -> str:
    return f"reasoningDataset/{args.dataset}/test_selector_base_prior_stacker_{'_'.join(selected_model_types(args))}_{result_suffix(args)}_{safe_name(args.selector_select_metric)}.json"


def prior_cmd(args) -> List[str]:
    return [
        py(),
        "calibrate_prior_ensemble.py",
        "--input_json",
        fusion_output_path(args),
        "--label_map",
        train_label_map(args),
        "--methods",
        "blend",
        "--strengths",
        args.prior_strengths,
        "--gate_modes",
        args.prior_gate_modes,
        "--gate_thresholds",
        args.prior_gate_thresholds,
        "--pool_strategy",
        args.prior_pool_strategy,
        "--top_k",
        str(args.prior_top_k),
        "--hard_prior_kl_cap",
        str(args.prior_hard_kl_cap),
        "--ensemble_mode",
        args.prior_ensemble_mode,
        "--include_identity_candidate",
        "--output_json",
        prior_candidate_output_path(args),
    ]


def safe_prior_residual_cmd(args) -> List[str]:
    return [
        py(),
        "fuse_prediction_jsons.py",
        "--input",
        "base",
        fusion_output_path(args),
        "--input",
        "prior",
        prior_candidate_output_path(args),
        "--label_map",
        train_label_map(args),
        "--simplex_step",
        str(args.safe_prior_simplex_step),
        "--select_metric",
        "accuracy",
        "--min_weight",
        "base",
        str(args.safe_prior_min_base_weight),
        "--output_json",
        safe_prior_output_path(args),
    ]


def selector_cmd(args) -> List[str]:
    cmd = [
        py(),
        "validation_gated_selector.py",
        "--input",
        "base",
        fusion_output_path(args),
        "--input",
        "prior",
        safe_prior_output_path(args),
        "--input",
        "stacker",
        stacker_output_path(args),
        "--label_map",
        train_label_map(args),
        "--select_metric",
        args.selector_select_metric,
        "--rank_select_metric",
        args.selector_rank_select_metric or args.selector_select_metric,
        "--rank_metric",
        args.selector_rank_metric,
        "--strategies",
        args.selector_strategies,
        "--base_input",
        args.selector_base_input,
        "--unified_expert_slots",
        args.selector_unified_expert_slots,
        "--min_valid_gain_over_base",
        str(args.selector_min_valid_gain_over_base),
        "--bootstrap_samples",
        str(args.selector_bootstrap_samples),
        "--rank_bootstrap_samples",
        str(args.selector_rank_bootstrap_samples),
        "--rank_candidate_limit",
        str(args.selector_rank_candidate_limit),
        "--bootstrap_min_win_rate",
        str(args.selector_bootstrap_min_win_rate),
        "--bootstrap_min_gain_quantile",
        str(args.selector_bootstrap_min_gain_quantile),
        "--max_prediction_change_rate",
        str(args.selector_max_prediction_change_rate),
        "--max_prediction_js_divergence",
        str(args.selector_max_prediction_js_divergence),
        "--calibration_penalty_weight",
        str(args.selector_calibration_penalty_weight),
        "--output_json",
        selector_output_path(args),
    ]
    return cmd


def selected_model_types(args) -> List[str]:
    out: List[str] = []
    for raw in args.model_types.split(","):
        model_type = raw.strip()
        if not model_type:
            continue
        if model_type not in {"graph", "seq"}:
            raise ValueError(f"Unknown model_type={model_type}; use graph,seq")
        if model_type not in out:
            out.append(model_type)
    return out or ["graph", "seq"]


def packet_index_policy(path: Path) -> str:
    if not path.exists():
        return "missing"
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                return str(json.loads(line).get("embedding_header_policy") or "unknown")
    return "empty"


def packet_index_context_policy(path: Path) -> str:
    if not path.exists():
        return "missing"
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                return str(
                    json.loads(line).get("packet_context_policy") or "unknown"
                )
    return "empty"


def embedding_policy_evidence(
    args,
    *,
    embedding_suffix: str | None = None,
    expected_policy: str | None = None,
) -> dict:
    splits = {}
    expected = str(expected_policy or args.embedding_header_policy)
    expected_context = (
        "flow_context"
        if args.packet_context_policy == "auto"
        else str(args.packet_context_policy)
    )
    expected_mode = str(getattr(args, "embedding_mode", "concat"))
    expected_batch_size = int(getattr(args, "embedding_batch_size", 8))
    expected_flow_batch_packets = int(
        getattr(args, "embedding_flow_batch_packets", 128)
    )
    expected_scheduler = (
        "cross_flow_length_bucketed_v1"
        if expected_flow_batch_packets > 0
        else "legacy_per_flow_v1"
    )
    suffix = str(embedding_suffix or args.embedding_suffix)
    for split in selected_splits(args.splits):
        config_path = (
            Path("reasoningDataset")
            / args.dataset
            / f"{split}_embeddings_{suffix}"
            / "embedding_config.json"
        )
        audit = audit_report_evidence(config_path.parent / "embedding_audit.json")
        actual = "missing"
        actual_context = "missing"
        actual_mode = "missing"
        actual_scheduler = "missing"
        actual_batch_size = None
        actual_flow_batch_packets = None
        actual_num_shards = 1
        actual_merge_scheduler = "none"
        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                    actual = str(config.get("packet_index_header_policy") or "unknown")
                    actual_context = str(
                        config.get("packet_index_context_policy") or "unknown"
                    )
                    actual_mode = str(config.get("embedding_mode") or "unknown")
                    actual_scheduler = str(config.get("scheduler") or "unknown")
                    actual_batch_size = config.get("batch_size")
                    actual_flow_batch_packets = config.get("flow_batch_packets")
                    actual_num_shards = int(config.get("num_shards", 1) or 1)
                    actual_merge_scheduler = str(
                        config.get("merge_scheduler") or "none"
                    )
            except (OSError, ValueError, TypeError):
                actual = "invalid"
                actual_context = "invalid"
                actual_mode = "invalid"
                actual_scheduler = "invalid"
                actual_num_shards = 0
                actual_merge_scheduler = "invalid"
        shard_execution_verified = bool(
            actual_num_shards == 1
            or actual_merge_scheduler == "deterministic_flow_sha1_v1"
        )
        splits[split] = {
            "config_path": str(config_path),
            "expected": expected,
            "actual": actual,
            "expected_context": expected_context,
            "actual_context": actual_context,
            "expected_mode": expected_mode,
            "actual_mode": actual_mode,
            "expected_scheduler": expected_scheduler,
            "actual_scheduler": actual_scheduler,
            "expected_batch_size": expected_batch_size,
            "actual_batch_size": actual_batch_size,
            "expected_flow_batch_packets": expected_flow_batch_packets,
            "actual_flow_batch_packets": actual_flow_batch_packets,
            "actual_num_shards": actual_num_shards,
            "actual_merge_scheduler": actual_merge_scheduler,
            "shard_execution_verified": shard_execution_verified,
            **audit,
            "header_verified": actual == expected,
            "context_verified": actual_context == expected_context,
            "verified": bool(
                actual == expected
                and actual_context == expected_context
                and actual_mode == expected_mode
                and actual_scheduler == expected_scheduler
                and actual_batch_size == expected_batch_size
                and actual_flow_batch_packets == expected_flow_batch_packets
                and shard_execution_verified
                and audit["audit_verified"]
            ),
        }
    return {
        "expected": expected,
        "expected_context": expected_context,
        "expected_mode": expected_mode,
        "expected_scheduler": expected_scheduler,
        "expected_batch_size": expected_batch_size,
        "expected_flow_batch_packets": expected_flow_batch_packets,
        "embedding_audits_verified": bool(splits)
        and all(row["audit_verified"] for row in splits.values()),
        "verified": bool(splits) and all(row["verified"] for row in splits.values()),
        "splits": splits,
    }


def intervention_embedding_suffix(args) -> str:
    return f"{args.embedding_suffix}_mask_ip_port_intervention"


def intervention_policy_evidence(args) -> dict | None:
    if not getattr(args, "use_intervention_views", False):
        return None
    return embedding_policy_evidence(
        args,
        embedding_suffix=intervention_embedding_suffix(args),
        expected_policy=args.intervened_embedding_header_policy,
    )


def field_aware_policy_status(
    args,
    factual_evidence: dict,
    intervention_evidence: dict | None,
) -> str:
    if args.framework_profile == "paper_unified":
        if not factual_evidence["verified"] or not intervention_evidence or not intervention_evidence["verified"]:
            return "unverified"
        return (
            f"factual_{factual_evidence['expected']}_plus_"
            f"{intervention_evidence['expected']}_intervention"
        )
    return factual_evidence["expected"] if factual_evidence["verified"] else "unverified"


def require_paper_unified_inputs(args) -> None:
    missing: list[str] = []
    label_map = Path(train_label_map(args))
    if not label_map.exists():
        missing.append(str(label_map))
    for split in selected_splits(args.splits):
        index_path = Path(embedding_output_dir(args, split)) / "flow_embedding_index.jsonl"
        if not index_path.exists():
            missing.append(str(index_path))
    if missing:
        joined = "\n  ".join(missing)
        raise SystemExit(
            "paper_unified flow stage starts after Tower-1 semantic embedding extraction. "
            "Missing required inputs:\n  "
            + joined
            + "\nRun the explicit tower1_preprocess/tower1_train/embeddings stages first, "
            "or pass an embedding_suffix that already contains train/valid/test embeddings."
        )
    policy = embedding_policy_evidence(args)
    if not policy["verified"]:
        details = "\n  ".join(
            f"{split}: expected={row['expected']} actual={row['actual']} config={row['config_path']}"
            for split, row in policy["splits"].items()
            if not row["verified"]
        )
        raise SystemExit(
            "paper_unified requires embedding provenance that matches the shared "
            f"header policy {args.embedding_header_policy!r}. Mismatches:\n  {details}\n"
            "Rerun tower1_preprocess and embeddings with the paper_unified profile; "
            "legacy embedding configs without policy evidence cannot be promoted."
        )
    intervention_policy = intervention_policy_evidence(args)
    if intervention_policy is not None and not intervention_policy["verified"]:
        details = "\n  ".join(
            f"{split}: expected={row['expected']} actual={row['actual']} config={row['config_path']}"
            for split, row in intervention_policy["splits"].items()
            if not row["verified"]
        )
        raise SystemExit(
            "paper_unified requires aligned intervention embedding provenance. "
            "Mismatches:\n  "
            + details
        )


def write_manifest(args, completed: bool = False) -> None:
    root = Path("reasoningDataset") / args.dataset
    root.mkdir(parents=True, exist_ok=True)
    source_root = Path(__file__).resolve().parent
    if not hasattr(args, "_algorithm_source_launch"):
        args._algorithm_source_launch = unified_method_source_snapshot(source_root)
    if completed:
        source_evidence = complete_source_stability(
            args._algorithm_source_launch, source_root
        )
    else:
        source_evidence = {
            "schema": "algorithm_source_stability_evidence_v1",
            "status": "running",
            "scope": args._algorithm_source_launch["scope"],
            "launch_fingerprint": args._algorithm_source_launch["fingerprint"],
            "completion_fingerprint": None,
            "num_launch_files": args._algorithm_source_launch["num_files"],
            "num_completion_files": 0,
            "changed_paths": [],
            "launch_snapshot": args._algorithm_source_launch,
        }
    effective_completed = completed and source_evidence["status"] == "pass"
    manifest_path = root / f"stage8_flowaware_manifest_{result_suffix(args)}.json"
    existing_payload = None
    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                existing_payload = json.load(f)
        except Exception:
            existing_payload = None
    if args.stage == "paper_unified":
        requested_paper_unified_stages = selected_paper_unified_stages(
            args.paper_unified_stages
        )
    elif args.stage == "all":
        requested_paper_unified_stages = selected_paper_unified_stages(
            args.paper_unified_stages
        )
    elif args.stage in PAPER_UNIFIED_POST_STAGES:
        requested_paper_unified_stages = [args.stage]
    else:
        requested_paper_unified_stages = []
    completed_paper_unified_stages = merge_completed_paper_unified_stages(
        existing_payload,
        requested_paper_unified_stages,
        completed
        and (
            args.stage == "paper_unified"
            or args.stage == "all"
            or args.stage in PAPER_UNIFIED_POST_STAGES
        ),
    )
    payload = vars(args).copy()
    payload.pop("_algorithm_source_launch", None)
    payload["splits"] = selected_splits(args.splits)
    payload["cuda_available_at_launch"] = cuda_available()
    policy_evidence = embedding_policy_evidence(args)
    intervention_evidence = intervention_policy_evidence(args)
    shared_status = profile_shared_status(args.framework_profile)
    shared_status.update({
        "field_aware_header_intervention": (
            field_aware_policy_status(
                args, policy_evidence, intervention_evidence
            )
        ),
        "semantic_tower1_channel": "present",
        "label_free_protocol_content_pretraining": "native_flow_multitask_v1",
        "current_packet_structural_encoder": "strict_current_packet_13d",
        "bounded_tri_channel_router": "semantic_anchor_residual_0_25",
        "content_group_empirical_risk": args.content_group_loss_reduction,
        "fixed_cross_fold_consensus": "equal_log_mean",
    })
    payload["framework"] = build_framework_manifest(
        task="flow-level",
        dataset=args.dataset,
        input_unit="flow_packet_sequence",
        stage=args.stage,
        shared_module_status=shared_status,
        task_module_status={
            "packet_to_window_flow_aggregator": args.flow_pooling,
            "flow_level_classifier": ",".join(selected_model_types(args)),
        },
        notes={
            "framework_profile": args.framework_profile,
            "fold": args.fold,
            "profile_overrides": getattr(args, "framework_profile_overrides", {}),
            "shared_core_config": args.shared_core_config,
            "shared_core_method_sha256": getattr(
                args, "shared_core_method_sha256", ""
            ),
            "shared_core_config_sha256": getattr(args, "shared_core_config_sha256", ""),
            "shared_core_overrides": getattr(args, "shared_core_overrides", {}),
            "algorithm_source_evidence": source_evidence,
            "window_size": args.window_size,
            "stride": args.stride,
            "model_types": selected_model_types(args),
            "paper_unified_stages": requested_paper_unified_stages,
            "completed_paper_unified_stages": completed_paper_unified_stages,
            "result_paths": [
                *(
                    [
                        model_prob_path(args, model_type, split)
                        for model_type in selected_model_types(args)
                        for split in selected_eval_splits(args.eval_splits)
                    ]
                    if "eval" in requested_paper_unified_stages
                    else []
                ),
                *([fusion_output_path(args)] if "fusion" in requested_paper_unified_stages else []),
                *([stacker_output_path(args)] if "stacker" in requested_paper_unified_stages else []),
                *(
                    [prior_candidate_output_path(args), safe_prior_output_path(args)]
                    if "prior" in requested_paper_unified_stages else []
                ),
                *([selector_output_path(args)] if "selector" in requested_paper_unified_stages else []),
            ],
            "paper_main_experts": [],
            "semantic_fusion_level": "representation",
            "paired_embedding_suffix": args.paired_embedding_suffix,
            "tower2_data_suffix": tower2_data_suffix(args),
            "native_structural_suffix": args.native_structural_suffix,
            "native_checkpoint": native_checkpoint_path(args) if args.native_structural_suffix else "",
            "tower2_checkpoints": {
                model_type: (
                    f"checkpoints/tower2_{model_type}_flow_"
                    f"{safe_name(args.dataset)}_{result_suffix(args)}/best.pt"
                )
                for model_type in selected_model_types(args)
            },
            "tower1_training_contract": tower1_training_contract(
                args, "flow-level"
            ),
            "tower1_execution_evidence": tower1_execution_evidence(
                args.tower1_output_dir,
                tower1_training_contract(args, "flow-level"),
            ),
            "packet_module_training_source": "flow_task_train_split_packets",
            "semantic_packet_context_policy": args.packet_context_policy,
            "cross_task_trained_weights_reused": False,
            "train_ablate_input_channel": args.train_ablate_input_channel,
            "train_ablate_intervention_view": args.train_ablate_intervention_view,
            "train_fixed_channel_fusion": args.train_fixed_channel_fusion,
            "train_row_risk_ablation": args.train_row_risk_ablation,
            "packet_evidence_max_weight": args.packet_evidence_max_weight,
            "packet_evidence_ablation_control": args.packet_evidence_ablation_control,
            "packet_evidence_training_label_source": (
                "flow_train_split_labels_broadcast_to_member_packets"
                if args.packet_evidence_max_weight > 0
                else "disabled"
            ),
            "native_structural_dim": effective_native_structural_dim(args),
            "tower2_select_metric": args.tower2_select_metric,
            "content_group_guard": args.content_group_guard,
            "split_group_key": args.split_group_key,
            "content_group_unique_batches": args.content_group_unique_batches,
            "content_group_loss_reduction": args.content_group_loss_reduction,
            "meta_dropout_prob": args.meta_dropout_prob,
            "edge_attr_dropout_prob": args.edge_attr_dropout_prob,
            "embedding_dropout_prob": args.embedding_dropout_prob,
            "window_dropout_prob": args.window_dropout_prob,
            "embedding_header_policy_evidence": policy_evidence,
            "intervention_header_policy_evidence": intervention_evidence,
            "dry_run": args.dry_run,
            "completed": effective_completed,
        },
    )
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    if completed and source_evidence["status"] != "pass":
        raise RuntimeError(
            "executable Python sources changed during flow pipeline execution: "
            + ", ".join(source_evidence["changed_paths"])
        )


def commands(args) -> Iterable[List[str]]:
    splits = selected_splits(args.splits)
    if args.stage in {"tower1_preprocess", "all"}:
        for split in splits:
            yield tower1_preprocess_cmd(args, split)
    if args.stage in {"tower1_train", "all"}:
        yield tower1_train_cmd(args)
    if args.stage in {"native_pretrain", "all"} and args.native_structural_suffix and not args.native_checkpoint:
        yield native_pretrain_cmd(args)
    if args.stage in {"native_embeddings", "all"} and args.native_structural_suffix:
        for split in splits:
            yield native_extract_cmd(args, split)
    if args.stage in {"tower2_preprocess", "all"}:
        for split in splits:
            yield tower2_preprocess_cmd(args, split)
    if args.stage in {"tower2_train", "all"}:
        for model_type in selected_model_types(args):
            yield tower2_train_cmd(args, model_type)
    if args.stage in {"eval", "all"}:
        for model_type in selected_model_types(args):
            for split in selected_eval_splits(args.eval_splits):
                yield tower2_eval_cmd(args, model_type, split)
    if args.stage in {"fusion", "stacker", "all"}:
        for model_type in selected_model_types(args):
            yield make_fusion_payload_cmd(args, model_type)
    if args.stage in {"fusion", "all"}:
        yield fusion_cmd(args)
    if args.stage in {"stacker", "all"}:
        yield stacker_cmd(args)
    if args.stage in {"prior", "all"}:
        yield prior_cmd(args)
        yield safe_prior_residual_cmd(args)
    if args.stage in {"selector", "all"}:
        yield selector_cmd(args)


def run_post_embedding_pipeline(args) -> None:
    if not args.dry_run:
        require_paper_unified_inputs(args)
    if args.native_structural_suffix:
        native_stages = ["native_embeddings"] if args.native_checkpoint else ["native_pretrain", "native_embeddings"]
        for stage in native_stages:
            for cmd in commands(argparse.Namespace(**{**vars(args), "stage": stage})):
                run(cmd, dry_run=args.dry_run)
    for stage in selected_paper_unified_stages(args.paper_unified_stages):
        for cmd in commands(argparse.Namespace(**{**vars(args), "stage": stage})):
            run(cmd, dry_run=args.dry_run)


def run_post_tower1_pipeline(
    args,
    intervention_args,
    *,
    intervention_preprocessed: bool = False,
) -> None:
    """Resume the unified pipeline from an already trained Tower-1 checkpoint."""
    for split in selected_splits(args.splits):
        run_embedding_stage(args, split)
    if intervention_args is None:
        run_post_embedding_pipeline(args)
        return

    if not intervention_preprocessed:
        for split in selected_splits(args.splits):
            run(tower1_preprocess_cmd(intervention_args, split), dry_run=args.dry_run)
    for split in selected_splits(args.splits):
        run_embedding_stage(intervention_args, split)

    if args.native_structural_suffix:
        native_stages = (
            ["native_embeddings"]
            if args.native_checkpoint
            else ["native_pretrain", "native_embeddings"]
        )
        for native_stage in native_stages:
            native_args = argparse.Namespace(**{**vars(args), "stage": native_stage})
            for cmd in commands(native_args):
                run(cmd, dry_run=args.dry_run)

    for split in selected_splits(args.splits):
        run(tower2_preprocess_cmd(args, split), dry_run=args.dry_run)
        run(tower2_preprocess_cmd(intervention_args, split), dry_run=args.dry_run)
    for stage in selected_paper_unified_stages(args.paper_unified_stages):
        if stage == "tower2_preprocess":
            continue
        stage_args = argparse.Namespace(**{**vars(args), "stage": stage})
        for cmd in commands(stage_args):
            run(cmd, dry_run=args.dry_run)


def main() -> None:
    ap = argparse.ArgumentParser()
    default_tower1_output_dir = "checkpoints/tower1_qwen_multitask_flowaware_change_weight"
    ap.add_argument("--dataset", default="vpn-app")
    ap.add_argument("--fold", type=int, default=0, help="SWEET train/valid fold recorded in paper provenance and used by default split paths.")
    ap.add_argument("--num_classes", type=int, default=16)
    ap.add_argument(
        "--stage",
        choices=[
            "tower1_train",
            "tower1_preprocess",
            "embeddings",
            "native_pretrain",
            "native_embeddings",
            "tower2_preprocess",
            "tower2_train",
            "eval",
            "fusion",
            "stacker",
            "prior",
            "selector",
            "paper_unified",
            "post_tower1",
            "all",
        ],
        required=True,
    )
    ap.add_argument("--splits", default="train,valid,test")
    ap.add_argument(
        "--eval_splits",
        default="valid,test",
        help="Comma-separated Tower-2 evaluation splits; use 'valid' for candidate selection.",
    )
    ap.add_argument("--train_dir", default="")
    ap.add_argument("--valid_dir", default="")
    ap.add_argument("--test_dir", default="")
    ap.add_argument("--source_suffix", default="change_weight")
    ap.add_argument("--output_suffix", default="flowaware_change_weight")
    ap.add_argument("--tower1_data_suffix", default="", help="Tower-1 train/eval label-map suffix. Defaults to --source_suffix after dataset-specific normalization.")
    ap.add_argument("--embedding_suffix", default="rawproj_flowaware_change_weight")
    ap.add_argument("--tower2_suffix", default="", help="Optional Tower-2 dataset suffix. Defaults to --embedding_suffix, or --embedding_suffix_native_<native_structural_suffix> when native structural embeddings are attached.")
    ap.add_argument("--native_structural_suffix", default="", help="Enable native structural embeddings and store them under <split>_native_structural_<suffix>.")
    ap.add_argument("--native_checkpoint", default="", help="Existing native-flow encoder checkpoint. Defaults to <native_pretrain_output_dir>/best.pt.")
    ap.add_argument("--native_pretrain_output_dir", default="", help="Output directory for --stage native_pretrain.")
    ap.add_argument("--native_max_packets", type=int, default=64)
    ap.add_argument("--native_max_bytes", type=int, default=128)
    ap.add_argument("--native_epochs", type=int, default=20)
    ap.add_argument("--native_batch_size", type=int, default=8)
    ap.add_argument("--native_eval_batch_size", type=int, default=16)
    ap.add_argument("--native_num_workers", type=int, default=0)
    ap.add_argument("--native_learning_rate", type=float, default=3e-4)
    ap.add_argument("--native_weight_decay", type=float, default=1e-2)
    ap.add_argument("--native_extract_batch_size", type=int, default=16)
    ap.add_argument("--native_hidden_dim", type=int, default=128)
    ap.add_argument("--native_projection_dim", type=int, default=128)
    ap.add_argument("--native_byte_layers", type=int, default=2)
    ap.add_argument("--native_flow_layers", type=int, default=2)
    ap.add_argument("--native_num_heads", type=int, default=4)
    ap.add_argument("--native_dropout", type=float, default=0.1)
    ap.add_argument("--native_field_mask_probability", type=float, default=0.2)
    ap.add_argument("--native_payload_dropout_probability", type=float, default=0.5)
    ap.add_argument("--native_session_mask_probability", type=float, default=1.0)
    ap.add_argument("--native_masked_byte_weight", type=float, default=1.0)
    ap.add_argument("--native_relative_order_weight", type=float, default=0.25)
    ap.add_argument("--native_same_flow_weight", type=float, default=0.25)
    ap.add_argument("--native_next_length_weight", type=float, default=0.2)
    ap.add_argument("--native_next_iat_weight", type=float, default=0.2)
    ap.add_argument("--native_direction_weight", type=float, default=0.1)
    ap.add_argument("--native_flow_contrastive_weight", type=float, default=0.25)
    ap.add_argument("--native_packet_consistency_weight", type=float, default=0.25)
    ap.add_argument("--native_temperature", type=float, default=0.1)
    ap.add_argument("--native_patience", type=int, default=4)
    ap.add_argument(
        "--native_structural_dim",
        type=int,
        default=0,
        help=(
            "Leading native structural dimensions inside Tower-2's structural "
            "channel. Default 0 treats extracted native embeddings as ordinary "
            "concatenated input features; pass a positive value only for explicit "
            "native-adapter ablations."
        ),
    )
    ap.add_argument("--native_device", default="cuda" if cuda_available() else "cpu")
    ap.add_argument("--dual_channel_mode", choices=["concat", "residual"], default="concat")
    ap.add_argument("--dual_channel_gate_mode", choices=["global", "adaptive"], default="global")
    ap.add_argument("--channel_fusion_base_mode", choices=["legacy", "semantic_anchor"], default="legacy")
    ap.add_argument("--dual_channel_max_weight", type=float, default=0.25)
    ap.add_argument("--run_tag", default="", help="Optional suffix appended to Stage-8 training/eval/fusion outputs, useful for ablations.")
    ap.add_argument("--tower1_output_dir", default=default_tower1_output_dir)
    ap.add_argument("--tower1_device", default="cuda", help="Explicit device for Tower-1 LoRA training, e.g. cuda:5 on a shared GPU host.")
    ap.add_argument("--base_model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--max_packets_per_flow", type=int, default=64)
    ap.add_argument("--payload_prefix_len", type=int, default=128)
    ap.add_argument("--l3_prefix_len", type=int, default=512)
    ap.add_argument("--preprocess_max_flows", type=int, default=0)
    ap.add_argument("--embedding_header_policy", choices=["full", "randomize_ip_port", "mask_ip_port"], default="full", help="Header policy for packet-index prompts used by embedding extraction.")
    ap.add_argument(
        "--packet_context_policy",
        choices=["auto", "single_packet", "flow_context"],
        default="auto",
        help="Context available to each Tower1 semantic packet prompt.",
    )
    ap.add_argument("--intervened_embedding_header_policy", choices=["randomize_ip_port", "mask_ip_port"], default="mask_ip_port", help="Second semantic prompt view used by the shared intervention router.")
    ap.add_argument("--tower1_epochs", type=int, default=2)
    ap.add_argument("--tower1_use_sft", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--tower1_max_steps", type=int, default=0)
    ap.add_argument("--tower1_save_steps", type=int, default=0)
    ap.add_argument("--tower1_init_checkpoint_dir", default="")
    ap.add_argument("--sft_batch_size", type=int, default=2)
    ap.add_argument("--packet_batch_size", type=int, default=16)
    ap.add_argument("--tower1_valid_batch_size", type=int, default=64)
    ap.add_argument("--tower1_valid_packets_per_flow", type=int, default=2)
    ap.add_argument("--tower1_early_stop_patience", type=int, default=0)
    ap.add_argument("--tower1_gradient_accumulation_steps", type=int, default=1)
    ap.add_argument("--max_sft_length", type=int, default=1792)
    ap.add_argument("--max_packet_length", type=int, default=1024)
    ap.add_argument("--cls_weight", type=float, default=0.1)
    ap.add_argument("--contrastive_weight", type=float, default=0.3)
    ap.add_argument("--temperature", type=float, default=0.07)
    ap.add_argument("--same_flow_positive_weight", type=float, default=2.0)
    ap.add_argument("--same_label_positive_weight", type=float, default=1.0)
    ap.add_argument("--flow_proto_weight", type=float, default=0.0)
    ap.add_argument("--flow_proto_positive", choices=["own_flow", "same_class"], default="same_class")
    ap.add_argument(
        "--flow_proto_context",
        choices=["inclusive", "leave_one_out"],
        default="inclusive",
    )
    ap.add_argument("--tower1_paired_data_suffix", default="", help="Optional Tower-1 second-view data suffix for paired packet consistency.")
    ap.add_argument("--tower1_paired_consistency_weight", type=float, default=0.0, help="Tower-1 full-header vs paired-view consistency weight.")
    ap.add_argument("--tower1_paired_cls_weight", type=float, default=0.0, help="Extra paired-view packet CE multiplier in Tower-1.")
    ap.add_argument("--tower1_paired_logit_kl_weight", type=float, default=0.5, help="Logit symmetric-KL weight inside Tower-1 paired consistency.")
    ap.add_argument("--tower1_paired_raw_consistency_weight", type=float, default=1.0, help="Raw last-token cosine term inside Tower-1 paired consistency.")
    ap.add_argument("--flow_balanced_packet_batches", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--packets_per_flow", type=int, default=2)
    ap.add_argument(
        "--packet_batch_scheduler",
        choices=["epoch_resampled_dataloader_v1", "coverage_cycle_dataloader_v1"],
        default="epoch_resampled_dataloader_v1",
    )
    ap.add_argument("--tower1_lr", type=float, default=2e-5)
    ap.add_argument("--tower1_head_lr", type=float, default=1e-4)
    ap.add_argument("--tower1_class_weighting", choices=["none", "inverse", "effective"], default="none")
    ap.add_argument("--tower1_class_weight_basis", choices=["packet", "flow"], default="packet")
    ap.add_argument("--tower1_class_weight_strength", type=float, default=1.0)
    ap.add_argument("--tower1_disable_packet_information_weights", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    ap.add_argument("--local_files_only", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--gradient_checkpointing", action="store_true")
    ap.add_argument("--embedding_mode", choices=["raw", "projected", "concat"], default="concat")
    ap.add_argument("--embedding_batch_size", type=int, default=8)
    ap.add_argument("--embedding_flow_batch_packets", type=int, default=128)
    ap.add_argument("--embedding_max_length", type=int, default=1024)
    ap.add_argument("--embedding_device", default="auto", help="Device for unsharded embedding extraction.")
    ap.add_argument("--embedding_num_shards", type=int, default=1, help="Run embedding extraction as N deterministic flow-id shards and merge the shard indexes.")
    ap.add_argument("--embedding_cuda_devices", default="", help="Comma-separated CUDA device ids assigned round-robin to embedding shards, e.g. 0,1,2,3.")
    ap.add_argument("--embedding_resume_shards", action=argparse.BooleanOptionalAction, default=True, help="Skip embedding shards whose index row count matches the expected flow count.")
    ap.add_argument("--embedding_resume_existing", action=argparse.BooleanOptionalAction, default=True, help="Resume interrupted embedding extraction by skipping flows with existing .npy files.")
    ap.add_argument("--window_size", type=int, default=32)
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--model_types", default="graph,seq")
    ap.add_argument(
        "--exact_shared_packet_encoder",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Reuse the exact packet representation module before the seq-only "
            "flow aggregation boundary. Graph remains a structural ablation."
        ),
    )
    ap.add_argument(
        "--shared_packet_hidden_dim",
        type=int,
        default=128,
        help="Width of the exact packet representation shared with packet-level training.",
    )
    ap.add_argument(
        "--packet_evidence_max_weight",
        type=float,
        default=0.0,
        help=(
            "Flow-only upper bound for the learned reuse of the packet classifier; "
            "0 preserves exact shared-core v2."
        ),
    )
    ap.add_argument(
        "--train_ablate_input_channel",
        choices=["none", "semantic", "content", "structural"],
        default="none",
        help="Retrain Flow with one channel removed from the exact shared packet core.",
    )
    ap.add_argument(
        "--train_ablate_intervention_view",
        choices=["none", "factual_only", "intervened_only"],
        default="none",
        help="Retrain Flow with only one shared intervention view.",
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
    ap.add_argument(
        "--packet_evidence_ablation_control",
        action="store_true",
        help=(
            "Validation-only matched control: keep late_fusion while disabling "
            "the Packet evidence head."
        ),
    )
    ap.add_argument("--tower2_epochs", type=int, default=30)
    ap.add_argument("--tower2_early_stop_patience", type=int, default=8)
    ap.add_argument("--tower2_batch_size", type=int, default=16)
    ap.add_argument(
        "--tower2_select_metric",
        choices=[
            "accuracy",
            "acc",
            "macro_f1",
            "flow_acc",
            "flow_macro_f1",
            "window_acc",
            "window_macro_f1",
            "content_group_acc",
            "content_group_accuracy",
            "content_group_macro_f1",
            "flow_content_group_acc",
            "flow_content_group_macro_f1",
        ],
        default="flow_macro_f1",
        help="Validation metric used by train_tower2.py to save best.pt.",
    )
    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--num_heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.15)
    ap.add_argument("--tower2_lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.03)
    ap.add_argument("--flow_pooling", default="mean", choices=["mean", "attention", "late_fusion", "transformer", "multi_view"])
    ap.add_argument("--multi_view_gate_entropy_weight", type=float, default=0.0, help="Entropy-minimization weight for trainable multi-view flow pooling gates.")
    ap.add_argument("--flow_stat_expert_weight", type=float, default=0.0, help="Fuse the trainable flow metadata/statistics expert into Tower-2.")
    ap.add_argument("--flow_stat_aux_weight", type=float, default=0.0, help="Auxiliary CE weight for the trainable flow metadata/statistics expert.")
    ap.add_argument("--window_loss_weight", type=float, default=0.3)
    ap.add_argument("--class_weight_strength", type=float, default=0.6)
    ap.add_argument("--label_smoothing", type=float, default=0.05)
    ap.add_argument("--confidence_penalty_weight", type=float, default=0.0, help="KL-to-uniform confidence penalty for Tower-2 classification logits.")
    ap.add_argument("--hierarchical_weight", type=float, default=0.2)
    ap.add_argument("--hierarchical_logit_weight", type=float, default=0.5)
    ap.add_argument("--coarse_groups", default="vpn_app")
    ap.add_argument("--samples_per_class", type=int, default=2)
    ap.add_argument(
        "--content_group_guard",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Attach exact-PCAP SHA256 content_group_id metadata during Tower-2 preprocessing.",
    )
    ap.add_argument(
        "--split_group_key",
        choices=["flow_id", "content_group_id"],
        default="flow_id",
        help="Group key for train_tower2.py internal validation splits.",
    )
    ap.add_argument(
        "--content_group_unique_batches",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Prefer at most one flow per exact-PCAP content group in balanced Tower-2 batches.",
    )
    ap.add_argument(
        "--content_group_loss_reduction",
        choices=["none", "group_mean"],
        default="none",
        help="Average Tower-2 flow CE over exact-PCAP content groups before the batch mean.",
    )
    ap.add_argument("--contrastive_mode", choices=["standard", "confusion", "confusion_weighted"], default="confusion")
    ap.add_argument("--confusion_groups", default="vpn_app")
    ap.add_argument("--flow_contrastive_weight", type=float, default=0.03)
    ap.add_argument("--flow_temperature", type=float, default=0.07)
    ap.add_argument("--window_contrastive_weight", type=float, default=0.0)
    ap.add_argument("--window_contrastive_temperature", type=float, default=0.07)
    ap.add_argument("--window_contrastive_positive", choices=["own_flow", "same_class"], default="same_class")
    ap.add_argument("--consistency_weight", type=float, default=0.0, help="KL consistency between clean and augmented Tower-2 flow logits.")
    ap.add_argument("--meta_dropout_prob", type=float, default=0.0, help="Training-only dropout on trailing Tower-2 metadata features.")
    ap.add_argument("--meta_feature_dim", type=int, default=13, help="Number of strict current-packet metadata features in Tower-2 x.")
    ap.add_argument("--embedding_dropout_prob", type=float, default=0.0, help="Training-only dropout on packet embedding features.")
    ap.add_argument("--window_dropout_prob", type=float, default=0.0, help="Training-only random dropping of windows before flow aggregation.")
    ap.add_argument("--edge_attr_dropout_prob", type=float, default=0.0, help="Training-only dropout on graph edge attributes.")
    ap.add_argument("--paired_embedding_suffix", default="", help="Optional second-view Tower-2 dataset suffix aligned by flow_id.")
    ap.add_argument("--use_intervention_views", action=argparse.BooleanOptionalAction, default=False, help="Use the shared representation-level factual/intervened router instead of separate flow predictions.")
    ap.add_argument("--intervention_max_residual_weight", type=float, default=0.25)
    ap.add_argument(
        "--intervention_view_base_mode",
        choices=["symmetric_mean", "factual_anchor"],
        default="symmetric_mean",
    )
    ap.add_argument("--paired_view_weight", type=float, default=0.0, help="Flow CE weight for the paired view.")
    ap.add_argument("--paired_consistency_weight", type=float, default=0.0, help="Symmetric KL weight between primary and paired-view flow logits.")
    ap.add_argument("--paired_alignment_weight", type=float, default=0.0, help="Cosine alignment weight for clean/randomized-header flow embeddings.")
    ap.add_argument("--paired_crossview_contrastive_weight", type=float, default=0.0, help="Cross-view supervised contrastive weight for endpoint-invariant flow semantics.")
    ap.add_argument("--paired_crossview_temperature", type=float, default=0.07)
    ap.add_argument("--paired_variance_weight", type=float, default=0.0, help="Anti-collapse variance weight for paired flow embeddings.")
    ap.add_argument("--paired_variance_target", type=float, default=0.04)
    ap.add_argument("--paired_covariance_weight", type=float, default=0.0, help="Off-diagonal covariance penalty for paired flow embeddings.")
    ap.add_argument("--view_domain_adversarial_weight", type=float, default=0.0, help="Adversarially remove primary-vs-paired view information from Tower-2 flow embeddings.")
    ap.add_argument("--domain_adversarial_lambda", type=float, default=1.0, help="Gradient reversal strength for Tower-2 view-domain adversarial training.")
    ap.add_argument("--environment_map_json", default="", help="Optional flow environment map generated by build_flow_environment_map.py.")
    ap.add_argument("--environment_risk_weight", type=float, default=0.0, help="Variance penalty for Tower-2 source-environment classification risks.")
    ap.add_argument("--environment_alignment_weight", type=float, default=0.0, help="Class-conditional Tower-2 embedding alignment across source environments.")
    ap.add_argument("--distill_targets_json", default="", help="Optional flow-id soft teacher JSON for consensus-guided Tower-2 student training.")
    ap.add_argument("--distill_weight", type=float, default=0.0, help="KL weight for flow-id consensus distillation.")
    ap.add_argument("--distill_class_prior_weight", type=float, default=0.0, help="KL weight for class-conditional teacher priors.")
    ap.add_argument("--distill_temperature", type=float, default=2.0)
    ap.add_argument("--distill_min_confidence", type=float, default=0.0)
    ap.add_argument("--distill_max_confidence", type=float, default=0.0)
    ap.add_argument("--distill_confidence_power", type=float, default=0.0)
    ap.add_argument("--distill_min_teachers_per_flow", type=int, default=1)
    ap.add_argument("--distill_require_oof_exclusion_proof", action="store_true")
    ap.add_argument("--distill_min_coverage", type=float, default=0.0)
    ap.add_argument("--distill_low_coverage_action", choices=["warn", "disable_flow", "fail"], default="warn")
    ap.add_argument("--prior_strengths", default="0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0,1.05,1.1,1.15,1.2")
    ap.add_argument("--prior_gate_modes", default="none,low_margin,high_entropy,low_confidence")
    ap.add_argument("--prior_gate_thresholds", default="0.4,0.45,0.5,0.55,0.6,0.62,0.64,0.66,0.68,0.7,0.72,0.75,0.78,0.8")
    ap.add_argument("--prior_pool_strategy", choices=["valid", "valid_weighted", "prior_softcap", "prior_softcap_valid", "prior_band"], default="prior_softcap_valid")
    ap.add_argument("--prior_top_k", type=int, default=1)
    ap.add_argument("--prior_hard_kl_cap", type=float, default=0.017)
    ap.add_argument("--prior_ensemble_mode", choices=["mean", "log_mean", "vote"], default="mean")
    ap.add_argument("--safe_prior_min_base_weight", type=float, default=0.90)
    ap.add_argument("--safe_prior_simplex_step", type=float, default=0.01)
    ap.add_argument("--stacker_select_metric", choices=["accuracy", "macro_f1"], default="accuracy")
    ap.add_argument("--stacker_c_grid", default="0.01,0.03,0.1,0.3,1,3,10")
    ap.add_argument("--stacker_class_weight_grid", default="none,balanced")
    ap.add_argument("--stacker_include_logits", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--stacker_include_confidence", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--stacker_base_input", default="")
    ap.add_argument("--stacker_unified_expert_slots", default="")
    ap.add_argument("--stacker_min_valid_gain_over_base", type=float, default=0.0)
    ap.add_argument("--stacker_max_prediction_change_rate", type=float, default=1.0)
    ap.add_argument("--stacker_max_prediction_js_divergence", type=float, default=1.0)
    ap.add_argument("--selector_select_metric", choices=["accuracy", "macro_f1"], default="accuracy")
    ap.add_argument("--selector_rank_select_metric", choices=["", "accuracy", "macro_f1"], default="")
    ap.add_argument(
        "--selector_rank_metric",
        choices=["select_metric", "accuracy", "macro_f1", "bootstrap_gain_quantile", "bootstrap_mean_gain", "bootstrap_win_rate"],
        default="bootstrap_gain_quantile",
    )
    ap.add_argument("--selector_strategies", default="always,class_bias_calibration")
    ap.add_argument("--selector_base_input", default="base")
    ap.add_argument("--selector_unified_expert_slots", default="base,prior,stacker")
    ap.add_argument("--selector_min_valid_gain_over_base", type=float, default=0.0)
    ap.add_argument("--selector_bootstrap_samples", type=int, default=50)
    ap.add_argument("--selector_rank_bootstrap_samples", type=int, default=50)
    ap.add_argument("--selector_rank_candidate_limit", type=int, default=48)
    ap.add_argument("--selector_bootstrap_min_win_rate", type=float, default=0.55)
    ap.add_argument("--selector_bootstrap_min_gain_quantile", type=float, default=-0.001)
    ap.add_argument("--selector_max_prediction_change_rate", type=float, default=0.25)
    ap.add_argument("--selector_max_prediction_js_divergence", type=float, default=0.03)
    ap.add_argument("--selector_calibration_penalty_weight", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
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
    ap.add_argument(
        "--paper_unified_stages",
        default="model",
        help=(
            "Comma-separated post-embedding stages for --stage paper_unified. "
            "The paper default is model (Tower2 single head only). Use model_fusion "
            "or all only for fusion/calibration/selector ablations, "
            "or explicit values from tower2_preprocess,tower2_train,eval,fusion,stacker,prior,selector."
        ),
    )
    ap.add_argument("--require_cuda", action="store_true", help="Fail before long stages if CUDA is unavailable.")
    ap.add_argument("--no_progress", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()
    if args.distill_min_teachers_per_flow <= 0:
        ap.error("--distill_min_teachers_per_flow must be positive")
    if not 0.0 <= args.tower1_class_weight_strength <= 1.0:
        ap.error("--tower1_class_weight_strength must be in [0, 1]")
    if args.tower1_paired_raw_consistency_weight < 0:
        ap.error("--tower1_paired_raw_consistency_weight must be non-negative")
    if args.tower1_paired_consistency_weight < 0 or args.tower1_paired_cls_weight < 0:
        ap.error("Tower-1 paired loss weights must be non-negative")
    if args.tower1_paired_logit_kl_weight < 0:
        ap.error("--tower1_paired_logit_kl_weight must be non-negative")
    if args.tower1_paired_cls_weight > 0 and args.tower1_paired_consistency_weight <= 0:
        ap.error("--tower1_paired_cls_weight requires --tower1_paired_consistency_weight > 0")

    if args.dataset in {"ustc-app", "ustc-binary"} and args.source_suffix == "change_weight":
        args.source_suffix = args.output_suffix
    if not args.tower1_data_suffix:
        args.tower1_data_suffix = args.source_suffix
    if args.tower1_output_dir == default_tower1_output_dir:
        args.tower1_output_dir = f"checkpoints/tower1_qwen_multitask_{safe_name(args.dataset)}_flowaware_change_weight"
    if args.dataset != "vpn-app":
        if args.coarse_groups == "vpn_app":
            args.coarse_groups = "none"
            args.hierarchical_weight = 0.0
            args.hierarchical_logit_weight = 0.0
        if args.confusion_groups == "vpn_app":
            args.confusion_groups = "none"
    independent_training_hyperparameters = capture_training_hyperparameter_overrides(
        args, "flow-level", args.training_hyperparameter_overrides
    )
    args.framework_profile_overrides = apply_framework_profile(args, "flow-level")
    args.shared_core_overrides = {}
    args.shared_core_config_sha256 = ""
    args.shared_core_method_sha256 = ""
    if args.shared_core_config:
        if args.framework_profile != "paper_unified":
            ap.error("--shared_core_config requires --framework_profile paper_unified")
        frozen = load_frozen_shared_core(args.shared_core_config)
        args.shared_core_overrides = apply_frozen_shared_core(
            args,
            "flow-level",
            frozen,
            training_hyperparameter_overrides=independent_training_hyperparameters,
        )
        args.shared_core_method_sha256 = frozen["config_sha256"]
        args.shared_core_config_sha256 = effective_shared_core_sha256(
            frozen,
            "flow-level",
            independent_training_hyperparameters,
        )
    elif independent_training_hyperparameters:
        args.shared_core_overrides = restore_profile_training_hyperparameters(
            args,
            "flow-level",
            independent_training_hyperparameters,
        )
    if args.train_row_risk_ablation:
        args.content_group_loss_reduction = "none"
    if not 0.0 <= args.packet_evidence_max_weight <= 1.0:
        ap.error("--packet_evidence_max_weight must be in [0, 1]")
    if args.train_ablate_input_channel != "none" and not args.exact_shared_packet_encoder:
        ap.error("--train_ablate_input_channel requires --exact_shared_packet_encoder")
    if args.train_ablate_intervention_view != "none" and (
        not args.exact_shared_packet_encoder or not args.use_intervention_views
    ):
        ap.error(
            "--train_ablate_intervention_view requires exact shared intervention views"
        )
    if args.train_fixed_channel_fusion and not args.exact_shared_packet_encoder:
        ap.error("--train_fixed_channel_fusion requires --exact_shared_packet_encoder")
    if args.packet_evidence_ablation_control:
        if args.packet_evidence_max_weight != 0:
            ap.error(
                "--packet_evidence_ablation_control requires "
                "--packet_evidence_max_weight 0"
            )
        if not args.exact_shared_packet_encoder:
            ap.error(
                "--packet_evidence_ablation_control requires "
                "--exact_shared_packet_encoder"
            )
        args.flow_pooling = "late_fusion"
    elif args.packet_evidence_max_weight > 0:
        if not args.exact_shared_packet_encoder:
            ap.error(
                "--packet_evidence_max_weight requires --exact_shared_packet_encoder"
            )
        # Packet evidence reaches the final flow prediction through the existing
        # learned late-fusion path; this is one registered cross-dataset candidate.
        args.flow_pooling = "late_fusion"
    if (
        args.framework_profile == "paper_unified"
        and args.intervention_view_base_mode != "symmetric_mean"
    ):
        ap.error(
            "paper_unified fixes --intervention_view_base_mode symmetric_mean; "
            "use a direct train_tower2.py ablation for factual_anchor"
        )
    if args.framework_profile == "paper_unified" and args.native_structural_suffix == "shared_content":
        # Native current-packet encoders are trained inside each fold. A static
        # suffix would silently share fold-0 weights with folds 1/2.
        args.native_structural_suffix = f"shared_content_{safe_name(args.output_suffix)}"

    intervention_args = None
    if args.framework_profile == "paper_unified" and args.use_intervention_views:
        intervention_suffix = f"{args.output_suffix}_mask_ip_port_intervention"
        intervention_embedding_suffix = f"{args.embedding_suffix}_mask_ip_port_intervention"
        intervention_args = argparse.Namespace(
            **{
                **vars(args),
                "output_suffix": intervention_suffix,
                "tower1_data_suffix": intervention_suffix,
                "embedding_suffix": intervention_embedding_suffix,
                "embedding_header_policy": args.intervened_embedding_header_policy,
                "tower2_suffix": "",
            }
        )
        if not args.paired_embedding_suffix:
            args.paired_embedding_suffix = tower2_data_suffix(intervention_args)
        if args.tower1_paired_consistency_weight > 0:
            args.tower1_paired_data_suffix = intervention_suffix

    if not args.train_dir or not args.valid_dir or not args.test_dir:
        train_dir, valid_dir, test_dir = default_split_paths(args.dataset, args.fold)
        args.train_dir = args.train_dir or train_dir
        args.valid_dir = args.valid_dir or valid_dir
        args.test_dir = args.test_dir or test_dir
    if intervention_args is not None:
        intervention_args.train_dir = args.train_dir
        intervention_args.valid_dir = args.valid_dir
        intervention_args.test_dir = args.test_dir
    if args.embedding_num_shards <= 0:
        raise SystemExit("--embedding_num_shards must be positive")

    long_stages = {"tower1_train", "embeddings", "native_pretrain", "native_embeddings", "all", "paper_unified", "post_tower1"}
    if args.require_cuda and args.stage in long_stages and not cuda_available():
        raise SystemExit("CUDA is unavailable; refusing to run long Stage-8 GPU stage.")

    write_manifest(args, completed=False)
    pre_embedding_stages = {"tower1_preprocess", "tower1_train"}
    if args.stage in pre_embedding_stages:
        for cmd in commands(args):
            run(cmd, dry_run=args.dry_run)
        write_manifest(args, completed=not args.dry_run)
        return

    if args.stage == "embeddings":
        for split in selected_splits(args.splits):
            run_embedding_stage(args, split)
        write_manifest(args, completed=not args.dry_run)
        return

    if args.stage == "native_embeddings" and not args.native_structural_suffix:
        raise SystemExit("--stage native_embeddings requires --native_structural_suffix")

    if args.stage == "paper_unified":
        run_post_embedding_pipeline(args)
        write_manifest(args, completed=not args.dry_run)
        return


    if args.stage == "post_tower1":
        run_post_tower1_pipeline(args, intervention_args)
        write_manifest(args, completed=not args.dry_run)
        return

    if args.stage == "all":
        for split in selected_splits(args.splits):
            run(tower1_preprocess_cmd(args, split), dry_run=args.dry_run)
        paired_tower1_preprocessed = bool(
            intervention_args is not None and args.tower1_paired_data_suffix
        )
        if paired_tower1_preprocessed:
            for split in selected_splits(args.splits):
                run(tower1_preprocess_cmd(intervention_args, split), dry_run=args.dry_run)
        run(tower1_train_cmd(args), dry_run=args.dry_run)
        run_post_tower1_pipeline(
            args,
            intervention_args,
            intervention_preprocessed=paired_tower1_preprocessed,
        )
        write_manifest(args, completed=not args.dry_run)
        return

    for cmd in commands(args):
        run(cmd, dry_run=args.dry_run)
    write_manifest(args, completed=not args.dry_run)


if __name__ == "__main__":
    main()
