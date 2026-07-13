#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import Iterable, List


def run(cmd: List[str], dry_run: bool = False) -> None:
    print("$ " + " ".join(shlex.quote(x) for x in cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True)


def cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def default_split_paths(dataset: str) -> tuple[str, str, str]:
    root = "/home/jing/download/sweet/flow-level-classification"
    if dataset == "vpn-app":
        base = f"{root}/vpn-app"
        return f"{base}/train_val_split_0/train", f"{base}/train_val_split_0/val", f"{base}/test"
    if dataset in {"tls", "tls-120"}:
        base = f"{root}/tls"
        return f"{base}/train_val_split_0/train", f"{base}/train_val_split_0/val", f"{base}/test"
    packet_root = "/home/jing/download/sweet/packet-level-classification/per-flow-split"
    if dataset in {"ustc-app", "ustc-binary"}:
        base = f"{packet_root}/{dataset}"
        return f"{base}/train_val_split_0/train", f"{base}/train_val_split_0/val", f"{base}/test"
    raise ValueError(f"No default paths for dataset={dataset}; pass --train_dir/--valid_dir/--test_dir.")


def selected_splits(raw: str) -> List[str]:
    out = []
    for split in raw.split(","):
        split = split.strip()
        if not split:
            continue
        if split not in {"train", "valid", "test"}:
            raise ValueError(f"Unknown split {split}; use train,valid,test")
        if split not in out:
            out.append(split)
    return out or ["train", "valid", "test"]


def py() -> str:
    return "python"


def safe_name(value: str) -> str:
    return value.replace("/", "_").replace("-", "_")


def train_tower1_dir(args) -> str:
    return f"reasoningDataset/{args.dataset}/train_tower1_{args.tower1_data_suffix}"


def train_label_map(args) -> str:
    return f"{train_tower1_dir(args)}/label_map.json"


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
        "--sft_jsonl",
        f"{train_tower1_dir(args)}/packet_instruction.jsonl",
        f"{train_tower1_dir(args)}/packet_validity.jsonl",
        "--output_dir",
        args.tower1_output_dir,
        "--epochs",
        str(args.tower1_epochs),
        "--sft_batch_size",
        str(args.sft_batch_size),
        "--packet_batch_size",
        str(args.packet_batch_size),
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
        "--flow_proto_weight",
        str(args.flow_proto_weight),
        "--flow_proto_positive",
        args.flow_proto_positive,
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
        "--seed",
        str(args.seed),
    ]
    if args.tower1_max_steps > 0:
        cmd += ["--max_steps", str(args.tower1_max_steps)]
    if args.tower1_save_steps > 0:
        cmd += ["--save_steps", str(args.tower1_save_steps)]
    if args.flow_balanced_packet_batches:
        cmd += ["--flow_balanced_packet_batches", "--packets_per_flow", str(args.packets_per_flow)]
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
    ]
    if args.preprocess_max_flows > 0:
        cmd += ["--max_flows", str(args.preprocess_max_flows)]
    if split == "train":
        cmd.append("--write_label_map")
        if Path(source_label_map).exists():
            cmd += ["--label_map_in", source_label_map]
    else:
        cmd += ["--label_map_in", train_label_map(args)]
    if args.no_progress:
        cmd.append("--no_progress")
    return cmd


def embedding_cmd(args, split: str) -> List[str]:
    return [
        py(),
        "extract_packet_embeddings_qwen.py",
        "--packet_index",
        f"reasoningDataset/{args.dataset}/{split}_tower1_{args.output_suffix}/packet_index.jsonl",
        "--output_dir",
        f"reasoningDataset/{args.dataset}/{split}_embeddings_{args.embedding_suffix}",
        "--base_model",
        args.base_model,
        "--lora_path",
        f"{args.tower1_output_dir}/adapter",
        "--tower1_heads",
        f"{args.tower1_output_dir}/tower1_heads.pt",
        "--embedding_mode",
        args.embedding_mode,
        "--batch_size",
        str(args.embedding_batch_size),
        "--max_length",
        str(args.embedding_max_length),
        "--local_files_only",
    ] + (["--no_progress"] if args.no_progress else [])


def tower2_preprocess_cmd(args, split: str) -> List[str]:
    return [
        py(),
        "preprocess_tower2.py",
        "--flow_embedding_index",
        f"reasoningDataset/{args.dataset}/{split}_embeddings_{args.embedding_suffix}/flow_embedding_index.jsonl",
        "--output_dir",
        f"reasoningDataset/{args.dataset}/{split}_tower2_{args.embedding_suffix}",
        "--window_size",
        str(args.window_size),
        "--stride",
        str(args.stride),
    ]


def tower2_train_cmd(args, model_type: str) -> List[str]:
    return [
        py(),
        "train_tower2.py",
        "--model_type",
        model_type,
        "--dataset",
        f"reasoningDataset/{args.dataset}/train_tower2_{args.embedding_suffix}/{model_type}_dataset.pt",
        "--valid_dataset",
        f"reasoningDataset/{args.dataset}/valid_tower2_{args.embedding_suffix}/{model_type}_dataset.pt",
        "--output_dir",
        f"checkpoints/tower2_{model_type}_flow_{safe_name(args.dataset)}_{args.embedding_suffix}_stage8_flowaware",
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
        "flow_macro_f1",
        "--early_stop_patience",
        str(args.tower2_early_stop_patience),
        "--flow_pooling",
        args.flow_pooling,
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
        "--hierarchical_weight",
        str(args.hierarchical_weight),
        "--hierarchical_logit_weight",
        str(args.hierarchical_logit_weight),
        "--coarse_groups",
        args.coarse_groups,
        "--balanced_flow_batches",
        "--samples_per_class",
        str(args.samples_per_class),
        "--contrastive_mode",
        "confusion",
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
        "--aux_weight",
        "0",
        "--coherence_weight",
        "0",
        "--seed",
        str(args.seed),
    ]


def tower2_eval_cmd(args, model_type: str, split: str) -> List[str]:
    prefix = "valid" if split == "valid" else "test"
    return [
        py(),
        "test_tower2.py",
        "--checkpoint",
        f"checkpoints/tower2_{model_type}_flow_{safe_name(args.dataset)}_{args.embedding_suffix}_stage8_flowaware/best.pt",
        "--dataset",
        f"reasoningDataset/{args.dataset}/{split}_tower2_{args.embedding_suffix}/{model_type}_dataset.pt",
        "--label_map",
        train_label_map(args),
        "--output_json",
        f"reasoningDataset/{args.dataset}/{prefix}_{model_type}_metrics_flow_{args.embedding_suffix}_stage8_flowaware_probs.json",
        "--no_report",
    ]


def model_prob_path(args, model_type: str, split: str) -> str:
    prefix = "valid" if split == "valid" else "test"
    return f"reasoningDataset/{args.dataset}/{prefix}_{model_type}_metrics_flow_{args.embedding_suffix}_stage8_flowaware_probs.json"


def fusion_payload_path(args, model_type: str) -> str:
    return f"reasoningDataset/{args.dataset}/{model_type}_metrics_flow_{args.embedding_suffix}_stage8_flowaware_fusion_payload.json"


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
    return f"reasoningDataset/{args.dataset}/test_fusion_{'_'.join(selected_model_types(args))}_{args.embedding_suffix}_stage8_flowaware_valid_acc.json"


def prior_candidate_output_path(args) -> str:
    return f"reasoningDataset/{args.dataset}/test_fusion_{'_'.join(selected_model_types(args))}_{args.embedding_suffix}_stage8_flowaware_safe_prior_candidate.json"


def safe_prior_output_path(args) -> str:
    return f"reasoningDataset/{args.dataset}/test_fusion_{'_'.join(selected_model_types(args))}_{args.embedding_suffix}_stage8_flowaware_safe_prior_residual.json"


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


def write_manifest(args) -> None:
    root = Path("reasoningDataset") / args.dataset
    root.mkdir(parents=True, exist_ok=True)
    payload = vars(args).copy()
    payload["splits"] = selected_splits(args.splits)
    payload["cuda_available_at_launch"] = cuda_available()
    with open(root / f"stage8_flowaware_manifest_{args.embedding_suffix}.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def commands(args) -> Iterable[List[str]]:
    splits = selected_splits(args.splits)
    if args.stage in {"tower1_preprocess", "all"}:
        for split in splits:
            yield tower1_preprocess_cmd(args, split)
    if args.stage in {"tower1_train", "all"}:
        yield tower1_train_cmd(args)
    if args.stage in {"embeddings", "all"}:
        for split in splits:
            yield embedding_cmd(args, split)
    if args.stage in {"tower2_preprocess", "all"}:
        for split in splits:
            yield tower2_preprocess_cmd(args, split)
    if args.stage in {"tower2_train", "all"}:
        for model_type in selected_model_types(args):
            yield tower2_train_cmd(args, model_type)
    if args.stage in {"eval", "all"}:
        for model_type in selected_model_types(args):
            yield tower2_eval_cmd(args, model_type, "valid")
            yield tower2_eval_cmd(args, model_type, "test")
    if args.stage in {"fusion", "all"}:
        for model_type in selected_model_types(args):
            yield make_fusion_payload_cmd(args, model_type)
        yield fusion_cmd(args)
    if args.stage in {"prior", "all"}:
        yield prior_cmd(args)
        yield safe_prior_residual_cmd(args)


def main() -> None:
    ap = argparse.ArgumentParser()
    default_tower1_output_dir = "checkpoints/tower1_qwen_multitask_flowaware_change_weight"
    ap.add_argument("--dataset", default="vpn-app")
    ap.add_argument("--num_classes", type=int, default=16)
    ap.add_argument("--stage", choices=["tower1_train", "tower1_preprocess", "embeddings", "tower2_preprocess", "tower2_train", "eval", "fusion", "prior", "all"], required=True)
    ap.add_argument("--splits", default="train,valid,test")
    ap.add_argument("--train_dir", default="")
    ap.add_argument("--valid_dir", default="")
    ap.add_argument("--test_dir", default="")
    ap.add_argument("--source_suffix", default="change_weight")
    ap.add_argument("--output_suffix", default="flowaware_change_weight")
    ap.add_argument("--tower1_data_suffix", default="", help="Tower-1 train/eval label-map suffix. Defaults to --source_suffix after dataset-specific normalization.")
    ap.add_argument("--embedding_suffix", default="rawproj_flowaware_change_weight")
    ap.add_argument("--tower1_output_dir", default=default_tower1_output_dir)
    ap.add_argument("--base_model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--max_packets_per_flow", type=int, default=64)
    ap.add_argument("--payload_prefix_len", type=int, default=128)
    ap.add_argument("--l3_prefix_len", type=int, default=512)
    ap.add_argument("--preprocess_max_flows", type=int, default=0)
    ap.add_argument("--tower1_epochs", type=int, default=2)
    ap.add_argument("--tower1_max_steps", type=int, default=0)
    ap.add_argument("--tower1_save_steps", type=int, default=0)
    ap.add_argument("--sft_batch_size", type=int, default=2)
    ap.add_argument("--packet_batch_size", type=int, default=16)
    ap.add_argument("--max_sft_length", type=int, default=1792)
    ap.add_argument("--max_packet_length", type=int, default=1024)
    ap.add_argument("--cls_weight", type=float, default=0.1)
    ap.add_argument("--contrastive_weight", type=float, default=0.3)
    ap.add_argument("--temperature", type=float, default=0.07)
    ap.add_argument("--same_flow_positive_weight", type=float, default=2.0)
    ap.add_argument("--same_label_positive_weight", type=float, default=1.0)
    ap.add_argument("--flow_proto_weight", type=float, default=0.0)
    ap.add_argument("--flow_proto_positive", choices=["own_flow", "same_class"], default="same_class")
    ap.add_argument("--flow_balanced_packet_batches", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--packets_per_flow", type=int, default=2)
    ap.add_argument("--tower1_lr", type=float, default=2e-5)
    ap.add_argument("--tower1_head_lr", type=float, default=1e-4)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    ap.add_argument("--local_files_only", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--gradient_checkpointing", action="store_true")
    ap.add_argument("--embedding_mode", choices=["raw", "projected", "concat"], default="concat")
    ap.add_argument("--embedding_batch_size", type=int, default=8)
    ap.add_argument("--embedding_max_length", type=int, default=1024)
    ap.add_argument("--window_size", type=int, default=32)
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--model_types", default="graph,seq")
    ap.add_argument("--tower2_epochs", type=int, default=30)
    ap.add_argument("--tower2_early_stop_patience", type=int, default=8)
    ap.add_argument("--tower2_batch_size", type=int, default=16)
    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--num_heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.15)
    ap.add_argument("--tower2_lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.03)
    ap.add_argument("--flow_pooling", default="mean", choices=["mean", "attention", "late_fusion", "transformer", "multi_view"])
    ap.add_argument("--window_loss_weight", type=float, default=0.3)
    ap.add_argument("--class_weight_strength", type=float, default=0.6)
    ap.add_argument("--label_smoothing", type=float, default=0.05)
    ap.add_argument("--hierarchical_weight", type=float, default=0.2)
    ap.add_argument("--hierarchical_logit_weight", type=float, default=0.5)
    ap.add_argument("--coarse_groups", default="vpn_app")
    ap.add_argument("--samples_per_class", type=int, default=2)
    ap.add_argument("--confusion_groups", default="vpn_app")
    ap.add_argument("--flow_contrastive_weight", type=float, default=0.03)
    ap.add_argument("--flow_temperature", type=float, default=0.07)
    ap.add_argument("--window_contrastive_weight", type=float, default=0.0)
    ap.add_argument("--window_contrastive_temperature", type=float, default=0.07)
    ap.add_argument("--window_contrastive_positive", choices=["own_flow", "same_class"], default="same_class")
    ap.add_argument("--prior_strengths", default="0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0,1.05,1.1,1.15,1.2")
    ap.add_argument("--prior_gate_modes", default="none,low_margin,high_entropy,low_confidence")
    ap.add_argument("--prior_gate_thresholds", default="0.4,0.45,0.5,0.55,0.6,0.62,0.64,0.66,0.68,0.7,0.72,0.75,0.78,0.8")
    ap.add_argument("--prior_pool_strategy", choices=["valid", "valid_weighted", "prior_softcap", "prior_softcap_valid", "prior_band"], default="prior_softcap_valid")
    ap.add_argument("--prior_top_k", type=int, default=1)
    ap.add_argument("--prior_hard_kl_cap", type=float, default=0.017)
    ap.add_argument("--prior_ensemble_mode", choices=["mean", "log_mean", "vote"], default="mean")
    ap.add_argument("--safe_prior_min_base_weight", type=float, default=0.90)
    ap.add_argument("--safe_prior_simplex_step", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--require_cuda", action="store_true", help="Fail before long stages if CUDA is unavailable.")
    ap.add_argument("--no_progress", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

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

    if not args.train_dir or not args.valid_dir or not args.test_dir:
        train_dir, valid_dir, test_dir = default_split_paths(args.dataset)
        args.train_dir = args.train_dir or train_dir
        args.valid_dir = args.valid_dir or valid_dir
        args.test_dir = args.test_dir or test_dir

    long_stages = {"tower1_train", "embeddings", "all"}
    if args.require_cuda and args.stage in long_stages and not cuda_available():
        raise SystemExit("CUDA is unavailable; refusing to run long Stage-8 GPU stage.")

    write_manifest(args)
    for cmd in commands(args):
        run(cmd, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
