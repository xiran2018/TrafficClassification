#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import Iterable, List


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


def result_suffix(args) -> str:
    suffix = f"{args.embedding_suffix}_stage8_flowaware"
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
        ]
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
        f"{args.tower1_output_dir}/adapter",
        "--tower1_heads",
        f"{args.tower1_output_dir}/tower1_heads.pt",
        "--embedding_mode",
        args.embedding_mode,
        "--batch_size",
        str(args.embedding_batch_size),
        "--max_length",
        str(args.embedding_max_length),
        "--device",
        device or args.embedding_device,
        "--local_files_only",
    ]
    if shard_index is not None:
        cmd += ["--num_shards", str(args.embedding_num_shards), "--shard_index", str(shard_index)]
    if args.no_progress:
        cmd.append("--no_progress")
    return cmd


def merge_embedding_shards(args, split: str) -> None:
    out_dir = Path(embedding_output_dir(args, split))
    shard_root = out_dir / "_shards"
    out_dir.mkdir(parents=True, exist_ok=True)
    merged_index = out_dir / "flow_embedding_index.jsonl"
    total = 0
    with open(merged_index, "w", encoding="utf-8") as outf:
        for shard_index in range(args.embedding_num_shards):
            shard_index_path = shard_root / f"shard_{shard_index}" / "flow_embedding_index.jsonl"
            if not shard_index_path.exists():
                raise FileNotFoundError(f"Missing shard index: {shard_index_path}")
            with open(shard_index_path, "r", encoding="utf-8") as inf:
                for line in inf:
                    if line.strip():
                        outf.write(line)
                        total += 1
    with open(out_dir / "embedding_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "base_model": args.base_model,
                "lora_path": f"{args.tower1_output_dir}/adapter",
                "tower1_heads": f"{args.tower1_output_dir}/tower1_heads.pt",
                "embedding_mode": args.embedding_mode,
                "max_length": args.embedding_max_length,
                "num_shards": args.embedding_num_shards,
                "shard_root": str(shard_root),
                "merged_flows": total,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"merged embedding shards: split={split}, flows={total}, index={merged_index}", flush=True)


def expected_embedding_shard_counts(packet_index: Path, num_shards: int) -> List[int]:
    counts = [0 for _ in range(num_shards)]
    seen_flows = set()
    with open(packet_index, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            flow_id = str(json.loads(line)["flow_id"])
            if flow_id in seen_flows:
                continue
            seen_flows.add(flow_id)
            digest = hashlib.sha1(flow_id.encode("utf-8", errors="ignore")).digest()
            shard_index = int.from_bytes(digest[:8], "big") % num_shards
            counts[shard_index] += 1
    return counts


def jsonl_row_count(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def run_embedding_stage(args, split: str) -> None:
    if args.embedding_num_shards <= 1:
        run(embedding_cmd(args, split), dry_run=args.dry_run)
        return

    out_dir = Path(embedding_output_dir(args, split))
    shard_root = out_dir / "_shards"
    devices = [x.strip() for x in args.embedding_cuda_devices.split(",") if x.strip()]
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
        return
    failed: list[tuple[int, int]] = []
    for shard_index, proc in procs:
        code = proc.wait()
        if code != 0:
            failed.append((shard_index, code))
    if failed:
        raise subprocess.CalledProcessError(failed[0][1], f"embedding shards failed: {failed}")
    merge_embedding_shards(args, split)


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
    cmd = [
        py(),
        "train_tower2.py",
        "--model_type",
        model_type,
        "--dataset",
        f"reasoningDataset/{args.dataset}/train_tower2_{args.embedding_suffix}/{model_type}_dataset.pt",
        "--valid_dataset",
        f"reasoningDataset/{args.dataset}/valid_tower2_{args.embedding_suffix}/{model_type}_dataset.pt",
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
        "flow_macro_f1",
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
        "--consistency_weight",
        str(args.consistency_weight),
        "--meta_dropout_prob",
        str(args.meta_dropout_prob),
        "--meta_feature_dim",
        str(args.meta_feature_dim),
        "--embedding_dropout_prob",
        str(args.embedding_dropout_prob),
        "--window_dropout_prob",
        str(args.window_dropout_prob),
        "--edge_attr_dropout_prob",
        str(args.edge_attr_dropout_prob),
        "--aux_weight",
        "0",
        "--coherence_weight",
        "0",
        "--seed",
        str(args.seed),
    ]
    if args.paired_embedding_suffix:
        cmd += [
            "--paired_view_dataset",
            f"reasoningDataset/{args.dataset}/train_tower2_{args.paired_embedding_suffix}/{model_type}_dataset.pt",
            "--paired_view_weight",
            str(args.paired_view_weight),
            "--paired_consistency_weight",
            str(args.paired_consistency_weight),
            "--paired_alignment_weight",
            str(args.paired_alignment_weight),
            "--paired_crossview_contrastive_weight",
            str(args.paired_crossview_contrastive_weight),
            "--paired_crossview_temperature",
            str(args.paired_crossview_temperature),
            "--paired_variance_weight",
            str(args.paired_variance_weight),
            "--paired_variance_target",
            str(args.paired_variance_target),
            "--paired_covariance_weight",
            str(args.paired_covariance_weight),
            "--view_domain_adversarial_weight",
            str(args.view_domain_adversarial_weight),
            "--domain_adversarial_lambda",
            str(args.domain_adversarial_lambda),
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
    return cmd


def tower2_eval_cmd(args, model_type: str, split: str) -> List[str]:
    prefix = "valid" if split == "valid" else "test"
    return [
        py(),
        "test_tower2.py",
        "--checkpoint",
        f"checkpoints/tower2_{model_type}_flow_{safe_name(args.dataset)}_{result_suffix(args)}/best.pt",
        "--dataset",
        f"reasoningDataset/{args.dataset}/{split}_tower2_{args.embedding_suffix}/{model_type}_dataset.pt",
        "--label_map",
        train_label_map(args),
        "--output_json",
        f"reasoningDataset/{args.dataset}/{prefix}_{model_type}_metrics_flow_{result_suffix(args)}_probs.json",
        "--no_report",
    ]


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


def write_manifest(args) -> None:
    root = Path("reasoningDataset") / args.dataset
    root.mkdir(parents=True, exist_ok=True)
    payload = vars(args).copy()
    payload["splits"] = selected_splits(args.splits)
    payload["cuda_available_at_launch"] = cuda_available()
    with open(root / f"stage8_flowaware_manifest_{result_suffix(args)}.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def commands(args) -> Iterable[List[str]]:
    splits = selected_splits(args.splits)
    if args.stage in {"tower1_preprocess", "all"}:
        for split in splits:
            yield tower1_preprocess_cmd(args, split)
    if args.stage in {"tower1_train", "all"}:
        yield tower1_train_cmd(args)
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


def main() -> None:
    ap = argparse.ArgumentParser()
    default_tower1_output_dir = "checkpoints/tower1_qwen_multitask_flowaware_change_weight"
    ap.add_argument("--dataset", default="vpn-app")
    ap.add_argument("--num_classes", type=int, default=16)
    ap.add_argument("--stage", choices=["tower1_train", "tower1_preprocess", "embeddings", "tower2_preprocess", "tower2_train", "eval", "fusion", "stacker", "prior", "selector", "all"], required=True)
    ap.add_argument("--splits", default="train,valid,test")
    ap.add_argument("--train_dir", default="")
    ap.add_argument("--valid_dir", default="")
    ap.add_argument("--test_dir", default="")
    ap.add_argument("--source_suffix", default="change_weight")
    ap.add_argument("--output_suffix", default="flowaware_change_weight")
    ap.add_argument("--tower1_data_suffix", default="", help="Tower-1 train/eval label-map suffix. Defaults to --source_suffix after dataset-specific normalization.")
    ap.add_argument("--embedding_suffix", default="rawproj_flowaware_change_weight")
    ap.add_argument("--run_tag", default="", help="Optional suffix appended to Stage-8 training/eval/fusion outputs, useful for ablations.")
    ap.add_argument("--tower1_output_dir", default=default_tower1_output_dir)
    ap.add_argument("--base_model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--max_packets_per_flow", type=int, default=64)
    ap.add_argument("--payload_prefix_len", type=int, default=128)
    ap.add_argument("--l3_prefix_len", type=int, default=512)
    ap.add_argument("--preprocess_max_flows", type=int, default=0)
    ap.add_argument("--embedding_header_policy", choices=["full", "randomize_ip_port", "mask_ip_port"], default="full", help="Header policy for packet-index prompts used by embedding extraction.")
    ap.add_argument("--tower1_epochs", type=int, default=2)
    ap.add_argument("--tower1_max_steps", type=int, default=0)
    ap.add_argument("--tower1_save_steps", type=int, default=0)
    ap.add_argument("--tower1_init_checkpoint_dir", default="")
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
    ap.add_argument("--tower1_paired_data_suffix", default="", help="Optional Tower-1 second-view data suffix for paired packet consistency.")
    ap.add_argument("--tower1_paired_consistency_weight", type=float, default=0.0, help="Tower-1 full-header vs paired-view consistency weight.")
    ap.add_argument("--tower1_paired_cls_weight", type=float, default=0.0, help="Extra paired-view packet CE multiplier in Tower-1.")
    ap.add_argument("--tower1_paired_logit_kl_weight", type=float, default=0.5, help="Logit symmetric-KL weight inside Tower-1 paired consistency.")
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
    ap.add_argument("--embedding_device", default="auto", help="Device for unsharded embedding extraction.")
    ap.add_argument("--embedding_num_shards", type=int, default=1, help="Run embedding extraction as N deterministic flow-id shards and merge the shard indexes.")
    ap.add_argument("--embedding_cuda_devices", default="", help="Comma-separated CUDA device ids assigned round-robin to embedding shards, e.g. 0,1,2,3.")
    ap.add_argument("--embedding_resume_shards", action=argparse.BooleanOptionalAction, default=True, help="Skip embedding shards whose index row count matches the expected flow count.")
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
    ap.add_argument("--confusion_groups", default="vpn_app")
    ap.add_argument("--flow_contrastive_weight", type=float, default=0.03)
    ap.add_argument("--flow_temperature", type=float, default=0.07)
    ap.add_argument("--window_contrastive_weight", type=float, default=0.0)
    ap.add_argument("--window_contrastive_temperature", type=float, default=0.07)
    ap.add_argument("--window_contrastive_positive", choices=["own_flow", "same_class"], default="same_class")
    ap.add_argument("--consistency_weight", type=float, default=0.0, help="KL consistency between clean and augmented Tower-2 flow logits.")
    ap.add_argument("--meta_dropout_prob", type=float, default=0.0, help="Training-only dropout on trailing Tower-2 metadata features.")
    ap.add_argument("--meta_feature_dim", type=int, default=14, help="Number of trailing metadata features in Tower-2 x.")
    ap.add_argument("--embedding_dropout_prob", type=float, default=0.0, help="Training-only dropout on packet embedding features.")
    ap.add_argument("--window_dropout_prob", type=float, default=0.0, help="Training-only random dropping of windows before flow aggregation.")
    ap.add_argument("--edge_attr_dropout_prob", type=float, default=0.0, help="Training-only dropout on graph edge attributes.")
    ap.add_argument("--paired_embedding_suffix", default="", help="Optional second-view Tower-2 dataset suffix aligned by flow_id.")
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
    if args.embedding_num_shards <= 0:
        raise SystemExit("--embedding_num_shards must be positive")

    long_stages = {"tower1_train", "embeddings", "all"}
    if args.require_cuda and args.stage in long_stages and not cuda_available():
        raise SystemExit("CUDA is unavailable; refusing to run long Stage-8 GPU stage.")

    write_manifest(args)
    pre_embedding_stages = {"tower1_preprocess", "tower1_train"}
    if args.stage in pre_embedding_stages:
        for cmd in commands(args):
            run(cmd, dry_run=args.dry_run)
        return

    if args.stage == "embeddings":
        for split in selected_splits(args.splits):
            run_embedding_stage(args, split)
        return

    if args.stage == "all":
        for split in selected_splits(args.splits):
            run(tower1_preprocess_cmd(args, split), dry_run=args.dry_run)
        run(tower1_train_cmd(args), dry_run=args.dry_run)
        for split in selected_splits(args.splits):
            run_embedding_stage(args, split)
        for cmd in commands(argparse.Namespace(**{**vars(args), "stage": "tower2_preprocess"})):
            run(cmd, dry_run=args.dry_run)
        for cmd in commands(argparse.Namespace(**{**vars(args), "stage": "tower2_train"})):
            run(cmd, dry_run=args.dry_run)
        for cmd in commands(argparse.Namespace(**{**vars(args), "stage": "eval"})):
            run(cmd, dry_run=args.dry_run)
        for cmd in commands(argparse.Namespace(**{**vars(args), "stage": "fusion"})):
            run(cmd, dry_run=args.dry_run)
        for cmd in commands(argparse.Namespace(**{**vars(args), "stage": "stacker"})):
            run(cmd, dry_run=args.dry_run)
        for cmd in commands(argparse.Namespace(**{**vars(args), "stage": "prior"})):
            run(cmd, dry_run=args.dry_run)
        for cmd in commands(argparse.Namespace(**{**vars(args), "stage": "selector"})):
            run(cmd, dry_run=args.dry_run)
        return

    for cmd in commands(args):
        run(cmd, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
