#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path
from typing import Iterable, List


def run(cmd: List[str], dry_run: bool = False) -> None:
    print("$ " + " ".join(shlex.quote(x) for x in cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True)


def default_split_paths(dataset: str) -> tuple[str, str, str]:
    root = "/home/jing/download/sweet/flow-level-classification"
    if dataset == "vpn-app":
        base = f"{root}/vpn-app"
        return f"{base}/train_val_split_0/train", f"{base}/train_val_split_0/val", f"{base}/test"
    if dataset in {"tls-120", "tls"}:
        base = f"{root}/tls"
        return f"{base}/train_val_split_0/train", f"{base}/train_val_split_0/val", f"{base}/test"
    raise ValueError(f"No default paths for dataset={dataset}; pass --train_dir/--valid_dir/--test_dir.")


def common_tower1_args(args, split: str) -> List[str]:
    cmd = [
        "python",
        "preprocess_tower1.py",
        "--input_dir",
        getattr(args, f"{split}_dir"),
        "--output_dir",
        f"reasoningDataset/{args.dataset}/{split}_tower1_{args.suffix}",
        "--max_packets_per_flow",
        str(args.max_packets_per_flow),
        "--payload_prefix_len",
        str(args.payload_prefix_len),
        "--l3_prefix_len",
        str(args.l3_prefix_len),
    ]
    if args.no_progress:
        cmd.append("--no_progress")
    return cmd


def selected_splits(args) -> List[str]:
    splits = []
    for raw in args.splits.split(","):
        split = raw.strip()
        if not split:
            continue
        if split not in {"train", "valid", "test"}:
            raise ValueError(f"Unknown split {split}; use train,valid,test")
        if split not in splits:
            splits.append(split)
    if not splits:
        raise ValueError("--splits must contain at least one of train,valid,test")
    return splits


def tower1_preprocess_commands(args) -> Iterable[List[str]]:
    train_out = f"reasoningDataset/{args.dataset}/train_tower1_{args.suffix}"
    splits = selected_splits(args)
    if "train" in splits:
        train_cmd = common_tower1_args(args, "train") + ["--write_label_map"]
        if args.embedding_header_policy != "full":
            train_cmd += ["--embedding_header_policy", args.embedding_header_policy]
        yield train_cmd
    for split in [s for s in splits if s != "train"]:
        cmd = common_tower1_args(args, split) + ["--label_map_in", f"{train_out}/label_map.json"]
        if args.embedding_header_policy != "full":
            cmd += ["--embedding_header_policy", args.embedding_header_policy]
        yield cmd


def embedding_commands(args) -> Iterable[List[str]]:
    for split in selected_splits(args):
        cmd = [
            "python",
            "extract_packet_embeddings_qwen.py",
            "--packet_index",
            f"reasoningDataset/{args.dataset}/{split}_tower1_{args.suffix}/packet_index.jsonl",
            "--output_dir",
            f"reasoningDataset/{args.dataset}/{split}_embeddings_{args.embedding_suffix}",
            "--base_model",
            args.base_model,
            "--lora_path",
            args.lora_path,
            "--tower1_heads",
            args.tower1_heads,
            "--embedding_mode",
            args.embedding_mode,
            "--batch_size",
            str(args.embedding_batch_size),
            "--max_length",
            str(args.embedding_max_length),
        ]
        if not args.allow_remote_model_download:
            cmd.append("--local_files_only")
        if args.no_progress:
            cmd.append("--no_progress")
        yield cmd


def tower2_preprocess_commands(args) -> Iterable[List[str]]:
    for split in selected_splits(args):
        yield [
            "python",
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


def write_manifest(args) -> None:
    root = Path("reasoningDataset") / args.dataset
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": args.dataset,
        "suffix": args.suffix,
        "embedding_suffix": args.embedding_suffix,
        "train_dir": args.train_dir,
        "valid_dir": args.valid_dir,
        "test_dir": args.test_dir,
        "max_packets_per_flow": args.max_packets_per_flow,
        "payload_prefix_len": args.payload_prefix_len,
        "l3_prefix_len": args.l3_prefix_len,
        "embedding_header_policy": args.embedding_header_policy,
        "embedding_mode": args.embedding_mode,
        "embedding_max_length": args.embedding_max_length,
        "allow_remote_model_download": args.allow_remote_model_download,
        "window_size": args.window_size,
        "stride": args.stride,
        "no_progress": args.no_progress,
        "splits": selected_splits(args),
    }
    with open(root / f"pipeline_manifest_{args.embedding_suffix}.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="Dataset key, e.g. vpn-app or tls-120.")
    ap.add_argument("--stage", choices=["tower1", "embeddings", "tower2", "all"], required=True)
    ap.add_argument("--splits", default="train,valid,test", help="Comma-separated split subset to run, e.g. train or test. Order is normalized to train/valid/test where needed.")
    ap.add_argument("--train_dir", default="")
    ap.add_argument("--valid_dir", default="")
    ap.add_argument("--test_dir", default="")
    ap.add_argument("--suffix", default="change_weight")
    ap.add_argument("--embedding_suffix", default="rawproj_change_weight")
    ap.add_argument("--max_packets_per_flow", type=int, default=64)
    ap.add_argument("--payload_prefix_len", type=int, default=128)
    ap.add_argument("--l3_prefix_len", type=int, default=512)
    ap.add_argument("--embedding_header_policy", choices=["full", "randomize_ip_port", "mask_ip_port"], default="full")
    ap.add_argument("--base_model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--lora_path", default="checkpoints/tower1_qwen_multitask_change_weight/adapter")
    ap.add_argument("--tower1_heads", default="checkpoints/tower1_qwen_multitask_change_weight/tower1_heads.pt")
    ap.add_argument("--embedding_mode", choices=["raw", "projected", "concat"], default="concat")
    ap.add_argument("--embedding_batch_size", type=int, default=8)
    ap.add_argument("--embedding_max_length", type=int, default=1792)
    ap.add_argument("--allow_remote_model_download", action="store_true", help="Allow HuggingFace downloads during embedding extraction; default is local cache only.")
    ap.add_argument("--window_size", type=int, default=32)
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--no_progress", action="store_true", help="Disable child progress bars where supported.")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    if not args.train_dir or not args.valid_dir or not args.test_dir:
        train_dir, valid_dir, test_dir = default_split_paths(args.dataset)
        args.train_dir = args.train_dir or train_dir
        args.valid_dir = args.valid_dir or valid_dir
        args.test_dir = args.test_dir or test_dir

    write_manifest(args)
    if args.stage in {"tower1", "all"}:
        for cmd in tower1_preprocess_commands(args):
            run(cmd, args.dry_run)
    if args.stage in {"embeddings", "all"}:
        for cmd in embedding_commands(args):
            run(cmd, args.dry_run)
    if args.stage in {"tower2", "all"}:
        for cmd in tower2_preprocess_commands(args):
            run(cmd, args.dry_run)


if __name__ == "__main__":
    main()
