#!/usr/bin/env python3
"""Run one fold of the unified Tower-1 Per-flow Split packet pipeline."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DATASET_DIRS = {
    "vpn-app": "vpn-app",
    "vpn-binary": "vpn-binary",
    "vpn-service": "vpn-service",
    "tls-120": "tls",
    "ustc-app": "ustc-app",
    "ustc-binary": "ustc-binary",
}


def run(command: list[str], dry_run: bool) -> None:
    print("+ " + " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(DATASET_DIRS), required=True)
    ap.add_argument("--fold", type=int, choices=[0, 1, 2], required=True)
    ap.add_argument(
        "--stage",
        choices=["preprocess", "audit", "feature", "byte", "fusion", "packet_best", "train", "test", "all"],
        default="all",
    )
    ap.add_argument("--source_root", default="/home/jing/download/sweet/packet-level-classification/per-flow-split")
    ap.add_argument("--artifact_root", default="reasoningDataset/packet-level")
    ap.add_argument("--checkpoint_root", default="checkpoints/packet-level")
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
    ap.add_argument("--eval_batch_size", type=int, default=64)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=4)
    ap.add_argument("--max_packet_length", type=int, default=640)
    ap.add_argument("--max_packets_per_flow", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--head_lr", type=float, default=1e-4)
    ap.add_argument("--cls_weight", type=float, default=1.0)
    ap.add_argument("--contrastive_weight", type=float, default=0.1)
    ap.add_argument("--same_flow_positive_weight", type=float, default=1.0)
    ap.add_argument("--class_weighting", choices=["none", "inverse", "effective"], default="effective")
    ap.add_argument("--byte_max_bytes", type=int, default=64)
    ap.add_argument(
        "--byte_use_payload_channel",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the shared current-packet payload branch in the byte expert.",
    )
    ap.add_argument("--byte_max_payload_bytes", type=int, default=128)
    ap.add_argument("--byte_epochs", type=int, default=12)
    ap.add_argument("--byte_batch_size", type=int, default=512)
    ap.add_argument("--byte_eval_batch_size", type=int, default=2048)
    ap.add_argument(
        "--embedding_header_policy",
        choices=["full", "randomize_ip_port", "mask_ip_port", "mask_session_fields"],
        default="full",
    )
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--local_files_only", action="store_true")
    args = ap.parse_args()

    dataset_dir = DATASET_DIRS[args.dataset]
    source = Path(args.source_root) / dataset_dir
    run_name = f"{args.dataset}_fold{args.fold}"
    artifacts = Path(args.artifact_root) / args.dataset / f"fold{args.fold}"
    checkpoint = Path(args.checkpoint_root) / run_name
    train_source = source / f"train_val_split_{args.fold}" / "train"
    valid_source = source / f"train_val_split_{args.fold}" / "val"
    test_source = source / "test"
    train_dir, valid_dir = artifacts / "train", artifacts / "valid"
    test_dir = Path(args.shared_test_dir) if args.shared_test_dir else artifacts / "test"
    label_map = train_dir / "label_map.json"

    if args.stage in {"preprocess", "all"}:
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
            command = [
                sys.executable,
                "preprocess_tower1.py",
                "--input_dir", str(input_dir),
                "--output_dir", str(output_dir),
                "--input_layout", "class_packet_pcaps",
                "--max_packets_per_flow", str(args.max_packets_per_flow),
                "--payload_prefix_len", "128",
                "--l3_prefix_len", "512",
                "--embedding_header_policy", args.embedding_header_policy,
                "--classification_only",
            ]
            if split == "train":
                command.append("--write_label_map")
            else:
                command.extend(["--label_map_in", str(label_map)])
            run(command, args.dry_run)

    if args.stage in {"audit", "all"}:
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

    if args.stage in {"byte", "packet_best", "all"}:
        byte_checkpoint = checkpoint / "byte_transformer"
        byte_command = [
                sys.executable,
                "train_packet_byte_transformer.py",
                "--train_index", str(train_dir / "packet_index.jsonl"),
                "--valid_index", str(valid_dir / "packet_index.jsonl"),
                "--label_map", str(label_map),
                "--output_dir", str(byte_checkpoint),
                "--output_json", str(artifacts / "packet_byte_transformer_validation.json"),
                "--max_bytes", str(args.byte_max_bytes),
                "--epochs", str(args.byte_epochs),
                "--batch_size", str(args.byte_batch_size),
                "--eval_batch_size", str(args.byte_eval_batch_size),
                "--packets_per_flow", "8",
                "--mask_probability", "0",
                "--masked_ce_weight", "0",
                "--consistency_weight", "0",
                "--contrastive_weight", "0",
            ]
        if args.byte_use_payload_channel:
            byte_command.extend(
                ["--use_payload_channel", "--max_payload_bytes", str(args.byte_max_payload_bytes)]
            )
        run(byte_command, args.dry_run)
        for split_name, split_dir in (("valid", valid_dir), ("test", test_dir)):
            run(
                [
                    sys.executable,
                    "test_packet_byte_transformer.py",
                    "--checkpoint", str(byte_checkpoint / "best.pt"),
                    "--packet_index", str(split_dir / "packet_index.jsonl"),
                    "--label_map", str(label_map),
                    "--output_json", str(artifacts / f"{split_name}_byte_probs.json"),
                    "--output_npz", str(artifacts / f"{split_name}_byte_probs.npz"),
                    "--batch_size", str(args.byte_eval_batch_size),
                ],
                args.dry_run,
            )

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
                "--output_json", str(artifacts / "test_byte_structural_nested_one_se.json"),
                "--output_npz", str(artifacts / "test_byte_structural_nested_one_se.npz"),
                "--gate_out", str(checkpoint / "byte_structural_nested_one_se_gate.joblib"),
            ],
            args.dry_run,
        )

    if args.stage in {"train", "all"}:
        command = [
            sys.executable,
            "train_tower1_multitask.py",
            "--base_model", args.base_model,
            "--label_map", str(label_map),
            "--packet_aux_jsonl", str(train_dir / "packet_auxiliary.jsonl"),
            "--valid_packet_aux_jsonl", str(valid_dir / "packet_auxiliary.jsonl"),
            "--output_dir", str(checkpoint),
            "--epochs", str(args.epochs),
            "--packet_batch_size", str(args.packet_batch_size),
            "--valid_batch_size", str(args.eval_batch_size),
            "--max_packet_length", str(args.max_packet_length),
            "--lr", str(args.lr),
            "--head_lr", str(args.head_lr),
            "--cls_weight", str(args.cls_weight),
            "--contrastive_weight", str(args.contrastive_weight),
            "--same_flow_positive_weight", str(args.same_flow_positive_weight),
            "--class_weighting", args.class_weighting,
            "--disable_packet_information_weights",
            "--select_metric", "macro_f1",
            "--gradient_accumulation_steps", str(args.gradient_accumulation_steps),
            "--gradient_checkpointing",
            "--no_sft",
        ]
        if args.init_checkpoint_dir:
            command.extend(["--init_checkpoint_dir", args.init_checkpoint_dir])
            if args.init_adapter_only:
                command.append("--init_adapter_only")
        if args.local_files_only:
            command.append("--local_files_only")
        run(command, args.dry_run)

    if args.stage in {"test", "all"}:
        command = [
            sys.executable,
            "test_tower1_packet.py",
            "--checkpoint_dir", str(checkpoint / "best"),
            "--packet_aux_jsonl", str(test_dir / "packet_auxiliary.jsonl"),
            "--label_map", str(label_map),
            "--output_json", str(artifacts / "test_packet_metrics.json"),
            "--batch_size", str(args.eval_batch_size),
            "--max_packet_length", str(args.max_packet_length),
        ]
        if args.local_files_only:
            command.append("--local_files_only")
        run(command, args.dry_run)


if __name__ == "__main__":
    main()
