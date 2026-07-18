#!/usr/bin/env python3
"""Evaluate a saved strict-one-packet byte Transformer checkpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.packet_byte_transformer import PacketByteTransformer
from train_packet_byte_transformer import PacketByteDataset, evaluate
from train_tower1_multitask import load_label_names


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--packet_index", required=True)
    ap.add_argument("--label_map", required=True)
    ap.add_argument("--output_json", required=True)
    ap.add_argument("--output_npz", default="")
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = PacketByteTransformer(**config).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    dataset = PacketByteDataset(
        args.packet_index,
        int(config["max_bytes"]),
        include_augmented=False,
        max_payload_bytes=(
            int(config.get("max_payload_bytes", 0))
            if bool(config.get("use_payload_channel", False)) else 0
        ),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    label_names = load_label_names(args.label_map)
    metrics, probabilities, y_true = evaluate(
        model, loader, device, len(label_names), label_names, return_probabilities=True
    )
    payload = {
        "task": "packet-level-classification",
        "sample_unit": "one_packet",
        "architecture": (
            "dual-channel-byte-payload-transformer-meta-gated"
            if bool(config.get("use_payload_channel", False))
            else "local-byte-transformer-meta-gated"
        ),
        "checkpoint": args.checkpoint,
        "packet_index": args.packet_index,
        "metrics": metrics,
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    if args.output_npz:
        output_npz = Path(args.output_npz)
        output_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_npz, y_true=y_true, probabilities=probabilities)
    print(f"accuracy={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f}")
    print(f"saved {output_json}" + (f" and {args.output_npz}" if args.output_npz else ""))


if __name__ == "__main__":
    main()
