#!/usr/bin/env python3
"""Evaluate a Tower-1 packet head on a held-out Per-flow Split packet set."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from packet_eval_utils import evaluate_packet_model
from train_tower1_multitask import PacketAuxCollator, PacketAuxDataset, load_label_names


def infer_projection_dim(head_state: dict) -> int:
    weight = head_state["projection_head"].get("net.3.weight")
    return int(weight.shape[0]) if weight is not None else 256


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_dir", required=True, help="Tower-1 directory containing adapter/, tower1_heads.pt, and tower1_config.json.")
    ap.add_argument("--packet_aux_jsonl", required=True)
    ap.add_argument("--label_map", required=True)
    ap.add_argument("--output_json", required=True)
    ap.add_argument("--base_model", default="", help="Override the base model recorded in tower1_config.json.")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_packet_length", type=int, default=1024)
    ap.add_argument("--start_sample", type=int, default=0, help="Zero-based first packet for deterministic sharded evaluation.")
    ap.add_argument("--max_samples", type=int, default=0, help="Evaluate only the first N packets for a smoke test.")
    ap.add_argument("--save_probabilities", action="store_true")
    ap.add_argument("--output_npz", default="", help="Optional compact y_true/probabilities output for expert fusion.")
    ap.add_argument("--local_files_only", action="store_true")
    ap.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    with open(checkpoint_dir / "tower1_config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
    label_names = load_label_names(args.label_map)
    head_state = torch.load(checkpoint_dir / "tower1_heads.pt", map_location="cpu")
    num_classes = len(label_names)
    if int(head_state.get("num_classes", num_classes)) != num_classes:
        raise ValueError(f"checkpoint classes={head_state.get('num_classes')} label_map classes={num_classes}")

    device = torch.device(args.device)
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    if device.type == "cpu" and dtype != torch.float32:
        dtype = torch.float32
    from models.qwen_packet_multitask import QwenPacketMultiTaskModel

    model = QwenPacketMultiTaskModel(
        base_model_name_or_path=args.base_model or config["base_model"],
        num_classes=num_classes,
        torch_dtype=dtype,
        lora_path=str(checkpoint_dir / "adapter"),
        create_lora=False,
        projection_dim=int(config.get("projection_dim", infer_projection_dim(head_state))),
        local_files_only=args.local_files_only,
    )
    model.packet_classifier.load_state_dict(head_state["packet_classifier"])
    model.projection_head.load_state_dict(head_state["projection_head"])
    model.to(device)

    dataset = PacketAuxDataset(args.packet_aux_jsonl)
    start = max(0, args.start_sample)
    if start >= len(dataset):
        raise ValueError(f"start_sample={start} is outside dataset of size {len(dataset)}")
    stop = len(dataset) if args.max_samples <= 0 else min(len(dataset), start + args.max_samples)
    indices = range(start, stop)
    eval_dataset = dataset if start == 0 and stop == len(dataset) else Subset(dataset, indices)
    rows = dataset.rows[start:stop]
    loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=PacketAuxCollator(model.tokenizer, args.max_packet_length),
    )
    metrics = evaluate_packet_model(
        model,
        loader,
        device=device,
        num_classes=num_classes,
        label_names=label_names,
        desc="test packets",
        return_probabilities=args.save_probabilities or bool(args.output_npz),
    )
    y_true = metrics.pop("y_true")
    y_pred = metrics.pop("y_pred")
    probabilities = metrics.pop("probabilities", None)
    predictions = []
    for i, (row, true, pred) in enumerate(zip(rows, y_true, y_pred)):
        item = {
            "packet_uid": row.get("packet_uid"),
            "flow_id": row.get("flow_id"),
            "label": row.get("label"),
            "label_id": int(true),
            "pred_id": int(pred),
            "pred_label": label_names[int(pred)],
        }
        if args.save_probabilities and probabilities is not None:
            item["probabilities"] = probabilities[i]
        predictions.append(item)
    result = {
        "task": "packet-level-classification",
        "split_protocol": "per-flow-split",
        "checkpoint_dir": str(checkpoint_dir),
        "packet_aux_jsonl": args.packet_aux_jsonl,
        "sample_range": {"start": start, "stop": stop, "dataset_size": len(dataset)},
        "metrics": metrics,
        "predictions": predictions,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    if args.output_npz:
        if probabilities is None:
            raise RuntimeError("probabilities were not returned")
        npz_path = Path(args.output_npz)
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            npz_path,
            y_true=np.asarray(y_true, dtype=np.int64),
            probabilities=np.asarray(probabilities, dtype=np.float32),
            sample_start=np.asarray(start, dtype=np.int64),
            sample_stop=np.asarray(stop, dtype=np.int64),
            dataset_size=np.asarray(len(dataset), dtype=np.int64),
        )
        print(f"saved {npz_path}")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"saved {output_path}")


if __name__ == "__main__":
    main()
