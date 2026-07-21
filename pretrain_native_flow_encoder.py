#!/usr/bin/env python3
"""Self-supervised pre-training for protocol-aware packet/flow representations."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.native_flow_encoder import (
    NATIVE_PACKET_PRETRAINING_PROTOCOL,
    NativeFlowEncoder,
    nt_xent_loss,
    sample_relative_pairs,
    sample_same_flow_pairs,
)
from native_flow_data import (
    NUM_FIELD_TYPES,
    PacketIndexFlowDataset,
    apply_payload_dropout,
    apply_session_invariant_mask,
    mask_protocol_fields,
)


def valid_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    valid = targets >= 0
    if not valid.any():
        return logits.sum() * 0.0
    return F.cross_entropy(logits[valid], targets[valid])


def accuracy_count(logits: torch.Tensor, targets: torch.Tensor) -> tuple[int, int]:
    valid = targets >= 0
    if not valid.any():
        return 0, 0
    correct = int((logits[valid].argmax(dim=-1) == targets[valid]).sum().item())
    return correct, int(valid.sum().item())


def compute_objectives(
    model: NativeFlowEncoder,
    batch: dict,
    device: torch.device,
    args,
    training: bool,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    tokens = batch["byte_tokens"].to(device, non_blocking=True)
    field_ids = batch["field_ids"].to(device, non_blocking=True)
    byte_mask = batch["byte_mask"].to(device, non_blocking=True)
    packet_mask = batch["packet_mask"].to(device, non_blocking=True)
    directions = batch["directions"].to(device, non_blocking=True)
    packet_meta = batch["packet_meta"].to(device, non_blocking=True)
    next_length = batch["next_length"].to(device, non_blocking=True)
    next_iat = batch["next_iat"].to(device, non_blocking=True)

    view1, prediction_mask1 = mask_protocol_fields(
        tokens, field_ids, byte_mask, args.field_mask_probability
    )
    view2, prediction_mask2 = mask_protocol_fields(
        tokens, field_ids, byte_mask, args.field_mask_probability
    )
    view1, _ = apply_payload_dropout(
        view1, field_ids, byte_mask, args.payload_dropout_probability
    )
    view2, _ = apply_payload_dropout(
        view2, field_ids, byte_mask, args.payload_dropout_probability
    )
    view1, _ = apply_session_invariant_mask(
        view1, field_ids, byte_mask, args.session_mask_probability
    )
    view2, _ = apply_session_invariant_mask(
        view2, field_ids, byte_mask, args.session_mask_probability
    )
    first = model(view1, field_ids, byte_mask, packet_mask, directions, packet_meta)
    second = model(view2, field_ids, byte_mask, packet_mask, directions, packet_meta)

    byte_logits1 = model.masked_byte_logits(first["token_repr"], prediction_mask1)
    byte_logits2 = model.masked_byte_logits(second["token_repr"], prediction_mask2)
    byte_targets1 = tokens[prediction_mask1].long()
    byte_targets2 = tokens[prediction_mask2].long()
    masked_byte_loss = 0.5 * (
        F.cross_entropy(byte_logits1, byte_targets1)
        + F.cross_entropy(byte_logits2, byte_targets2)
    )

    rel_first, rel_second, rel_targets = sample_relative_pairs(
        first["packet_observation"], packet_mask
    )
    relative_logits = model.relative_order_logits(rel_first, rel_second)
    relative_loss = valid_cross_entropy(relative_logits, rel_targets)

    flow_first, flow_second, flow_targets = sample_same_flow_pairs(
        first["packet_content"], packet_mask
    )
    same_flow_logits = model.same_flow_logits(flow_first, flow_second)
    same_flow_loss = valid_cross_entropy(same_flow_logits, flow_targets)

    next_length_logits, next_iat_logits = model.next_packet_logits(first["packet_repr"])
    next_length_loss = valid_cross_entropy(next_length_logits, next_length)
    next_iat_loss = valid_cross_entropy(next_iat_logits, next_iat)

    direction_targets = directions - 1
    direction_targets = torch.where(
        packet_mask & (directions > 0), direction_targets, torch.full_like(direction_targets, -1)
    )
    direction_loss = valid_cross_entropy(first["direction_logits"], direction_targets)
    valid_packets = packet_mask.bool()
    if valid_packets.any():
        packet_consistency_loss = (
            1.0
            - F.cosine_similarity(
                first["packet_repr"][valid_packets].float(),
                second["packet_repr"][valid_packets].float(),
                dim=-1,
            )
        ).mean()
    else:
        packet_consistency_loss = first["packet_repr"].sum() * 0.0
    contrastive_loss = nt_xent_loss(
        first["flow_projection"], second["flow_projection"], args.temperature
    )

    losses = {
        "masked_byte": masked_byte_loss,
        "relative_order": relative_loss,
        "same_flow": same_flow_loss,
        "next_length": next_length_loss,
        "next_iat": next_iat_loss,
        "direction": direction_loss,
        "packet_consistency": packet_consistency_loss,
        "flow_contrastive": contrastive_loss,
    }
    total = (
        args.masked_byte_weight * masked_byte_loss
        + args.relative_order_weight * relative_loss
        + args.same_flow_weight * same_flow_loss
        + args.next_length_weight * next_length_loss
        + args.next_iat_weight * next_iat_loss
        + args.direction_weight * direction_loss
        + args.packet_consistency_weight * packet_consistency_loss
        + args.flow_contrastive_weight * contrastive_loss
    )
    metrics: dict[str, float | int] = {
        "loss": float(total.detach()),
        **{f"loss_{name}": float(value.detach()) for name, value in losses.items()},
    }
    for name, logits, targets in (
        ("masked_byte_1", byte_logits1, byte_targets1),
        ("relative_order", relative_logits, rel_targets),
        ("same_flow", same_flow_logits, flow_targets),
        ("next_length", next_length_logits, next_length),
        ("next_iat", next_iat_logits, next_iat),
        ("direction", first["direction_logits"], direction_targets),
    ):
        correct, count = accuracy_count(logits, targets)
        metrics[f"correct_{name}"] = correct
        metrics[f"count_{name}"] = count
    return total, metrics


def aggregate_metrics(rows: list[dict[str, float | int]]) -> dict[str, float]:
    if not rows:
        return {}
    output = {
        key: float(np.mean([float(row[key]) for row in rows]))
        for key in rows[0]
        if key.startswith("loss")
    }
    for suffix in (
        "masked_byte_1",
        "relative_order",
        "same_flow",
        "next_length",
        "next_iat",
        "direction",
    ):
        correct = sum(int(row[f"correct_{suffix}"]) for row in rows)
        count = sum(int(row[f"count_{suffix}"]) for row in rows)
        output[f"accuracy_{suffix}"] = correct / count if count else 0.0
        output[f"count_{suffix}"] = float(count)
    return output


@torch.no_grad()
def evaluate(model, loader, device, args) -> dict[str, float]:
    model.eval()
    rows = []
    fork_devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=fork_devices):
        torch.manual_seed(args.seed + 100003)
        for step, batch in enumerate(tqdm(loader, desc="native valid", leave=False), start=1):
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                _, metrics = compute_objectives(model, batch, device, args, training=False)
            rows.append(metrics)
            if args.valid_steps and step >= args.valid_steps:
                break
    return aggregate_metrics(rows)


def save_checkpoint(path: Path, model, model_config: dict, args, history: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_config": model_config,
            "pretraining_config": vars(args),
            "history": history,
            "objective": "label_free_protocol_structural_pretraining",
            "pretraining_protocol": NATIVE_PACKET_PRETRAINING_PROTOCOL,
        },
        path,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_index", required=True)
    ap.add_argument("--valid_index", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--max_packets", type=int, default=64)
    ap.add_argument("--max_bytes", type=int, default=128)
    ap.add_argument("--max_train_flows", type=int, default=0)
    ap.add_argument("--max_valid_flows", type=int, default=0)
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--projection_dim", type=int, default=128)
    ap.add_argument("--byte_layers", type=int, default=2)
    ap.add_argument("--flow_layers", type=int, default=2)
    ap.add_argument("--num_heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--eval_batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--learning_rate", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--field_mask_probability", type=float, default=0.2)
    ap.add_argument(
        "--payload_dropout_probability",
        type=float,
        default=0.5,
        help="Erase encrypted payload by packet without using it as a reconstruction target.",
    )
    ap.add_argument(
        "--session_mask_probability",
        type=float,
        default=1.0,
        help="Mask endpoint, checksum, sequence, IP-ID, and TTL fields in both views without reconstructing them.",
    )
    ap.add_argument("--masked_byte_weight", type=float, default=1.0)
    ap.add_argument("--relative_order_weight", type=float, default=0.25)
    ap.add_argument("--same_flow_weight", type=float, default=0.25)
    ap.add_argument("--next_length_weight", type=float, default=0.2)
    ap.add_argument("--next_iat_weight", type=float, default=0.2)
    ap.add_argument("--direction_weight", type=float, default=0.1)
    ap.add_argument(
        "--packet_consistency_weight",
        type=float,
        default=0.25,
        help="Align corresponding packet representations across independent protocol-field interventions.",
    )
    ap.add_argument("--flow_contrastive_weight", type=float, default=0.25)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--steps_per_epoch", type=int, default=0)
    ap.add_argument("--valid_steps", type=int, default=0)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--initialize_only",
        action="store_true",
        help="Save a deterministic untrained checkpoint for frozen-probe controls and exit.",
    )
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    if not 0.0 < args.field_mask_probability <= 1.0:
        ap.error("--field_mask_probability must be in (0, 1]")
    if not 0.0 <= args.session_mask_probability <= 1.0:
        ap.error("--session_mask_probability must be in [0, 1]")
    if not 0.0 <= args.payload_dropout_probability <= 1.0:
        ap.error("--payload_dropout_probability must be in [0, 1]")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    train_dataset = PacketIndexFlowDataset(
        args.train_index,
        args.max_packets,
        args.max_bytes,
        args.max_train_flows,
    )
    valid_dataset = PacketIndexFlowDataset(
        args.valid_index,
        args.max_packets,
        args.max_bytes,
        args.max_valid_flows,
    )
    print(
        json.dumps(
            {
                "train_flows": len(train_dataset),
                "valid_flows": len(valid_dataset),
                "uses_downstream_labels": False,
            }
        ),
        flush=True,
    )
    train_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=train_generator,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    model_config = {
        "max_bytes": args.max_bytes,
        "max_packets": args.max_packets,
        "hidden_dim": args.hidden_dim,
        "byte_layers": args.byte_layers,
        "flow_layers": args.flow_layers,
        "num_heads": args.num_heads,
        "dropout": args.dropout,
        "num_field_types": NUM_FIELD_TYPES,
        "projection_dim": args.projection_dim,
    }
    model = NativeFlowEncoder(**model_config).to(device)
    output_dir = Path(args.output_dir)
    if args.initialize_only:
        save_checkpoint(output_dir / "initial.pt", model, model_config, args, [])
        print(f"saved {output_dir / 'initial.pt'}", flush=True)
        return
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    estimated_steps = args.steps_per_epoch or len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs * estimated_steps)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history: list[dict] = []
    best_loss = float("inf")
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        rows = []
        progress = tqdm(train_loader, desc=f"native pretrain epoch {epoch}")
        for step, batch in enumerate(progress, start=1):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                loss, metrics = compute_objectives(model, batch, device, args, training=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            rows.append(metrics)
            progress.set_postfix(loss=f"{metrics['loss']:.3f}")
            if args.steps_per_epoch and step >= args.steps_per_epoch:
                break
        train_metrics = aggregate_metrics(rows)
        valid_metrics = evaluate(model, valid_loader, device, args)
        record = {
            "epoch": epoch,
            "train": train_metrics,
            "valid": valid_metrics,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        if valid_metrics["loss"] < best_loss:
            best_loss = valid_metrics["loss"]
            best_epoch = epoch
            save_checkpoint(output_dir / "best.pt", model, model_config, args, history)
        save_checkpoint(output_dir / "last.pt", model, model_config, args, history)
        if epoch - best_epoch >= args.patience:
            print(f"early stopping epoch={epoch} best_epoch={best_epoch}", flush=True)
            break

    summary = {
        "method": "traffic-native-protocol-structural-pretraining",
        "uses_downstream_labels": False,
        "best_epoch": best_epoch,
        "best_valid_loss": best_loss,
        "model_config": model_config,
        "pretraining_config": vars(args),
        "history": history,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "pretraining_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(f"saved {output_dir / 'best.pt'}", flush=True)


if __name__ == "__main__":
    main()
