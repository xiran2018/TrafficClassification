#!/usr/bin/env python3
"""Train a shared strict-one-packet byte Transformer under Per-flow Split."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from models.packet_byte_transformer import PacketByteTransformer
from models.qwen_packet_multitask import flow_aware_contrastive_loss
from packet_eval_utils import packet_classification_metrics
from train_packet_feature_expert import packet_features
from train_tower1_multitask import FlowBalancedPacketBatchSampler, load_label_names, stable_flow_id


PAD_TOKEN = 256
MASK_TOKEN = 257


def extract_packet_payload(raw: bytes, meta: dict) -> bytes:
    """Return only the current packet payload without using flow context."""
    if not raw:
        return b""
    version = raw[0] >> 4
    if version == 4:
        l4_offset = (raw[0] & 0x0F) * 4
        protocol = raw[9] if len(raw) > 9 else int(meta.get("ip_proto", -1))
    elif version == 6:
        l4_offset = 40
        protocol = raw[6] if len(raw) > 6 else int(meta.get("ip_proto", -1))
    else:
        return b""
    if l4_offset >= len(raw):
        return b""
    if protocol == 6:
        if len(raw) <= l4_offset + 12:
            return b""
        transport_header_len = (raw[l4_offset + 12] >> 4) * 4
    elif protocol == 17:
        transport_header_len = 8
    else:
        transport_header_len = 0
    payload_offset = min(len(raw), l4_offset + transport_header_len)
    return raw[payload_offset:]


def mask_session_tokens(tokens: np.ndarray, length: int) -> np.ndarray:
    masked = tokens.copy()
    if length <= 0:
        return masked
    raw = tokens[:length]
    version = int(raw[0]) >> 4
    positions: list[tuple[int, int]] = []
    if version == 4:
        ihl = (int(raw[0]) & 0x0F) * 4
        positions.extend([(4, 6), (8, 9), (10, 12), (12, 20)])
        protocol = int(raw[9]) if length > 9 else -1
        l4_offset = ihl
    elif version == 6:
        positions.extend([(1, 4), (7, 8), (8, 40)])
        protocol = int(raw[6]) if length > 6 else -1
        l4_offset = 40
    else:
        return masked
    positions.append((l4_offset, l4_offset + 4))
    if protocol == 6:
        positions.extend([(l4_offset + 4, l4_offset + 12), (l4_offset + 16, l4_offset + 18)])
    elif protocol == 17:
        positions.append((l4_offset + 6, l4_offset + 8))
    for start, stop in positions:
        if start < length:
            masked[start:min(stop, length)] = MASK_TOKEN
    return masked


class PacketByteDataset(Dataset):
    def __init__(
        self,
        index_path: str,
        max_bytes: int,
        include_augmented: bool = True,
        max_payload_bytes: int = 0,
    ) -> None:
        self.rows: list[dict] = []
        tokens, masked_tokens, lengths = [], [], []
        payload_tokens, payload_lengths = [], []
        metas, masked_metas, labels, flow_ids = [], [], [], []
        payload_width = max(1, int(max_payload_bytes))
        with open(index_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                full_raw = bytes.fromhex(str(row["meta"].get("l3_hex_prefix", "")).replace(" ", ""))
                raw = full_raw[:max_bytes]
                item = np.full(max_bytes, PAD_TOKEN, dtype=np.int64)
                if raw:
                    item[:len(raw)] = np.frombuffer(raw, dtype=np.uint8).astype(np.int64)
                tokens.append(item)
                if include_augmented:
                    masked_tokens.append(mask_session_tokens(item, len(raw)))
                lengths.append(len(raw))
                payload = extract_packet_payload(full_raw, row["meta"])[:payload_width]
                payload_item = np.full(payload_width, PAD_TOKEN, dtype=np.int64)
                if payload:
                    payload_item[:len(payload)] = np.frombuffer(payload, dtype=np.uint8).astype(np.int64)
                payload_tokens.append(payload_item)
                payload_lengths.append(len(payload))
                metas.append(packet_features(row, 0, False, False, False))
                if include_augmented:
                    masked_metas.append(packet_features(row, 0, False, False, True))
                labels.append(int(row["label_id"]))
                flow_ids.append(stable_flow_id(str(row.get("flow_id", len(labels) - 1))))
                self.rows.append({"flow_id": str(row.get("flow_id", len(labels) - 1))})
        self.tokens = torch.from_numpy(np.stack(tokens))
        self.masked_tokens = torch.from_numpy(np.stack(masked_tokens)) if include_augmented else self.tokens
        self.lengths = torch.tensor(lengths, dtype=torch.long)
        self.payload_tokens = torch.from_numpy(np.stack(payload_tokens))
        self.payload_lengths = torch.tensor(payload_lengths, dtype=torch.long)
        self.metas = torch.from_numpy(np.stack(metas)).float()
        self.masked_metas = torch.from_numpy(np.stack(masked_metas)).float() if include_augmented else self.metas
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.flow_ids = torch.tensor(flow_ids, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "tokens": self.tokens[index],
            "masked_tokens": self.masked_tokens[index],
            "length": self.lengths[index],
            "payload_tokens": self.payload_tokens[index],
            "payload_length": self.payload_lengths[index],
            "meta": self.metas[index],
            "masked_meta": self.masked_metas[index],
            "label": self.labels[index],
            "flow_id": self.flow_ids[index],
        }


def effective_class_weights(labels: torch.Tensor, num_classes: int, beta: float) -> torch.Tensor:
    counts = torch.bincount(labels, minlength=num_classes).float().clamp(min=1)
    weights = (1.0 - beta) / (1.0 - torch.pow(torch.tensor(beta), counts))
    return weights * num_classes / weights.sum()


@torch.no_grad()
def evaluate(model, loader, device, num_classes, label_names, return_probabilities=False):
    model.eval()
    ys, preds, probs = [], [], []
    for batch in tqdm(loader, desc="eval byte transformer", leave=False):
        logits, _, _ = model(
            batch["tokens"].to(device),
            batch["length"].to(device),
            batch["meta"].to(device),
            batch["payload_tokens"].to(device) if model.use_payload_channel else None,
            batch["payload_length"].to(device) if model.use_payload_channel else None,
        )
        probability = torch.softmax(logits.float(), dim=-1).cpu()
        ys.extend(batch["label"].tolist())
        preds.extend(probability.argmax(dim=-1).tolist())
        if return_probabilities:
            probs.append(probability.numpy())
    metrics = packet_classification_metrics(ys, preds, num_classes, label_names)
    return metrics, np.concatenate(probs) if probs else None, np.asarray(ys, dtype=np.int64)


def save_checkpoint(path: Path, model, config, metrics) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": config, "validation_metrics": metrics}, path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_index", required=True)
    ap.add_argument("--valid_index", required=True)
    ap.add_argument("--test_index", default="")
    ap.add_argument("--label_map", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--output_json", required=True)
    ap.add_argument("--output_npz", default="")
    ap.add_argument("--max_bytes", type=int, default=256)
    ap.add_argument("--use_payload_channel", action="store_true")
    ap.add_argument("--max_payload_bytes", type=int, default=128)
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--num_layers", type=int, default=3)
    ap.add_argument("--num_heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.15)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--eval_batch_size", type=int, default=512)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--learning_rate", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--class_weight_beta", type=float, default=0.9999)
    ap.add_argument("--mask_probability", type=float, default=0.5)
    ap.add_argument("--masked_ce_weight", type=float, default=0.3)
    ap.add_argument("--consistency_weight", type=float, default=0.1)
    ap.add_argument("--contrastive_weight", type=float, default=0.05)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--packets_per_flow", type=int, default=2)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    label_names = load_label_names(args.label_map)
    num_classes = len(label_names)
    device = torch.device(args.device)
    needs_second_view = args.masked_ce_weight > 0 or args.consistency_weight > 0
    payload_width = args.max_payload_bytes if args.use_payload_channel else 0
    train_dataset = PacketByteDataset(
        args.train_index,
        args.max_bytes,
        include_augmented=needs_second_view,
        max_payload_bytes=payload_width,
    )
    valid_dataset = PacketByteDataset(
        args.valid_index,
        args.max_bytes,
        include_augmented=False,
        max_payload_bytes=payload_width,
    )
    sampler = FlowBalancedPacketBatchSampler(
        train_dataset.rows, args.batch_size, args.packets_per_flow, seed=args.seed
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=sampler,
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
    config = {
        "num_classes": num_classes,
        "max_bytes": args.max_bytes,
        "meta_dim": int(train_dataset.metas.shape[1]),
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "dropout": args.dropout,
        "use_payload_channel": args.use_payload_channel,
        "max_payload_bytes": args.max_payload_bytes,
    }
    model = PacketByteTransformer(**config).to(device)
    class_weights = effective_class_weights(train_dataset.labels, num_classes, args.class_weight_beta).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = max(1, args.epochs * len(train_loader))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    output_dir = Path(args.output_dir)
    best_key = None
    best_epoch = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in tqdm(train_loader, desc=f"byte transformer epoch {epoch}"):
            clean_tokens = batch["tokens"].to(device)
            lengths = batch["length"].to(device)
            clean_meta = batch["meta"].to(device)
            labels = batch["label"].to(device)
            flow_ids = batch["flow_id"].to(device)
            if needs_second_view:
                masked_tokens = batch["masked_tokens"].to(device)
                masked_meta = batch["masked_meta"].to(device)
                use_mask = torch.rand(len(labels), device=device) < args.mask_probability
                view_tokens = torch.where(use_mask[:, None], masked_tokens, clean_tokens)
                view_meta = torch.where(use_mask[:, None], masked_meta, clean_meta)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                payload_tokens = batch["payload_tokens"].to(device) if model.use_payload_channel else None
                payload_lengths = batch["payload_length"].to(device) if model.use_payload_channel else None
                clean_logits, clean_z, _ = model(
                    clean_tokens, lengths, clean_meta, payload_tokens, payload_lengths
                )
                ce = F.cross_entropy(clean_logits, labels, weight=class_weights)
                masked_ce = clean_logits.sum() * 0.0
                consistency = clean_logits.sum() * 0.0
                if needs_second_view:
                    view_logits, _, _ = model(
                        view_tokens, lengths, view_meta, payload_tokens, payload_lengths
                    )
                    masked_ce = F.cross_entropy(view_logits, labels, weight=class_weights)
                    clean_log = F.log_softmax(clean_logits.float(), dim=-1)
                    view_log = F.log_softmax(view_logits.float(), dim=-1)
                    clean_prob = clean_log.exp().detach()
                    view_prob = view_log.exp().detach()
                    consistency = 0.5 * (
                        F.kl_div(clean_log, view_prob, reduction="batchmean")
                        + F.kl_div(view_log, clean_prob, reduction="batchmean")
                    )
                contrastive = flow_aware_contrastive_loss(
                    clean_z, labels, flow_ids, temperature=args.temperature,
                    same_flow_weight=2.0, same_label_weight=1.0,
                )
                loss = ce + args.masked_ce_weight * masked_ce + args.consistency_weight * consistency
                loss = loss + args.contrastive_weight * contrastive
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            losses.append(float(loss.detach().cpu()))
        valid_metrics, _, _ = evaluate(model, valid_loader, device, num_classes, label_names)
        record = {"epoch": epoch, "loss": float(np.mean(losses)), "validation_metrics": valid_metrics}
        history.append(record)
        key = (valid_metrics["macro_f1"], valid_metrics["accuracy"])
        print(
            f"epoch={epoch} loss={record['loss']:.4f} valid_acc={valid_metrics['accuracy']:.4f} "
            f"valid_macro_f1={valid_metrics['macro_f1']:.4f}", flush=True
        )
        if best_key is None or key > best_key:
            best_key = key
            best_epoch = epoch
            save_checkpoint(output_dir / "best.pt", model, config, valid_metrics)
        if epoch - best_epoch >= args.patience:
            print(f"early stopping after epoch={epoch}; best_epoch={best_epoch}", flush=True)
            break

    checkpoint = torch.load(output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    result = {
        "task": "packet-level-classification",
        "sample_unit": "one_packet",
        "architecture": (
            "dual-channel-byte-payload-transformer-meta-gated"
            if args.use_payload_channel else "local-byte-transformer-meta-gated"
        ),
        "config": vars(args),
        "model_config": config,
        "best_epoch": best_epoch,
        "validation_metrics": checkpoint["validation_metrics"],
        "history": history,
    }
    if args.test_index:
        test_dataset = PacketByteDataset(
            args.test_index,
            args.max_bytes,
            include_augmented=False,
            max_payload_bytes=payload_width,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        test_metrics, probabilities, y_true = evaluate(
            model, test_loader, device, num_classes, label_names, return_probabilities=True
        )
        result["test_metrics"] = test_metrics
        if args.output_npz:
            Path(args.output_npz).parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(args.output_npz, y_true=y_true, probabilities=probabilities)
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"saved {output_path} and {output_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
