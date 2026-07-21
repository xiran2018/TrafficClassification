#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from models.intervention_transport import (
    LowRankInterventionTransport,
    transport_alignment_loss,
)


def load_pt(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def sample_key(item: Dict[str, Any]) -> tuple[str, tuple[int, int]]:
    window = item.get("window")
    if not isinstance(window, (tuple, list)) or len(window) != 2:
        raise ValueError("every paired sample must include a two-value window")
    return str(item["flow_id"]), (int(window[0]), int(window[1]))


def align_packet_embeddings(
    clean_items: Sequence[Dict[str, Any]],
    intervened_items: Sequence[Dict[str, Any]],
    meta_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    clean_map = {sample_key(item): item for item in clean_items}
    intervened_map = {sample_key(item): item for item in intervened_items}
    if len(clean_map) != len(clean_items) or len(intervened_map) != len(intervened_items):
        raise ValueError("duplicate (flow_id, window) keys in paired datasets")
    if set(clean_map) != set(intervened_map):
        missing_intervened = sorted(set(clean_map) - set(intervened_map))[:3]
        missing_clean = sorted(set(intervened_map) - set(clean_map))[:3]
        raise ValueError(
            "paired datasets have different window keys: "
            f"missing_intervened={missing_intervened} missing_clean={missing_clean}"
        )

    clean_packets: List[torch.Tensor] = []
    intervened_packets: List[torch.Tensor] = []
    feature_dim = None
    for key in sorted(clean_map):
        clean = clean_map[key]
        intervened = intervened_map[key]
        if int(clean["label"]) != int(intervened["label"]):
            raise ValueError(f"paired label mismatch for key={key}")
        clean_x = clean["x"]
        intervened_x = intervened["x"]
        if clean_x.shape != intervened_x.shape:
            raise ValueError(
                f"paired tensor shape mismatch for key={key}: "
                f"clean={tuple(clean_x.shape)} intervened={tuple(intervened_x.shape)}"
            )
        current_dim = int(clean_x.shape[-1]) - int(meta_dim)
        if current_dim <= 0:
            raise ValueError("meta_dim leaves no embedding features")
        if feature_dim is None:
            feature_dim = current_dim
        elif feature_dim != current_dim:
            raise ValueError("inconsistent embedding dimensions")
        clean_packets.append(clean_x[:, :current_dim].to(dtype=torch.float16))
        intervened_packets.append(intervened_x[:, :current_dim].to(dtype=torch.float16))

    clean_tensor = torch.cat(clean_packets, dim=0)
    intervened_tensor = torch.cat(intervened_packets, dim=0)
    report = {
        "windows": len(clean_map),
        "flows": len({key[0] for key in clean_map}),
        "packets": int(clean_tensor.shape[0]),
        "embedding_dim": int(clean_tensor.shape[1]),
        "meta_dim": int(meta_dim),
    }
    return clean_tensor, intervened_tensor, report


@torch.no_grad()
def evaluate(model, loader, device, args) -> Dict[str, float]:
    model.eval()
    totals = {"loss": 0.0, "cosine_loss": 0.0, "normalized_mse": 0.0, "moment_loss": 0.0}
    count = 0
    for intervened, clean in loader:
        intervened = intervened.to(device=device, dtype=torch.float32)
        clean = clean.to(device=device, dtype=torch.float32)
        transported = model(intervened)
        loss, parts = transport_alignment_loss(
            transported,
            clean,
            args.cosine_weight,
            args.normalized_mse_weight,
            args.moment_weight,
        )
        batch_size = clean.size(0)
        totals["loss"] += float(loss.item()) * batch_size
        for key, value in parts.items():
            totals[key] += float(value.item()) * batch_size
        count += batch_size
    return {key: value / max(1, count) for key, value in totals.items()}


def train(args) -> None:
    clean_train, intervened_train, train_report = align_packet_embeddings(
        load_pt(args.train_clean), load_pt(args.train_intervened), args.meta_dim
    )
    clean_valid, intervened_valid, valid_report = align_packet_embeddings(
        load_pt(args.valid_clean), load_pt(args.valid_intervened), args.meta_dim
    )
    print("transport_alignment " + json.dumps({"train": train_report, "valid": valid_report}), flush=True)
    train_loader = DataLoader(
        TensorDataset(intervened_train, clean_train),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    valid_loader = DataLoader(
        TensorDataset(intervened_valid, clean_valid),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    model = LowRankInterventionTransport(
        train_report["embedding_dim"], args.rank, args.dropout
    ).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    stale = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        count = 0
        for intervened, clean in train_loader:
            intervened = intervened.to(args.device, dtype=torch.float32)
            clean = clean.to(args.device, dtype=torch.float32)
            transported = model(intervened)
            loss, _ = transport_alignment_loss(
                transported,
                clean,
                args.cosine_weight,
                args.normalized_mse_weight,
                args.moment_weight,
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.item()) * clean.size(0)
            count += clean.size(0)
        valid = evaluate(model, valid_loader, args.device, args)
        train_loss = total / max(1, count)
        print(
            f"epoch={epoch} train_loss={train_loss:.6f} "
            f"valid_loss={valid['loss']:.6f} valid_cosine={1.0-valid['cosine_loss']:.6f} "
            f"valid_nmse={valid['normalized_mse']:.6f}",
            flush=True,
        )
        if valid["loss"] < best:
            best = valid["loss"]
            stale = 0
            torch.save(
                {
                    "state": model.state_dict(),
                    "embedding_dim": train_report["embedding_dim"],
                    "rank": args.rank,
                    "dropout": args.dropout,
                    "meta_dim": args.meta_dim,
                    "train_report": train_report,
                    "valid_report": valid_report,
                    "best_valid": valid,
                    "objective": {
                        "cosine_weight": args.cosine_weight,
                        "normalized_mse_weight": args.normalized_mse_weight,
                        "moment_weight": args.moment_weight,
                    },
                },
                output_dir / "best.pt",
            )
        else:
            stale += 1
            if args.early_stop_patience > 0 and stale >= args.early_stop_patience:
                print(f"early_stop epoch={epoch} best_valid_loss={best:.6f}", flush=True)
                break


def apply_transport(args) -> None:
    checkpoint = load_pt(args.checkpoint)
    model = LowRankInterventionTransport(
        checkpoint["embedding_dim"], checkpoint["rank"], checkpoint.get("dropout", 0.0)
    ).to(args.device)
    model.load_state_dict(checkpoint["state"])
    model.eval()
    items = load_pt(args.apply_input)
    output = []
    with torch.no_grad():
        for item in items:
            copied = dict(item)
            x = item["x"].clone()
            embedding_dim = int(checkpoint["embedding_dim"])
            if x.shape[-1] - int(checkpoint["meta_dim"]) != embedding_dim:
                raise ValueError(
                    f"apply input dimension mismatch: x={x.shape[-1]} "
                    f"embedding={embedding_dim} meta={checkpoint['meta_dim']}"
                )
            chunks = []
            for start in range(0, x.size(0), args.batch_size):
                chunk = x[start:start + args.batch_size, :embedding_dim].to(
                    args.device, dtype=torch.float32
                )
                chunks.append(model(chunk).cpu())
            x[:, :embedding_dim] = torch.cat(chunks, dim=0).to(dtype=x.dtype)
            copied["x"] = x
            copied["intervention_transport"] = str(args.checkpoint)
            output.append(copied)
    output_path = Path(args.apply_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, output_path)
    print(f"transported samples={len(output)} to {output_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train", "apply"], required=True)
    ap.add_argument("--train_clean", default="")
    ap.add_argument("--train_intervened", default="")
    ap.add_argument("--valid_clean", default="")
    ap.add_argument("--valid_intervened", default="")
    ap.add_argument("--output_dir", default="")
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--apply_input", default="")
    ap.add_argument("--apply_output", default="")
    ap.add_argument("--meta_dim", type=int, default=14)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-3)
    ap.add_argument("--cosine_weight", type=float, default=1.0)
    ap.add_argument("--normalized_mse_weight", type=float, default=1.0)
    ap.add_argument("--moment_weight", type=float, default=0.1)
    ap.add_argument("--early_stop_patience", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if args.mode == "train":
        required = [args.train_clean, args.train_intervened, args.valid_clean, args.valid_intervened, args.output_dir]
        if not all(required):
            ap.error("train mode requires train/valid clean/intervened paths and --output_dir")
        train(args)
    else:
        if not args.checkpoint or not args.apply_input or not args.apply_output:
            ap.error("apply mode requires --checkpoint, --apply_input, and --apply_output")
        apply_transport(args)


if __name__ == "__main__":
    main()
