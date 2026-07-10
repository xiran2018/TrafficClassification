#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from models.flow_transformer import FlowTransformerClassifier
from models.flow_graph_transformer import FlowGraphTransformerClassifier


class FlowAggregationHead(nn.Module):
    """Pool multiple window embeddings into one flow-level prediction."""

    def __init__(self, hidden_dim: int, num_classes: int, pooling: str = "attention"):
        super().__init__()
        self.pooling = pooling
        self.score = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
        self.cls = nn.Linear(hidden_dim, num_classes)

    def pool(self, h: torch.Tensor) -> torch.Tensor:
        if h.numel() == 0:
            raise ValueError("Cannot pool an empty flow.")
        if self.pooling == "mean":
            return h.mean(dim=0)
        weights = torch.softmax(self.score(h).squeeze(-1), dim=0)
        return torch.sum(h * weights.unsqueeze(-1), dim=0)

    def forward(self, h: torch.Tensor):
        emb = self.pool(h)
        return {"embedding": emb, "logits": self.cls(emb)}


def load_pt(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


class SeqDataset(Dataset):
    def __init__(self, path: str):
        self.data = load_pt(path)
    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i]


class GraphDataset(Dataset):
    def __init__(self, path: str):
        self.data = load_pt(path)
    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i]


def group_split_by_flow(ds: Dataset, valid_ratio: float, seed: int):
    """Split by flow_id, not by window, to avoid validation leakage."""
    flow_to_indices: Dict[str, List[int]] = defaultdict(list)
    for i in range(len(ds)):
        flow_to_indices[str(ds[i].get("flow_id", i))].append(i)
    flows = list(flow_to_indices.keys())
    rng = random.Random(seed)
    rng.shuffle(flows)
    n_val = max(1, int(len(flows) * valid_ratio)) if len(flows) > 1 and valid_ratio > 0 else 0
    val_flows = set(flows[:n_val])
    tr_idx, va_idx = [], []
    for f, idxs in flow_to_indices.items():
        (va_idx if f in val_flows else tr_idx).extend(idxs)
    return Subset(ds, tr_idx), Subset(ds, va_idx)


def build_flow_groups(ds: Dataset):
    flow_to_items: Dict[str, List[dict]] = defaultdict(list)
    flow_labels: Dict[str, int] = {}
    for i in range(len(ds)):
        item = ds[i]
        label = int(item.get("label", -1))
        if label < 0:
            continue
        flow_id = str(item.get("flow_id", i))
        flow_to_items[flow_id].append(item)
        flow_labels[flow_id] = label
    return [
        {"flow_id": flow_id, "label": flow_labels[flow_id], "items": items}
        for flow_id, items in flow_to_items.items()
        if items
    ]


def split_flow_groups(groups: List[dict], valid_ratio: float, seed: int):
    groups = list(groups)
    rng = random.Random(seed)
    rng.shuffle(groups)
    n_val = max(1, int(len(groups) * valid_ratio)) if len(groups) > 1 and valid_ratio > 0 else 0
    return groups[n_val:], groups[:n_val]


def iter_group_batches(groups: List[dict], batch_size: int, shuffle: bool = False):
    order = list(range(len(groups)))
    if shuffle:
        random.shuffle(order)
    batch_size = max(1, batch_size)
    for start in range(0, len(order), batch_size):
        yield [groups[i] for i in order[start:start + batch_size]]


def split_or_external_window_dataset(ds: Dataset, dataset_cls, args):
    if args.valid_dataset:
        return ds, dataset_cls(args.valid_dataset)
    return group_split_by_flow(ds, args.valid_ratio, args.seed)


def split_or_external_flow_groups(ds: Dataset, dataset_cls, args):
    groups = build_flow_groups(ds)
    if args.valid_dataset:
        return groups, build_flow_groups(dataset_cls(args.valid_dataset))
    return split_flow_groups(groups, args.valid_ratio, args.seed)


def collate_seq(batch):
    max_n = max(item["x"].shape[0] for item in batch)
    d = batch[0]["x"].shape[1]
    x = torch.zeros(len(batch), max_n, d)
    mask = torch.zeros(len(batch), max_n, dtype=torch.bool)
    labels = torch.tensor([int(item.get("label", -1)) for item in batch], dtype=torch.long)
    coh = torch.stack([item.get("coherence_label", torch.tensor(1)) for item in batch]).long()
    nd = torch.stack([item.get("next_direction", torch.tensor(-1)) for item in batch]).long()
    nl = torch.stack([item.get("next_length_bin", torch.tensor(-1)) for item in batch]).long()
    ni = torch.stack([item.get("next_iat_bin", torch.tensor(-1)) for item in batch]).long()
    flow_ids = [str(item.get("flow_id", "")) for item in batch]
    for b, item in enumerate(batch):
        n = item["x"].shape[0]
        x[b, :n] = item["x"]
        mask[b, :n] = True
    return {"x": x, "mask": mask, "label": labels, "coherence_label": coh, "next_direction": nd, "next_length_bin": nl, "next_iat_bin": ni, "flow_id": flow_ids}


def classification_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    valid = y >= 0
    if valid.any():
        return F.cross_entropy(logits[valid], y[valid])
    return torch.tensor(0.0, device=logits.device)


def next_aux_loss(out, targets: Dict[str, torch.Tensor], weight: float) -> torch.Tensor:
    if weight <= 0:
        return torch.tensor(0.0, device=out["logits"].device)
    loss = torch.tensor(0.0, device=out["logits"].device)
    for key, logit_key in [("next_direction", "next_direction_logits"), ("next_length_bin", "next_length_logits"), ("next_iat_bin", "next_iat_logits")]:
        y = targets[key].to(out["logits"].device)
        valid = y >= 0
        if valid.any():
            loss = loss + F.cross_entropy(out[logit_key][valid], y[valid])
    return loss * weight


def coherence_loss(out, y: torch.Tensor, weight: float) -> torch.Tensor:
    if weight <= 0 or "coherence_logits" not in out:
        return torch.tensor(0.0, device=out["logits"].device)
    return weight * F.cross_entropy(out["coherence_logits"], y.to(out["logits"].device))


def edge_aux_loss(out, item, device: str, weight: float) -> torch.Tensor:
    if weight <= 0 or out.get("edge_logits") is None:
        return torch.tensor(0.0, device=out["logits"].device)
    loss = torch.tensor(0.0, device=out["logits"].device)
    for target_name in ("ack_labels", "same_burst_labels", "retrans_labels"):
        labels = item.get(target_name)
        if labels is None:
            continue
        labels = labels.to(device)
        valid = labels >= 0
        if valid.any():
            loss = loss + F.cross_entropy(out["edge_logits"][valid], labels[valid])
    return loss * weight


def supervised_contrastive_loss(z: torch.Tensor, labels: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    if z.size(0) <= 1:
        return z.sum() * 0.0
    z = F.normalize(z.float(), p=2, dim=-1)
    labels = labels.view(-1, 1)
    logits = torch.matmul(z, z.T) / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    self_mask = torch.eye(z.size(0), dtype=torch.bool, device=z.device)
    pos_mask = torch.eq(labels, labels.T).to(z.device) & ~self_mask
    valid = pos_mask.sum(dim=1) > 0
    if valid.sum() == 0:
        return z.sum() * 0.0
    logits = logits.masked_fill(self_mask, -1e9)
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    mean_log_prob_pos = (pos_mask.float() * log_prob).sum(dim=1) / pos_mask.sum(dim=1).clamp(min=1)
    return -mean_log_prob_pos[valid].mean()


def train_seq(args):
    ds = SeqDataset(args.dataset)
    tr, va = split_or_external_window_dataset(ds, SeqDataset, args)
    dl = DataLoader(tr, batch_size=args.batch_size, shuffle=True, collate_fn=collate_seq)
    vdl = DataLoader(va, batch_size=args.batch_size, shuffle=False, collate_fn=collate_seq) if len(va) else None
    sample = ds[0]
    model = FlowTransformerClassifier(sample["x"].shape[1], args.num_classes, args.hidden_dim, args.num_layers, args.num_heads, args.dropout).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train(); total = 0.0; ok = 0; cnt = 0; skipped_no_grad = 0
        for batch in dl:
            x, mask, y = batch["x"].to(args.device), batch["mask"].to(args.device), batch["label"].to(args.device)
            out = model(x, mask)
            loss = classification_loss(out["logits"], y)
            loss = loss + next_aux_loss(out, {k: v.to(args.device) if torch.is_tensor(v) else v for k, v in batch.items()}, args.aux_weight)
            loss = loss + coherence_loss(out, batch["coherence_label"], args.coherence_weight)
            if not loss.requires_grad:
                skipped_no_grad += 1
                continue
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            valid = y >= 0
            if valid.any():
                total += float(loss.item()) * int(valid.sum())
                pred = out["logits"].argmax(-1)
                ok += int((pred[valid] == y[valid]).sum()); cnt += int(valid.sum())
        val_acc = evaluate_seq(model, vdl, args.device) if vdl else ok / max(1, cnt)
        print(f"epoch={epoch} train_loss={total/max(1,cnt):.4f} train_acc={ok/max(1,cnt):.4f} val_acc={val_acc:.4f} skipped_no_grad={skipped_no_grad}")
        if val_acc >= best:
            best = val_acc; save_ckpt(args, model, sample["x"].shape[1])


def mean_logits_by_flow(window_logits: torch.Tensor, owners: torch.Tensor, num_flows: int) -> torch.Tensor:
    flow_logits = []
    for flow_idx in range(num_flows):
        flow_logits.append(window_logits[owners == flow_idx].mean(dim=0))
    return torch.stack(flow_logits, dim=0)


def train_seq_flow(args):
    ds = SeqDataset(args.dataset)
    tr_groups, va_groups = split_or_external_flow_groups(ds, SeqDataset, args)
    sample = (tr_groups or va_groups)[0]["items"][0]
    model = FlowTransformerClassifier(sample["x"].shape[1], args.num_classes, args.hidden_dim, args.num_layers, args.num_heads, args.dropout).to(args.device)
    flow_head = FlowAggregationHead(args.hidden_dim, args.num_classes, pooling=args.flow_pooling).to(args.device)
    opt = torch.optim.AdamW(list(model.parameters()) + list(flow_head.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    best = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train(); total = 0.0; ok = 0; cnt = 0
        flow_head.train()
        for flow_batch in iter_group_batches(tr_groups, args.batch_size, shuffle=True):
            windows = []
            owners = []
            labels = []
            for flow_idx, group in enumerate(flow_batch):
                labels.append(int(group["label"]))
                for item in group["items"]:
                    windows.append(item)
                    owners.append(flow_idx)
            if not windows:
                continue
            batch = collate_seq(windows)
            out = model(batch["x"].to(args.device), batch["mask"].to(args.device))
            owners_t = torch.tensor(owners, dtype=torch.long, device=args.device)
            flow_logits = []
            flow_embs = []
            for flow_idx in range(len(flow_batch)):
                pooled = flow_head(out["embedding"][owners_t == flow_idx])
                flow_logits.append(pooled["logits"])
                flow_embs.append(pooled["embedding"])
            flow_logits = torch.stack(flow_logits, dim=0)
            flow_embs = torch.stack(flow_embs, dim=0)
            y = torch.tensor(labels, dtype=torch.long, device=args.device)
            loss = F.cross_entropy(flow_logits, y)
            if args.flow_contrastive_weight > 0:
                loss = loss + args.flow_contrastive_weight * supervised_contrastive_loss(flow_embs, y, args.flow_temperature)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(flow_head.parameters()), 1.0)
            opt.step()
            total += float(loss.item()) * len(labels)
            ok += int((flow_logits.argmax(-1) == y).sum()); cnt += len(labels)
        val_acc = evaluate_seq_flow(model, va_groups, args.device, args.batch_size, flow_head=flow_head) if va_groups else ok / max(1, cnt)
        print(f"epoch={epoch} train_loss={total/max(1,cnt):.4f} train_acc={ok/max(1,cnt):.4f} val_acc={val_acc:.4f} flows={cnt}")
        if val_acc >= best:
            best = val_acc; save_ckpt(args, model, sample["x"].shape[1], flow_head=flow_head)


@torch.no_grad()
def evaluate_seq(model, dl, device):
    if dl is None:
        return 0.0
    model.eval(); ok = 0; cnt = 0
    for batch in dl:
        out = model(batch["x"].to(device), batch["mask"].to(device))
        y = batch["label"].to(device)
        valid = y >= 0
        if valid.any():
            ok += int((out["logits"].argmax(-1)[valid] == y[valid]).sum()); cnt += int(valid.sum())
    return ok / max(1, cnt)


@torch.no_grad()
def evaluate_seq_flow(model, groups: List[dict], device: str, batch_size: int, flow_head: FlowAggregationHead | None = None):
    model.eval(); ok = 0; cnt = 0
    if flow_head is not None:
        flow_head.eval()
    for flow_batch in iter_group_batches(groups, batch_size, shuffle=False):
        windows = []
        owners = []
        labels = []
        for flow_idx, group in enumerate(flow_batch):
            labels.append(int(group["label"]))
            for item in group["items"]:
                windows.append(item)
                owners.append(flow_idx)
        if not windows:
            continue
        batch = collate_seq(windows)
        out = model(batch["x"].to(device), batch["mask"].to(device))
        owners_t = torch.tensor(owners, dtype=torch.long, device=device)
        if flow_head is None:
            flow_logits = mean_logits_by_flow(out["logits"], owners_t, len(flow_batch))
        else:
            flow_logits = torch.stack(
                [flow_head(out["embedding"][owners_t == flow_idx])["logits"] for flow_idx in range(len(flow_batch))],
                dim=0,
            )
        y = torch.tensor(labels, dtype=torch.long, device=device)
        ok += int((flow_logits.argmax(-1) == y).sum()); cnt += len(labels)
    return ok / max(1, cnt)


def train_graph(args):
    ds = GraphDataset(args.dataset)
    tr, va = split_or_external_window_dataset(ds, GraphDataset, args)
    sample = ds[0]
    edge_attr_dim = int(sample.get("edge_attr", torch.zeros((0, 4))).shape[1])
    model = FlowGraphTransformerClassifier(
        sample["x"].shape[1],
        args.num_classes,
        args.hidden_dim,
        args.num_layers,
        args.num_heads,
        edge_attr_dim=edge_attr_dim,
        dropout=args.dropout,
    ).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train(); total = 0.0; ok = 0; cnt = 0; skipped_no_grad = 0
        order = list(range(len(tr)))
        random.shuffle(order)
        for idx in order:
            item = tr[idx]
            x = item["x"].to(args.device)
            edge_index = item["edge_index"].to(args.device)
            edge_attr = item["edge_attr"].to(args.device)
            y = torch.tensor([int(item.get("label", -1))], dtype=torch.long, device=args.device)
            out = model(x, edge_index, edge_attr)
            loss = classification_loss(out["logits"], y)
            targets = {
                "next_direction": item.get("next_direction", torch.tensor(-1)).view(1),
                "next_length_bin": item.get("next_length_bin", torch.tensor(-1)).view(1),
                "next_iat_bin": item.get("next_iat_bin", torch.tensor(-1)).view(1),
            }
            loss = loss + next_aux_loss(out, targets, args.aux_weight)
            coh = item.get("coherence_label", torch.tensor(1)).view(1)
            loss = loss + coherence_loss(out, coh, args.coherence_weight)
            loss = loss + edge_aux_loss(out, item, args.device, args.aux_weight)
            if not loss.requires_grad:
                skipped_no_grad += 1
                continue
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            if y.item() >= 0:
                total += float(loss.item()); ok += int(out["logits"].argmax(-1).item() == int(y.item())); cnt += 1
        val_acc = evaluate_graph(model, va, args.device) if len(va) else ok / max(1, cnt)
        print(f"epoch={epoch} train_loss={total/max(1,cnt):.4f} train_acc={ok/max(1,cnt):.4f} val_acc={val_acc:.4f} skipped_no_grad={skipped_no_grad}")
        if val_acc >= best:
            best = val_acc; save_ckpt(args, model, sample["x"].shape[1], edge_attr_dim=edge_attr_dim)


def train_graph_flow(args):
    ds = GraphDataset(args.dataset)
    tr_groups, va_groups = split_or_external_flow_groups(ds, GraphDataset, args)
    sample = (tr_groups or va_groups)[0]["items"][0]
    edge_attr_dim = int(sample.get("edge_attr", torch.zeros((0, 4))).shape[1])
    model = FlowGraphTransformerClassifier(
        sample["x"].shape[1],
        args.num_classes,
        args.hidden_dim,
        args.num_layers,
        args.num_heads,
        edge_attr_dim=edge_attr_dim,
        dropout=args.dropout,
    ).to(args.device)
    flow_head = FlowAggregationHead(args.hidden_dim, args.num_classes, pooling=args.flow_pooling).to(args.device)
    opt = torch.optim.AdamW(list(model.parameters()) + list(flow_head.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    best = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train(); total = 0.0; ok = 0; cnt = 0
        flow_head.train()
        for flow_batch in iter_group_batches(tr_groups, args.batch_size, shuffle=True):
            flow_logits = []
            flow_embs = []
            labels = []
            for group in flow_batch:
                window_embs = []
                for item in group["items"]:
                    out = model(item["x"].to(args.device), item["edge_index"].to(args.device), item["edge_attr"].to(args.device))
                    window_embs.append(out["embedding"])
                if window_embs:
                    pooled = flow_head(torch.stack(window_embs, dim=0))
                    flow_logits.append(pooled["logits"])
                    flow_embs.append(pooled["embedding"])
                    labels.append(int(group["label"]))
            if not flow_logits:
                continue
            logits = torch.stack(flow_logits, dim=0)
            embs = torch.stack(flow_embs, dim=0)
            y = torch.tensor(labels, dtype=torch.long, device=args.device)
            loss = F.cross_entropy(logits, y)
            if args.flow_contrastive_weight > 0:
                loss = loss + args.flow_contrastive_weight * supervised_contrastive_loss(embs, y, args.flow_temperature)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(flow_head.parameters()), 1.0)
            opt.step()
            total += float(loss.item()) * len(labels)
            ok += int((logits.argmax(-1) == y).sum()); cnt += len(labels)
        val_acc = evaluate_graph_flow(model, va_groups, args.device, flow_head=flow_head) if va_groups else ok / max(1, cnt)
        print(f"epoch={epoch} train_loss={total/max(1,cnt):.4f} train_acc={ok/max(1,cnt):.4f} val_acc={val_acc:.4f} flows={cnt}")
        if val_acc >= best:
            best = val_acc; save_ckpt(args, model, sample["x"].shape[1], edge_attr_dim=edge_attr_dim, flow_head=flow_head)


@torch.no_grad()
def evaluate_graph(model, ds, device):
    model.eval(); ok = 0; cnt = 0
    for item in ds:
        y = int(item.get("label", -1))
        if y < 0:
            continue
        out = model(item["x"].to(device), item["edge_index"].to(device), item["edge_attr"].to(device))
        ok += int(out["logits"].argmax(-1).item() == y); cnt += 1
    return ok / max(1, cnt)


@torch.no_grad()
def evaluate_graph_flow(model, groups: List[dict], device: str, flow_head: FlowAggregationHead | None = None):
    model.eval(); ok = 0; cnt = 0
    if flow_head is not None:
        flow_head.eval()
    for group in groups:
        window_logits = []
        window_embs = []
        for item in group["items"]:
            out = model(item["x"].to(device), item["edge_index"].to(device), item["edge_attr"].to(device))
            window_logits.append(out["logits"].squeeze(0))
            window_embs.append(out["embedding"])
        if not window_logits:
            continue
        if flow_head is None:
            logits = torch.stack(window_logits, dim=0).mean(dim=0)
        else:
            logits = flow_head(torch.stack(window_embs, dim=0))["logits"]
        y = int(group["label"])
        ok += int(logits.argmax(-1).item() == y); cnt += 1
    return ok / max(1, cnt)


def save_ckpt(args, model, input_dim, edge_attr_dim=None, flow_head: FlowAggregationHead | None = None):
    os.makedirs(args.output_dir, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "model_type": args.model_type,
        "input_dim": input_dim,
        "num_classes": args.num_classes,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "dropout": args.dropout,
        "train_level": args.train_level,
        "flow_pooling": args.flow_pooling,
        "flow_contrastive_weight": args.flow_contrastive_weight,
        "flow_temperature": args.flow_temperature,
    }
    if edge_attr_dim is not None:
        payload["edge_attr_dim"] = edge_attr_dim
    if flow_head is not None:
        payload["flow_head_state"] = flow_head.state_dict()
    torch.save(payload, Path(args.output_dir) / "best.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_type", choices=["seq", "graph"], required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--valid_dataset", default="", help="Optional external validation dataset. If set, train uses the full --dataset and best.pt is selected on this split.")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--num_classes", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--num_heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--aux_weight", type=float, default=0.1)
    ap.add_argument("--coherence_weight", type=float, default=0.1)
    ap.add_argument("--train_level", choices=["window", "flow"], default="window", help="window: classify each window; flow: pool window embeddings and optimize flow labels directly.")
    ap.add_argument("--flow_pooling", choices=["attention", "mean"], default="attention", help="Pooling used by --train_level flow over window embeddings.")
    ap.add_argument("--flow_contrastive_weight", type=float, default=0.0, help="Weight for supervised contrastive loss over flow embeddings in --train_level flow.")
    ap.add_argument("--flow_temperature", type=float, default=0.07)
    ap.add_argument("--valid_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    if args.train_level == "flow" and (args.aux_weight > 0 or args.coherence_weight > 0):
        print("WARNING: --train_level flow optimizes flow-level classification and optional flow contrastive loss only; aux/coherence weights are ignored in this mode.")
    if args.model_type == "seq":
        train_seq_flow(args) if args.train_level == "flow" else train_seq(args)
    else:
        train_graph_flow(args) if args.train_level == "flow" else train_graph(args)


if __name__ == "__main__":
    main()
