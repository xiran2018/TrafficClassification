#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
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


VPN_APP_GROUPS = "0,2,5,6,10,14;1,4;3,8,9;7,11,13,15;12"


def sinusoidal_position_encoding(length: int, dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    pos = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
    div = torch.exp(torch.arange(0, dim, 2, device=device, dtype=dtype) * (-math.log(10000.0) / dim))
    pe = torch.zeros(length, dim, device=device, dtype=dtype)
    pe[:, 0::2] = torch.sin(pos * div)
    if dim > 1:
        pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
    return pe


class FlowAggregationHead(nn.Module):
    """Pool multiple window embeddings into one flow-level prediction."""

    def __init__(
        self,
        hidden_dim: int,
        num_classes: int,
        pooling: str = "attention",
        num_coarse_classes: int = 0,
        class_groups: Sequence[Sequence[int]] | None = None,
        hierarchical_mode: str = "logit",
        flow_transformer_layers: int = 1,
        flow_transformer_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.pooling = pooling
        self.num_classes = num_classes
        self.num_coarse_classes = num_coarse_classes
        self.hierarchical_mode = hierarchical_mode
        self.class_groups = [list(group) for group in (class_groups or [])]
        self.score = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
        if pooling == "transformer":
            if hidden_dim % flow_transformer_heads != 0:
                raise ValueError("--hidden_dim must be divisible by --flow_transformer_heads")
            enc_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=flow_transformer_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
                norm_first=True,
            )
            self.flow_encoder = nn.TransformerEncoder(enc_layer, num_layers=flow_transformer_layers)
        else:
            self.flow_encoder = None
        self.cls = nn.Linear(hidden_dim, num_classes)
        self.coarse_cls = nn.Linear(hidden_dim, num_coarse_classes) if num_coarse_classes > 0 else None
        self.use_expert_heads = hierarchical_mode == "expert" and self.coarse_cls is not None and bool(self.class_groups)
        if self.use_expert_heads and len(self.class_groups) != num_coarse_classes:
            raise ValueError("class_groups must match num_coarse_classes when --hierarchical_mode expert is used")
        self.expert_heads = nn.ModuleList(
            [nn.Linear(hidden_dim, len(group)) for group in self.class_groups]
        ) if self.use_expert_heads else nn.ModuleList()
        if pooling == "late_fusion":
            self.fusion_logit_scale = nn.Parameter(torch.tensor(1.0))

    def pool(self, h: torch.Tensor) -> torch.Tensor:
        if h.numel() == 0:
            raise ValueError("Cannot pool an empty flow.")
        if self.pooling == "mean":
            return h.mean(dim=0)
        if self.pooling == "transformer":
            pos = sinusoidal_position_encoding(h.size(0), h.size(1), h.device, h.dtype)
            h = self.flow_encoder((h + pos).unsqueeze(0)).squeeze(0)
        weights = torch.softmax(self.score(h).squeeze(-1), dim=0)
        return torch.sum(h * weights.unsqueeze(-1), dim=0)

    def expert_logits(self, emb: torch.Tensor, coarse_logits: torch.Tensor) -> torch.Tensor:
        coarse_log_prob = F.log_softmax(coarse_logits, dim=-1)
        logits = emb.new_full((self.num_classes,), -1e9)
        for coarse_idx, (group, head) in enumerate(zip(self.class_groups, self.expert_heads)):
            fine_log_prob = F.log_softmax(head(emb), dim=-1)
            for local_idx, class_id in enumerate(group):
                logits[class_id] = coarse_log_prob[coarse_idx] + fine_log_prob[local_idx]
        return logits

    def forward(self, h: torch.Tensor, window_logits: torch.Tensor | None = None):
        emb = self.pool(h)
        coarse_logits = self.coarse_cls(emb) if self.coarse_cls is not None else None
        if self.use_expert_heads:
            logits = self.expert_logits(emb, coarse_logits)
        else:
            logits = self.cls(emb)
        if self.pooling == "late_fusion" and window_logits is not None:
            logits = logits + self.fusion_logit_scale * window_logits.mean(dim=0)
        out = {"embedding": emb, "logits": logits}
        if coarse_logits is not None:
            out["coarse_logits"] = coarse_logits
        return out


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


def iter_balanced_group_batches(
    groups: List[dict],
    batch_size: int,
    classes_per_batch: int = 0,
    samples_per_class: int = 2,
):
    """Sample flow batches with repeated classes so SupCon has positives."""
    label_to_groups: Dict[int, List[dict]] = defaultdict(list)
    for group in groups:
        label_to_groups[int(group["label"])].append(group)
    labels = list(label_to_groups.keys())
    if not labels:
        return
    samples_per_class = max(2, samples_per_class)
    classes_per_batch = classes_per_batch if classes_per_batch > 0 else max(1, batch_size // samples_per_class)
    classes_per_batch = max(1, min(classes_per_batch, len(labels)))
    effective_batch = max(1, classes_per_batch * samples_per_class)
    steps = max(1, int(np.ceil(len(groups) / effective_batch)))
    for _ in range(steps):
        if len(labels) >= classes_per_batch:
            chosen_labels = random.sample(labels, classes_per_batch)
        else:
            chosen_labels = [random.choice(labels) for _ in range(classes_per_batch)]
        batch = []
        for label in chosen_labels:
            candidates = label_to_groups[label]
            if len(candidates) >= samples_per_class:
                batch.extend(random.sample(candidates, samples_per_class))
            else:
                batch.extend(random.choice(candidates) for _ in range(samples_per_class))
        random.shuffle(batch)
        yield batch


def iter_train_flow_batches(groups: List[dict], args):
    if args.balanced_flow_batches:
        yield from iter_balanced_group_batches(
            groups,
            args.batch_size,
            classes_per_batch=args.classes_per_batch,
            samples_per_class=args.samples_per_class,
        )
    else:
        yield from iter_group_batches(groups, args.batch_size, shuffle=True)


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


def class_weighted_loss(logits: torch.Tensor, y: torch.Tensor, class_weight: torch.Tensor | None = None) -> torch.Tensor:
    valid = y >= 0
    if valid.any():
        return F.cross_entropy(logits[valid], y[valid], weight=class_weight)
    return torch.tensor(0.0, device=logits.device)


def compute_class_weights(labels: Sequence[int], num_classes: int, device: str, scheme: str = "none", beta: float = 0.9999):
    if scheme == "none":
        return None
    counts = torch.zeros(num_classes, dtype=torch.float32)
    for label in labels:
        label = int(label)
        if 0 <= label < num_classes:
            counts[label] += 1.0
    weights = torch.zeros_like(counts)
    present = counts > 0
    if not present.any():
        return None
    if scheme == "inverse":
        weights[present] = 1.0 / counts[present]
    elif scheme == "effective":
        beta = min(max(float(beta), 0.0), 0.999999)
        weights[present] = (1.0 - beta) / (1.0 - torch.pow(torch.tensor(beta), counts[present]))
    else:
        raise ValueError(f"Unknown class weighting scheme: {scheme}")
    weights[present] = weights[present] / weights[present].mean().clamp(min=1e-12)
    return weights.to(device)


def dataset_labels(ds: Dataset):
    labels = []
    for i in range(len(ds)):
        label = int(ds[i].get("label", -1))
        if label >= 0:
            labels.append(label)
    return labels


def parse_label_groups(spec: str, num_classes: int) -> List[List[int]]:
    spec = (spec or "none").strip()
    if spec in {"", "none"}:
        return []
    if spec == "vpn_app":
        spec = VPN_APP_GROUPS
    groups = []
    assigned = set()
    for raw_group in spec.split(";"):
        group = []
        for raw_label in raw_group.split(","):
            raw_label = raw_label.strip()
            if not raw_label:
                continue
            label = int(raw_label)
            if 0 <= label < num_classes and label not in group:
                group.append(label)
        if group:
            groups.append(group)
            assigned.update(group)
    for label in range(num_classes):
        if label not in assigned:
            groups.append([label])
    return groups


def build_class_to_coarse(spec: str, num_classes: int, device: str | torch.device = "cpu"):
    groups = parse_label_groups(spec, num_classes)
    if not groups:
        return None, 0
    class_to_coarse = torch.empty(num_classes, dtype=torch.long)
    class_to_coarse.fill_(-1)
    for coarse_id, group in enumerate(groups):
        for label in group:
            class_to_coarse[label] = coarse_id
    missing = class_to_coarse < 0
    if missing.any():
        start = len(groups)
        for offset, label in enumerate(torch.nonzero(missing, as_tuple=False).view(-1).tolist()):
            class_to_coarse[label] = start + offset
    return class_to_coarse.to(device), int(class_to_coarse.max().item() + 1)


def build_confusion_weights(
    spec: str,
    num_classes: int,
    device: str | torch.device = "cpu",
    metrics_json: str = "",
    level: str = "flow",
    power: float = 1.0,
):
    weights = torch.zeros(num_classes, num_classes, dtype=torch.float32)
    if metrics_json:
        with open(metrics_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        prefix = "flow" if level == "flow" else "window"
        y_true = data.get(f"{prefix}_y_true", [])
        y_pred = data.get(f"{prefix}_y_pred", [])
        counts = torch.zeros(num_classes, num_classes, dtype=torch.float32)
        for true, pred in zip(y_true, y_pred):
            true = int(true)
            pred = int(pred)
            if 0 <= true < num_classes and 0 <= pred < num_classes and true != pred:
                counts[true, pred] += 1.0
        weights = counts + counts.T
        if weights.max() > 0:
            weights = weights / weights.max().clamp(min=1.0)
    if weights.max() <= 0:
        groups = parse_label_groups(spec, num_classes)
        for group in groups:
            for a in group:
                for b in group:
                    if a != b:
                        weights[a, b] = 1.0
    weights.fill_diagonal_(0.0)
    if power != 1.0 and weights.max() > 0:
        weights = weights.clamp(min=0.0).pow(power)
    return weights.to(device) if weights.max() > 0 else None


def apply_hierarchical_logits(
    fine_logits: torch.Tensor,
    coarse_logits: torch.Tensor | None,
    class_to_coarse: torch.Tensor | None,
    weight: float,
) -> torch.Tensor:
    if coarse_logits is None or class_to_coarse is None or weight <= 0:
        return fine_logits
    class_to_coarse = class_to_coarse.to(fine_logits.device)
    if fine_logits.dim() == 1:
        coarse_log_prob = F.log_softmax(coarse_logits, dim=-1)
        return fine_logits + weight * coarse_log_prob[class_to_coarse]
    coarse_log_prob = F.log_softmax(coarse_logits, dim=-1)
    return fine_logits + weight * coarse_log_prob[:, class_to_coarse]


def needs_hierarchical_head(args) -> bool:
    return args.hierarchical_mode == "expert" or args.hierarchical_weight > 0 or args.hierarchical_logit_weight > 0


def build_hierarchical_mapping(args):
    if not needs_hierarchical_head(args):
        return None, 0, []
    class_to_coarse, num_coarse_classes = build_class_to_coarse(args.coarse_groups, args.num_classes, args.device)
    class_groups = parse_label_groups(args.coarse_groups, args.num_classes)
    if args.hierarchical_mode == "expert" and (class_to_coarse is None or num_coarse_classes <= 0):
        raise ValueError("--hierarchical_mode expert requires non-empty --coarse_groups")
    return class_to_coarse, num_coarse_classes, class_groups


def effective_hierarchical_logit_weight(args) -> float:
    return 0.0 if args.hierarchical_mode == "expert" else args.hierarchical_logit_weight


def classification_metrics_from_lists(y_true: Sequence[int], y_pred: Sequence[int], num_classes: int):
    if not y_true:
        return {"accuracy": 0.0, "macro_f1": 0.0}
    correct = sum(int(a == b) for a, b in zip(y_true, y_pred))
    f1s = []
    for label in range(num_classes):
        tp = sum(1 for a, b in zip(y_true, y_pred) if a == label and b == label)
        fp = sum(1 for a, b in zip(y_true, y_pred) if a != label and b == label)
        fn = sum(1 for a, b in zip(y_true, y_pred) if a == label and b != label)
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1s.append(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))
    return {"accuracy": correct / len(y_true), "macro_f1": float(np.mean(f1s))}


def selected_metric_value(metrics: Dict[str, float], metric_name: str) -> float:
    aliases = {
        "accuracy": "accuracy",
        "acc": "accuracy",
        "flow_acc": "accuracy",
        "window_acc": "accuracy",
        "macro_f1": "macro_f1",
        "flow_macro_f1": "macro_f1",
        "window_macro_f1": "macro_f1",
    }
    return float(metrics[aliases[metric_name]])


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


def supervised_contrastive_loss(
    z: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07,
    negative_mask: torch.Tensor | None = None,
    negative_weight: torch.Tensor | None = None,
) -> torch.Tensor:
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
    if negative_weight is not None:
        denom_weight = pos_mask.float() + negative_weight.to(z.device, dtype=logits.dtype)
    elif negative_mask is None:
        denom_weight = (~self_mask).float()
    else:
        denom_weight = (pos_mask | negative_mask.to(z.device)).float()
    denom_weight = denom_weight.masked_fill(self_mask, 0.0)
    weighted_logits = logits + torch.log(denom_weight.clamp(min=1e-12))
    weighted_logits = weighted_logits.masked_fill(denom_weight <= 0, -1e9)
    log_prob = logits - torch.logsumexp(weighted_logits, dim=1, keepdim=True)
    mean_log_prob_pos = (pos_mask.float() * log_prob).sum(dim=1) / pos_mask.sum(dim=1).clamp(min=1)
    return -mean_log_prob_pos[valid].mean()


def contrastive_loss(z: torch.Tensor, labels: torch.Tensor, args, confusion_weights: torch.Tensor | None = None) -> torch.Tensor:
    negative_mask = None
    negative_weight = None
    if args.contrastive_mode in {"confusion", "confusion_weighted"} and confusion_weights is not None:
        label_ids = labels.long().clamp(min=0, max=confusion_weights.size(0) - 1)
        negative_weight = confusion_weights[label_ids][:, label_ids]
        negative_weight = negative_weight * (labels.view(-1, 1) != labels.view(1, -1)).float()
        if args.contrastive_mode == "confusion":
            negative_mask = negative_weight > 0
            negative_weight = None
    return supervised_contrastive_loss(
        z,
        labels,
        args.flow_temperature,
        negative_mask=negative_mask,
        negative_weight=negative_weight,
    )


def train_seq(args):
    ds = SeqDataset(args.dataset)
    tr, va = split_or_external_window_dataset(ds, SeqDataset, args)
    dl = DataLoader(tr, batch_size=args.batch_size, shuffle=True, collate_fn=collate_seq)
    vdl = DataLoader(va, batch_size=args.batch_size, shuffle=False, collate_fn=collate_seq) if len(va) else None
    sample = ds[0]
    model = FlowTransformerClassifier(sample["x"].shape[1], args.num_classes, args.hidden_dim, args.num_layers, args.num_heads, args.dropout).to(args.device)
    class_weight = compute_class_weights(dataset_labels(tr), args.num_classes, args.device, args.class_weighting, args.class_weight_beta)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train(); total = 0.0; ok = 0; cnt = 0; skipped_no_grad = 0
        for batch in dl:
            x = apply_meta_dropout(batch["x"].to(args.device), args.meta_dropout_prob, args.meta_feature_dim)
            mask, y = batch["mask"].to(args.device), batch["label"].to(args.device)
            out = model(x, mask)
            loss = class_weighted_loss(out["logits"], y, class_weight)
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
        val_metrics = evaluate_seq(model, vdl, args.device, args.num_classes) if vdl else {"accuracy": ok / max(1, cnt), "macro_f1": ok / max(1, cnt)}
        select_score = selected_metric_value(val_metrics, args.select_metric)
        print(f"epoch={epoch} train_loss={total/max(1,cnt):.4f} train_acc={ok/max(1,cnt):.4f} val_acc={val_metrics['accuracy']:.4f} val_macro_f1={val_metrics['macro_f1']:.4f} select={args.select_metric}:{select_score:.4f} skipped_no_grad={skipped_no_grad}")
        if select_score > best:
            best = select_score; save_ckpt(args, model, sample["x"].shape[1])


def mean_logits_by_flow(window_logits: torch.Tensor, owners: torch.Tensor, num_flows: int) -> torch.Tensor:
    flow_logits = []
    for flow_idx in range(num_flows):
        flow_logits.append(window_logits[owners == flow_idx].mean(dim=0))
    return torch.stack(flow_logits, dim=0)


def apply_meta_dropout(x: torch.Tensor, prob: float, meta_dim: int) -> torch.Tensor:
    if prob <= 0 or meta_dim <= 0:
        return x
    meta_dim = min(meta_dim, x.shape[-1])
    if meta_dim <= 0:
        return x
    x = x.clone()
    x[..., -meta_dim:] = F.dropout(x[..., -meta_dim:], p=prob, training=True)
    return x


def apply_edge_attr_dropout(edge_attr: torch.Tensor, prob: float) -> torch.Tensor:
    if prob <= 0 or edge_attr.numel() == 0 or edge_attr.shape[-1] <= 1:
        return edge_attr
    edge_attr = edge_attr.clone()
    edge_attr[:, 1:] = F.dropout(edge_attr[:, 1:], p=prob, training=True)
    return edge_attr


def train_seq_flow(args):
    ds = SeqDataset(args.dataset)
    tr_groups, va_groups = split_or_external_flow_groups(ds, SeqDataset, args)
    sample = (tr_groups or va_groups)[0]["items"][0]
    class_to_coarse, num_coarse_classes, class_groups = build_hierarchical_mapping(args)
    confusion_weights = build_confusion_weights(
        args.confusion_groups,
        args.num_classes,
        args.device,
        metrics_json=args.confusion_matrix_json,
        level=args.confusion_matrix_level,
        power=args.confusion_weight_power,
    ) if args.contrastive_mode in {"confusion", "confusion_weighted"} else None
    model = FlowTransformerClassifier(sample["x"].shape[1], args.num_classes, args.hidden_dim, args.num_layers, args.num_heads, args.dropout).to(args.device)
    flow_head = FlowAggregationHead(
        args.hidden_dim,
        args.num_classes,
        pooling=args.flow_pooling,
        num_coarse_classes=num_coarse_classes,
        class_groups=class_groups if args.hierarchical_mode == "expert" else None,
        hierarchical_mode=args.hierarchical_mode,
        flow_transformer_layers=args.flow_transformer_layers,
        flow_transformer_heads=args.flow_transformer_heads,
        dropout=args.dropout,
    ).to(args.device)
    class_weight = compute_class_weights([g["label"] for g in tr_groups], args.num_classes, args.device, args.class_weighting, args.class_weight_beta)
    opt = torch.optim.AdamW(list(model.parameters()) + list(flow_head.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    best = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train(); total = 0.0; ok = 0; cnt = 0
        flow_head.train()
        for flow_batch in iter_train_flow_batches(tr_groups, args):
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
            x = apply_meta_dropout(batch["x"].to(args.device), args.meta_dropout_prob, args.meta_feature_dim)
            out = model(x, batch["mask"].to(args.device))
            owners_t = torch.tensor(owners, dtype=torch.long, device=args.device)
            window_labels = batch["label"].to(args.device)
            flow_logits = []
            flow_coarse_logits = []
            flow_embs = []
            for flow_idx in range(len(flow_batch)):
                win_mask = owners_t == flow_idx
                pooled = flow_head(out["embedding"][win_mask], window_logits=out["logits"][win_mask])
                flow_logits.append(pooled["logits"])
                if pooled.get("coarse_logits") is not None:
                    flow_coarse_logits.append(pooled["coarse_logits"])
                flow_embs.append(pooled["embedding"])
            flow_logits = torch.stack(flow_logits, dim=0)
            coarse_logits = torch.stack(flow_coarse_logits, dim=0) if flow_coarse_logits else None
            flow_logits = apply_hierarchical_logits(flow_logits, coarse_logits, class_to_coarse, effective_hierarchical_logit_weight(args))
            flow_embs = torch.stack(flow_embs, dim=0)
            y = torch.tensor(labels, dtype=torch.long, device=args.device)
            loss = class_weighted_loss(flow_logits, y, class_weight)
            if coarse_logits is not None and args.hierarchical_weight > 0:
                loss = loss + args.hierarchical_weight * F.cross_entropy(coarse_logits, class_to_coarse[y])
            if args.window_loss_weight > 0:
                loss = loss + args.window_loss_weight * class_weighted_loss(out["logits"], window_labels, class_weight)
            if args.flow_contrastive_weight > 0:
                loss = loss + args.flow_contrastive_weight * contrastive_loss(flow_embs, y, args, confusion_weights)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(flow_head.parameters()), 1.0)
            opt.step()
            total += float(loss.item()) * len(labels)
            ok += int((flow_logits.argmax(-1) == y).sum()); cnt += len(labels)
        val_metrics = evaluate_seq_flow(
            model,
            va_groups,
            args.device,
            args.batch_size,
            args.num_classes,
            flow_head=flow_head,
            class_to_coarse=class_to_coarse,
            hierarchical_logit_weight=effective_hierarchical_logit_weight(args),
            hierarchical_mode=args.hierarchical_mode,
        ) if va_groups else {"accuracy": ok / max(1, cnt), "macro_f1": ok / max(1, cnt)}
        select_score = selected_metric_value(val_metrics, args.select_metric)
        print(f"epoch={epoch} train_loss={total/max(1,cnt):.4f} train_acc={ok/max(1,cnt):.4f} val_acc={val_metrics['accuracy']:.4f} val_macro_f1={val_metrics['macro_f1']:.4f} select={args.select_metric}:{select_score:.4f} flows={cnt}")
        if select_score > best:
            best = select_score; save_ckpt(args, model, sample["x"].shape[1], flow_head=flow_head, num_coarse_classes=num_coarse_classes)


@torch.no_grad()
def evaluate_seq(model, dl, device, num_classes: int):
    if dl is None:
        return {"accuracy": 0.0, "macro_f1": 0.0}
    model.eval(); y_true = []; y_pred = []
    for batch in dl:
        out = model(batch["x"].to(device), batch["mask"].to(device))
        y = batch["label"].to(device)
        valid = y >= 0
        if valid.any():
            pred = out["logits"].argmax(-1)
            y_true.extend(y[valid].detach().cpu().tolist())
            y_pred.extend(pred[valid].detach().cpu().tolist())
    return classification_metrics_from_lists(y_true, y_pred, num_classes)


@torch.no_grad()
def evaluate_seq_flow(
    model,
    groups: List[dict],
    device: str,
    batch_size: int,
    num_classes: int,
    flow_head: FlowAggregationHead | None = None,
    class_to_coarse: torch.Tensor | None = None,
    hierarchical_logit_weight: float = 0.0,
    hierarchical_mode: str = "logit",
):
    model.eval(); y_true = []; y_pred = []
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
            pooled_outputs = [
                flow_head(
                    out["embedding"][owners_t == flow_idx],
                    window_logits=out["logits"][owners_t == flow_idx],
                )
                for flow_idx in range(len(flow_batch))
            ]
            flow_logits = torch.stack([pooled["logits"] for pooled in pooled_outputs], dim=0)
            if pooled_outputs and pooled_outputs[0].get("coarse_logits") is not None:
                coarse_logits = torch.stack([pooled["coarse_logits"] for pooled in pooled_outputs], dim=0)
                flow_logits = apply_hierarchical_logits(flow_logits, coarse_logits, class_to_coarse, hierarchical_logit_weight)
        y = torch.tensor(labels, dtype=torch.long, device=device)
        y_true.extend(y.detach().cpu().tolist())
        y_pred.extend(flow_logits.argmax(-1).detach().cpu().tolist())
    return classification_metrics_from_lists(y_true, y_pred, num_classes)


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
    class_weight = compute_class_weights(dataset_labels(tr), args.num_classes, args.device, args.class_weighting, args.class_weight_beta)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train(); total = 0.0; ok = 0; cnt = 0; skipped_no_grad = 0
        order = list(range(len(tr)))
        random.shuffle(order)
        for idx in order:
            item = tr[idx]
            x = apply_meta_dropout(item["x"].to(args.device), args.meta_dropout_prob, args.meta_feature_dim)
            edge_index = item["edge_index"].to(args.device)
            edge_attr = apply_edge_attr_dropout(item["edge_attr"].to(args.device), args.edge_attr_dropout_prob)
            y = torch.tensor([int(item.get("label", -1))], dtype=torch.long, device=args.device)
            out = model(x, edge_index, edge_attr)
            loss = class_weighted_loss(out["logits"], y, class_weight)
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
        val_metrics = evaluate_graph(model, va, args.device, args.num_classes) if len(va) else {"accuracy": ok / max(1, cnt), "macro_f1": ok / max(1, cnt)}
        select_score = selected_metric_value(val_metrics, args.select_metric)
        print(f"epoch={epoch} train_loss={total/max(1,cnt):.4f} train_acc={ok/max(1,cnt):.4f} val_acc={val_metrics['accuracy']:.4f} val_macro_f1={val_metrics['macro_f1']:.4f} select={args.select_metric}:{select_score:.4f} skipped_no_grad={skipped_no_grad}")
        if select_score > best:
            best = select_score; save_ckpt(args, model, sample["x"].shape[1], edge_attr_dim=edge_attr_dim)


def train_graph_flow(args):
    ds = GraphDataset(args.dataset)
    tr_groups, va_groups = split_or_external_flow_groups(ds, GraphDataset, args)
    sample = (tr_groups or va_groups)[0]["items"][0]
    edge_attr_dim = int(sample.get("edge_attr", torch.zeros((0, 4))).shape[1])
    class_to_coarse, num_coarse_classes, class_groups = build_hierarchical_mapping(args)
    confusion_weights = build_confusion_weights(
        args.confusion_groups,
        args.num_classes,
        args.device,
        metrics_json=args.confusion_matrix_json,
        level=args.confusion_matrix_level,
        power=args.confusion_weight_power,
    ) if args.contrastive_mode in {"confusion", "confusion_weighted"} else None
    model = FlowGraphTransformerClassifier(
        sample["x"].shape[1],
        args.num_classes,
        args.hidden_dim,
        args.num_layers,
        args.num_heads,
        edge_attr_dim=edge_attr_dim,
        dropout=args.dropout,
    ).to(args.device)
    flow_head = FlowAggregationHead(
        args.hidden_dim,
        args.num_classes,
        pooling=args.flow_pooling,
        num_coarse_classes=num_coarse_classes,
        class_groups=class_groups if args.hierarchical_mode == "expert" else None,
        hierarchical_mode=args.hierarchical_mode,
        flow_transformer_layers=args.flow_transformer_layers,
        flow_transformer_heads=args.flow_transformer_heads,
        dropout=args.dropout,
    ).to(args.device)
    class_weight = compute_class_weights([g["label"] for g in tr_groups], args.num_classes, args.device, args.class_weighting, args.class_weight_beta)
    opt = torch.optim.AdamW(list(model.parameters()) + list(flow_head.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    best = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train(); total = 0.0; ok = 0; cnt = 0
        flow_head.train()
        for flow_batch in iter_train_flow_batches(tr_groups, args):
            flow_logits = []
            flow_coarse_logits = []
            flow_embs = []
            window_logits_all = []
            window_labels = []
            labels = []
            for group in flow_batch:
                window_embs = []
                window_logits = []
                for item in group["items"]:
                    x = apply_meta_dropout(item["x"].to(args.device), args.meta_dropout_prob, args.meta_feature_dim)
                    edge_attr = apply_edge_attr_dropout(item["edge_attr"].to(args.device), args.edge_attr_dropout_prob)
                    out = model(x, item["edge_index"].to(args.device), edge_attr)
                    window_embs.append(out["embedding"])
                    window_logits.append(out["logits"].squeeze(0))
                    window_logits_all.append(out["logits"].squeeze(0))
                    window_labels.append(int(item.get("label", -1)))
                if window_embs:
                    pooled = flow_head(torch.stack(window_embs, dim=0), window_logits=torch.stack(window_logits, dim=0))
                    flow_logits.append(pooled["logits"])
                    if pooled.get("coarse_logits") is not None:
                        flow_coarse_logits.append(pooled["coarse_logits"])
                    flow_embs.append(pooled["embedding"])
                    labels.append(int(group["label"]))
            if not flow_logits:
                continue
            logits = torch.stack(flow_logits, dim=0)
            coarse_logits = torch.stack(flow_coarse_logits, dim=0) if flow_coarse_logits else None
            logits = apply_hierarchical_logits(logits, coarse_logits, class_to_coarse, effective_hierarchical_logit_weight(args))
            embs = torch.stack(flow_embs, dim=0)
            y = torch.tensor(labels, dtype=torch.long, device=args.device)
            loss = class_weighted_loss(logits, y, class_weight)
            if coarse_logits is not None and args.hierarchical_weight > 0:
                loss = loss + args.hierarchical_weight * F.cross_entropy(coarse_logits, class_to_coarse[y])
            if args.window_loss_weight > 0 and window_logits_all:
                win_logits = torch.stack(window_logits_all, dim=0)
                win_y = torch.tensor(window_labels, dtype=torch.long, device=args.device)
                loss = loss + args.window_loss_weight * class_weighted_loss(win_logits, win_y, class_weight)
            if args.flow_contrastive_weight > 0:
                loss = loss + args.flow_contrastive_weight * contrastive_loss(embs, y, args, confusion_weights)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(flow_head.parameters()), 1.0)
            opt.step()
            total += float(loss.item()) * len(labels)
            ok += int((logits.argmax(-1) == y).sum()); cnt += len(labels)
        val_metrics = evaluate_graph_flow(
            model,
            va_groups,
            args.device,
            args.num_classes,
            flow_head=flow_head,
            class_to_coarse=class_to_coarse,
            hierarchical_logit_weight=effective_hierarchical_logit_weight(args),
            hierarchical_mode=args.hierarchical_mode,
        ) if va_groups else {"accuracy": ok / max(1, cnt), "macro_f1": ok / max(1, cnt)}
        select_score = selected_metric_value(val_metrics, args.select_metric)
        print(f"epoch={epoch} train_loss={total/max(1,cnt):.4f} train_acc={ok/max(1,cnt):.4f} val_acc={val_metrics['accuracy']:.4f} val_macro_f1={val_metrics['macro_f1']:.4f} select={args.select_metric}:{select_score:.4f} flows={cnt}")
        if select_score > best:
            best = select_score; save_ckpt(args, model, sample["x"].shape[1], edge_attr_dim=edge_attr_dim, flow_head=flow_head, num_coarse_classes=num_coarse_classes)


@torch.no_grad()
def evaluate_graph(model, ds, device, num_classes: int):
    model.eval(); y_true = []; y_pred = []
    for item in ds:
        y = int(item.get("label", -1))
        if y < 0:
            continue
        out = model(item["x"].to(device), item["edge_index"].to(device), item["edge_attr"].to(device))
        y_true.append(y)
        y_pred.append(int(out["logits"].argmax(-1).item()))
    return classification_metrics_from_lists(y_true, y_pred, num_classes)


@torch.no_grad()
def evaluate_graph_flow(
    model,
    groups: List[dict],
    device: str,
    num_classes: int,
    flow_head: FlowAggregationHead | None = None,
    class_to_coarse: torch.Tensor | None = None,
    hierarchical_logit_weight: float = 0.0,
    hierarchical_mode: str = "logit",
):
    model.eval(); y_true = []; y_pred = []
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
            pooled = flow_head(torch.stack(window_embs, dim=0), window_logits=torch.stack(window_logits, dim=0))
            logits = apply_hierarchical_logits(pooled["logits"], pooled.get("coarse_logits"), class_to_coarse, hierarchical_logit_weight)
        y = int(group["label"])
        y_true.append(y)
        y_pred.append(int(logits.argmax(-1).item()))
    return classification_metrics_from_lists(y_true, y_pred, num_classes)


def save_ckpt(args, model, input_dim, edge_attr_dim=None, flow_head: FlowAggregationHead | None = None, num_coarse_classes: int = 0):
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
        "select_metric": args.select_metric,
        "window_loss_weight": args.window_loss_weight,
        "class_weighting": args.class_weighting,
        "class_weight_beta": args.class_weight_beta,
        "balanced_flow_batches": args.balanced_flow_batches,
        "classes_per_batch": args.classes_per_batch,
        "samples_per_class": args.samples_per_class,
        "hierarchical_mode": args.hierarchical_mode,
        "hierarchical_weight": args.hierarchical_weight,
        "hierarchical_logit_weight": args.hierarchical_logit_weight,
        "coarse_groups": args.coarse_groups,
        "class_groups": parse_label_groups(args.coarse_groups, args.num_classes) if num_coarse_classes > 0 else [],
        "num_coarse_classes": num_coarse_classes,
        "contrastive_mode": args.contrastive_mode,
        "confusion_groups": args.confusion_groups,
        "confusion_matrix_json": args.confusion_matrix_json,
        "confusion_matrix_level": args.confusion_matrix_level,
        "confusion_weight_power": args.confusion_weight_power,
        "flow_contrastive_weight": args.flow_contrastive_weight,
        "flow_temperature": args.flow_temperature,
        "flow_transformer_layers": args.flow_transformer_layers,
        "flow_transformer_heads": args.flow_transformer_heads,
        "meta_dropout_prob": args.meta_dropout_prob,
        "meta_feature_dim": args.meta_feature_dim,
        "edge_attr_dropout_prob": args.edge_attr_dropout_prob,
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
    ap.add_argument("--select_metric", choices=["accuracy", "acc", "macro_f1", "flow_acc", "flow_macro_f1", "window_acc", "window_macro_f1"], default="accuracy", help="Validation metric used to save best.pt.")
    ap.add_argument("--train_level", choices=["window", "flow"], default="window", help="window: classify each window; flow: pool window embeddings and optimize flow labels directly.")
    ap.add_argument("--window_loss_weight", type=float, default=0.0, help="Extra window-level CE weight used together with flow CE in --train_level flow.")
    ap.add_argument("--class_weighting", choices=["none", "inverse", "effective"], default="none", help="Class-balanced CE weighting scheme.")
    ap.add_argument("--class_weight_beta", type=float, default=0.9999, help="Beta used by --class_weighting effective.")
    ap.add_argument("--balanced_flow_batches", action="store_true", help="Sample each flow batch from multiple classes with repeated positives, useful for flow SupCon.")
    ap.add_argument("--classes_per_batch", type=int, default=0, help="Classes per balanced flow batch. 0 derives it from batch_size / samples_per_class.")
    ap.add_argument("--samples_per_class", type=int, default=2, help="Flows per class in balanced flow batches.")
    ap.add_argument("--hierarchical_mode", choices=["logit", "expert"], default="logit", help="logit: add coarse log-probability to flat logits; expert: use group-specific fine heads P(coarse)*P(class|coarse).")
    ap.add_argument("--hierarchical_weight", type=float, default=0.0, help="Coarse-label CE weight for hierarchical coarse-to-fine flow classification.")
    ap.add_argument("--hierarchical_logit_weight", type=float, default=0.0, help="Add coarse log-probability to fine logits during training/evaluation.")
    ap.add_argument("--coarse_groups", default="vpn_app", help="Coarse groups. Use 'vpn_app', 'none', or semicolon groups like '0,2;1,4'.")
    ap.add_argument("--flow_pooling", choices=["mean", "attention", "late_fusion", "transformer"], default="attention", help="Pooling used by --train_level flow over window embeddings.")
    ap.add_argument("--flow_transformer_layers", type=int, default=1, help="Number of Transformer layers for --flow_pooling transformer.")
    ap.add_argument("--flow_transformer_heads", type=int, default=4, help="Attention heads for --flow_pooling transformer.")
    ap.add_argument("--contrastive_mode", choices=["standard", "confusion", "confusion_weighted"], default="standard", help="standard uses all non-self negatives; confusion uses configured hard negatives; confusion_weighted weights hard negatives by a confusion matrix.")
    ap.add_argument("--confusion_groups", default="vpn_app", help="Groups used by --contrastive_mode confusion. Use 'vpn_app', 'none', or semicolon groups.")
    ap.add_argument("--confusion_matrix_json", default="", help="Optional previous test/valid metrics JSON. Its flow/window y_true/y_pred arrays define weighted SupCon negatives.")
    ap.add_argument("--confusion_matrix_level", choices=["flow", "window"], default="flow", help="Use flow or window predictions from --confusion_matrix_json.")
    ap.add_argument("--confusion_weight_power", type=float, default=1.0, help="Power applied to normalized confusion weights; >1 focuses more on the strongest confusions.")
    ap.add_argument("--flow_contrastive_weight", type=float, default=0.0, help="Weight for supervised contrastive loss over flow embeddings in --train_level flow.")
    ap.add_argument("--flow_temperature", type=float, default=0.07)
    ap.add_argument("--meta_dropout_prob", type=float, default=0.0, help="Training-only dropout on the trailing metadata feature dimensions of x.")
    ap.add_argument("--meta_feature_dim", type=int, default=14, help="Number of trailing metadata feature dimensions appended by preprocess_tower2.py.")
    ap.add_argument("--edge_attr_dropout_prob", type=float, default=0.0, help="Training-only dropout on continuous graph edge attributes; edge type id is kept.")
    ap.add_argument("--valid_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    if args.train_level == "flow" and (args.aux_weight > 0 or args.coherence_weight > 0):
        print("WARNING: --train_level flow ignores aux/coherence weights; use --window_loss_weight for dual flow/window classification.")
    if args.flow_contrastive_weight > 0 and not args.balanced_flow_batches:
        print("WARNING: flow SupCon is usually more effective with --balanced_flow_batches so each batch has positive pairs.")
    if args.hierarchical_mode == "expert" and args.hierarchical_logit_weight > 0:
        print("WARNING: --hierarchical_mode expert already uses coarse probabilities; --hierarchical_logit_weight is ignored.")
    if args.hierarchical_mode == "logit" and args.hierarchical_logit_weight > 0 and args.hierarchical_weight <= 0:
        print("WARNING: --hierarchical_logit_weight is enabled but --hierarchical_weight is 0, so the coarse head has no direct supervision.")
    if args.contrastive_mode == "confusion_weighted" and not args.confusion_matrix_json:
        print("WARNING: --contrastive_mode confusion_weighted has no --confusion_matrix_json; falling back to binary group weights.")
    if args.model_type == "seq":
        train_seq_flow(args) if args.train_level == "flow" else train_seq(args)
    else:
        train_graph_flow(args) if args.train_level == "flow" else train_graph(args)


if __name__ == "__main__":
    main()
