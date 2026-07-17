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
        flow_stat_meta_dim: int = 0,
        flow_stat_expert_weight: float = 0.0,
        flow_stat_aux_weight: float = 0.0,
    ):
        super().__init__()
        self.pooling = pooling
        self.num_classes = num_classes
        self.num_coarse_classes = num_coarse_classes
        self.hierarchical_mode = hierarchical_mode
        self.class_groups = [list(group) for group in (class_groups or [])]
        self.flow_stat_meta_dim = int(max(0, flow_stat_meta_dim))
        self.flow_stat_expert_weight = float(max(0.0, flow_stat_expert_weight))
        self.flow_stat_aux_weight = float(max(0.0, flow_stat_aux_weight))
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
        if pooling == "multi_view":
            self.multi_view_gate = nn.Sequential(
                nn.LayerNorm(hidden_dim * 4),
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 4),
            )
        else:
            self.multi_view_gate = None
        self.use_expert_heads = hierarchical_mode == "expert" and self.coarse_cls is not None and bool(self.class_groups)
        if self.use_expert_heads and len(self.class_groups) != num_coarse_classes:
            raise ValueError("class_groups must match num_coarse_classes when --hierarchical_mode expert is used")
        self.expert_heads = nn.ModuleList(
            [nn.Linear(hidden_dim, len(group)) for group in self.class_groups]
        ) if self.use_expert_heads else nn.ModuleList()
        if pooling == "late_fusion":
            self.fusion_logit_scale = nn.Parameter(torch.tensor(1.0))
        self.use_flow_stat_expert = self.flow_stat_meta_dim > 0 and (
            self.flow_stat_expert_weight > 0 or self.flow_stat_aux_weight > 0
        )
        stat_dim = self.flow_stat_meta_dim * 4 + 2
        if self.use_flow_stat_expert:
            self.stat_proj = nn.Sequential(
                nn.LayerNorm(stat_dim),
                nn.Linear(stat_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
            )
            self.stat_cls = nn.Linear(hidden_dim, num_classes)
            self.stat_gate = nn.Sequential(
                nn.LayerNorm(hidden_dim * 2),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )

    def pool(self, h: torch.Tensor) -> torch.Tensor:
        if h.numel() == 0:
            raise ValueError("Cannot pool an empty flow.")
        if self.pooling == "mean":
            return h.mean(dim=0)
        if self.pooling == "multi_view":
            mean = h.mean(dim=0)
            maxv = h.max(dim=0).values
            std = h.std(dim=0, unbiased=False) if h.size(0) > 1 else torch.zeros_like(mean)
            weights = torch.softmax(self.score(h).squeeze(-1), dim=0)
            attn = torch.sum(h * weights.unsqueeze(-1), dim=0)
            views = torch.stack([mean, maxv, std, attn], dim=0)
            gate = torch.softmax(self.multi_view_gate(torch.cat([mean, maxv, std, attn], dim=-1)), dim=-1)
            return torch.sum(views * gate.unsqueeze(-1), dim=0), gate
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

    def _flow_stat_features(self, window_x: torch.Tensor | Sequence[torch.Tensor] | None, like: torch.Tensor) -> torch.Tensor | None:
        if not self.use_flow_stat_expert or window_x is None:
            return None
        if isinstance(window_x, torch.Tensor):
            x = window_x.reshape(-1, window_x.shape[-1])
        else:
            pieces = [t.reshape(-1, t.shape[-1]) for t in window_x if isinstance(t, torch.Tensor) and t.numel() > 0]
            if not pieces:
                return None
            x = torch.cat(pieces, dim=0)
        if x.numel() == 0:
            return None
        meta_dim = min(self.flow_stat_meta_dim, x.shape[-1])
        if meta_dim <= 0:
            return None
        meta = x[:, -meta_dim:].float()
        if meta_dim < self.flow_stat_meta_dim:
            pad = meta.new_zeros((meta.shape[0], self.flow_stat_meta_dim - meta_dim))
            meta = torch.cat([meta, pad], dim=-1)
        mean = meta.mean(dim=0)
        std = meta.std(dim=0, unbiased=False) if meta.size(0) > 1 else torch.zeros_like(mean)
        minv = meta.min(dim=0).values
        maxv = meta.max(dim=0).values
        extras = meta.new_tensor([
            math.log1p(float(meta.size(0))),
            math.log1p(float(x.shape[0])),
        ])
        return torch.cat([mean, std, minv, maxv, extras], dim=0).to(device=like.device, dtype=like.dtype)

    def forward(
        self,
        h: torch.Tensor,
        window_logits: torch.Tensor | None = None,
        window_x: torch.Tensor | Sequence[torch.Tensor] | None = None,
    ):
        pooled = self.pool(h)
        gate = None
        if isinstance(pooled, tuple):
            emb, gate = pooled
        else:
            emb = pooled
        coarse_logits = self.coarse_cls(emb) if self.coarse_cls is not None else None
        if self.use_expert_heads:
            logits = self.expert_logits(emb, coarse_logits)
        else:
            logits = self.cls(emb)
        if self.pooling == "late_fusion" and window_logits is not None:
            logits = logits + self.fusion_logit_scale * window_logits.mean(dim=0)
        out = {"embedding": emb, "logits": logits}
        stat_features = self._flow_stat_features(window_x, emb)
        if stat_features is not None:
            stat_emb = self.stat_proj(stat_features)
            stat_logits = self.stat_cls(stat_emb)
            stat_gate = torch.sigmoid(self.stat_gate(torch.cat([emb, stat_emb], dim=-1))).squeeze(-1)
            if self.flow_stat_expert_weight > 0:
                logits = logits + self.flow_stat_expert_weight * stat_gate * stat_logits
                out["logits"] = logits
            out["stat_logits"] = stat_logits
            out["stat_gate"] = stat_gate
        if gate is not None:
            out["multi_view_gate"] = gate
        if coarse_logits is not None:
            out["coarse_logits"] = coarse_logits
        return out


class GradientReverseFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float):
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambd * grad_output, None


def gradient_reverse(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    return GradientReverseFn.apply(x, lambd)


class DomainAdversarialHead(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
        return self.net(gradient_reverse(x, lambd))


def load_pt(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_distillation_targets(path: str, num_classes: int, device: str | torch.device) -> Dict[str, torch.Tensor]:
    """Load flow-id keyed soft targets from a prediction/consensus JSON file."""
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    flow_ids = data.get("flow_ids", [])
    flow_prob = data.get("flow_prob", [])
    if not flow_ids or not flow_prob:
        raise ValueError(f"{path} must contain flow_ids and flow_prob for distillation")
    if len(flow_ids) != len(flow_prob):
        raise ValueError(f"{path} has mismatched flow_ids/flow_prob lengths")
    targets: Dict[str, torch.Tensor] = {}
    for fid, prob in zip(flow_ids, flow_prob):
        p = torch.as_tensor(prob, dtype=torch.float32, device=device)
        if p.numel() != num_classes:
            raise ValueError(f"{path} target for flow_id={fid} has {p.numel()} classes, expected {num_classes}")
        p = p.clamp(min=1e-8)
        targets[str(fid)] = p / p.sum().clamp(min=1e-8)
    return targets


def load_distillation_class_priors(path: str, num_classes: int, device: str | torch.device) -> torch.Tensor | None:
    """Average teacher distributions by true class for class-conditional distillation."""
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    y_true = data.get("flow_y_true", [])
    flow_prob = data.get("flow_prob", [])
    if not y_true or not flow_prob or len(y_true) != len(flow_prob):
        return None
    sums = torch.zeros(num_classes, num_classes, dtype=torch.float32, device=device)
    counts = torch.zeros(num_classes, dtype=torch.float32, device=device)
    for label, prob in zip(y_true, flow_prob):
        label = int(label)
        if not (0 <= label < num_classes):
            continue
        p = torch.as_tensor(prob, dtype=torch.float32, device=device)
        if p.numel() != num_classes:
            continue
        p = p.clamp(min=1e-8)
        sums[label] += p / p.sum().clamp(min=1e-8)
        counts[label] += 1.0
    priors = torch.eye(num_classes, dtype=torch.float32, device=device)
    present = counts > 0
    if present.any():
        priors[present] = sums[present] / counts[present].unsqueeze(-1).clamp(min=1.0)
        priors[present] = priors[present].clamp(min=1e-8)
        priors[present] = priors[present] / priors[present].sum(dim=-1, keepdim=True).clamp(min=1e-8)
    return priors


def report_distillation_coverage(name: str, targets: Dict[str, torch.Tensor], flow_ids: Sequence[str]) -> None:
    if not targets:
        return
    unique_ids = sorted(set(map(str, flow_ids)))
    matched = sum(1 for fid in unique_ids if fid in targets)
    ratio = matched / max(1, len(unique_ids))
    msg = {
        "scope": name,
        "target_count": len(targets),
        "unique_flow_count": len(unique_ids),
        "matched_flow_count": matched,
        "coverage": ratio,
    }
    print("distillation_coverage " + json.dumps(msg, sort_keys=True), flush=True)
    if matched == 0:
        print("WARNING: distillation targets do not match any training flow_id; KL distillation will be inactive.", flush=True)


def report_class_prior_distillation(name: str, priors: torch.Tensor | None) -> None:
    if priors is None:
        return
    top1 = priors.argmax(dim=-1)
    diag_mass = priors.diag()
    msg = {
        "scope": name,
        "num_classes": int(priors.size(0)),
        "mean_diag_mass": float(diag_mass.mean().detach().cpu()),
        "min_diag_mass": float(diag_mass.min().detach().cpu()),
        "non_identity_top1_classes": int((top1 != torch.arange(priors.size(0), device=priors.device)).sum().detach().cpu()),
    }
    print("distillation_class_prior " + json.dumps(msg, sort_keys=True), flush=True)


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


def paired_flow_group_lookup(path: str, dataset_cls) -> Dict[str, dict]:
    if not path:
        return {}
    groups = build_flow_groups(dataset_cls(path))
    return {str(group["flow_id"]): group for group in groups}


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


def class_weighted_loss(
    logits: torch.Tensor,
    y: torch.Tensor,
    class_weight: torch.Tensor | None = None,
    label_smoothing: float = 0.0,
    focal_gamma: float = 0.0,
) -> torch.Tensor:
    valid = y >= 0
    if valid.any():
        selected_logits = logits[valid]
        selected_y = y[valid]
        if focal_gamma <= 0:
            return F.cross_entropy(selected_logits, selected_y, weight=class_weight, label_smoothing=label_smoothing)
        log_prob = F.log_softmax(selected_logits, dim=-1)
        ce = F.cross_entropy(
            selected_logits,
            selected_y,
            weight=class_weight,
            label_smoothing=label_smoothing,
            reduction="none",
        )
        pt = log_prob.gather(1, selected_y.view(-1, 1)).squeeze(1).exp().clamp(min=1e-6, max=1.0)
        focal = (1.0 - pt).pow(float(focal_gamma))
        return (focal * ce).mean()
    return torch.tensor(0.0, device=logits.device)


def confidence_penalty_loss(logits: torch.Tensor, y: torch.Tensor, weight: float = 0.0) -> torch.Tensor:
    if weight <= 0:
        return logits.sum() * 0.0
    valid = y >= 0
    if not valid.any():
        return logits.sum() * 0.0
    selected = logits[valid].float()
    log_prob = F.log_softmax(selected, dim=-1)
    prob = log_prob.exp()
    # KL(p || uniform) = sum p log p + log(C). Minimizing it discourages
    # overconfident predictions without changing the model architecture.
    kl_uniform = (prob * log_prob).sum(dim=-1) + math.log(max(1, selected.size(-1)))
    return weight * kl_uniform.mean()


def regularized_classification_loss(
    logits: torch.Tensor,
    y: torch.Tensor,
    class_weight: torch.Tensor | None,
    args,
) -> torch.Tensor:
    return class_weighted_loss(logits, y, class_weight, args.label_smoothing, args.focal_gamma) + confidence_penalty_loss(
        logits,
        y,
        args.confidence_penalty_weight,
    )


def soften_prob_targets(prob: torch.Tensor, temperature: float) -> torch.Tensor:
    temperature = max(float(temperature), 1e-6)
    if temperature == 1.0:
        return prob
    softened = prob.clamp(min=1e-8).pow(1.0 / temperature)
    return softened / softened.sum(dim=-1, keepdim=True).clamp(min=1e-8)


def distillation_kl_loss(
    logits: torch.Tensor,
    flow_ids: Sequence[str],
    targets: Dict[str, torch.Tensor],
    args,
) -> torch.Tensor:
    if args.distill_weight <= 0 or not targets or logits.numel() == 0:
        return logits.sum() * 0.0
    selected_idx = []
    selected_targets = []
    selected_conf = []
    for idx, flow_id in enumerate(flow_ids):
        target = targets.get(str(flow_id))
        if target is None:
            continue
        conf = float(target.max().detach().cpu().item())
        if conf < args.distill_min_confidence:
            continue
        selected_idx.append(idx)
        selected_targets.append(target)
        selected_conf.append(conf)
    if not selected_idx:
        return logits.sum() * 0.0
    idx_t = torch.tensor(selected_idx, dtype=torch.long, device=logits.device)
    teacher = torch.stack(selected_targets, dim=0).to(logits.device, dtype=logits.dtype)
    teacher = soften_prob_targets(teacher, args.distill_temperature)
    student_log_prob = F.log_softmax(logits.index_select(0, idx_t) / max(float(args.distill_temperature), 1e-6), dim=-1)
    per_sample = F.kl_div(student_log_prob, teacher, reduction="none").sum(dim=-1)
    if args.distill_confidence_power > 0:
        weights = torch.tensor(selected_conf, dtype=logits.dtype, device=logits.device).pow(args.distill_confidence_power)
        weights = weights / weights.mean().clamp(min=1e-8)
        per_sample = per_sample * weights
    return args.distill_weight * per_sample.mean() * (max(float(args.distill_temperature), 1e-6) ** 2)


def class_prior_distillation_loss(
    logits: torch.Tensor,
    y: torch.Tensor,
    class_priors: torch.Tensor | None,
    args,
) -> torch.Tensor:
    if args.distill_class_prior_weight <= 0 or class_priors is None or logits.numel() == 0:
        return logits.sum() * 0.0
    valid = y >= 0
    if not valid.any():
        return logits.sum() * 0.0
    selected_y = y[valid].long().clamp(min=0, max=class_priors.size(0) - 1)
    teacher = class_priors.to(logits.device, dtype=logits.dtype).index_select(0, selected_y)
    teacher = soften_prob_targets(teacher, args.distill_temperature)
    temperature = max(float(args.distill_temperature), 1e-6)
    student_log_prob = F.log_softmax(logits[valid] / temperature, dim=-1)
    return args.distill_class_prior_weight * F.kl_div(student_log_prob, teacher, reduction="batchmean") * (temperature ** 2)


def compute_class_weights(
    labels: Sequence[int],
    num_classes: int,
    device: str,
    scheme: str = "none",
    beta: float = 0.9999,
    strength: float = 1.0,
):
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
    strength = min(max(float(strength), 0.0), 1.0)
    if strength < 1.0:
        weights[present] = 1.0 + strength * (weights[present] - 1.0)
    return weights.to(device)


def dataset_labels(ds: Dataset):
    labels = []
    for i in range(len(ds)):
        label = int(ds[i].get("label", -1))
        if label >= 0:
            labels.append(label)
    return labels


def dataset_flow_ids(ds: Dataset) -> List[str]:
    return [str(ds[i].get("flow_id", i)) for i in range(len(ds))]


def group_flow_ids(groups: Sequence[dict]) -> List[str]:
    return [str(group["flow_id"]) for group in groups]


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


def maybe_load_init_checkpoint(args, model: nn.Module, flow_head: FlowAggregationHead | None = None) -> None:
    if not args.init_checkpoint:
        return
    ckpt = torch.load(args.init_checkpoint, map_location=args.device, weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
    msg = {
        "checkpoint": args.init_checkpoint,
        "model_missing": list(missing),
        "model_unexpected": list(unexpected),
    }
    if flow_head is not None and ckpt.get("flow_head_state") is not None:
        flow_missing, flow_unexpected = flow_head.load_state_dict(ckpt["flow_head_state"], strict=False)
        msg["flow_head_missing"] = list(flow_missing)
        msg["flow_head_unexpected"] = list(flow_unexpected)
    print("loaded_init_checkpoint " + json.dumps(msg, sort_keys=True), flush=True)


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


def early_stop_update(select_score: float, best: float, stale_epochs: int, epoch: int, args) -> tuple[bool, float, int, bool]:
    improved = select_score > best
    if improved:
        return False, select_score, 0, True
    stale_epochs += 1
    if args.early_stop_patience > 0 and stale_epochs >= args.early_stop_patience:
        print(
            f"early_stop epoch={epoch} best_{args.select_metric}={best:.4f} "
            f"patience={args.early_stop_patience}",
            flush=True,
        )
        return True, best, stale_epochs, False
    return False, best, stale_epochs, False


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


def window_to_flow_contrastive_loss(
    window_embs: torch.Tensor,
    owners: torch.Tensor,
    flow_embs: torch.Tensor,
    flow_labels: torch.Tensor,
    temperature: float = 0.07,
    positive_mode: str = "same_class",
) -> torch.Tensor:
    """Pull local window embeddings toward same-flow or same-class flow prototypes."""
    if window_embs.numel() == 0 or flow_embs.numel() == 0 or owners.numel() == 0:
        return flow_embs.sum() * 0.0
    temperature = max(float(temperature), 1e-6)
    owners = owners.long().clamp(min=0, max=flow_embs.size(0) - 1)
    z_win = F.normalize(window_embs.float(), p=2, dim=-1)
    z_flow = F.normalize(flow_embs.float(), p=2, dim=-1)
    logits = torch.matmul(z_win, z_flow.T) / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    flow_ids = torch.arange(flow_embs.size(0), device=flow_embs.device)
    own_flow_mask = flow_ids.unsqueeze(0) == owners.unsqueeze(1)
    if positive_mode == "own_flow":
        pos_mask = own_flow_mask
    else:
        owner_labels = flow_labels[owners].view(-1, 1)
        pos_mask = owner_labels.eq(flow_labels.view(1, -1)) | own_flow_mask
    valid = pos_mask.sum(dim=1) > 0
    if valid.sum() == 0:
        return flow_embs.sum() * 0.0
    pos_logits = logits.masked_fill(~pos_mask, -1e9)
    return -(torch.logsumexp(pos_logits[valid], dim=1) - torch.logsumexp(logits[valid], dim=1)).mean()


def train_seq(args):
    ds = SeqDataset(args.dataset)
    tr, va = split_or_external_window_dataset(ds, SeqDataset, args)
    dl = DataLoader(tr, batch_size=args.batch_size, shuffle=True, collate_fn=collate_seq)
    vdl = DataLoader(va, batch_size=args.batch_size, shuffle=False, collate_fn=collate_seq) if len(va) else None
    sample = ds[0]
    model = FlowTransformerClassifier(sample["x"].shape[1], args.num_classes, args.hidden_dim, args.num_layers, args.num_heads, args.dropout).to(args.device)
    maybe_load_init_checkpoint(args, model)
    distill_targets = load_distillation_targets(args.distill_targets_json, args.num_classes, args.device)
    report_distillation_coverage("train_seq", distill_targets, dataset_flow_ids(tr))
    distill_class_priors = load_distillation_class_priors(args.distill_targets_json, args.num_classes, args.device)
    report_class_prior_distillation("train_seq", distill_class_priors)
    class_weight = compute_class_weights(dataset_labels(tr), args.num_classes, args.device, args.class_weighting, args.class_weight_beta, args.class_weight_strength)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best = -1.0
    stale_epochs = 0
    for epoch in range(1, args.epochs + 1):
        model.train(); total = 0.0; ok = 0; cnt = 0; skipped_no_grad = 0
        for batch in dl:
            x = apply_input_augmentation(batch["x"].to(args.device), args)
            mask, y = batch["mask"].to(args.device), batch["label"].to(args.device)
            out = model(x, mask)
            loss = regularized_classification_loss(out["logits"], y, class_weight, args)
            loss = loss + distillation_kl_loss(out["logits"], batch["flow_id"], distill_targets, args)
            loss = loss + class_prior_distillation_loss(out["logits"], y, distill_class_priors, args)
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
        print(f"epoch={epoch} train_loss={total/max(1,cnt):.4f} train_acc={ok/max(1,cnt):.4f} val_acc={val_metrics['accuracy']:.4f} val_macro_f1={val_metrics['macro_f1']:.4f} select={args.select_metric}:{select_score:.4f} skipped_no_grad={skipped_no_grad}", flush=True)
        stop, best, stale_epochs, improved = early_stop_update(select_score, best, stale_epochs, epoch, args)
        if improved:
            save_ckpt(args, model, sample["x"].shape[1])
        if stop:
            break


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


def apply_embedding_dropout(x: torch.Tensor, prob: float, meta_dim: int) -> torch.Tensor:
    if prob <= 0:
        return x
    embed_dim = x.shape[-1] - max(0, meta_dim)
    if embed_dim <= 0:
        return x
    x = x.clone()
    x[..., :embed_dim] = F.dropout(x[..., :embed_dim], p=prob, training=True)
    return x


def apply_edge_attr_dropout(edge_attr: torch.Tensor, prob: float) -> torch.Tensor:
    if prob <= 0 or edge_attr.numel() == 0 or edge_attr.shape[-1] <= 1:
        return edge_attr
    edge_attr = edge_attr.clone()
    edge_attr[:, 1:] = F.dropout(edge_attr[:, 1:], p=prob, training=True)
    return edge_attr


def maybe_drop_windows(items: Sequence[dict], prob: float) -> List[dict]:
    items = list(items)
    if prob <= 0 or len(items) <= 1:
        return items
    kept = [item for item in items if random.random() >= prob]
    if kept:
        return kept
    return [random.choice(items)]


def apply_input_augmentation(x: torch.Tensor, args) -> torch.Tensor:
    x = apply_meta_dropout(x, args.meta_dropout_prob, args.meta_feature_dim)
    x = apply_embedding_dropout(x, args.embedding_dropout_prob, args.meta_feature_dim)
    return x


def consistency_kl_loss(reference_logits: torch.Tensor, augmented_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if reference_logits.numel() == 0 or augmented_logits.numel() == 0:
        return reference_logits.sum() * 0.0
    temperature = max(float(temperature), 1e-6)
    target = F.softmax(reference_logits.detach() / temperature, dim=-1)
    log_prob = F.log_softmax(augmented_logits / temperature, dim=-1)
    return F.kl_div(log_prob, target, reduction="batchmean") * (temperature ** 2)


def symmetric_consistency_kl(logits_a: torch.Tensor, logits_b: torch.Tensor, temperature: float) -> torch.Tensor:
    return 0.5 * (
        consistency_kl_loss(logits_a, logits_b, temperature)
        + consistency_kl_loss(logits_b, logits_a, temperature)
    )


def multi_view_gate_entropy_loss(gates: List[torch.Tensor], weight: float) -> torch.Tensor | None:
    if weight <= 0 or not gates:
        return None
    gate = torch.stack(gates, dim=0).float().clamp(min=1e-8)
    entropy = -(gate * gate.log()).sum(dim=-1).mean()
    return weight * entropy


def flow_stat_aux_loss(stat_logits: List[torch.Tensor], y: torch.Tensor, class_weight: torch.Tensor | None, args) -> torch.Tensor | None:
    if args.flow_stat_aux_weight <= 0 or not stat_logits:
        return None
    if len(stat_logits) != y.numel():
        return None
    logits = torch.stack(stat_logits, dim=0)
    return args.flow_stat_aux_weight * regularized_classification_loss(logits, y, class_weight, args)


def view_domain_adversarial_loss(
    domain_head: DomainAdversarialHead | None,
    clean_embs: torch.Tensor,
    paired_embs: torch.Tensor | None,
    weight: float,
    lambd: float,
) -> torch.Tensor:
    if domain_head is None or weight <= 0 or paired_embs is None or paired_embs.numel() == 0:
        return clean_embs.sum() * 0.0
    if clean_embs.size(0) != paired_embs.size(0):
        return clean_embs.sum() * 0.0
    z = torch.cat([clean_embs, paired_embs], dim=0)
    y = torch.cat(
        [
            torch.zeros(clean_embs.size(0), dtype=torch.long, device=clean_embs.device),
            torch.ones(paired_embs.size(0), dtype=torch.long, device=clean_embs.device),
        ],
        dim=0,
    )
    return weight * F.cross_entropy(domain_head(z, lambd), y)


def train_seq_flow(args):
    ds = SeqDataset(args.dataset)
    tr_groups, va_groups = split_or_external_flow_groups(ds, SeqDataset, args)
    paired_groups = paired_flow_group_lookup(args.paired_view_dataset, SeqDataset)
    distill_targets = load_distillation_targets(args.distill_targets_json, args.num_classes, args.device)
    report_distillation_coverage("train_seq_flow", distill_targets, group_flow_ids(tr_groups))
    distill_class_priors = load_distillation_class_priors(args.distill_targets_json, args.num_classes, args.device)
    report_class_prior_distillation("train_seq_flow", distill_class_priors)
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
        flow_stat_meta_dim=args.meta_feature_dim,
        flow_stat_expert_weight=args.flow_stat_expert_weight,
        flow_stat_aux_weight=args.flow_stat_aux_weight,
    ).to(args.device)
    domain_head = (
        DomainAdversarialHead(args.hidden_dim, args.dropout).to(args.device)
        if args.view_domain_adversarial_weight > 0 and args.paired_view_dataset
        else None
    )
    maybe_load_init_checkpoint(args, model, flow_head)
    class_weight = compute_class_weights([g["label"] for g in tr_groups], args.num_classes, args.device, args.class_weighting, args.class_weight_beta, args.class_weight_strength)
    opt_params = list(model.parameters()) + list(flow_head.parameters())
    if domain_head is not None:
        opt_params += list(domain_head.parameters())
    opt = torch.optim.AdamW(opt_params, lr=args.lr, weight_decay=args.weight_decay)
    best = -1.0
    stale_epochs = 0
    for epoch in range(1, args.epochs + 1):
        model.train(); total = 0.0; ok = 0; cnt = 0
        flow_head.train()
        for flow_batch in iter_train_flow_batches(tr_groups, args):
            windows = []
            owners = []
            labels = []
            flow_ids = []
            paired_windows = []
            paired_owners = []
            paired_labels = []
            for flow_idx, group in enumerate(flow_batch):
                labels.append(int(group["label"]))
                flow_ids.append(str(group["flow_id"]))
                for item in maybe_drop_windows(group["items"], args.window_dropout_prob):
                    windows.append(item)
                    owners.append(flow_idx)
                paired_group = paired_groups.get(str(group["flow_id"]))
                if paired_group is not None and int(paired_group["label"]) == int(group["label"]):
                    paired_labels.append(int(paired_group["label"]))
                    for item in maybe_drop_windows(paired_group["items"], args.window_dropout_prob):
                        paired_windows.append(item)
                        paired_owners.append(flow_idx)
            if not windows:
                continue
            batch = collate_seq(windows)
            x_clean = batch["x"].to(args.device)
            mask = batch["mask"].to(args.device)
            if args.consistency_weight > 0:
                out = model(x_clean, mask)
                out_aug = model(apply_input_augmentation(x_clean, args), mask)
            else:
                out = model(apply_input_augmentation(x_clean, args), mask)
                out_aug = None
            owners_t = torch.tensor(owners, dtype=torch.long, device=args.device)
            window_labels = batch["label"].to(args.device)
            flow_logits = []
            flow_coarse_logits = []
            flow_embs = []
            flow_logits_aug = []
            multi_view_gates = []
            stat_logits = []
            for flow_idx in range(len(flow_batch)):
                win_mask = owners_t == flow_idx
                pooled = flow_head(
                    out["embedding"][win_mask],
                    window_logits=out["logits"][win_mask],
                    window_x=x_clean[win_mask],
                )
                flow_logits.append(pooled["logits"])
                if pooled.get("multi_view_gate") is not None:
                    multi_view_gates.append(pooled["multi_view_gate"])
                if pooled.get("stat_logits") is not None:
                    stat_logits.append(pooled["stat_logits"])
                if pooled.get("coarse_logits") is not None:
                    flow_coarse_logits.append(pooled["coarse_logits"])
                flow_embs.append(pooled["embedding"])
                if out_aug is not None:
                    pooled_aug = flow_head(
                        out_aug["embedding"][win_mask],
                        window_logits=out_aug["logits"][win_mask],
                        window_x=x_clean[win_mask],
                    )
                    flow_logits_aug.append(pooled_aug["logits"])
            flow_logits = torch.stack(flow_logits, dim=0)
            coarse_logits = torch.stack(flow_coarse_logits, dim=0) if flow_coarse_logits else None
            flow_logits = apply_hierarchical_logits(flow_logits, coarse_logits, class_to_coarse, effective_hierarchical_logit_weight(args))
            if flow_logits_aug:
                flow_logits_aug = torch.stack(flow_logits_aug, dim=0)
            flow_embs = torch.stack(flow_embs, dim=0)
            y = torch.tensor(labels, dtype=torch.long, device=args.device)
            loss = regularized_classification_loss(flow_logits, y, class_weight, args)
            loss = loss + distillation_kl_loss(flow_logits, flow_ids, distill_targets, args)
            loss = loss + class_prior_distillation_loss(flow_logits, y, distill_class_priors, args)
            gate_loss = multi_view_gate_entropy_loss(multi_view_gates, args.multi_view_gate_entropy_weight)
            if gate_loss is not None:
                loss = loss + gate_loss
            stat_loss = flow_stat_aux_loss(stat_logits, y, class_weight, args)
            if stat_loss is not None:
                loss = loss + stat_loss
            paired_logits = None
            paired_embs = None
            if paired_windows and (args.paired_view_weight > 0 or args.paired_consistency_weight > 0 or args.view_domain_adversarial_weight > 0):
                paired_batch = collate_seq(paired_windows)
                paired_out = model(
                    apply_input_augmentation(paired_batch["x"].to(args.device), args),
                    paired_batch["mask"].to(args.device),
                )
                paired_owners_t = torch.tensor(paired_owners, dtype=torch.long, device=args.device)
                paired_flow_logits = []
                paired_flow_coarse_logits = []
                paired_flow_embs = []
                for flow_idx in range(len(flow_batch)):
                    win_mask = paired_owners_t == flow_idx
                    if not win_mask.any():
                        continue
                    pooled = flow_head(
                        paired_out["embedding"][win_mask],
                        window_logits=paired_out["logits"][win_mask],
                        window_x=paired_batch["x"].to(args.device)[win_mask],
                    )
                    paired_flow_logits.append(pooled["logits"])
                    paired_flow_embs.append(pooled["embedding"])
                    if pooled.get("coarse_logits") is not None:
                        paired_flow_coarse_logits.append(pooled["coarse_logits"])
                if paired_flow_logits and len(paired_flow_logits) == len(flow_batch):
                    paired_logits = torch.stack(paired_flow_logits, dim=0)
                    paired_embs = torch.stack(paired_flow_embs, dim=0)
                    paired_coarse = torch.stack(paired_flow_coarse_logits, dim=0) if paired_flow_coarse_logits else None
                    paired_logits = apply_hierarchical_logits(paired_logits, paired_coarse, class_to_coarse, effective_hierarchical_logit_weight(args))
                    if args.paired_view_weight > 0:
                        loss = loss + args.paired_view_weight * regularized_classification_loss(paired_logits, y, class_weight, args)
                    if args.paired_consistency_weight > 0:
                        loss = loss + args.paired_consistency_weight * symmetric_consistency_kl(flow_logits, paired_logits, args.consistency_temperature)
            if args.view_domain_adversarial_weight > 0:
                loss = loss + view_domain_adversarial_loss(
                    domain_head,
                    flow_embs,
                    paired_embs,
                    args.view_domain_adversarial_weight,
                    args.domain_adversarial_lambda,
                )
            if args.consistency_weight > 0 and isinstance(flow_logits_aug, torch.Tensor):
                loss = loss + args.consistency_weight * consistency_kl_loss(flow_logits, flow_logits_aug, args.consistency_temperature)
            if coarse_logits is not None and args.hierarchical_weight > 0:
                loss = loss + args.hierarchical_weight * F.cross_entropy(coarse_logits, class_to_coarse[y])
            if args.window_loss_weight > 0:
                loss = loss + args.window_loss_weight * regularized_classification_loss(out["logits"], window_labels, class_weight, args)
            if args.window_contrastive_weight > 0:
                loss = loss + args.window_contrastive_weight * window_to_flow_contrastive_loss(
                    out["embedding"],
                    owners_t,
                    flow_embs,
                    y,
                    temperature=args.window_contrastive_temperature,
                    positive_mode=args.window_contrastive_positive,
                )
            if args.flow_contrastive_weight > 0:
                loss = loss + args.flow_contrastive_weight * contrastive_loss(flow_embs, y, args, confusion_weights)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(opt_params, 1.0)
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
        print(f"epoch={epoch} train_loss={total/max(1,cnt):.4f} train_acc={ok/max(1,cnt):.4f} val_acc={val_metrics['accuracy']:.4f} val_macro_f1={val_metrics['macro_f1']:.4f} select={args.select_metric}:{select_score:.4f} flows={cnt}", flush=True)
        stop, best, stale_epochs, improved = early_stop_update(select_score, best, stale_epochs, epoch, args)
        if improved:
            save_ckpt(args, model, sample["x"].shape[1], flow_head=flow_head, num_coarse_classes=num_coarse_classes)
        if stop:
            break


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
                    window_x=batch["x"].to(device)[owners_t == flow_idx],
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
    maybe_load_init_checkpoint(args, model)
    distill_targets = load_distillation_targets(args.distill_targets_json, args.num_classes, args.device)
    report_distillation_coverage("train_graph", distill_targets, dataset_flow_ids(tr))
    distill_class_priors = load_distillation_class_priors(args.distill_targets_json, args.num_classes, args.device)
    report_class_prior_distillation("train_graph", distill_class_priors)
    class_weight = compute_class_weights(dataset_labels(tr), args.num_classes, args.device, args.class_weighting, args.class_weight_beta, args.class_weight_strength)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best = -1.0
    stale_epochs = 0
    for epoch in range(1, args.epochs + 1):
        model.train(); total = 0.0; ok = 0; cnt = 0; skipped_no_grad = 0
        order = list(range(len(tr)))
        random.shuffle(order)
        for idx in order:
            item = tr[idx]
            x = apply_input_augmentation(item["x"].to(args.device), args)
            edge_index = item["edge_index"].to(args.device)
            edge_attr = apply_edge_attr_dropout(item["edge_attr"].to(args.device), args.edge_attr_dropout_prob)
            y = torch.tensor([int(item.get("label", -1))], dtype=torch.long, device=args.device)
            out = model(x, edge_index, edge_attr)
            loss = regularized_classification_loss(out["logits"], y, class_weight, args)
            loss = loss + distillation_kl_loss(out["logits"], [str(item.get("flow_id", idx))], distill_targets, args)
            loss = loss + class_prior_distillation_loss(out["logits"], y, distill_class_priors, args)
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
        print(f"epoch={epoch} train_loss={total/max(1,cnt):.4f} train_acc={ok/max(1,cnt):.4f} val_acc={val_metrics['accuracy']:.4f} val_macro_f1={val_metrics['macro_f1']:.4f} select={args.select_metric}:{select_score:.4f} skipped_no_grad={skipped_no_grad}", flush=True)
        stop, best, stale_epochs, improved = early_stop_update(select_score, best, stale_epochs, epoch, args)
        if improved:
            save_ckpt(args, model, sample["x"].shape[1], edge_attr_dim=edge_attr_dim)
        if stop:
            break


def train_graph_flow(args):
    ds = GraphDataset(args.dataset)
    tr_groups, va_groups = split_or_external_flow_groups(ds, GraphDataset, args)
    paired_groups = paired_flow_group_lookup(args.paired_view_dataset, GraphDataset)
    distill_targets = load_distillation_targets(args.distill_targets_json, args.num_classes, args.device)
    report_distillation_coverage("train_graph_flow", distill_targets, group_flow_ids(tr_groups))
    distill_class_priors = load_distillation_class_priors(args.distill_targets_json, args.num_classes, args.device)
    report_class_prior_distillation("train_graph_flow", distill_class_priors)
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
        flow_stat_meta_dim=args.meta_feature_dim,
        flow_stat_expert_weight=args.flow_stat_expert_weight,
        flow_stat_aux_weight=args.flow_stat_aux_weight,
    ).to(args.device)
    domain_head = (
        DomainAdversarialHead(args.hidden_dim, args.dropout).to(args.device)
        if args.view_domain_adversarial_weight > 0 and args.paired_view_dataset
        else None
    )
    maybe_load_init_checkpoint(args, model, flow_head)
    class_weight = compute_class_weights([g["label"] for g in tr_groups], args.num_classes, args.device, args.class_weighting, args.class_weight_beta, args.class_weight_strength)
    opt_params = list(model.parameters()) + list(flow_head.parameters())
    if domain_head is not None:
        opt_params += list(domain_head.parameters())
    opt = torch.optim.AdamW(opt_params, lr=args.lr, weight_decay=args.weight_decay)
    best = -1.0
    stale_epochs = 0
    for epoch in range(1, args.epochs + 1):
        model.train(); total = 0.0; ok = 0; cnt = 0
        flow_head.train()
        for flow_batch in iter_train_flow_batches(tr_groups, args):
            flow_logits = []
            flow_coarse_logits = []
            flow_embs = []
            flow_logits_aug = []
            paired_flow_logits = []
            paired_flow_coarse_logits = []
            paired_flow_embs = []
            window_logits_all = []
            window_labels = []
            window_embs_all = []
            window_owners = []
            multi_view_gates = []
            stat_logits = []
            labels = []
            flow_ids = []
            for group in flow_batch:
                window_embs = []
                window_logits = []
                window_embs_aug = []
                window_logits_aug = []
                window_xs = []
                for item in maybe_drop_windows(group["items"], args.window_dropout_prob):
                    x_clean = item["x"].to(args.device)
                    window_xs.append(x_clean)
                    edge_attr = apply_edge_attr_dropout(item["edge_attr"].to(args.device), args.edge_attr_dropout_prob)
                    if args.consistency_weight > 0:
                        out = model(x_clean, item["edge_index"].to(args.device), item["edge_attr"].to(args.device))
                        out_aug = model(apply_input_augmentation(x_clean, args), item["edge_index"].to(args.device), edge_attr)
                    else:
                        out = model(apply_input_augmentation(x_clean, args), item["edge_index"].to(args.device), edge_attr)
                        out_aug = None
                    window_embs.append(out["embedding"])
                    window_logits.append(out["logits"].squeeze(0))
                    window_logits_all.append(out["logits"].squeeze(0))
                    window_labels.append(int(item.get("label", -1)))
                    if out_aug is not None:
                        window_embs_aug.append(out_aug["embedding"])
                        window_logits_aug.append(out_aug["logits"].squeeze(0))
                if window_embs:
                    active_flow_idx = len(labels)
                    window_embs_all.extend(window_embs)
                    window_owners.extend([active_flow_idx] * len(window_embs))
                    pooled = flow_head(
                        torch.stack(window_embs, dim=0),
                        window_logits=torch.stack(window_logits, dim=0),
                        window_x=window_xs,
                    )
                    flow_logits.append(pooled["logits"])
                    if pooled.get("multi_view_gate") is not None:
                        multi_view_gates.append(pooled["multi_view_gate"])
                    if pooled.get("stat_logits") is not None:
                        stat_logits.append(pooled["stat_logits"])
                    if pooled.get("coarse_logits") is not None:
                        flow_coarse_logits.append(pooled["coarse_logits"])
                    flow_embs.append(pooled["embedding"])
                    if window_embs_aug:
                        pooled_aug = flow_head(torch.stack(window_embs_aug, dim=0), window_logits=torch.stack(window_logits_aug, dim=0))
                        flow_logits_aug.append(pooled_aug["logits"])
                    labels.append(int(group["label"]))
                    flow_ids.append(str(group["flow_id"]))
                    paired_group = paired_groups.get(str(group["flow_id"]))
                    if paired_group is not None and int(paired_group["label"]) == int(group["label"]) and (
                        args.paired_view_weight > 0
                        or args.paired_consistency_weight > 0
                        or args.view_domain_adversarial_weight > 0
                    ):
                        paired_window_embs = []
                        paired_window_logits = []
                        for paired_item in maybe_drop_windows(paired_group["items"], args.window_dropout_prob):
                            paired_x = apply_input_augmentation(paired_item["x"].to(args.device), args)
                            paired_edge_attr = apply_edge_attr_dropout(paired_item["edge_attr"].to(args.device), args.edge_attr_dropout_prob)
                            paired_out = model(paired_x, paired_item["edge_index"].to(args.device), paired_edge_attr)
                            paired_window_embs.append(paired_out["embedding"])
                            paired_window_logits.append(paired_out["logits"].squeeze(0))
                        if paired_window_embs:
                            paired_pooled = flow_head(
                                torch.stack(paired_window_embs, dim=0),
                                window_logits=torch.stack(paired_window_logits, dim=0),
                                window_x=[paired_item["x"].to(args.device) for paired_item in paired_group["items"]],
                            )
                            paired_flow_logits.append(paired_pooled["logits"])
                            paired_flow_embs.append(paired_pooled["embedding"])
                            if paired_pooled.get("coarse_logits") is not None:
                                paired_flow_coarse_logits.append(paired_pooled["coarse_logits"])
            if not flow_logits:
                continue
            logits = torch.stack(flow_logits, dim=0)
            coarse_logits = torch.stack(flow_coarse_logits, dim=0) if flow_coarse_logits else None
            logits = apply_hierarchical_logits(logits, coarse_logits, class_to_coarse, effective_hierarchical_logit_weight(args))
            paired_logits = None
            paired_embs = None
            if paired_flow_logits and len(paired_flow_logits) == len(flow_logits):
                paired_logits = torch.stack(paired_flow_logits, dim=0)
                paired_embs = torch.stack(paired_flow_embs, dim=0)
                paired_coarse_logits = torch.stack(paired_flow_coarse_logits, dim=0) if paired_flow_coarse_logits else None
                paired_logits = apply_hierarchical_logits(paired_logits, paired_coarse_logits, class_to_coarse, effective_hierarchical_logit_weight(args))
            embs = torch.stack(flow_embs, dim=0)
            y = torch.tensor(labels, dtype=torch.long, device=args.device)
            loss = regularized_classification_loss(logits, y, class_weight, args)
            loss = loss + distillation_kl_loss(logits, flow_ids, distill_targets, args)
            loss = loss + class_prior_distillation_loss(logits, y, distill_class_priors, args)
            gate_loss = multi_view_gate_entropy_loss(multi_view_gates, args.multi_view_gate_entropy_weight)
            if gate_loss is not None:
                loss = loss + gate_loss
            stat_loss = flow_stat_aux_loss(stat_logits, y, class_weight, args)
            if stat_loss is not None:
                loss = loss + stat_loss
            if paired_logits is not None:
                if args.paired_view_weight > 0:
                    loss = loss + args.paired_view_weight * regularized_classification_loss(paired_logits, y, class_weight, args)
                if args.paired_consistency_weight > 0:
                    loss = loss + args.paired_consistency_weight * symmetric_consistency_kl(logits, paired_logits, args.consistency_temperature)
            if args.view_domain_adversarial_weight > 0:
                loss = loss + view_domain_adversarial_loss(
                    domain_head,
                    embs,
                    paired_embs,
                    args.view_domain_adversarial_weight,
                    args.domain_adversarial_lambda,
                )
            if args.consistency_weight > 0 and flow_logits_aug:
                aug_logits = torch.stack(flow_logits_aug, dim=0)
                loss = loss + args.consistency_weight * consistency_kl_loss(logits, aug_logits, args.consistency_temperature)
            if coarse_logits is not None and args.hierarchical_weight > 0:
                loss = loss + args.hierarchical_weight * F.cross_entropy(coarse_logits, class_to_coarse[y])
            if args.window_loss_weight > 0 and window_logits_all:
                win_logits = torch.stack(window_logits_all, dim=0)
                win_y = torch.tensor(window_labels, dtype=torch.long, device=args.device)
                loss = loss + args.window_loss_weight * regularized_classification_loss(win_logits, win_y, class_weight, args)
            if args.window_contrastive_weight > 0 and window_embs_all:
                loss = loss + args.window_contrastive_weight * window_to_flow_contrastive_loss(
                    torch.stack(window_embs_all, dim=0),
                    torch.tensor(window_owners, dtype=torch.long, device=args.device),
                    embs,
                    y,
                    temperature=args.window_contrastive_temperature,
                    positive_mode=args.window_contrastive_positive,
                )
            if args.flow_contrastive_weight > 0:
                loss = loss + args.flow_contrastive_weight * contrastive_loss(embs, y, args, confusion_weights)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(opt_params, 1.0)
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
        print(f"epoch={epoch} train_loss={total/max(1,cnt):.4f} train_acc={ok/max(1,cnt):.4f} val_acc={val_metrics['accuracy']:.4f} val_macro_f1={val_metrics['macro_f1']:.4f} select={args.select_metric}:{select_score:.4f} flows={cnt}", flush=True)
        stop, best, stale_epochs, improved = early_stop_update(select_score, best, stale_epochs, epoch, args)
        if improved:
            save_ckpt(args, model, sample["x"].shape[1], edge_attr_dim=edge_attr_dim, flow_head=flow_head, num_coarse_classes=num_coarse_classes)
        if stop:
            break


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
            pooled = flow_head(
                torch.stack(window_embs, dim=0),
                window_logits=torch.stack(window_logits, dim=0),
                window_x=[item["x"].to(device) for item in group["items"]],
            )
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
        "init_checkpoint": args.init_checkpoint,
        "train_level": args.train_level,
        "flow_pooling": args.flow_pooling,
        "multi_view_branches": ["mean", "max", "std", "attention"] if args.flow_pooling == "multi_view" else [],
        "multi_view_gate_entropy_weight": args.multi_view_gate_entropy_weight,
        "select_metric": args.select_metric,
        "window_loss_weight": args.window_loss_weight,
        "class_weighting": args.class_weighting,
        "class_weight_beta": args.class_weight_beta,
        "class_weight_strength": args.class_weight_strength,
        "label_smoothing": args.label_smoothing,
        "focal_gamma": args.focal_gamma,
        "confidence_penalty_weight": args.confidence_penalty_weight,
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
        "window_contrastive_weight": args.window_contrastive_weight,
        "window_contrastive_temperature": args.window_contrastive_temperature,
        "window_contrastive_positive": args.window_contrastive_positive,
        "flow_transformer_layers": args.flow_transformer_layers,
        "flow_transformer_heads": args.flow_transformer_heads,
        "flow_stat_expert_weight": args.flow_stat_expert_weight,
        "flow_stat_aux_weight": args.flow_stat_aux_weight,
        "meta_dropout_prob": args.meta_dropout_prob,
        "meta_feature_dim": args.meta_feature_dim,
        "embedding_dropout_prob": args.embedding_dropout_prob,
        "window_dropout_prob": args.window_dropout_prob,
        "edge_attr_dropout_prob": args.edge_attr_dropout_prob,
        "consistency_weight": args.consistency_weight,
        "consistency_temperature": args.consistency_temperature,
        "paired_view_dataset": args.paired_view_dataset,
        "paired_view_weight": args.paired_view_weight,
        "paired_consistency_weight": args.paired_consistency_weight,
        "view_domain_adversarial_weight": args.view_domain_adversarial_weight,
        "domain_adversarial_lambda": args.domain_adversarial_lambda,
        "distill_targets_json": args.distill_targets_json,
        "distill_weight": args.distill_weight,
        "distill_class_prior_weight": args.distill_class_prior_weight,
        "distill_temperature": args.distill_temperature,
        "distill_min_confidence": args.distill_min_confidence,
        "distill_confidence_power": args.distill_confidence_power,
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
    ap.add_argument("--init_checkpoint", default="", help="Optional best.pt checkpoint used to initialize model/flow_head before fine-tuning.")
    ap.add_argument("--aux_weight", type=float, default=0.1)
    ap.add_argument("--coherence_weight", type=float, default=0.1)
    ap.add_argument("--select_metric", choices=["accuracy", "acc", "macro_f1", "flow_acc", "flow_macro_f1", "window_acc", "window_macro_f1"], default="accuracy", help="Validation metric used to save best.pt.")
    ap.add_argument("--early_stop_patience", type=int, default=0, help="Stop when --select_metric does not improve for this many epochs. 0 disables early stopping.")
    ap.add_argument("--train_level", choices=["window", "flow"], default="window", help="window: classify each window; flow: pool window embeddings and optimize flow labels directly.")
    ap.add_argument("--window_loss_weight", type=float, default=0.0, help="Extra window-level CE weight used together with flow CE in --train_level flow.")
    ap.add_argument("--class_weighting", choices=["none", "inverse", "effective"], default="none", help="Class-balanced CE weighting scheme.")
    ap.add_argument("--class_weight_beta", type=float, default=0.9999, help="Beta used by --class_weighting effective.")
    ap.add_argument("--class_weight_strength", type=float, default=1.0, help="Interpolate class weights toward 1.0. 1 keeps full weighting, 0 disables the weighting effect.")
    ap.add_argument("--label_smoothing", type=float, default=0.0, help="Label smoothing for main flow/window classification CE losses.")
    ap.add_argument("--focal_gamma", type=float, default=0.0, help="Focal CE gamma. 0 disables focal loss; positive values focus Tower-2 CE on hard examples.")
    ap.add_argument("--confidence_penalty_weight", type=float, default=0.0, help="KL-to-uniform penalty on classification logits. Positive values reduce overconfident predictions and can improve CI stability.")
    ap.add_argument("--balanced_flow_batches", action="store_true", help="Sample each flow batch from multiple classes with repeated positives, useful for flow SupCon.")
    ap.add_argument("--classes_per_batch", type=int, default=0, help="Classes per balanced flow batch. 0 derives it from batch_size / samples_per_class.")
    ap.add_argument("--samples_per_class", type=int, default=2, help="Flows per class in balanced flow batches.")
    ap.add_argument("--hierarchical_mode", choices=["logit", "expert"], default="logit", help="logit: add coarse log-probability to flat logits; expert: use group-specific fine heads P(coarse)*P(class|coarse).")
    ap.add_argument("--hierarchical_weight", type=float, default=0.0, help="Coarse-label CE weight for hierarchical coarse-to-fine flow classification.")
    ap.add_argument("--hierarchical_logit_weight", type=float, default=0.0, help="Add coarse log-probability to fine logits during training/evaluation.")
    ap.add_argument("--coarse_groups", default="vpn_app", help="Coarse groups. Use 'vpn_app', 'none', or semicolon groups like '0,2;1,4'.")
    ap.add_argument("--flow_pooling", choices=["mean", "attention", "late_fusion", "transformer", "multi_view"], default="attention", help="Pooling used by --train_level flow over window embeddings.")
    ap.add_argument("--flow_transformer_layers", type=int, default=1, help="Number of Transformer layers for --flow_pooling transformer.")
    ap.add_argument("--flow_transformer_heads", type=int, default=4, help="Attention heads for --flow_pooling transformer.")
    ap.add_argument("--flow_stat_expert_weight", type=float, default=0.0, help="Fuse a trainable flow-level metadata/statistics expert into the Tower-2 flow head. 0 disables it.")
    ap.add_argument("--flow_stat_aux_weight", type=float, default=0.0, help="Auxiliary CE weight for the flow metadata/statistics expert.")
    ap.add_argument("--multi_view_gate_entropy_weight", type=float, default=0.0, help="Positive weight minimizes entropy of --flow_pooling multi_view gates, encouraging automatic branch down-weighting.")
    ap.add_argument("--contrastive_mode", choices=["standard", "confusion", "confusion_weighted"], default="standard", help="standard uses all non-self negatives; confusion uses configured hard negatives; confusion_weighted weights hard negatives by a confusion matrix.")
    ap.add_argument("--confusion_groups", default="vpn_app", help="Groups used by --contrastive_mode confusion. Use 'vpn_app', 'none', or semicolon groups.")
    ap.add_argument("--confusion_matrix_json", default="", help="Optional previous test/valid metrics JSON. Its flow/window y_true/y_pred arrays define weighted SupCon negatives.")
    ap.add_argument("--confusion_matrix_level", choices=["flow", "window"], default="flow", help="Use flow or window predictions from --confusion_matrix_json.")
    ap.add_argument("--confusion_weight_power", type=float, default=1.0, help="Power applied to normalized confusion weights; >1 focuses more on the strongest confusions.")
    ap.add_argument("--flow_contrastive_weight", type=float, default=0.0, help="Weight for supervised contrastive loss over flow embeddings in --train_level flow.")
    ap.add_argument("--flow_temperature", type=float, default=0.07)
    ap.add_argument("--window_contrastive_weight", type=float, default=0.0, help="Weight for window-to-flow prototype contrastive loss in --train_level flow.")
    ap.add_argument("--window_contrastive_temperature", type=float, default=0.07, help="Temperature for --window_contrastive_weight.")
    ap.add_argument("--window_contrastive_positive", choices=["own_flow", "same_class"], default="same_class", help="Positive flow prototypes for window-to-flow contrastive loss.")
    ap.add_argument("--meta_dropout_prob", type=float, default=0.0, help="Training-only dropout on the trailing metadata feature dimensions of x.")
    ap.add_argument("--meta_feature_dim", type=int, default=14, help="Number of trailing metadata feature dimensions appended by preprocess_tower2.py.")
    ap.add_argument("--embedding_dropout_prob", type=float, default=0.0, help="Training-only dropout on packet embedding dimensions; trailing metadata dimensions are controlled by --meta_dropout_prob.")
    ap.add_argument("--window_dropout_prob", type=float, default=0.0, help="Training-only random window dropping for --train_level flow; at least one window per flow is kept.")
    ap.add_argument("--edge_attr_dropout_prob", type=float, default=0.0, help="Training-only dropout on continuous graph edge attributes; edge type id is kept.")
    ap.add_argument("--consistency_weight", type=float, default=0.0, help="Weight for clean/augmented view KL consistency. CE is computed on the clean view when this is >0.")
    ap.add_argument("--consistency_temperature", type=float, default=2.0, help="Temperature for clean/augmented consistency KL.")
    ap.add_argument("--paired_view_dataset", default="", help="Optional second-view Tower2 dataset aligned by flow_id, e.g. ip/port randomized or masked view.")
    ap.add_argument("--paired_view_weight", type=float, default=0.0, help="Flow CE weight for --paired_view_dataset.")
    ap.add_argument("--paired_consistency_weight", type=float, default=0.0, help="Symmetric KL weight between primary and paired flow logits.")
    ap.add_argument("--view_domain_adversarial_weight", type=float, default=0.0, help="Adversarially remove clean-vs-paired view information from flow embeddings in --train_level flow.")
    ap.add_argument("--domain_adversarial_lambda", type=float, default=1.0, help="Gradient reversal strength for --view_domain_adversarial_weight.")
    ap.add_argument("--distill_targets_json", default="", help="Optional prediction/consensus JSON with flow_ids and flow_prob used as soft distillation targets.")
    ap.add_argument("--distill_weight", type=float, default=0.0, help="KL weight for consensus/self-distillation targets. 0 disables it.")
    ap.add_argument("--distill_class_prior_weight", type=float, default=0.0, help="KL weight for class-conditional consensus priors averaged from teacher flow_y_true/flow_prob.")
    ap.add_argument("--distill_temperature", type=float, default=2.0, help="Temperature for probability-target distillation.")
    ap.add_argument("--distill_min_confidence", type=float, default=0.0, help="Ignore teacher targets whose max probability is below this confidence.")
    ap.add_argument("--distill_confidence_power", type=float, default=0.0, help="If >0, weight distillation samples by teacher max-probability^power.")
    ap.add_argument("--valid_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    if args.train_level == "flow" and (args.aux_weight > 0 or args.coherence_weight > 0):
        print("WARNING: --train_level flow ignores aux/coherence weights; use --window_loss_weight for dual flow/window classification.")
    if args.flow_contrastive_weight > 0 and not args.balanced_flow_batches:
        print("WARNING: flow SupCon is usually more effective with --balanced_flow_batches so each batch has positive pairs.")
    if args.window_contrastive_weight > 0 and args.train_level != "flow":
        print("WARNING: --window_contrastive_weight is only used with --train_level flow.")
    if args.window_contrastive_weight > 0 and not args.balanced_flow_batches and args.window_contrastive_positive == "same_class":
        print("WARNING: same-class window-to-flow contrastive works best with --balanced_flow_batches.")
    if args.hierarchical_mode == "expert" and args.hierarchical_logit_weight > 0:
        print("WARNING: --hierarchical_mode expert already uses coarse probabilities; --hierarchical_logit_weight is ignored.")
    if args.hierarchical_mode == "logit" and args.hierarchical_logit_weight > 0 and args.hierarchical_weight <= 0:
        print("WARNING: --hierarchical_logit_weight is enabled but --hierarchical_weight is 0, so the coarse head has no direct supervision.")
    if args.contrastive_mode == "confusion_weighted" and not args.confusion_matrix_json:
        print("WARNING: --contrastive_mode confusion_weighted has no --confusion_matrix_json; falling back to binary group weights.")
    if args.paired_view_dataset and args.train_level != "flow":
        print("WARNING: --paired_view_dataset is only used with --train_level flow.")
    if (
        args.paired_view_dataset
        and args.paired_view_weight <= 0
        and args.paired_consistency_weight <= 0
        and args.view_domain_adversarial_weight <= 0
    ):
        print("WARNING: --paired_view_dataset is set but paired losses are both 0.")
    if args.view_domain_adversarial_weight > 0 and not args.paired_view_dataset:
        print("WARNING: --view_domain_adversarial_weight requires --paired_view_dataset and will be inactive.")
    if args.distill_targets_json and args.distill_weight <= 0:
        print("WARNING: --distill_targets_json is set but --distill_weight <= 0, so distillation is disabled.")
    if args.distill_weight > 0 and not args.distill_targets_json:
        print("WARNING: --distill_weight > 0 but no --distill_targets_json was provided.")
    if args.distill_class_prior_weight > 0 and not args.distill_targets_json:
        print("WARNING: --distill_class_prior_weight > 0 but no --distill_targets_json was provided.")
    if args.model_type == "seq":
        train_seq_flow(args) if args.train_level == "flow" else train_seq(args)
    else:
        train_graph_flow(args) if args.train_level == "flow" else train_graph(args)


if __name__ == "__main__":
    main()
