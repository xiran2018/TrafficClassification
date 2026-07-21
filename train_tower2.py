#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from models.flow_transformer import FlowTransformerClassifier
from models.flow_graph_transformer import FlowGraphTransformerClassifier
from models.counterfactual_flow_fusion import (
    CounterfactualFlowFusion,
    counterfactual_regularization,
    intervention_routing_loss,
)


VPN_APP_GROUPS = "0,2,5,6,10,14;1,4;3,8,9;7,11,13,15;12"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_evidence(path: str | Path) -> Dict[str, Any]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise ValueError(f"provenance input is not a file: {resolved}")
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "size_bytes": int(resolved.stat().st_size),
    }


def tower2_training_input_evidence(args) -> Dict[str, Any]:
    evidence: Dict[str, Any] = {
        "train_dataset": file_evidence(args.dataset),
        "trainer_source": file_evidence(Path(__file__).resolve()),
    }
    optional = {
        "valid_dataset": getattr(args, "valid_dataset", ""),
        "paired_train_dataset": getattr(args, "paired_view_dataset", ""),
        "paired_valid_dataset": getattr(args, "paired_valid_dataset", ""),
        "distillation_targets": getattr(args, "distill_targets_json", ""),
    }
    for name, path in optional.items():
        if path:
            evidence[name] = file_evidence(path)
    return evidence


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
        identifiability_attention_prior: bool = False,
        identifiability_prior_init: float = 0.1,
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
        self.identifiability_attention_prior = bool(identifiability_attention_prior)
        if self.identifiability_attention_prior:
            initial_scale = max(float(identifiability_prior_init), 1e-6)
            self.identifiability_prior_raw_scale = nn.Parameter(
                torch.tensor(math.log(math.expm1(initial_scale)))
            )
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

    def _attention_weights(
        self,
        h: torch.Tensor,
        window_reliability: torch.Tensor | None = None,
    ) -> torch.Tensor:
        scores = self.score(h).squeeze(-1)
        if self.identifiability_attention_prior and window_reliability is not None:
            reliability = window_reliability.to(device=h.device, dtype=h.dtype).flatten()
            if reliability.numel() != h.size(0):
                raise ValueError("window_reliability must have one value per window")
            scale = F.softplus(self.identifiability_prior_raw_scale)
            scores = scores + scale * reliability.clamp(min=0.05, max=1.0).log()
        return torch.softmax(scores, dim=0)

    def pool(
        self,
        h: torch.Tensor,
        window_reliability: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if h.numel() == 0:
            raise ValueError("Cannot pool an empty flow.")
        if self.pooling == "mean":
            return h.mean(dim=0)
        if self.pooling == "multi_view":
            mean = h.mean(dim=0)
            maxv = h.max(dim=0).values
            std = h.std(dim=0, unbiased=False) if h.size(0) > 1 else torch.zeros_like(mean)
            weights = self._attention_weights(h, window_reliability)
            attn = torch.sum(h * weights.unsqueeze(-1), dim=0)
            views = torch.stack([mean, maxv, std, attn], dim=0)
            gate = torch.softmax(self.multi_view_gate(torch.cat([mean, maxv, std, attn], dim=-1)), dim=-1)
            return torch.sum(views * gate.unsqueeze(-1), dim=0), gate
        if self.pooling == "transformer":
            pos = sinusoidal_position_encoding(h.size(0), h.size(1), h.device, h.dtype)
            h = self.flow_encoder((h + pos).unsqueeze(0)).squeeze(0)
        weights = self._attention_weights(h, window_reliability)
        return torch.sum(h * weights.unsqueeze(-1), dim=0)

    def expert_logits(self, emb: torch.Tensor, coarse_logits: torch.Tensor) -> torch.Tensor:
        coarse_log_prob = F.log_softmax(coarse_logits, dim=-1)
        logits = emb.new_full((self.num_classes,), -1e9)
        for coarse_idx, (group, head) in enumerate(zip(self.class_groups, self.expert_heads)):
            fine_log_prob = F.log_softmax(head(emb), dim=-1)
            for local_idx, class_id in enumerate(group):
                logits[class_id] = coarse_log_prob[coarse_idx] + fine_log_prob[local_idx]
        return logits

    def _flow_stat_features(
        self,
        window_x: torch.Tensor | Sequence[torch.Tensor] | None,
        like: torch.Tensor,
        window_mask: torch.Tensor | Sequence[torch.Tensor] | None = None,
        window_ranges: Sequence[Sequence[int] | torch.Tensor] | None = None,
    ) -> torch.Tensor | None:
        if not self.use_flow_stat_expert or window_x is None:
            return None
        if isinstance(window_x, torch.Tensor):
            tensor_windows = (
                [window_x]
                if window_x.ndim == 2
                else [window_x[index] for index in range(window_x.shape[0])]
            )
        else:
            tensor_windows = [
                tensor for tensor in window_x
                if isinstance(tensor, torch.Tensor) and tensor.numel() > 0
            ]
        if not tensor_windows:
            return None
        mask_windows: list[torch.Tensor | None]
        if window_mask is None:
            mask_windows = [None] * len(tensor_windows)
        elif isinstance(window_mask, torch.Tensor):
            mask_windows = (
                [window_mask]
                if window_mask.ndim == 1
                else [window_mask[index] for index in range(window_mask.shape[0])]
            )
        else:
            mask_windows = list(window_mask)
        if len(mask_windows) != len(tensor_windows):
            raise ValueError("window_mask must align with window_x")
        pieces = []
        for tensor, valid_mask in zip(tensor_windows, mask_windows):
            piece = tensor.reshape(-1, tensor.shape[-1])
            if valid_mask is not None:
                valid = valid_mask.reshape(-1).bool()
                if valid.numel() != piece.shape[0]:
                    raise ValueError("each window_mask must match its packet rows")
                piece = piece[valid]
            if piece.numel() > 0:
                pieces.append(piece)
        if not pieces:
            return None
        num_windows = len(pieces)
        if window_ranges is not None:
            if len(window_ranges) != len(pieces):
                raise ValueError("window_ranges must align with non-empty window_x")
            packet_rows: dict[int, torch.Tensor] = {}
            for piece, raw_range in zip(pieces, window_ranges):
                if isinstance(raw_range, torch.Tensor):
                    raw_range = raw_range.detach().cpu().tolist()
                if len(raw_range) != 2:
                    raise ValueError("each window range must contain start and end")
                start, end = int(raw_range[0]), int(raw_range[1])
                if end < start or end - start != piece.shape[0]:
                    raise ValueError("window range length must match valid packet rows")
                for offset, row in enumerate(piece):
                    packet_rows.setdefault(start + offset, row)
            x = torch.stack([packet_rows[index] for index in sorted(packet_rows)], dim=0)
        else:
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
            math.log1p(float(num_windows)),
        ])
        return torch.cat([mean, std, minv, maxv, extras], dim=0).to(device=like.device, dtype=like.dtype)

    def forward(
        self,
        h: torch.Tensor,
        window_logits: torch.Tensor | None = None,
        window_x: torch.Tensor | Sequence[torch.Tensor] | None = None,
        window_mask: torch.Tensor | Sequence[torch.Tensor] | None = None,
        window_ranges: Sequence[Sequence[int] | torch.Tensor] | None = None,
        window_reliability: torch.Tensor | None = None,
    ):
        pooled = self.pool(h, window_reliability=window_reliability)
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
        if self.identifiability_attention_prior:
            out["identifiability_prior_scale"] = F.softplus(
                self.identifiability_prior_raw_scale
            )
        stat_features = self._flow_stat_features(
            window_x,
            emb,
            window_mask=window_mask,
            window_ranges=window_ranges,
        )
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


def distillation_teacher_counts(
    data: Dict[str, Any], path: str, min_teachers_per_flow: int, require_oof_exclusion_proof: bool
) -> Dict[str, int] | None:
    if min_teachers_per_flow <= 0:
        raise ValueError("distill_min_teachers_per_flow must be positive")
    flow_ids = [str(fid) for fid in data.get("flow_ids", [])]
    multiplicity = data.get("teacher_multiplicity")
    contract_required = min_teachers_per_flow > 1 or require_oof_exclusion_proof
    if not isinstance(multiplicity, dict):
        if contract_required:
            raise ValueError(f"{path} lacks teacher_multiplicity required by strict distillation")
        return None
    count_ids = [str(fid) for fid in multiplicity.get("flow_ids", [])]
    raw_counts = multiplicity.get("teacher_counts", [])
    if len(count_ids) != len(raw_counts) or len(set(count_ids)) != len(count_ids):
        raise ValueError(f"{path} has invalid teacher_multiplicity alignment")
    counts = {fid: int(count) for fid, count in zip(count_ids, raw_counts)}
    if set(counts) != set(flow_ids):
        raise ValueError(f"{path} teacher_multiplicity flow IDs do not match target flow IDs")
    if require_oof_exclusion_proof and multiplicity.get("oof_exclusion_proven") is not True:
        raise ValueError(f"{path} does not prove per-flow OOF teacher exclusion")
    return counts


def load_distillation_targets(
    path: str,
    num_classes: int,
    device: str | torch.device,
    min_teachers_per_flow: int = 1,
    require_oof_exclusion_proof: bool = False,
) -> Dict[str, torch.Tensor]:
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
    teacher_counts = distillation_teacher_counts(
        data, path, min_teachers_per_flow, require_oof_exclusion_proof
    )
    targets: Dict[str, torch.Tensor] = {}
    for fid, prob in zip(flow_ids, flow_prob):
        if teacher_counts is not None and teacher_counts[str(fid)] < min_teachers_per_flow:
            continue
        p = torch.as_tensor(prob, dtype=torch.float32, device=device)
        if p.numel() != num_classes:
            raise ValueError(f"{path} target for flow_id={fid} has {p.numel()} classes, expected {num_classes}")
        p = p.clamp(min=1e-8)
        targets[str(fid)] = p / p.sum().clamp(min=1e-8)
    return targets


def load_distillation_class_priors(
    path: str,
    num_classes: int,
    device: str | torch.device,
    min_teachers_per_flow: int = 1,
    require_oof_exclusion_proof: bool = False,
) -> torch.Tensor | None:
    """Average teacher distributions by true class for class-conditional distillation."""
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    y_true = data.get("flow_y_true", [])
    flow_prob = data.get("flow_prob", [])
    flow_ids = [str(fid) for fid in data.get("flow_ids", [])]
    if not y_true or not flow_prob or len(y_true) != len(flow_prob):
        return None
    if len(flow_ids) != len(flow_prob):
        raise ValueError(f"{path} has mismatched flow_ids/flow_prob lengths")
    teacher_counts = distillation_teacher_counts(
        data, path, min_teachers_per_flow, require_oof_exclusion_proof
    )
    sums = torch.zeros(num_classes, num_classes, dtype=torch.float32, device=device)
    counts = torch.zeros(num_classes, dtype=torch.float32, device=device)
    for flow_id, label, prob in zip(flow_ids, y_true, flow_prob):
        if teacher_counts is not None and teacher_counts[flow_id] < min_teachers_per_flow:
            continue
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


def report_distillation_coverage(name: str, targets: Dict[str, torch.Tensor], flow_ids: Sequence[str]) -> Dict[str, Any]:
    if not targets:
        return {
            "scope": name,
            "target_count": 0,
            "unique_flow_count": len(set(map(str, flow_ids))),
            "matched_flow_count": 0,
            "coverage": 0.0,
        }
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
    return msg


def apply_distillation_coverage_policy(
    name: str,
    targets: Dict[str, torch.Tensor],
    flow_ids: Sequence[str],
    args,
) -> Dict[str, torch.Tensor]:
    msg = report_distillation_coverage(name, targets, flow_ids)
    if not targets or args.distill_min_coverage <= 0:
        return targets
    coverage = float(msg.get("coverage", 0.0))
    if coverage >= float(args.distill_min_coverage):
        return targets
    action = str(args.distill_low_coverage_action)
    policy_msg = {
        "scope": name,
        "coverage": coverage,
        "min_coverage": float(args.distill_min_coverage),
        "action": action,
    }
    print("distillation_coverage_policy " + json.dumps(policy_msg, sort_keys=True), flush=True)
    if action == "fail":
        raise ValueError(
            f"{name} distillation coverage {coverage:.4f} is below "
            f"--distill_min_coverage={args.distill_min_coverage:.4f}"
        )
    if action == "disable_flow":
        return {}
    return targets


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


def sample_split_key(item: dict, index: int, group_key: str) -> str:
    if group_key == "flow_id":
        return str(item.get("flow_id", index))
    if group_key == "content_group_id":
        if "content_group_id" not in item:
            raise ValueError(
                "--split_group_key content_group_id requires Tower-2 datasets "
                "built with preprocess_tower2.py --content_group_index"
            )
        return f"content::{item['content_group_id']}"
    raise ValueError(f"unknown split group key: {group_key}")


def group_split_by_flow(ds: Dataset, valid_ratio: float, seed: int, group_key: str = "flow_id"):
    """Split by a stable group key, not by window, to avoid validation leakage."""
    group_to_indices: Dict[str, List[int]] = defaultdict(list)
    for i in range(len(ds)):
        group_to_indices[sample_split_key(ds[i], i, group_key)].append(i)
    flows = list(group_to_indices.keys())
    rng = random.Random(seed)
    rng.shuffle(flows)
    n_val = max(1, int(len(flows) * valid_ratio)) if len(flows) > 1 and valid_ratio > 0 else 0
    val_flows = set(flows[:n_val])
    tr_idx, va_idx = [], []
    for f, idxs in group_to_indices.items():
        (va_idx if f in val_flows else tr_idx).extend(idxs)
    return Subset(ds, tr_idx), Subset(ds, va_idx)


def build_flow_groups(ds: Dataset):
    flow_to_items: Dict[str, List[dict]] = defaultdict(list)
    flow_labels: Dict[str, int] = {}
    flow_content_group_ids: Dict[str, int] = {}
    flow_content_hashes: Dict[str, str] = {}
    for i in range(len(ds)):
        item = ds[i]
        label = int(item.get("label", -1))
        if label < 0:
            continue
        flow_id = str(item.get("flow_id", i))
        flow_to_items[flow_id].append(item)
        flow_labels[flow_id] = label
        if "content_group_id" in item:
            flow_content_group_ids[flow_id] = int(item["content_group_id"])
        if "content_hash" in item:
            flow_content_hashes[flow_id] = str(item["content_hash"])
    groups = []
    for flow_id, items in flow_to_items.items():
        if not items:
            continue
        group = {"flow_id": flow_id, "label": flow_labels[flow_id], "items": items}
        if flow_id in flow_content_group_ids:
            group["content_group_id"] = flow_content_group_ids[flow_id]
        if flow_id in flow_content_hashes:
            group["content_hash"] = flow_content_hashes[flow_id]
        groups.append(group)
    return groups


def flow_group_split_key(group: dict, group_key: str) -> str:
    if group_key == "flow_id":
        return str(group["flow_id"])
    if group_key == "content_group_id":
        if "content_group_id" not in group:
            raise ValueError(
                "--split_group_key content_group_id requires Tower-2 datasets "
                "built with preprocess_tower2.py --content_group_index"
            )
        return f"content::{group['content_group_id']}"
    raise ValueError(f"unknown split group key: {group_key}")


def split_flow_groups(groups: List[dict], valid_ratio: float, seed: int, group_key: str = "flow_id"):
    split_to_groups: Dict[str, List[dict]] = defaultdict(list)
    for group in groups:
        split_to_groups[flow_group_split_key(group, group_key)].append(group)
    split_keys = list(split_to_groups.keys())
    rng = random.Random(seed)
    rng.shuffle(split_keys)
    n_val = max(1, int(len(split_keys) * valid_ratio)) if len(split_keys) > 1 and valid_ratio > 0 else 0
    val_keys = set(split_keys[:n_val])
    tr_groups: List[dict] = []
    va_groups: List[dict] = []
    for key, bucket in split_to_groups.items():
        (va_groups if key in val_keys else tr_groups).extend(bucket)
    return tr_groups, va_groups


def content_group_key(group: dict) -> str | None:
    if "content_group_id" not in group:
        return None
    return str(group["content_group_id"])


def choose_content_unique(
    candidates: Sequence[dict],
    used_content_groups: set[str],
    require_unique: bool,
) -> dict:
    if not candidates:
        raise ValueError("cannot choose from empty candidates")
    pool = list(candidates)
    if require_unique:
        unique_pool = [
            group for group in pool
            if content_group_key(group) is None or content_group_key(group) not in used_content_groups
        ]
        if unique_pool:
            pool = unique_pool
    chosen = random.choice(pool)
    key = content_group_key(chosen)
    if key is not None:
        used_content_groups.add(key)
    return chosen


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
    content_group_unique: bool = False,
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
        used_content_groups: set[str] = set()
        for label in chosen_labels:
            candidates = label_to_groups[label]
            environment_to_groups: Dict[int, List[dict]] = defaultdict(list)
            for candidate in candidates:
                environment = int(candidate.get("environment", -1))
                if environment >= 0:
                    environment_to_groups[environment].append(candidate)
            if len(environment_to_groups) >= 2 and samples_per_class >= 2:
                environments = list(environment_to_groups)
                random.shuffle(environments)
                selected = [
                    choose_content_unique(
                        environment_to_groups[env],
                        used_content_groups,
                        content_group_unique,
                    )
                    for env in environments[:samples_per_class]
                ]
                while len(selected) < samples_per_class:
                    selected.append(
                        choose_content_unique(
                            candidates,
                            used_content_groups,
                            content_group_unique,
                        )
                    )
                batch.extend(selected)
            elif len(candidates) >= samples_per_class:
                if content_group_unique:
                    selected = [
                        choose_content_unique(candidates, used_content_groups, True)
                        for _ in range(samples_per_class)
                    ]
                else:
                    selected = random.sample(candidates, samples_per_class)
                batch.extend(selected)
            else:
                batch.extend(
                    choose_content_unique(candidates, used_content_groups, content_group_unique)
                    for _ in range(samples_per_class)
                )
        random.shuffle(batch)
        yield batch


def iter_train_flow_batches(groups: List[dict], args):
    if args.balanced_flow_batches:
        yield from iter_balanced_group_batches(
            groups,
            args.batch_size,
            classes_per_batch=args.classes_per_batch,
            samples_per_class=args.samples_per_class,
            content_group_unique=args.content_group_unique_batches,
        )
    else:
        yield from iter_group_batches(groups, args.batch_size, shuffle=True)


def split_or_external_window_dataset(ds: Dataset, dataset_cls, args):
    if args.valid_dataset:
        return ds, dataset_cls(args.valid_dataset)
    return group_split_by_flow(ds, args.valid_ratio, args.seed, args.split_group_key)


def split_or_external_flow_groups(ds: Dataset, dataset_cls, args):
    groups = build_flow_groups(ds)
    if args.valid_dataset:
        return groups, build_flow_groups(dataset_cls(args.valid_dataset))
    return split_flow_groups(groups, args.valid_ratio, args.seed, args.split_group_key)


def paired_flow_group_lookup(path: str, dataset_cls) -> Dict[str, dict]:
    if not path:
        return {}
    groups = build_flow_groups(dataset_cls(path))
    return {str(group["flow_id"]): group for group in groups}


def window_identifiability(item: dict) -> float:
    reliability = item.get("packet_identifiability")
    if not isinstance(reliability, torch.Tensor) or reliability.numel() == 0:
        return 1.0
    return float(reliability.float().mean().clamp(0.0, 1.0).item())


def identifiability_feature_index(input_dim: int, args) -> int:
    if not (
        args.packet_identifiability_pooling
        or args.packet_identifiability_dual_pooling
        or args.packet_identifiability_evidence_adapter
        or args.identifiability_feature_mode != "observed"
    ):
        return -1
    index = int(input_dim) - int(args.meta_feature_dim) - 2
    if index < 0:
        raise ValueError(
            "packet identifiability pooling requires the two profile features "
            "inserted before trailing metadata by preprocess_tower2.py"
        )
    return index


def configure_identifiability_adapter_only(
    model: nn.Module,
    flow_head: nn.Module | None,
) -> List[str]:
    for parameter in model.parameters():
        parameter.requires_grad = False
    if flow_head is not None:
        for parameter in flow_head.parameters():
            parameter.requires_grad = False
    prefixes = (
        "pool.reliability_prior_raw_scale",
        "pool.unidentifiable_prior_raw_scale",
        "pool.reliability_residual_raw_gate",
        "pool.dual_gate.",
        "pool.evidence_adapter.",
    )
    trainable = []
    for name, parameter in model.named_parameters():
        if name.startswith(prefixes):
            parameter.requires_grad = True
            trainable.append(name)
    if not trainable:
        raise ValueError(
            "--identifiability_adapter_only requires dual identifiability pooling"
        )
    return trainable


def configure_dual_channel_train_scope(
    model: nn.Module,
    flow_head: nn.Module | None,
    scope: str,
) -> List[str]:
    if scope == "full":
        return [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if getattr(model, "dual_channel_mode", "concat") != "residual":
        raise ValueError("dual-channel restricted scopes require --dual_channel_mode residual")
    for parameter in model.parameters():
        parameter.requires_grad = False
    if flow_head is not None:
        for parameter in flow_head.parameters():
            parameter.requires_grad = False
    prefixes = [
        "channel_interaction.",
        "shared_packet_fusion.interaction.",
        "shared_packet_fusion.gate.",
    ]
    if scope == "native_interaction":
        if getattr(model, "native_structural_dim", 0) <= 0:
            raise ValueError(
                "--dual_channel_train_scope native_interaction requires "
                "--native_structural_dim > 0"
            )
        prefixes.append("native_structural_adapter.")
        prefixes.append("native_structural_raw_gate")
    if scope == "new_modules":
        prefixes.extend(
            [
                "semantic_proj.",
                "structural_proj.",
                "fusion_bias",
                "semantic_channel_cls.",
                "structural_channel_cls.",
            ]
        )
    for name, parameter in model.named_parameters():
        if any(name == prefix or name.startswith(prefix) for prefix in prefixes):
            parameter.requires_grad = True
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError(f"dual-channel scope {scope} selected no trainable parameters")
    return trainable


def set_dual_channel_restricted_train_mode(model: nn.Module, flow_head: nn.Module) -> None:
    model.eval()
    flow_head.eval()
    model.channel_interaction.train()
    if hasattr(model, "native_structural_adapter"):
        model.native_structural_adapter.train()
    model.semantic_channel_cls.train()
    model.structural_channel_cls.train()


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
    windows = [item.get("window") for item in batch]
    window_reliability = torch.tensor(
        [window_identifiability(item) for item in batch], dtype=torch.float32
    )
    for b, item in enumerate(batch):
        n = item["x"].shape[0]
        x[b, :n] = item["x"]
        mask[b, :n] = True
    return {"x": x, "mask": mask, "label": labels, "coherence_label": coh, "next_direction": nd, "next_length_bin": nl, "next_iat_bin": ni, "flow_id": flow_ids, "window": windows, "window_reliability": window_reliability}


def flow_window_ranges(batch: Dict[str, Any], owners: Sequence[int], flow_idx: int):
    ranges = [
        window_range
        for window_range, owner in zip(batch.get("window", []), owners)
        if owner == flow_idx
    ]
    return ranges if ranges and all(item is not None for item in ranges) else None


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
    content_group_ids: Sequence[Any] | None = None,
    content_group_reduction: str = "none",
) -> torch.Tensor:
    valid = y >= 0
    if valid.any():
        selected_logits = logits[valid]
        selected_y = y[valid]
        log_prob = F.log_softmax(selected_logits, dim=-1)
        ce = F.cross_entropy(
            selected_logits,
            selected_y,
            weight=class_weight,
            label_smoothing=label_smoothing,
            reduction="none",
        )
        if focal_gamma > 0:
            pt = log_prob.gather(1, selected_y.view(-1, 1)).squeeze(1).exp().clamp(min=1e-6, max=1.0)
            ce = (1.0 - pt).pow(float(focal_gamma)) * ce
        if content_group_reduction == "group_mean":
            return content_group_mean_loss(ce, valid, content_group_ids)
        return ce.mean()
    return torch.tensor(0.0, device=logits.device)


def content_group_mean_loss(
    per_sample_loss: torch.Tensor,
    valid_mask: torch.Tensor,
    content_group_ids: Sequence[Any] | None,
) -> torch.Tensor:
    if content_group_ids is None:
        raise ValueError(
            "--content_group_loss_reduction group_mean requires content_group_id "
            "metadata from preprocess_tower2.py --content_group_index"
        )
    if len(content_group_ids) != int(valid_mask.numel()):
        raise ValueError("content_group_ids length must match the logits/label batch")
    selected_groups = [
        str(content_group_ids[idx])
        for idx, is_valid in enumerate(valid_mask.detach().cpu().tolist())
        if is_valid and content_group_ids[idx] is not None
    ]
    if len(selected_groups) != int(per_sample_loss.numel()):
        raise ValueError(
            "--content_group_loss_reduction group_mean requires every valid sample "
            "to have content_group_id metadata"
        )
    buckets: Dict[str, List[int]] = defaultdict(list)
    for idx, group_id in enumerate(selected_groups):
        buckets[group_id].append(idx)
    if not buckets:
        raise ValueError(
            "--content_group_loss_reduction group_mean found no usable content groups"
        )
    group_losses = [
        per_sample_loss.new_tensor(0.0) + per_sample_loss[indices].mean()
        for indices in buckets.values()
    ]
    return torch.stack(group_losses).mean()


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
    content_group_ids: Sequence[Any] | None = None,
    content_group_reduction: str | None = None,
) -> torch.Tensor:
    return class_weighted_loss(
        logits,
        y,
        class_weight,
        args.label_smoothing,
        args.focal_gamma,
        content_group_ids=content_group_ids,
        content_group_reduction=(
            getattr(args, "content_group_loss_reduction", "none")
            if content_group_reduction is None
            else content_group_reduction
        ),
    ) + confidence_penalty_loss(
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


def cap_prob_target_confidence(prob: torch.Tensor, max_confidence: float = 0.0) -> torch.Tensor:
    """Soft-cap overconfident teacher targets by mixing with a uniform prior."""
    if max_confidence <= 0.0 or prob.numel() == 0:
        return prob
    num_classes = prob.size(-1)
    if num_classes <= 1:
        return prob
    uniform_value = 1.0 / float(num_classes)
    cap = float(max_confidence)
    if cap >= 1.0:
        return prob
    cap = max(cap, uniform_value)
    p = prob.clamp(min=1e-8)
    p = p / p.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    max_prob = p.max(dim=-1, keepdim=True).values
    denom = (max_prob - uniform_value).clamp(min=1e-8)
    alpha = ((cap - uniform_value) / denom).clamp(min=0.0, max=1.0)
    alpha = torch.where(max_prob > cap, alpha, torch.ones_like(alpha))
    capped = alpha * p + (1.0 - alpha) * uniform_value
    return capped / capped.sum(dim=-1, keepdim=True).clamp(min=1e-8)


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
    teacher = cap_prob_target_confidence(teacher, args.distill_max_confidence)
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
    teacher = cap_prob_target_confidence(teacher, args.distill_max_confidence)
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


def load_flow_environment_map(path: str) -> tuple[Dict[str, int], List[str]]:
    if not path:
        return {}, []
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    raw = payload.get("flow_to_environment", payload)
    mapping = {str(flow_id): int(environment) for flow_id, environment in raw.items()}
    names = [str(name) for name in payload.get("environment_names", [])]
    return mapping, names


def attach_flow_environments(groups: Sequence[dict], path: str, scope: str) -> tuple[int, int]:
    mapping, names = load_flow_environment_map(path)
    if not mapping:
        return 0, len(groups)
    matched = 0
    counts: Dict[int, int] = defaultdict(int)
    for group in groups:
        environment = mapping.get(str(group["flow_id"]), -1)
        group["environment"] = environment
        if environment >= 0:
            matched += 1
            counts[environment] += 1
    report = {
        "scope": scope,
        "matched": matched,
        "total": len(groups),
        "coverage": matched / max(1, len(groups)),
        "counts": {
            names[idx] if idx < len(names) else str(idx): count for idx, count in sorted(counts.items())
        },
    }
    print("environment_coverage " + json.dumps(report, sort_keys=True), flush=True)
    return matched, len(groups)


def environment_risk_variance_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    environments: torch.Tensor,
    class_weight: torch.Tensor | None,
    weight: float,
) -> torch.Tensor:
    if weight <= 0:
        return logits.sum() * 0.0
    valid = (labels >= 0) & (environments >= 0)
    unique = environments[valid].unique()
    if unique.numel() < 2:
        return logits.sum() * 0.0
    per_sample = F.cross_entropy(logits[valid], labels[valid], weight=class_weight, reduction="none")
    valid_environments = environments[valid]
    risks = torch.stack([per_sample[valid_environments == env].mean() for env in unique])
    return float(weight) * risks.var(unbiased=False)


def class_conditional_environment_alignment_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    environments: torch.Tensor,
    weight: float,
) -> torch.Tensor:
    if weight <= 0:
        return embeddings.sum() * 0.0
    losses = []
    for label in labels.unique():
        class_mask = labels == label
        class_environments = environments[class_mask]
        valid_environments = class_environments[class_environments >= 0].unique()
        if valid_environments.numel() < 2:
            continue
        class_embeddings = embeddings[class_mask]
        means = []
        for environment in valid_environments:
            means.append(class_embeddings[class_environments == environment].mean(dim=0))
        means = F.normalize(torch.stack(means, dim=0).float(), dim=-1)
        similarity = means @ means.T
        pair_mask = ~torch.eye(len(means), dtype=torch.bool, device=means.device)
        losses.append((1.0 - similarity[pair_mask]).mean())
    if not losses:
        return embeddings.sum() * 0.0
    return float(weight) * torch.stack(losses).mean()


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
    model_state = dict(ckpt["model_state"])
    current_state = model.state_dict()
    projection_key = "proj.weight"
    projection_adaptation = None
    if (
        getattr(model, "dual_channel_mode", "concat") == "residual"
        and projection_key in model_state
    ):
        old_weight = model_state.pop(projection_key)
        old_bias = model_state.pop("proj.bias")
        imported_weight = old_weight
        added_structural_columns = 0
        expected_width = model.embedding_feature_dim + model.meta_feature_dim
        if old_weight.size(1) != expected_width:
            old_meta_dim = int(ckpt.get("meta_feature_dim", 14))
            old_semantic_dim = old_weight.size(1) - old_meta_dim
            if (
                old_semantic_dim != model.embedding_feature_dim
                or model.meta_feature_dim < old_meta_dim
            ):
                raise ValueError(
                    "cannot preserve concat checkpoint while expanding the structural "
                    f"channel: old={tuple(old_weight.shape)} semantic={model.embedding_feature_dim} "
                    f"structural={model.meta_feature_dim} old_meta={old_meta_dim}"
                )
            imported_weight = old_weight.new_zeros((old_weight.size(0), expected_width))
            imported_weight[:, :old_semantic_dim] = old_weight[:, :old_semantic_dim]
            if old_meta_dim > 0:
                imported_weight[:, -old_meta_dim:] = old_weight[:, -old_meta_dim:]
            added_structural_columns = model.meta_feature_dim - old_meta_dim
        model.import_concat_projection(imported_weight, old_bias)
        projection_adaptation = {
            "old_shape": list(old_weight.shape),
            "imported_shape": list(imported_weight.shape),
            "semantic_shape": list(model.semantic_proj.weight.shape),
            "structural_shape": list(model.structural_proj.weight.shape),
            "zero_initialized_structural_columns": added_structural_columns,
            "mode": (
                "exact_concat_to_expanded_dual_channel"
                if added_structural_columns
                else "exact_concat_to_dual_channel"
            ),
        }
    if (
        projection_key in model_state
        and projection_key in current_state
        and model_state[projection_key].shape != current_state[projection_key].shape
    ):
        old_weight = model_state[projection_key]
        new_weight = current_state[projection_key].clone()
        old_meta_dim = int(ckpt.get("meta_feature_dim", args.meta_feature_dim))
        added_columns = new_weight.size(1) - old_weight.size(1)
        old_embedding_dim = old_weight.size(1) - old_meta_dim
        if (
            old_weight.size(0) != new_weight.size(0)
            or added_columns <= 0
            or old_embedding_dim < 0
            or old_meta_dim > new_weight.size(1)
        ):
            raise ValueError(
                f"cannot adapt {projection_key} from {tuple(old_weight.shape)} "
                f"to {tuple(new_weight.shape)}"
            )
        new_weight.zero_()
        new_weight[:, :old_embedding_dim] = old_weight[:, :old_embedding_dim]
        if old_meta_dim > 0:
            new_weight[:, -old_meta_dim:] = old_weight[:, -old_meta_dim:]
        model_state[projection_key] = new_weight
        projection_adaptation = {
            "old_shape": list(old_weight.shape),
            "new_shape": list(new_weight.shape),
            "zero_initialized_columns": added_columns,
            "meta_feature_dim": old_meta_dim,
        }
    missing, unexpected = model.load_state_dict(model_state, strict=False)
    msg = {
        "checkpoint": args.init_checkpoint,
        "model_missing": list(missing),
        "model_unexpected": list(unexpected),
    }
    if projection_adaptation is not None:
        msg["input_projection_adaptation"] = projection_adaptation
    if flow_head is not None and ckpt.get("flow_head_state") is not None:
        flow_missing, flow_unexpected = flow_head.load_state_dict(ckpt["flow_head_state"], strict=False)
        msg["flow_head_missing"] = list(flow_missing)
        msg["flow_head_unexpected"] = list(flow_unexpected)
    if getattr(args, "counterfactual_head_only", False):
        incompatible = {
            key: values
            for key, values in msg.items()
            if key in {
                "model_missing",
                "model_unexpected",
                "flow_head_missing",
                "flow_head_unexpected",
            }
            and values
        }
        if flow_head is not None and "flow_head_missing" not in msg:
            incompatible["flow_head_state"] = ["missing from checkpoint"]
        if incompatible:
            raise ValueError(
                "counterfactual head-only training requires an exactly compatible "
                f"frozen checkpoint: {json.dumps(incompatible, sort_keys=True)}"
            )
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


def content_group_metrics_from_lists(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    content_group_ids: Sequence[Any],
    num_classes: int,
) -> Dict[str, float | int]:
    if not y_true or len(content_group_ids) != len(y_true):
        return {}
    buckets: Dict[str, List[int]] = defaultdict(list)
    for idx, group_id in enumerate(content_group_ids):
        if group_id is None:
            continue
        buckets[str(group_id)].append(idx)
    if not buckets:
        return {}
    group_y_true: List[int] = []
    group_y_pred: List[int] = []
    for group_id, indices in buckets.items():
        labels = {int(y_true[index]) for index in indices}
        if len(labels) != 1:
            raise ValueError(
                f"content_group_id={group_id} has conflicting labels: {sorted(labels)}"
            )
        votes = [int(y_pred[index]) for index in indices]
        counts = np.bincount(votes, minlength=num_classes)
        group_y_true.append(next(iter(labels)))
        group_y_pred.append(int(counts.argmax()))
    metrics = classification_metrics_from_lists(group_y_true, group_y_pred, num_classes)
    return {
        "content_group_accuracy": metrics["accuracy"],
        "content_group_macro_f1": metrics["macro_f1"],
        "content_group_count": len(group_y_true),
        "content_group_rows": sum(len(indices) for indices in buckets.values()),
    }


def merge_content_group_metrics(
    metrics: Dict[str, float],
    y_true: Sequence[int],
    y_pred: Sequence[int],
    content_group_ids: Sequence[Any],
    num_classes: int,
) -> Dict[str, float]:
    group_metrics = content_group_metrics_from_lists(
        y_true, y_pred, content_group_ids, num_classes
    )
    if group_metrics:
        metrics.update(group_metrics)
    return metrics


def selected_metric_value(metrics: Dict[str, float], metric_name: str) -> float:
    aliases = {
        "accuracy": "accuracy",
        "acc": "accuracy",
        "flow_acc": "accuracy",
        "window_acc": "accuracy",
        "macro_f1": "macro_f1",
        "flow_macro_f1": "macro_f1",
        "window_macro_f1": "macro_f1",
        "content_group_acc": "content_group_accuracy",
        "content_group_accuracy": "content_group_accuracy",
        "content_group_macro_f1": "content_group_macro_f1",
        "flow_content_group_acc": "content_group_accuracy",
        "flow_content_group_macro_f1": "content_group_macro_f1",
    }
    key = aliases[metric_name]
    if key not in metrics:
        raise ValueError(
            f"--select_metric {metric_name} requires metric '{key}', but it was "
            "not produced. Rebuild Tower-2 datasets with --content_group_index "
            "or choose a standard flow/window metric."
        )
    return float(metrics[key])


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
    model = FlowTransformerClassifier(
        sample["x"].shape[1], args.num_classes, args.hidden_dim, args.num_layers,
        args.num_heads, args.dropout,
        identifiability_feature_index=identifiability_feature_index(sample["x"].shape[1], args),
        identifiability_pooling=args.packet_identifiability_pooling,
        identifiability_feature_mode=args.identifiability_feature_mode,
        identifiability_prior_init=args.identifiability_prior_init,
        identifiability_dual_pooling=args.packet_identifiability_dual_pooling,
        identifiability_evidence_adapter=args.packet_identifiability_evidence_adapter,
        identifiability_adapter_max_delta=args.identifiability_adapter_max_delta,
        identifiability_residual_max_weight=args.identifiability_residual_max_weight,
        identifiability_residual_init=args.identifiability_residual_init,
        dual_channel_mode=args.dual_channel_mode,
        meta_feature_dim=args.meta_feature_dim,
        native_structural_dim=args.native_structural_dim,
        dual_channel_max_weight=args.dual_channel_max_weight,
        dual_channel_init=args.dual_channel_init,
        dual_channel_gate_mode=args.dual_channel_gate_mode,
        channel_fusion_base_mode=args.channel_fusion_base_mode,
        use_intervention_views=args.use_intervention_views,
        intervention_max_residual_weight=args.intervention_max_residual_weight,
        intervention_view_base_mode=args.intervention_view_base_mode,
        exact_shared_packet_encoder=args.exact_shared_packet_encoder,
        shared_packet_hidden_dim=args.shared_packet_hidden_dim,
        packet_evidence_max_weight=args.packet_evidence_max_weight,
        train_ablate_input_channel=args.train_ablate_input_channel,
        train_ablate_intervention_view=args.train_ablate_intervention_view,
        train_fixed_channel_fusion=args.train_fixed_channel_fusion,
    ).to(args.device)
    maybe_load_init_checkpoint(args, model)
    distill_targets = load_distillation_targets(
        args.distill_targets_json,
        args.num_classes,
        args.device,
        args.distill_min_teachers_per_flow,
        args.distill_require_oof_exclusion_proof,
    )
    distill_targets = apply_distillation_coverage_policy("train_seq", distill_targets, dataset_flow_ids(tr), args)
    distill_class_priors = load_distillation_class_priors(
        args.distill_targets_json,
        args.num_classes,
        args.device,
        args.distill_min_teachers_per_flow,
        args.distill_require_oof_exclusion_proof,
    )
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
            loss = loss + dual_channel_auxiliary_loss(out, y, class_weight, args)
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


def dual_channel_auxiliary_loss(
    out: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    class_weight: torch.Tensor | None,
    args,
    owners: torch.Tensor | None = None,
    num_flows: int | None = None,
    content_group_ids: Sequence[Any] | None = None,
) -> torch.Tensor:
    semantic = out.get("semantic_channel_logits")
    structural = out.get("structural_channel_logits")
    if semantic is None or structural is None:
        return out["logits"].sum() * 0.0
    fused = out["logits"]
    if owners is not None:
        if num_flows is None:
            raise ValueError("num_flows is required with dual-channel owners")
        semantic = mean_logits_by_flow(semantic, owners, num_flows)
        structural = mean_logits_by_flow(structural, owners, num_flows)
        fused = mean_logits_by_flow(fused, owners, num_flows)
    loss = fused.sum() * 0.0
    if args.dual_channel_semantic_aux_weight > 0:
        loss = loss + args.dual_channel_semantic_aux_weight * regularized_classification_loss(
            semantic, labels, class_weight, args, content_group_ids=content_group_ids
        )
    if args.dual_channel_structural_aux_weight > 0:
        loss = loss + args.dual_channel_structural_aux_weight * regularized_classification_loss(
            structural, labels, class_weight, args, content_group_ids=content_group_ids
        )
    if args.dual_channel_consistency_weight > 0:
        loss = loss + args.dual_channel_consistency_weight * 0.5 * (
            consistency_kl_loss(fused, semantic, args.consistency_temperature)
            + consistency_kl_loss(fused, structural, args.consistency_temperature)
        )
    return loss


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


def flow_stat_aux_loss(
    stat_logits: List[torch.Tensor],
    y: torch.Tensor,
    class_weight: torch.Tensor | None,
    args,
    content_group_ids: Sequence[Any] | None = None,
) -> torch.Tensor | None:
    if args.flow_stat_aux_weight <= 0 or not stat_logits:
        return None
    if len(stat_logits) != y.numel():
        return None
    logits = torch.stack(stat_logits, dim=0)
    return args.flow_stat_aux_weight * regularized_classification_loss(
        logits, y, class_weight, args, content_group_ids=content_group_ids
    )


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


def paired_semantic_alignment_enabled(args) -> bool:
    return any(
        float(value) > 0
        for value in (
            args.paired_alignment_weight,
            args.paired_crossview_contrastive_weight,
            args.paired_variance_weight,
            args.paired_covariance_weight,
        )
    )


def paired_semantic_alignment_loss(
    clean_embs: torch.Tensor,
    paired_embs: torch.Tensor | None,
    labels: torch.Tensor,
    args,
) -> torch.Tensor:
    """Learn endpoint-invariant flow semantics from two header views.

    The clean and randomized-header views are aligned per flow. A joint
    supervised contrastive term preserves class structure, while variance and
    covariance regularizers prevent an invariance-only representation from
    collapsing or concentrating on a few dimensions.
    """
    zero = clean_embs.sum() * 0.0
    if not paired_semantic_alignment_enabled(args) or paired_embs is None:
        return zero
    if clean_embs.shape != paired_embs.shape or clean_embs.numel() == 0:
        return zero

    clean = F.normalize(clean_embs.float(), p=2, dim=-1)
    paired = F.normalize(paired_embs.float(), p=2, dim=-1)
    loss = zero

    if args.paired_alignment_weight > 0:
        alignment = (1.0 - (clean * paired).sum(dim=-1)).mean()
        loss = loss + args.paired_alignment_weight * alignment

    if args.paired_crossview_contrastive_weight > 0:
        joint = torch.cat([clean, paired], dim=0)
        joint_labels = torch.cat([labels, labels], dim=0)
        crossview = supervised_contrastive_loss(
            joint,
            joint_labels,
            temperature=args.paired_crossview_temperature,
        )
        loss = loss + args.paired_crossview_contrastive_weight * crossview

    if args.paired_variance_weight > 0:
        target = max(float(args.paired_variance_target), 0.0)
        clean_std = torch.sqrt(clean.var(dim=0, unbiased=False) + 1e-4)
        paired_std = torch.sqrt(paired.var(dim=0, unbiased=False) + 1e-4)
        variance = 0.5 * (F.relu(target - clean_std).mean() + F.relu(target - paired_std).mean())
        loss = loss + args.paired_variance_weight * variance

    if args.paired_covariance_weight > 0 and clean.size(0) > 1:
        def covariance_penalty(z: torch.Tensor) -> torch.Tensor:
            centered = z - z.mean(dim=0, keepdim=True)
            cov = centered.T @ centered / float(max(1, z.size(0) - 1))
            eye = torch.eye(cov.size(0), dtype=torch.bool, device=cov.device)
            return cov.masked_select(~eye).pow(2).mean()

        covariance = 0.5 * (covariance_penalty(clean) + covariance_penalty(paired))
        loss = loss + args.paired_covariance_weight * covariance
    return loss


def build_counterfactual_fusion(args) -> CounterfactualFlowFusion | None:
    if args.counterfactual_fusion == "none":
        return None
    return CounterfactualFlowFusion(
        args.hidden_dim,
        args.num_classes,
        mode=args.counterfactual_fusion,
        base_mode=args.counterfactual_base_mode,
        max_residual_weight=args.counterfactual_max_residual_weight,
        initial_residual_fraction=args.counterfactual_initial_residual_fraction,
        dropout=args.dropout,
    ).to(args.device)


def configure_counterfactual_head_only(
    model: nn.Module,
    flow_head: nn.Module,
    fusion_head: CounterfactualFlowFusion | None,
) -> List[str]:
    if fusion_head is None or fusion_head.mode not in {"counterfactual", "router"}:
        raise ValueError(
            "--counterfactual_head_only requires --counterfactual_fusion counterfactual"
        )
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in flow_head.parameters():
        parameter.requires_grad = False
    trainable = []
    for name, parameter in fusion_head.named_parameters():
        parameter.requires_grad = True
        trainable.append(name)
    return trainable


def counterfactual_training_loss(
    fusion_head: CounterfactualFlowFusion | None,
    clean_embeddings: torch.Tensor,
    paired_embeddings: torch.Tensor | None,
    clean_logits: torch.Tensor,
    paired_logits: torch.Tensor | None,
    labels: torch.Tensor,
    class_weight: torch.Tensor | None,
    args,
    content_group_ids: Sequence[Any] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
    zero = clean_logits.sum() * 0.0
    if fusion_head is None or paired_embeddings is None or paired_logits is None:
        return zero, None
    output = fusion_head(
        clean_embeddings,
        paired_embeddings,
        clean_logits,
        paired_logits,
    )
    loss = args.counterfactual_fusion_weight * regularized_classification_loss(
        output["logits"], labels, class_weight, args, content_group_ids=content_group_ids
    )
    if args.counterfactual_shared_ce_weight > 0:
        loss = loss + args.counterfactual_shared_ce_weight * regularized_classification_loss(
            output["base_logits"], labels, class_weight, args, content_group_ids=content_group_ids
        )
    loss = loss + counterfactual_regularization(
        output,
        gate_weight=args.counterfactual_gate_weight,
        orthogonality_weight=args.counterfactual_orthogonality_weight,
    )
    loss = loss + intervention_routing_loss(
        output,
        clean_logits,
        paired_logits,
        labels,
        args.counterfactual_routing_weight,
    )
    return loss, output


def train_seq_flow(args):
    ds = SeqDataset(args.dataset)
    tr_groups, va_groups = split_or_external_flow_groups(ds, SeqDataset, args)
    attach_flow_environments(tr_groups, args.environment_map_json, "train_seq_flow")
    paired_groups = paired_flow_group_lookup(args.paired_view_dataset, SeqDataset)
    paired_valid_groups = paired_flow_group_lookup(args.paired_valid_dataset, SeqDataset)
    distill_targets = load_distillation_targets(
        args.distill_targets_json,
        args.num_classes,
        args.device,
        args.distill_min_teachers_per_flow,
        args.distill_require_oof_exclusion_proof,
    )
    distill_targets = apply_distillation_coverage_policy("train_seq_flow", distill_targets, group_flow_ids(tr_groups), args)
    distill_class_priors = load_distillation_class_priors(
        args.distill_targets_json,
        args.num_classes,
        args.device,
        args.distill_min_teachers_per_flow,
        args.distill_require_oof_exclusion_proof,
    )
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
    model = FlowTransformerClassifier(
        sample["x"].shape[1], args.num_classes, args.hidden_dim, args.num_layers,
        args.num_heads, args.dropout,
        identifiability_feature_index=identifiability_feature_index(sample["x"].shape[1], args),
        identifiability_pooling=args.packet_identifiability_pooling,
        identifiability_feature_mode=args.identifiability_feature_mode,
        identifiability_prior_init=args.identifiability_prior_init,
        identifiability_dual_pooling=args.packet_identifiability_dual_pooling,
        identifiability_evidence_adapter=args.packet_identifiability_evidence_adapter,
        identifiability_adapter_max_delta=args.identifiability_adapter_max_delta,
        identifiability_residual_max_weight=args.identifiability_residual_max_weight,
        identifiability_residual_init=args.identifiability_residual_init,
        dual_channel_mode=args.dual_channel_mode,
        meta_feature_dim=args.meta_feature_dim,
        native_structural_dim=args.native_structural_dim,
        dual_channel_max_weight=args.dual_channel_max_weight,
        dual_channel_init=args.dual_channel_init,
        dual_channel_gate_mode=args.dual_channel_gate_mode,
        channel_fusion_base_mode=args.channel_fusion_base_mode,
        use_intervention_views=args.use_intervention_views,
        intervention_max_residual_weight=args.intervention_max_residual_weight,
        intervention_view_base_mode=args.intervention_view_base_mode,
        exact_shared_packet_encoder=args.exact_shared_packet_encoder,
        shared_packet_hidden_dim=args.shared_packet_hidden_dim,
        packet_evidence_max_weight=args.packet_evidence_max_weight,
        train_ablate_input_channel=args.train_ablate_input_channel,
        train_ablate_intervention_view=args.train_ablate_intervention_view,
        train_fixed_channel_fusion=args.train_fixed_channel_fusion,
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
        # Aggregate only explicit packet-local structural fields. Native
        # structural embeddings are learned latent channels, not metadata.
        flow_stat_meta_dim=max(0, args.meta_feature_dim - args.native_structural_dim),
        flow_stat_expert_weight=args.flow_stat_expert_weight,
        flow_stat_aux_weight=args.flow_stat_aux_weight,
        identifiability_attention_prior=args.identifiability_attention_prior,
        identifiability_prior_init=args.identifiability_prior_init,
    ).to(args.device)
    domain_head = (
        DomainAdversarialHead(args.hidden_dim, args.dropout).to(args.device)
        if args.view_domain_adversarial_weight > 0 and args.paired_view_dataset
        else None
    )
    fusion_head = build_counterfactual_fusion(args)
    maybe_load_init_checkpoint(args, model, flow_head)
    if args.dual_channel_train_scope != "full":
        trainable_dual = configure_dual_channel_train_scope(
            model, flow_head, args.dual_channel_train_scope
        )
        print(
            "dual_channel_trainable "
            + json.dumps({"scope": args.dual_channel_train_scope, "parameters": trainable_dual}),
            flush=True,
        )
    if args.counterfactual_head_only:
        trainable_counterfactual = configure_counterfactual_head_only(
            model, flow_head, fusion_head
        )
        print(
            "counterfactual_head_trainable "
            + json.dumps(trainable_counterfactual),
            flush=True,
        )
    if args.identifiability_adapter_only:
        trainable_adapter = configure_identifiability_adapter_only(model, flow_head)
        print(
            "identifiability_adapter_trainable " + json.dumps(trainable_adapter),
            flush=True,
        )
    class_weight = compute_class_weights([g["label"] for g in tr_groups], args.num_classes, args.device, args.class_weighting, args.class_weight_beta, args.class_weight_strength)
    opt_params = [
        parameter
        for parameter in list(model.parameters()) + list(flow_head.parameters())
        if parameter.requires_grad
    ]
    if domain_head is not None and not args.identifiability_adapter_only:
        opt_params += list(domain_head.parameters())
    if fusion_head is not None and not args.identifiability_adapter_only:
        opt_params += list(fusion_head.parameters())
    opt = torch.optim.AdamW(opt_params, lr=args.lr, weight_decay=args.weight_decay)
    best = -1.0
    stale_epochs = 0
    if args.select_init_checkpoint and args.init_checkpoint and va_groups:
        init_metrics = evaluate_seq_flow(
            model,
            va_groups,
            args.device,
            args.batch_size,
            args.num_classes,
            flow_head=flow_head,
            class_to_coarse=class_to_coarse,
            hierarchical_logit_weight=effective_hierarchical_logit_weight(args),
            hierarchical_mode=args.hierarchical_mode,
            paired_groups=paired_valid_groups,
            fusion_head=fusion_head,
        )
        best = selected_metric_value(init_metrics, args.select_metric)
        save_ckpt(
            args,
            model,
            sample["x"].shape[1],
            flow_head=flow_head,
            num_coarse_classes=num_coarse_classes,
            counterfactual_fusion_head=fusion_head,
        )
        print(
            f"epoch=0 val_acc={init_metrics['accuracy']:.4f} "
            f"val_macro_f1={init_metrics['macro_f1']:.4f} "
            f"val_group_acc={init_metrics.get('content_group_accuracy', float('nan')):.4f} "
            f"val_group_macro_f1={init_metrics.get('content_group_macro_f1', float('nan')):.4f} "
            f"select={args.select_metric}:{best:.4f} init_checkpoint_candidate=1",
            flush=True,
        )
    for epoch in range(1, args.epochs + 1):
        model.train(); total = 0.0; ok = 0; cnt = 0
        flow_head.train()
        if args.dual_channel_train_scope != "full":
            set_dual_channel_restricted_train_mode(model, flow_head)
        if args.counterfactual_head_only:
            model.eval()
            flow_head.eval()
        if fusion_head is not None:
            fusion_head.train()
        for flow_batch in iter_train_flow_batches(tr_groups, args):
            windows = []
            owners = []
            labels = []
            environments = []
            flow_ids = []
            content_group_ids = []
            paired_windows = []
            paired_owners = []
            paired_labels = []
            for flow_idx, group in enumerate(flow_batch):
                labels.append(int(group["label"]))
                environments.append(int(group.get("environment", -1)))
                flow_ids.append(str(group["flow_id"]))
                content_group_ids.append(group.get("content_group_id"))
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
            intervened_x = None
            if args.use_intervention_views:
                if len(paired_windows) != len(windows) or paired_owners != owners:
                    raise ValueError(
                        "shared intervention views require exactly aligned flow windows"
                    )
                paired_batch = collate_seq(paired_windows)
                if paired_batch["x"].shape != batch["x"].shape:
                    raise ValueError("factual/intervened flow window shapes differ")
                intervened_x = paired_batch["x"].to(args.device)
            if args.consistency_weight > 0:
                out = model(x_clean, mask, intervened_x=intervened_x)
                out_aug = model(
                    apply_input_augmentation(x_clean, args),
                    mask,
                    intervened_x=intervened_x,
                )
            else:
                out = model(
                    apply_input_augmentation(x_clean, args),
                    mask,
                    intervened_x=intervened_x,
                )
                out_aug = None
            owners_t = torch.tensor(owners, dtype=torch.long, device=args.device)
            window_labels = batch["label"].to(args.device)
            window_reliability = batch["window_reliability"].to(args.device)
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
                    window_mask=mask[win_mask],
                    window_ranges=flow_window_ranges(batch, owners, flow_idx),
                    window_reliability=window_reliability[win_mask],
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
                        window_mask=mask[win_mask],
                        window_ranges=flow_window_ranges(batch, owners, flow_idx),
                        window_reliability=window_reliability[win_mask],
                    )
                    flow_logits_aug.append(pooled_aug["logits"])
            flow_logits = torch.stack(flow_logits, dim=0)
            coarse_logits = torch.stack(flow_coarse_logits, dim=0) if flow_coarse_logits else None
            flow_logits = apply_hierarchical_logits(flow_logits, coarse_logits, class_to_coarse, effective_hierarchical_logit_weight(args))
            if flow_logits_aug:
                flow_logits_aug = torch.stack(flow_logits_aug, dim=0)
            flow_embs = torch.stack(flow_embs, dim=0)
            y = torch.tensor(labels, dtype=torch.long, device=args.device)
            environment_ids = torch.tensor(environments, dtype=torch.long, device=args.device)
            loss = regularized_classification_loss(
                flow_logits,
                y,
                class_weight,
                args,
                content_group_ids=content_group_ids,
            )
            loss = loss + dual_channel_auxiliary_loss(
                out,
                y,
                class_weight,
                args,
                owners=owners_t,
                num_flows=len(flow_batch),
                content_group_ids=content_group_ids,
            )
            loss = loss + environment_risk_variance_loss(
                flow_logits, y, environment_ids, class_weight, args.environment_risk_weight
            )
            loss = loss + class_conditional_environment_alignment_loss(
                flow_embs, y, environment_ids, args.environment_alignment_weight
            )
            loss = loss + distillation_kl_loss(flow_logits, flow_ids, distill_targets, args)
            loss = loss + class_prior_distillation_loss(flow_logits, y, distill_class_priors, args)
            gate_loss = multi_view_gate_entropy_loss(multi_view_gates, args.multi_view_gate_entropy_weight)
            if gate_loss is not None:
                loss = loss + gate_loss
            stat_loss = flow_stat_aux_loss(stat_logits, y, class_weight, args, content_group_ids)
            if stat_loss is not None:
                loss = loss + stat_loss
            paired_logits = None
            paired_embs = None
            if (not args.use_intervention_views) and paired_windows and (
                args.paired_view_weight > 0
                or args.paired_consistency_weight > 0
                or args.view_domain_adversarial_weight > 0
                or paired_semantic_alignment_enabled(args)
                or fusion_head is not None
            ):
                paired_batch = collate_seq(paired_windows)
                paired_out = model(
                    apply_input_augmentation(paired_batch["x"].to(args.device), args),
                    paired_batch["mask"].to(args.device),
                )
                paired_owners_t = torch.tensor(paired_owners, dtype=torch.long, device=args.device)
                paired_window_reliability = paired_batch["window_reliability"].to(args.device)
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
                        window_mask=paired_batch["mask"].to(args.device)[win_mask],
                        window_ranges=flow_window_ranges(
                            paired_batch, paired_owners, flow_idx
                        ),
                        window_reliability=paired_window_reliability[win_mask],
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
                        loss = loss + args.paired_view_weight * regularized_classification_loss(
                            paired_logits,
                            y,
                            class_weight,
                            args,
                            content_group_ids=content_group_ids,
                        )
                    if args.paired_consistency_weight > 0:
                        loss = loss + args.paired_consistency_weight * symmetric_consistency_kl(flow_logits, paired_logits, args.consistency_temperature)
            counterfactual_loss, counterfactual_output = counterfactual_training_loss(
                fusion_head,
                flow_embs,
                paired_embs,
                flow_logits,
                paired_logits,
                y,
                class_weight,
                args,
                content_group_ids=content_group_ids,
            )
            loss = loss + counterfactual_loss
            loss = loss + paired_semantic_alignment_loss(flow_embs, paired_embs, y, args)
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
                loss = loss + args.window_loss_weight * regularized_classification_loss(
                    out["logits"],
                    window_labels,
                    class_weight,
                    args,
                    content_group_reduction="none",
                )
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
                contrastive_embeddings = (
                    counterfactual_output["embedding"]
                    if counterfactual_output is not None
                    else flow_embs
                )
                loss = loss + args.flow_contrastive_weight * contrastive_loss(contrastive_embeddings, y, args, confusion_weights)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(opt_params, 1.0)
            opt.step()
            total += float(loss.item()) * len(labels)
            train_logits = (
                counterfactual_output["logits"]
                if counterfactual_output is not None
                else flow_logits
            )
            ok += int((train_logits.argmax(-1) == y).sum()); cnt += len(labels)
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
            paired_groups=paired_valid_groups,
            fusion_head=fusion_head,
        ) if va_groups else {"accuracy": ok / max(1, cnt), "macro_f1": ok / max(1, cnt)}
        select_score = selected_metric_value(val_metrics, args.select_metric)
        print(
            f"epoch={epoch} train_loss={total/max(1,cnt):.4f} "
            f"train_acc={ok/max(1,cnt):.4f} val_acc={val_metrics['accuracy']:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f} "
            f"val_group_acc={val_metrics.get('content_group_accuracy', float('nan')):.4f} "
            f"val_group_macro_f1={val_metrics.get('content_group_macro_f1', float('nan')):.4f} "
            f"select={args.select_metric}:{select_score:.4f} flows={cnt}",
            flush=True,
        )
        stop, best, stale_epochs, improved = early_stop_update(select_score, best, stale_epochs, epoch, args)
        if improved:
            save_ckpt(
                args,
                model,
                sample["x"].shape[1],
                flow_head=flow_head,
                num_coarse_classes=num_coarse_classes,
                counterfactual_fusion_head=fusion_head,
            )
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
    paired_groups: Dict[str, dict] | None = None,
    fusion_head: CounterfactualFlowFusion | None = None,
):
    model.eval(); y_true = []; y_pred = []; content_group_ids = []
    if flow_head is not None:
        flow_head.eval()
    if fusion_head is not None:
        fusion_head.eval()
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
        intervened_x = None
        if getattr(model, "use_intervention_views", False):
            if not paired_groups:
                raise ValueError("intervention-aware evaluation requires paired_groups")
            paired_batch_groups = [paired_groups.get(str(group["flow_id"])) for group in flow_batch]
            if any(group is None for group in paired_batch_groups):
                raise ValueError("intervention-aware evaluation is missing paired flows")
            paired_windows = []
            paired_owners = []
            for flow_idx, paired_group in enumerate(paired_batch_groups):
                for item in paired_group["items"]:
                    paired_windows.append(item)
                    paired_owners.append(flow_idx)
            if paired_owners != owners:
                raise ValueError("factual/intervened evaluation windows are not aligned")
            paired_batch = collate_seq(paired_windows)
            if paired_batch["x"].shape != batch["x"].shape:
                raise ValueError("factual/intervened evaluation window shapes differ")
            intervened_x = paired_batch["x"].to(device)
        out = model(
            batch["x"].to(device),
            batch["mask"].to(device),
            intervened_x=intervened_x,
        )
        owners_t = torch.tensor(owners, dtype=torch.long, device=device)
        window_reliability = batch["window_reliability"].to(device)
        if flow_head is None:
            flow_logits = mean_logits_by_flow(out["logits"], owners_t, len(flow_batch))
        else:
            pooled_outputs = [
                flow_head(
                    out["embedding"][owners_t == flow_idx],
                    window_logits=out["logits"][owners_t == flow_idx],
                    window_x=batch["x"].to(device)[owners_t == flow_idx],
                    window_mask=batch["mask"].to(device)[owners_t == flow_idx],
                    window_ranges=flow_window_ranges(batch, owners, flow_idx),
                    window_reliability=window_reliability[owners_t == flow_idx],
                )
                for flow_idx in range(len(flow_batch))
            ]
            flow_logits = torch.stack([pooled["logits"] for pooled in pooled_outputs], dim=0)
            if pooled_outputs and pooled_outputs[0].get("coarse_logits") is not None:
                coarse_logits = torch.stack([pooled["coarse_logits"] for pooled in pooled_outputs], dim=0)
                flow_logits = apply_hierarchical_logits(flow_logits, coarse_logits, class_to_coarse, hierarchical_logit_weight)
        if fusion_head is not None and paired_groups and not getattr(model, "use_intervention_views", False):
            paired_batch_groups = [paired_groups.get(str(group["flow_id"])) for group in flow_batch]
            if all(group is not None for group in paired_batch_groups):
                paired_windows = []
                paired_owners = []
                for flow_idx, group in enumerate(paired_batch_groups):
                    for item in group["items"]:
                        paired_windows.append(item)
                        paired_owners.append(flow_idx)
                paired_batch = collate_seq(paired_windows)
                paired_out = model(
                    paired_batch["x"].to(device), paired_batch["mask"].to(device)
                )
                paired_owners_t = torch.tensor(paired_owners, dtype=torch.long, device=device)
                paired_reliability = paired_batch["window_reliability"].to(device)
                paired_pooled = [
                    flow_head(
                        paired_out["embedding"][paired_owners_t == flow_idx],
                        window_logits=paired_out["logits"][paired_owners_t == flow_idx],
                        window_x=paired_batch["x"].to(device)[paired_owners_t == flow_idx],
                        window_mask=paired_batch["mask"].to(device)[paired_owners_t == flow_idx],
                        window_ranges=flow_window_ranges(
                            paired_batch, paired_owners, flow_idx
                        ),
                        window_reliability=paired_reliability[paired_owners_t == flow_idx],
                    )
                    for flow_idx in range(len(flow_batch))
                ]
                paired_logits = torch.stack([item["logits"] for item in paired_pooled], dim=0)
                paired_embeddings = torch.stack([item["embedding"] for item in paired_pooled], dim=0)
                clean_embeddings = torch.stack([item["embedding"] for item in pooled_outputs], dim=0)
                flow_logits = fusion_head(
                    clean_embeddings, paired_embeddings, flow_logits, paired_logits
                )["logits"]
        y = torch.tensor(labels, dtype=torch.long, device=device)
        y_true.extend(y.detach().cpu().tolist())
        y_pred.extend(flow_logits.argmax(-1).detach().cpu().tolist())
        content_group_ids.extend([group.get("content_group_id") for group in flow_batch])
    return merge_content_group_metrics(
        classification_metrics_from_lists(y_true, y_pred, num_classes),
        y_true,
        y_pred,
        content_group_ids,
        num_classes,
    )


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
        identifiability_feature_index=identifiability_feature_index(sample["x"].shape[1], args),
        identifiability_pooling=args.packet_identifiability_pooling,
        identifiability_feature_mode=args.identifiability_feature_mode,
        identifiability_prior_init=args.identifiability_prior_init,
        identifiability_dual_pooling=args.packet_identifiability_dual_pooling,
        identifiability_evidence_adapter=args.packet_identifiability_evidence_adapter,
        identifiability_adapter_max_delta=args.identifiability_adapter_max_delta,
        identifiability_residual_max_weight=args.identifiability_residual_max_weight,
        identifiability_residual_init=args.identifiability_residual_init,
    ).to(args.device)
    maybe_load_init_checkpoint(args, model)
    distill_targets = load_distillation_targets(
        args.distill_targets_json,
        args.num_classes,
        args.device,
        args.distill_min_teachers_per_flow,
        args.distill_require_oof_exclusion_proof,
    )
    distill_targets = apply_distillation_coverage_policy("train_graph", distill_targets, dataset_flow_ids(tr), args)
    distill_class_priors = load_distillation_class_priors(
        args.distill_targets_json,
        args.num_classes,
        args.device,
        args.distill_min_teachers_per_flow,
        args.distill_require_oof_exclusion_proof,
    )
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
    attach_flow_environments(tr_groups, args.environment_map_json, "train_graph_flow")
    paired_groups = paired_flow_group_lookup(args.paired_view_dataset, GraphDataset)
    distill_targets = load_distillation_targets(
        args.distill_targets_json,
        args.num_classes,
        args.device,
        args.distill_min_teachers_per_flow,
        args.distill_require_oof_exclusion_proof,
    )
    distill_targets = apply_distillation_coverage_policy("train_graph_flow", distill_targets, group_flow_ids(tr_groups), args)
    distill_class_priors = load_distillation_class_priors(
        args.distill_targets_json,
        args.num_classes,
        args.device,
        args.distill_min_teachers_per_flow,
        args.distill_require_oof_exclusion_proof,
    )
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
        identifiability_feature_index=identifiability_feature_index(sample["x"].shape[1], args),
        identifiability_pooling=args.packet_identifiability_pooling,
        identifiability_feature_mode=args.identifiability_feature_mode,
        identifiability_prior_init=args.identifiability_prior_init,
        identifiability_dual_pooling=args.packet_identifiability_dual_pooling,
        identifiability_evidence_adapter=args.packet_identifiability_evidence_adapter,
        identifiability_adapter_max_delta=args.identifiability_adapter_max_delta,
        identifiability_residual_max_weight=args.identifiability_residual_max_weight,
        identifiability_residual_init=args.identifiability_residual_init,
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
        # Keep the graph and sequence variants on the same explicit
        # packet-local structural statistics contract.
        flow_stat_meta_dim=max(0, args.meta_feature_dim - args.native_structural_dim),
        flow_stat_expert_weight=args.flow_stat_expert_weight,
        flow_stat_aux_weight=args.flow_stat_aux_weight,
        identifiability_attention_prior=args.identifiability_attention_prior,
        identifiability_prior_init=args.identifiability_prior_init,
    ).to(args.device)
    domain_head = (
        DomainAdversarialHead(args.hidden_dim, args.dropout).to(args.device)
        if args.view_domain_adversarial_weight > 0 and args.paired_view_dataset
        else None
    )
    maybe_load_init_checkpoint(args, model, flow_head)
    if args.identifiability_adapter_only:
        trainable_adapter = configure_identifiability_adapter_only(model, flow_head)
        print(
            "identifiability_adapter_trainable " + json.dumps(trainable_adapter),
            flush=True,
        )
    class_weight = compute_class_weights([g["label"] for g in tr_groups], args.num_classes, args.device, args.class_weighting, args.class_weight_beta, args.class_weight_strength)
    opt_params = [
        parameter
        for parameter in list(model.parameters()) + list(flow_head.parameters())
        if parameter.requires_grad
    ]
    if domain_head is not None and not args.identifiability_adapter_only:
        opt_params += list(domain_head.parameters())
    opt = torch.optim.AdamW(opt_params, lr=args.lr, weight_decay=args.weight_decay)
    best = -1.0
    stale_epochs = 0
    if args.select_init_checkpoint and args.init_checkpoint and va_groups:
        init_metrics = evaluate_graph_flow(
            model,
            va_groups,
            args.device,
            args.num_classes,
            flow_head=flow_head,
            class_to_coarse=class_to_coarse,
            hierarchical_logit_weight=effective_hierarchical_logit_weight(args),
            hierarchical_mode=args.hierarchical_mode,
        )
        best = selected_metric_value(init_metrics, args.select_metric)
        save_ckpt(
            args,
            model,
            sample["x"].shape[1],
            edge_attr_dim=edge_attr_dim,
            flow_head=flow_head,
            num_coarse_classes=num_coarse_classes,
        )
        print(
            f"epoch=0 val_acc={init_metrics['accuracy']:.4f} "
            f"val_macro_f1={init_metrics['macro_f1']:.4f} "
            f"val_group_acc={init_metrics.get('content_group_accuracy', float('nan')):.4f} "
            f"val_group_macro_f1={init_metrics.get('content_group_macro_f1', float('nan')):.4f} "
            f"select={args.select_metric}:{best:.4f} init_checkpoint_candidate=1",
            flush=True,
        )
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
            environments = []
            flow_ids = []
            content_group_ids = []
            for group in flow_batch:
                window_embs = []
                window_logits = []
                window_embs_aug = []
                window_logits_aug = []
                window_xs = []
                window_reliability = []
                active_items = maybe_drop_windows(
                    group["items"], args.window_dropout_prob
                )
                for item in active_items:
                    x_clean = item["x"].to(args.device)
                    window_xs.append(x_clean)
                    window_reliability.append(window_identifiability(item))
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
                        window_ranges=[item.get("window") for item in active_items]
                        if all(item.get("window") is not None for item in active_items)
                        else None,
                        window_reliability=torch.tensor(
                            window_reliability, dtype=torch.float32, device=args.device
                        ),
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
                        pooled_aug = flow_head(
                            torch.stack(window_embs_aug, dim=0),
                            window_logits=torch.stack(window_logits_aug, dim=0),
                            window_x=window_xs,
                            window_ranges=[item.get("window") for item in active_items]
                            if all(item.get("window") is not None for item in active_items)
                            else None,
                            window_reliability=torch.tensor(
                                window_reliability, dtype=torch.float32, device=args.device
                            ),
                        )
                        flow_logits_aug.append(pooled_aug["logits"])
                    labels.append(int(group["label"]))
                    environments.append(int(group.get("environment", -1)))
                    flow_ids.append(str(group["flow_id"]))
                    content_group_ids.append(group.get("content_group_id"))
                    paired_group = paired_groups.get(str(group["flow_id"]))
                    if paired_group is not None and int(paired_group["label"]) == int(group["label"]) and (
                        args.paired_view_weight > 0
                        or args.paired_consistency_weight > 0
                        or args.view_domain_adversarial_weight > 0
                        or paired_semantic_alignment_enabled(args)
                    ):
                        paired_window_embs = []
                        paired_window_logits = []
                        paired_window_reliability = []
                        paired_active_items = maybe_drop_windows(
                            paired_group["items"], args.window_dropout_prob
                        )
                        for paired_item in paired_active_items:
                            paired_x = apply_input_augmentation(paired_item["x"].to(args.device), args)
                            paired_edge_attr = apply_edge_attr_dropout(paired_item["edge_attr"].to(args.device), args.edge_attr_dropout_prob)
                            paired_out = model(paired_x, paired_item["edge_index"].to(args.device), paired_edge_attr)
                            paired_window_embs.append(paired_out["embedding"])
                            paired_window_logits.append(paired_out["logits"].squeeze(0))
                            paired_window_reliability.append(window_identifiability(paired_item))
                        if paired_window_embs:
                            paired_pooled = flow_head(
                                torch.stack(paired_window_embs, dim=0),
                                window_logits=torch.stack(paired_window_logits, dim=0),
                                window_x=[paired_item["x"].to(args.device) for paired_item in paired_active_items],
                                window_ranges=[paired_item.get("window") for paired_item in paired_active_items]
                                if all(paired_item.get("window") is not None for paired_item in paired_active_items)
                                else None,
                                window_reliability=torch.tensor(
                                    paired_window_reliability,
                                    dtype=torch.float32,
                                    device=args.device,
                                ),
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
            environment_ids = torch.tensor(environments, dtype=torch.long, device=args.device)
            loss = regularized_classification_loss(
                logits,
                y,
                class_weight,
                args,
                content_group_ids=content_group_ids,
            )
            loss = loss + environment_risk_variance_loss(
                logits, y, environment_ids, class_weight, args.environment_risk_weight
            )
            loss = loss + class_conditional_environment_alignment_loss(
                embs, y, environment_ids, args.environment_alignment_weight
            )
            loss = loss + distillation_kl_loss(logits, flow_ids, distill_targets, args)
            loss = loss + class_prior_distillation_loss(logits, y, distill_class_priors, args)
            gate_loss = multi_view_gate_entropy_loss(multi_view_gates, args.multi_view_gate_entropy_weight)
            if gate_loss is not None:
                loss = loss + gate_loss
            stat_loss = flow_stat_aux_loss(stat_logits, y, class_weight, args, content_group_ids)
            if stat_loss is not None:
                loss = loss + stat_loss
            if paired_logits is not None:
                if args.paired_view_weight > 0:
                    loss = loss + args.paired_view_weight * regularized_classification_loss(
                        paired_logits,
                        y,
                        class_weight,
                        args,
                        content_group_ids=content_group_ids,
                    )
                if args.paired_consistency_weight > 0:
                    loss = loss + args.paired_consistency_weight * symmetric_consistency_kl(logits, paired_logits, args.consistency_temperature)
            loss = loss + paired_semantic_alignment_loss(embs, paired_embs, y, args)
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
                loss = loss + args.window_loss_weight * regularized_classification_loss(
                    win_logits,
                    win_y,
                    class_weight,
                    args,
                    content_group_reduction="none",
                )
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
        print(
            f"epoch={epoch} train_loss={total/max(1,cnt):.4f} "
            f"train_acc={ok/max(1,cnt):.4f} val_acc={val_metrics['accuracy']:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f} "
            f"val_group_acc={val_metrics.get('content_group_accuracy', float('nan')):.4f} "
            f"val_group_macro_f1={val_metrics.get('content_group_macro_f1', float('nan')):.4f} "
            f"select={args.select_metric}:{select_score:.4f} flows={cnt}",
            flush=True,
        )
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
    model.eval(); y_true = []; y_pred = []; content_group_ids = []
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
                window_ranges=[item.get("window") for item in group["items"]]
                if all(item.get("window") is not None for item in group["items"])
                else None,
                window_reliability=torch.tensor(
                    [window_identifiability(item) for item in group["items"]],
                    dtype=torch.float32,
                    device=device,
                ),
            )
            logits = apply_hierarchical_logits(pooled["logits"], pooled.get("coarse_logits"), class_to_coarse, hierarchical_logit_weight)
        y = int(group["label"])
        y_true.append(y)
        y_pred.append(int(logits.argmax(-1).item()))
        content_group_ids.append(group.get("content_group_id"))
    return merge_content_group_metrics(
        classification_metrics_from_lists(y_true, y_pred, num_classes),
        y_true,
        y_pred,
        content_group_ids,
        num_classes,
    )


def save_ckpt(
    args,
    model,
    input_dim,
    edge_attr_dim=None,
    flow_head: FlowAggregationHead | None = None,
    num_coarse_classes: int = 0,
    counterfactual_fusion_head: CounterfactualFlowFusion | None = None,
):
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
        "split_group_key": args.split_group_key,
        "content_group_unique_batches": args.content_group_unique_batches,
        "content_group_loss_reduction": args.content_group_loss_reduction,
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
        "flow_stat_meta_dim": (
            int(flow_head.flow_stat_meta_dim) if flow_head is not None else 0
        ),
        "identifiability_attention_prior": args.identifiability_attention_prior,
        "packet_identifiability_pooling": args.packet_identifiability_pooling,
        "packet_identifiability_dual_pooling": args.packet_identifiability_dual_pooling,
        "packet_identifiability_evidence_adapter": args.packet_identifiability_evidence_adapter,
        "identifiability_adapter_max_delta": args.identifiability_adapter_max_delta,
        "identifiability_feature_mode": args.identifiability_feature_mode,
        "identifiability_feature_index": identifiability_feature_index(input_dim, args),
        "identifiability_prior_init": args.identifiability_prior_init,
        "identifiability_residual_max_weight": args.identifiability_residual_max_weight,
        "identifiability_residual_init": args.identifiability_residual_init,
        "identifiability_adapter_only": args.identifiability_adapter_only,
        "dual_channel_mode": args.dual_channel_mode,
        "dual_channel_train_scope": args.dual_channel_train_scope,
        "dual_channel_gate_mode": args.dual_channel_gate_mode,
        "channel_fusion_base_mode": args.channel_fusion_base_mode,
        "dual_channel_max_weight": args.dual_channel_max_weight,
        "dual_channel_init": args.dual_channel_init,
        "use_intervention_views": args.use_intervention_views,
        "intervention_max_residual_weight": args.intervention_max_residual_weight,
        "intervention_view_base_mode": args.intervention_view_base_mode,
        "exact_shared_packet_encoder": args.exact_shared_packet_encoder,
        "shared_packet_hidden_dim": args.shared_packet_hidden_dim,
        "packet_evidence_max_weight": args.packet_evidence_max_weight,
        "train_ablate_input_channel": args.train_ablate_input_channel,
        "train_ablate_intervention_view": args.train_ablate_intervention_view,
        "train_fixed_channel_fusion": args.train_fixed_channel_fusion,
        "dual_channel_semantic_aux_weight": args.dual_channel_semantic_aux_weight,
        "dual_channel_structural_aux_weight": args.dual_channel_structural_aux_weight,
        "dual_channel_consistency_weight": args.dual_channel_consistency_weight,
        "meta_dropout_prob": args.meta_dropout_prob,
        "meta_feature_dim": args.meta_feature_dim,
        "native_structural_dim": args.native_structural_dim,
        "embedding_dropout_prob": args.embedding_dropout_prob,
        "window_dropout_prob": args.window_dropout_prob,
        "edge_attr_dropout_prob": args.edge_attr_dropout_prob,
        "consistency_weight": args.consistency_weight,
        "consistency_temperature": args.consistency_temperature,
        "paired_view_dataset": args.paired_view_dataset,
        "paired_valid_dataset": args.paired_valid_dataset,
        "paired_view_weight": args.paired_view_weight,
        "paired_consistency_weight": args.paired_consistency_weight,
        "paired_alignment_weight": args.paired_alignment_weight,
        "paired_crossview_contrastive_weight": args.paired_crossview_contrastive_weight,
        "paired_crossview_temperature": args.paired_crossview_temperature,
        "paired_variance_weight": args.paired_variance_weight,
        "paired_variance_target": args.paired_variance_target,
        "paired_covariance_weight": args.paired_covariance_weight,
        "view_domain_adversarial_weight": args.view_domain_adversarial_weight,
        "domain_adversarial_lambda": args.domain_adversarial_lambda,
        "distill_targets_json": args.distill_targets_json,
        "distill_weight": args.distill_weight,
        "distill_class_prior_weight": args.distill_class_prior_weight,
        "distill_temperature": args.distill_temperature,
        "distill_min_confidence": args.distill_min_confidence,
        "distill_max_confidence": args.distill_max_confidence,
        "distill_confidence_power": args.distill_confidence_power,
        "distill_min_teachers_per_flow": args.distill_min_teachers_per_flow,
        "distill_require_oof_exclusion_proof": args.distill_require_oof_exclusion_proof,
        "distill_min_coverage": args.distill_min_coverage,
        "distill_low_coverage_action": args.distill_low_coverage_action,
        "environment_map_json": args.environment_map_json,
        "environment_risk_weight": args.environment_risk_weight,
        "environment_alignment_weight": args.environment_alignment_weight,
        "counterfactual_fusion": args.counterfactual_fusion,
        "counterfactual_base_mode": args.counterfactual_base_mode,
        "counterfactual_fusion_weight": args.counterfactual_fusion_weight,
        "counterfactual_shared_ce_weight": args.counterfactual_shared_ce_weight,
        "counterfactual_max_residual_weight": args.counterfactual_max_residual_weight,
        "counterfactual_initial_residual_fraction": args.counterfactual_initial_residual_fraction,
        "counterfactual_gate_weight": args.counterfactual_gate_weight,
        "counterfactual_orthogonality_weight": args.counterfactual_orthogonality_weight,
        "counterfactual_head_only": args.counterfactual_head_only,
        "counterfactual_routing_weight": args.counterfactual_routing_weight,
        "training_input_evidence": getattr(args, "training_input_evidence", {}),
    }
    if edge_attr_dim is not None:
        payload["edge_attr_dim"] = edge_attr_dim
    if flow_head is not None:
        payload["flow_head_state"] = flow_head.state_dict()
    if counterfactual_fusion_head is not None:
        payload["counterfactual_fusion_state"] = counterfactual_fusion_head.state_dict()
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
    ap.add_argument("--dual_channel_mode", choices=["concat", "residual"], default="concat", help="concat uses the legacy joint projection; residual separately projects semantic embeddings and structural metadata, then adds an identity-initialized bounded interaction.")
    ap.add_argument("--use_intervention_views", action="store_true", help="Fuse aligned factual/intervened semantic packet views with the shared representation-level router before flow aggregation.")
    ap.add_argument("--intervention_max_residual_weight", type=float, default=0.25)
    ap.add_argument(
        "--intervention_view_base_mode",
        choices=["symmetric_mean", "factual_anchor"],
        default="symmetric_mean",
        help="Use the historical symmetric mean or a factual identity path with bounded intervention residual.",
    )
    ap.add_argument("--dual_channel_train_scope", choices=["full", "interaction", "native_interaction", "new_modules"], default="full", help="Train the full Tower2, only the bounded interaction, the zero-initialized native structural adapter plus interaction, or all newly introduced dual-channel modules while freezing the legacy Transformer and flow head.")
    ap.add_argument("--dual_channel_gate_mode", choices=["global", "adaptive"], default="global", help="Use one global bounded residual weight or a packet-conditioned semantic-structural reliability gate.")
    ap.add_argument("--channel_fusion_base_mode", choices=["legacy", "semantic_anchor"], default="legacy", help="Legacy preserves historical sum/mean fusion; semantic_anchor gives semantic evidence a stable path and routes other channels only through a bounded learned residual.")
    ap.add_argument("--dual_channel_max_weight", type=float, default=0.25, help="Maximum residual interaction contribution in dual-channel residual mode.")
    ap.add_argument("--dual_channel_init", type=float, default=0.1, help="Initial fraction of the maximum dual-channel residual weight; the zero-initialized interaction preserves exact checkpoint logits.")
    ap.add_argument("--dual_channel_semantic_aux_weight", type=float, default=0.0, help="Auxiliary flow/window CE for the semantic channel before interaction.")
    ap.add_argument("--dual_channel_structural_aux_weight", type=float, default=0.0, help="Auxiliary flow/window CE for the structural metadata channel before interaction.")
    ap.add_argument("--dual_channel_consistency_weight", type=float, default=0.0, help="KL weight aligning each channel prediction to the detached fused prediction.")
    ap.add_argument("--init_checkpoint", default="", help="Optional best.pt checkpoint used to initialize model/flow_head before fine-tuning.")
    ap.add_argument("--select_init_checkpoint", action="store_true", help="Evaluate and save the warm-start model as the epoch-0 candidate before fine-tuning.")
    ap.add_argument("--aux_weight", type=float, default=0.1)
    ap.add_argument("--coherence_weight", type=float, default=0.1)
    ap.add_argument(
        "--select_metric",
        choices=[
            "accuracy",
            "acc",
            "macro_f1",
            "flow_acc",
            "flow_macro_f1",
            "window_acc",
            "window_macro_f1",
            "content_group_acc",
            "content_group_accuracy",
            "content_group_macro_f1",
            "flow_content_group_acc",
            "flow_content_group_macro_f1",
        ],
        default="accuracy",
        help="Validation metric used to save best.pt. content_group_* requires Tower-2 datasets built with --content_group_index.",
    )
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
    ap.add_argument("--content_group_unique_batches", action="store_true", help="When balanced flow batches are enabled, prefer at most one flow from each exact-PCAP content_group_id per batch.")
    ap.add_argument(
        "--content_group_loss_reduction",
        choices=["none", "group_mean"],
        default="none",
        help="For flow-level main CE, group duplicate exact-PCAP content by content_group_id before averaging. Requires Tower-2 data built with --content_group_index.",
    )
    ap.add_argument("--hierarchical_mode", choices=["logit", "expert"], default="logit", help="logit: add coarse log-probability to flat logits; expert: use group-specific fine heads P(coarse)*P(class|coarse).")
    ap.add_argument("--hierarchical_weight", type=float, default=0.0, help="Coarse-label CE weight for hierarchical coarse-to-fine flow classification.")
    ap.add_argument("--hierarchical_logit_weight", type=float, default=0.0, help="Add coarse log-probability to fine logits during training/evaluation.")
    ap.add_argument("--coarse_groups", default="vpn_app", help="Coarse groups. Use 'vpn_app', 'none', or semicolon groups like '0,2;1,4'.")
    ap.add_argument("--flow_pooling", choices=["mean", "attention", "late_fusion", "transformer", "multi_view"], default="attention", help="Pooling used by --train_level flow over window embeddings.")
    ap.add_argument("--flow_transformer_layers", type=int, default=1, help="Number of Transformer layers for --flow_pooling transformer.")
    ap.add_argument("--flow_transformer_heads", type=int, default=4, help="Attention heads for --flow_pooling transformer.")
    ap.add_argument("--flow_stat_expert_weight", type=float, default=0.0, help="Fuse a trainable flow-level metadata/statistics expert into the Tower-2 flow head. 0 disables it.")
    ap.add_argument("--flow_stat_aux_weight", type=float, default=0.0, help="Auxiliary CE weight for the flow metadata/statistics expert.")
    ap.add_argument("--identifiability_attention_prior", action="store_true", help="Bias flow attention toward windows whose packet signatures are class-identifiable in the training split.")
    ap.add_argument("--packet_identifiability_pooling", action="store_true", help="Bias within-window packet/node pooling by train-split signature identifiability.")
    ap.add_argument("--packet_identifiability_dual_pooling", action="store_true", help="Pool all, identifiable, and low-identifiability packet/node views and fuse them with a learned sample-wise gate.")
    ap.add_argument("--packet_identifiability_evidence_adapter", action="store_true", help="Apply a zero-initialized residual adapter to complementary identifiable and contextual packet evidence.")
    ap.add_argument("--identifiability_adapter_max_delta", type=float, default=0.25, help="Maximum per-channel tanh correction applied by the identity-preserving evidence adapter.")
    ap.add_argument("--identifiability_feature_mode", choices=["observed", "zero"], default="observed", help="Use the train-profile reliability/support features or zero both columns for a dimension-matched causal ablation.")
    ap.add_argument("--identifiability_prior_init", type=float, default=0.1, help="Initial non-negative scale for the identifiability attention prior; the learned scale can shrink toward zero.")
    ap.add_argument("--identifiability_residual_max_weight", type=float, default=0.0, help="Maximum dual-view residual weight. 0 uses direct dual fusion; e.g. 0.25 keeps at least 75%% of base pooling.")
    ap.add_argument("--identifiability_residual_init", type=float, default=0.5, help="Initial fraction of --identifiability_residual_max_weight used by the learned global residual gate.")
    ap.add_argument("--identifiability_adapter_only", action="store_true", help="Freeze the pretrained Tower-2 backbone and flow head; train only dual identifiability scales and gates.")
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
    ap.add_argument("--native_structural_dim", type=int, default=0, help="Leading dimensions inside the structural channel produced by the native flow encoder. Used by the identity-initialized native adapter.")
    ap.add_argument("--exact_shared_packet_encoder", action="store_true", help="Use the exact reusable packet module before flow-only projection and aggregation.")
    ap.add_argument(
        "--train_ablate_input_channel",
        choices=["none", "semantic", "content", "structural"],
        default="none",
        help="Retrained shared-core ablation applied consistently in train/valid/test.",
    )
    ap.add_argument(
        "--train_ablate_intervention_view",
        choices=["none", "factual_only", "intervened_only"],
        default="none",
        help="Retrained intervention-view ablation applied consistently in train/valid/test.",
    )
    ap.add_argument(
        "--train_fixed_channel_fusion",
        action="store_true",
        help="Retrain with a fixed equal three-channel mean instead of the bounded router.",
    )
    ap.add_argument(
        "--packet_evidence_max_weight",
        type=float,
        default=0.0,
        help=(
            "Maximum learned convex weight of the reused fused-packet classifier "
            "inside each flow window; 0 disables this validation candidate."
        ),
    )
    ap.add_argument("--shared_packet_hidden_dim", type=int, default=128, help="Width of the exact shared packet representation before Tower2.")
    ap.add_argument("--embedding_dropout_prob", type=float, default=0.0, help="Training-only dropout on packet embedding dimensions; trailing metadata dimensions are controlled by --meta_dropout_prob.")
    ap.add_argument("--window_dropout_prob", type=float, default=0.0, help="Training-only random window dropping for --train_level flow; at least one window per flow is kept.")
    ap.add_argument("--edge_attr_dropout_prob", type=float, default=0.0, help="Training-only dropout on continuous graph edge attributes; edge type id is kept.")
    ap.add_argument("--consistency_weight", type=float, default=0.0, help="Weight for clean/augmented view KL consistency. CE is computed on the clean view when this is >0.")
    ap.add_argument("--consistency_temperature", type=float, default=2.0, help="Temperature for clean/augmented consistency KL.")
    ap.add_argument("--paired_view_dataset", default="", help="Optional second-view Tower2 dataset aligned by flow_id, e.g. ip/port randomized or masked view.")
    ap.add_argument("--paired_valid_dataset", default="", help="Validation counterpart for --paired_view_dataset; required for selecting a counterfactual two-view checkpoint without leakage.")
    ap.add_argument("--paired_view_weight", type=float, default=0.0, help="Flow CE weight for --paired_view_dataset.")
    ap.add_argument("--paired_consistency_weight", type=float, default=0.0, help="Symmetric KL weight between primary and paired flow logits.")
    ap.add_argument("--paired_alignment_weight", type=float, default=0.0, help="Cosine alignment weight for clean/randomized-header flow embeddings.")
    ap.add_argument("--paired_crossview_contrastive_weight", type=float, default=0.0, help="Supervised cross-view contrastive weight over clean/randomized-header flow embeddings.")
    ap.add_argument("--paired_crossview_temperature", type=float, default=0.07, help="Temperature for --paired_crossview_contrastive_weight.")
    ap.add_argument("--paired_variance_weight", type=float, default=0.0, help="Anti-collapse variance regularization weight for paired flow embeddings.")
    ap.add_argument("--paired_variance_target", type=float, default=0.04, help="Minimum per-dimension std for normalized paired flow embeddings.")
    ap.add_argument("--paired_covariance_weight", type=float, default=0.0, help="Off-diagonal covariance penalty for paired flow embeddings.")
    ap.add_argument("--view_domain_adversarial_weight", type=float, default=0.0, help="Adversarially remove clean-vs-paired view information from flow embeddings in --train_level flow.")
    ap.add_argument("--domain_adversarial_lambda", type=float, default=1.0, help="Gradient reversal strength for --view_domain_adversarial_weight.")
    ap.add_argument("--counterfactual_fusion", choices=["none", "clean", "mean", "counterfactual", "router"], default="none", help="Two-view flow fusion. clean and mean are controls; counterfactual learns a bounded residual; router learns when the intervened prediction is complementary.")
    ap.add_argument("--counterfactual_base_mode", choices=["clean", "mean"], default="clean", help="Identity path used by counterfactual fusion. clean preserves the deployable primary-view classifier; mean is the symmetric two-view ablation.")
    ap.add_argument("--counterfactual_fusion_weight", type=float, default=1.0, help="CE weight on the two-view counterfactual prediction.")
    ap.add_argument("--counterfactual_shared_ce_weight", type=float, default=0.25, help="Auxiliary CE on the simple mean-logit control to preserve intervention-invariant evidence.")
    ap.add_argument("--counterfactual_max_residual_weight", type=float, default=0.25, help="Hard upper bound on the intervention-sensitive residual contribution.")
    ap.add_argument("--counterfactual_initial_residual_fraction", type=float, default=0.1, help="Initial fraction of the residual upper bound; the output still exactly equals mean fusion because the residual classifier starts at zero.")
    ap.add_argument("--counterfactual_gate_weight", type=float, default=0.01, help="Sparsity penalty on intervention-sensitive residual usage.")
    ap.add_argument("--counterfactual_orthogonality_weight", type=float, default=0.01, help="Penalty separating shared semantics from intervention-sensitive evidence.")
    ap.add_argument("--counterfactual_head_only", action="store_true", help="Freeze the initialized Tower2 backbone and flow head; train only the identity-initialized bounded counterfactual correction.")
    ap.add_argument("--counterfactual_routing_weight", type=float, default=0.0, help="BCE weight supervising the paired-view routing gate only on flows where exactly one view predicts the training label correctly.")
    ap.add_argument("--distill_targets_json", default="", help="Optional prediction/consensus JSON with flow_ids and flow_prob used as soft distillation targets.")
    ap.add_argument("--distill_weight", type=float, default=0.0, help="KL weight for consensus/self-distillation targets. 0 disables it.")
    ap.add_argument("--distill_class_prior_weight", type=float, default=0.0, help="KL weight for class-conditional consensus priors averaged from teacher flow_y_true/flow_prob.")
    ap.add_argument("--distill_temperature", type=float, default=2.0, help="Temperature for probability-target distillation.")
    ap.add_argument("--distill_min_confidence", type=float, default=0.0, help="Ignore teacher targets whose max probability is below this confidence.")
    ap.add_argument("--distill_max_confidence", type=float, default=0.0, help="If >0 and <1, soft-cap overconfident teacher targets by mixing them with a uniform prior before temperature softening.")
    ap.add_argument("--distill_confidence_power", type=float, default=0.0, help="If >0, weight distillation samples by teacher max-probability^power.")
    ap.add_argument(
        "--distill_min_teachers_per_flow",
        type=int,
        default=1,
        help="Require this many source predictions for each loaded flow target. Values above 1 require teacher_multiplicity metadata.",
    )
    ap.add_argument(
        "--distill_require_oof_exclusion_proof",
        action="store_true",
        help="Reject targets unless their multiplicity contract proves per-flow teacher exclusion from training.",
    )
    ap.add_argument("--distill_min_coverage", type=float, default=0.0, help="Minimum fraction of training flow_ids that must have flow-id teacher targets.")
    ap.add_argument(
        "--distill_low_coverage_action",
        choices=["warn", "disable_flow", "fail"],
        default="warn",
        help="Action when flow-id teacher coverage is below --distill_min_coverage. disable_flow keeps class-prior distillation active.",
    )
    ap.add_argument("--environment_map_json", default="", help="Optional flow_id-to-source-environment map used for cross-environment flow training.")
    ap.add_argument("--environment_risk_weight", type=float, default=0.0, help="Weight for variance of flow classification risks across source environments.")
    ap.add_argument("--environment_alignment_weight", type=float, default=0.0, help="Weight for class-conditional flow embedding alignment across source environments.")
    ap.add_argument("--valid_ratio", type=float, default=0.1)
    ap.add_argument(
        "--split_group_key",
        choices=["flow_id", "content_group_id"],
        default="flow_id",
        help="Grouping key for the internal train/valid split when --valid_dataset is absent. content_group_id prevents exact-PCAP duplicate content from crossing the split.",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    if args.distill_min_teachers_per_flow <= 0:
        ap.error("--distill_min_teachers_per_flow must be positive")
    args.training_input_evidence = tower2_training_input_evidence(args)
    print(
        "tower2_training_input_evidence "
        + json.dumps(args.training_input_evidence, sort_keys=True),
        flush=True,
    )
    identifiability_modes = sum(
        int(enabled)
        for enabled in (
            args.packet_identifiability_pooling,
            args.packet_identifiability_dual_pooling,
            args.packet_identifiability_evidence_adapter,
        )
    )
    if identifiability_modes > 1:
        ap.error("use only one packet identifiability pooling mode")
    if args.identifiability_adapter_only and not (
        args.packet_identifiability_dual_pooling
        or args.packet_identifiability_evidence_adapter
    ):
        ap.error("--identifiability_adapter_only requires a trainable identifiability adapter")
    dual_channel_losses = (
        args.dual_channel_semantic_aux_weight
        + args.dual_channel_structural_aux_weight
        + args.dual_channel_consistency_weight
    )
    if args.dual_channel_mode != "residual" and (
        args.dual_channel_train_scope != "full" or dual_channel_losses > 0
    ):
        ap.error("dual-channel restricted training/losses require --dual_channel_mode residual")
    if args.use_intervention_views:
        if args.model_type != "seq" or args.train_level != "flow":
            ap.error("--use_intervention_views currently requires seq flow training")
        if args.dual_channel_mode != "residual":
            ap.error("--use_intervention_views requires --dual_channel_mode residual")
        if not args.paired_view_dataset or not args.paired_valid_dataset:
            ap.error("--use_intervention_views requires aligned paired train/valid datasets")
        if args.counterfactual_fusion != "none":
            ap.error("shared intervention views cannot be combined with flow-level counterfactual fusion")
    if args.dual_channel_mode == "residual" and args.model_type != "seq":
        ap.error("dual-channel residual mode is currently implemented for --model_type seq")
    if not 0.0 <= args.packet_evidence_max_weight <= 1.0:
        ap.error("--packet_evidence_max_weight must be in [0, 1]")
    if args.train_ablate_input_channel != "none" and not args.exact_shared_packet_encoder:
        ap.error("--train_ablate_input_channel requires --exact_shared_packet_encoder")
    if args.train_ablate_intervention_view != "none" and (
        not args.exact_shared_packet_encoder or not args.use_intervention_views
    ):
        ap.error(
            "--train_ablate_intervention_view requires exact shared intervention views"
        )
    if args.train_fixed_channel_fusion and not args.exact_shared_packet_encoder:
        ap.error("--train_fixed_channel_fusion requires --exact_shared_packet_encoder")
    if args.packet_evidence_max_weight > 0:
        if not args.exact_shared_packet_encoder:
            ap.error("--packet_evidence_max_weight requires --exact_shared_packet_encoder")
        if args.train_level != "flow" or args.flow_pooling != "late_fusion":
            ap.error(
                "packet evidence is a flow candidate and requires "
                "--train_level flow --flow_pooling late_fusion"
            )
    if args.dual_channel_train_scope != "full" and args.train_level != "flow":
        ap.error("dual-channel restricted training scopes require --train_level flow")
    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    if args.train_level == "flow" and (args.aux_weight > 0 or args.coherence_weight > 0):
        print("WARNING: --train_level flow ignores aux/coherence weights; use --window_loss_weight for dual flow/window classification.")
    if args.flow_contrastive_weight > 0 and not args.balanced_flow_batches:
        print("WARNING: flow SupCon is usually more effective with --balanced_flow_batches so each batch has positive pairs.")
    if args.content_group_unique_batches and not args.balanced_flow_batches:
        print("WARNING: --content_group_unique_batches only affects --balanced_flow_batches.")
    if args.content_group_loss_reduction != "none" and args.train_level != "flow":
        print("WARNING: --content_group_loss_reduction currently affects only --train_level flow main CE.")
    if args.valid_dataset and args.split_group_key != "flow_id":
        print("WARNING: --split_group_key only affects the internal split; --valid_dataset is external and already fixed.")
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
        and not args.use_intervention_views
        and args.paired_view_weight <= 0
        and args.paired_consistency_weight <= 0
        and args.view_domain_adversarial_weight <= 0
        and not paired_semantic_alignment_enabled(args)
        and args.counterfactual_fusion == "none"
    ):
        print("WARNING: --paired_view_dataset is set but all paired-view losses are 0.")
    if (args.environment_risk_weight > 0 or args.environment_alignment_weight > 0) and not args.environment_map_json:
        print("WARNING: environment regularization is enabled without --environment_map_json; both losses will be inactive.")
    if args.view_domain_adversarial_weight > 0 and not args.paired_view_dataset:
        print("WARNING: --view_domain_adversarial_weight requires --paired_view_dataset and will be inactive.")
    if args.counterfactual_fusion != "none" and not args.paired_view_dataset:
        ap.error("--counterfactual_fusion requires --paired_view_dataset")
    if args.counterfactual_fusion != "none" and not args.paired_valid_dataset:
        ap.error("--counterfactual_fusion requires --paired_valid_dataset for leakage-free checkpoint selection")
    if args.counterfactual_fusion != "none" and args.train_level != "flow":
        ap.error("--counterfactual_fusion requires --train_level flow")
    if args.counterfactual_fusion != "none" and args.model_type != "seq":
        ap.error("counterfactual fusion is currently implemented for seq flow training; graph support follows after the causal screening run")
    if args.counterfactual_head_only and args.counterfactual_fusion not in {"counterfactual", "router"}:
        ap.error("--counterfactual_head_only requires counterfactual or router fusion")
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
