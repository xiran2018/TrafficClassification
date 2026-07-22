#!/usr/bin/env python3
"""Train a shared strict-one-packet byte Transformer under Per-flow Split."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from audit_packet_identifiability import packet_signature
from models.packet_byte_transformer import PacketByteTransformer
from models.native_flow_encoder import NATIVE_PACKET_PRETRAINING_PROTOCOL
from models.qwen_packet_multitask import flow_aware_contrastive_loss
from packet_eval_utils import packet_classification_metrics
from train_tower1_multitask import FlowBalancedPacketBatchSampler, load_label_names, stable_flow_id
from native_flow_data import protocol_field_ids
from traffic_utils import current_packet_meta_feature_vector


PAD_TOKEN = 256
MASK_TOKEN = 257


def sha256_file(path: str, cache: dict[str, str]) -> str:
    key = str(Path(path).expanduser().resolve())
    cached = cache.get(key)
    if cached is not None:
        return cached
    h = hashlib.sha256()
    with open(key, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    digest = h.hexdigest()
    cache[key] = digest
    return digest


def packet_content_group_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    content_group_ids: np.ndarray | None,
    num_classes: int,
    label_names: list[str],
) -> dict:
    if content_group_ids is None or len(content_group_ids) != len(y_true):
        return {}
    buckets: dict[int, list[int]] = {}
    for idx, group_id in enumerate(content_group_ids.tolist()):
        buckets.setdefault(int(group_id), []).append(idx)
    if not buckets:
        return {}
    group_true, group_pred = [], []
    for group_id, indices in buckets.items():
        labels = {int(y_true[index]) for index in indices}
        if len(labels) != 1:
            raise ValueError(
                f"content_group_id={group_id} has conflicting labels: {sorted(labels)}"
            )
        votes = np.bincount([int(y_pred[index]) for index in indices], minlength=num_classes)
        group_true.append(next(iter(labels)))
        group_pred.append(int(votes.argmax()))
    metrics = packet_classification_metrics(
        np.asarray(group_true, dtype=np.int64),
        np.asarray(group_pred, dtype=np.int64),
        num_classes,
        label_names,
    )
    return {
        "content_group_accuracy": metrics["accuracy"],
        "content_group_macro_f1": metrics["macro_f1"],
        "content_group_count": len(group_true),
        "content_group_rows": int(len(y_true)),
    }


def packet_content_group_mean_loss(
    per_sample_loss: torch.Tensor,
    content_group_ids: torch.Tensor | None,
) -> torch.Tensor:
    if content_group_ids is None:
        raise ValueError(
            "--content_group_loss_reduction group_mean requires pcap_path-derived "
            "content_group_id metadata in packet_index.jsonl"
        )
    if int(content_group_ids.numel()) != int(per_sample_loss.numel()):
        raise ValueError("content_group_ids length must match packet losses")
    buckets: dict[int, list[int]] = {}
    for idx, group_id in enumerate(content_group_ids.detach().cpu().tolist()):
        buckets.setdefault(int(group_id), []).append(idx)
    if not buckets:
        raise ValueError("--content_group_loss_reduction group_mean found no content groups")
    return torch.stack([per_sample_loss[indices].mean() for indices in buckets.values()]).mean()


def packet_ce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_weights: torch.Tensor,
    content_group_ids: torch.Tensor | None,
    reduction: str,
) -> torch.Tensor:
    losses = F.cross_entropy(logits, labels, weight=class_weights, reduction="none")
    if reduction == "group_mean":
        return packet_content_group_mean_loss(losses, content_group_ids)
    return losses.mean()


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
        if length > l4_offset + 12:
            data_offset = (int(raw[l4_offset + 12]) >> 4) * 4
            positions.append((l4_offset + 20, l4_offset + data_offset))
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
        build_ambiguity_targets: bool = False,
        num_classes: int | None = None,
        semantic_embedding_cache: str = "",
        semantic_embedding_manifest: str = "",
        required_header_policy: str = "",
        required_packet_context_policy: str = "",
        intervened_semantic_embedding_cache: str = "",
        intervened_semantic_embedding_manifest: str = "",
        required_intervened_header_policy: str = "",
    ) -> None:
        self.rows: list[dict] = []
        self.semantic_embeddings = None
        self.intervened_semantic_embeddings = None
        self.semantic_dim = 0
        tokens, masked_tokens, field_rows, lengths = [], [], [], []
        payload_tokens, payload_lengths = [], []
        metas, masked_metas, labels, flow_ids = [], [], [], []
        content_hashes, content_group_ids = [], []
        invariant_signatures = []
        content_cache: dict[str, str] = {}
        hash_to_group_id: dict[str, int] = {}
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
                field_rows.append(protocol_field_ids(full_raw, max_bytes))
                if include_augmented:
                    masked_tokens.append(mask_session_tokens(item, len(raw)))
                lengths.append(len(raw))
                payload = extract_packet_payload(full_raw, row["meta"])[:payload_width]
                payload_item = np.full(payload_width, PAD_TOKEN, dtype=np.int64)
                if payload:
                    payload_item[:len(payload)] = np.frombuffer(payload, dtype=np.uint8).astype(np.int64)
                payload_tokens.append(payload_item)
                payload_lengths.append(len(payload))
                class Meta:
                    pass
                meta_obj = Meta()
                for key, value in {
                    "direction": "",
                    "l4": "",
                    "tcp_flags": "",
                    "packet_len": 0,
                    "payload_len": 0,
                    "iat": 0.0,
                    "payload_entropy": 0.0,
                    "tcp_window": -1,
                    "full_l3_captured": False,
                }.items():
                    setattr(meta_obj, key, value)
                for key, value in row["meta"].items():
                    setattr(meta_obj, key, value)
                metas.append(current_packet_meta_feature_vector(meta_obj))
                if include_augmented:
                    masked_metas.append(current_packet_meta_feature_vector(meta_obj))
                labels.append(int(row["label_id"]))
                flow_ids.append(stable_flow_id(str(row.get("flow_id", len(labels) - 1))))
                pcap_path = str(row.get("pcap_path", ""))
                if pcap_path:
                    digest = sha256_file(pcap_path, content_cache)
                else:
                    digest = stable_flow_id(str(row.get("flow_id", len(labels) - 1)))
                if digest not in hash_to_group_id:
                    hash_to_group_id[digest] = len(hash_to_group_id)
                content_hashes.append(digest)
                content_group_ids.append(hash_to_group_id[digest])
                if build_ambiguity_targets:
                    invariant_signatures.append(packet_signature(row, "session"))
                self.rows.append({
                    "flow_id": str(row.get("flow_id", len(labels) - 1)),
                    "content_group_id": hash_to_group_id[digest],
                })
        if semantic_embedding_cache:
            if not semantic_embedding_manifest:
                raise ValueError("semantic embedding cache requires its manifest")
            with open(semantic_embedding_manifest, "r", encoding="utf-8") as handle:
                semantic_manifest = json.load(handle)
            if semantic_manifest.get("scope") != "strict_current_packet_row_aligned_semantic_embedding":
                raise ValueError("semantic embedding cache has an unsupported scope")
            if str(semantic_manifest.get("packet_index_sha256")) != sha256_file(index_path, {}):
                raise ValueError("semantic embedding cache packet-index SHA256 mismatch")
            if required_header_policy and str(semantic_manifest.get("header_policy")) != required_header_policy:
                raise ValueError(
                    "semantic embedding cache header policy mismatch: "
                    f"observed={semantic_manifest.get('header_policy')!r} "
                    f"required={required_header_policy!r}"
                )
            if (
                required_packet_context_policy
                and str(semantic_manifest.get("packet_context_policy"))
                != required_packet_context_policy
            ):
                raise ValueError(
                    "semantic embedding cache packet context policy mismatch: "
                    f"observed={semantic_manifest.get('packet_context_policy')!r} "
                    f"required={required_packet_context_policy!r}"
                )
            semantic_values = np.load(semantic_embedding_cache, mmap_mode="r")
            if semantic_values.ndim != 2 or len(semantic_values) != len(labels):
                raise ValueError(
                    "semantic embedding cache shape mismatch: "
                    f"observed={semantic_values.shape} packets={len(labels)}"
                )
            if int(semantic_manifest.get("num_packets", -1)) != len(labels):
                raise ValueError("semantic embedding manifest packet count mismatch")
            if int(semantic_manifest.get("embedding_dim", -1)) != semantic_values.shape[1]:
                raise ValueError("semantic embedding manifest dimension mismatch")
            self.semantic_embeddings = semantic_values
            self.semantic_dim = int(semantic_values.shape[1])
        if intervened_semantic_embedding_cache:
            if self.semantic_embeddings is None:
                raise ValueError("intervened semantic cache requires a factual semantic cache")
            if not intervened_semantic_embedding_manifest:
                raise ValueError("intervened semantic embedding cache requires its manifest")
            with open(intervened_semantic_embedding_manifest, "r", encoding="utf-8") as handle:
                intervened_manifest = json.load(handle)
            if intervened_manifest.get("scope") != "strict_current_packet_row_aligned_semantic_embedding":
                raise ValueError("intervened semantic embedding cache has an unsupported scope")
            if str(intervened_manifest.get("packet_index_sha256")) != sha256_file(index_path, {}):
                raise ValueError("intervened semantic embedding packet-index SHA256 mismatch")
            if (
                required_intervened_header_policy
                and str(intervened_manifest.get("header_policy"))
                != required_intervened_header_policy
            ):
                raise ValueError(
                    "intervened semantic embedding header policy mismatch: "
                    f"observed={intervened_manifest.get('header_policy')!r} "
                    f"required={required_intervened_header_policy!r}"
                )
            if str(intervened_manifest.get("header_policy")) == str(
                semantic_manifest.get("header_policy")
            ):
                raise ValueError("factual and intervened semantic caches use the same header policy")
            if (
                required_packet_context_policy
                and str(intervened_manifest.get("packet_context_policy"))
                != required_packet_context_policy
            ):
                raise ValueError(
                    "intervened semantic embedding packet context policy mismatch: "
                    f"observed={intervened_manifest.get('packet_context_policy')!r} "
                    f"required={required_packet_context_policy!r}"
                )
            if str(intervened_manifest.get("packet_context_policy")) != str(
                semantic_manifest.get("packet_context_policy")
            ):
                raise ValueError(
                    "factual and intervened semantic caches use different packet context policies"
                )
            intervened_values = np.load(
                intervened_semantic_embedding_cache, mmap_mode="r"
            )
            expected_shape = (len(labels), self.semantic_dim)
            if intervened_values.ndim != 2 or tuple(intervened_values.shape) != expected_shape:
                raise ValueError(
                    "intervened semantic embedding cache shape mismatch: "
                    f"observed={intervened_values.shape} expected={expected_shape}"
                )
            if int(intervened_manifest.get("num_packets", -1)) != len(labels):
                raise ValueError("intervened semantic manifest packet count mismatch")
            if int(intervened_manifest.get("embedding_dim", -1)) != self.semantic_dim:
                raise ValueError("intervened semantic manifest dimension mismatch")
            self.intervened_semantic_embeddings = intervened_values
        self.tokens = torch.from_numpy(np.stack(tokens))
        self.field_ids = torch.from_numpy(np.stack(field_rows))
        self.masked_tokens = torch.from_numpy(np.stack(masked_tokens)) if include_augmented else self.tokens
        self.lengths = torch.tensor(lengths, dtype=torch.long)
        self.payload_tokens = torch.from_numpy(np.stack(payload_tokens))
        self.payload_lengths = torch.tensor(payload_lengths, dtype=torch.long)
        self.metas = torch.from_numpy(np.stack(metas)).float()
        self.masked_metas = torch.from_numpy(np.stack(masked_metas)).float() if include_augmented else self.metas
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.flow_ids = torch.tensor(flow_ids, dtype=torch.long)
        self.content_group_ids = torch.tensor(content_group_ids, dtype=torch.long)
        self.content_hashes = content_hashes
        group_packet_counts: dict[int, int] = {}
        for group_id in content_group_ids:
            group_packet_counts[int(group_id)] = group_packet_counts.get(int(group_id), 0) + 1
        self.content_group_manifest = {
            "version": 1,
            "method": "exact_pcap_sha256_content_groups",
            "source_index": index_path,
            "num_packets": len(labels),
            "num_content_groups": len(hash_to_group_id),
            "duplicate_content_groups": sum(1 for count in group_packet_counts.values() if count > 1),
        }
        self.build_ambiguity_targets = bool(build_ambiguity_targets)
        if self.build_ambiguity_targets:
            target_classes = int(num_classes or (max(labels, default=-1) + 1))
            signature_counts: dict[str, np.ndarray] = {}
            for signature, label in zip(invariant_signatures, labels):
                counts = signature_counts.setdefault(
                    signature, np.zeros(target_classes, dtype=np.float32)
                )
                counts[label] += 1.0
            targets, reliabilities, supports = [], [], []
            for signature in invariant_signatures:
                counts = signature_counts[signature]
                target = counts / counts.sum()
                entropy = float(-(target[target > 0] * np.log(target[target > 0])).sum())
                active_classes = int((counts > 0).sum())
                reliability = (
                    1.0 - entropy / np.log(active_classes) if active_classes > 1 else 1.0
                )
                targets.append(target)
                reliabilities.append(reliability)
                supports.append(int(counts.sum()))
            self.ambiguity_targets = torch.from_numpy(np.stack(targets)).float()
            self.invariant_reliability = torch.tensor(reliabilities, dtype=torch.float32)
            self.signature_support = torch.tensor(supports, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {
            "tokens": self.tokens[index],
            "masked_tokens": self.masked_tokens[index],
            "field_ids": self.field_ids[index],
            "length": self.lengths[index],
            "payload_tokens": self.payload_tokens[index],
            "payload_length": self.payload_lengths[index],
            "meta": self.metas[index],
            "masked_meta": self.masked_metas[index],
            "label": self.labels[index],
            "flow_id": self.flow_ids[index],
            "content_group_id": self.content_group_ids[index],
        }
        if self.build_ambiguity_targets:
            item.update(
                ambiguity_target=self.ambiguity_targets[index],
                invariant_reliability=self.invariant_reliability[index],
                signature_support=self.signature_support[index],
            )
        if self.semantic_embeddings is not None:
            item["semantic_embedding"] = torch.tensor(
                np.asarray(self.semantic_embeddings[index]), dtype=torch.float32
            )
        if self.intervened_semantic_embeddings is not None:
            item["intervened_semantic_embedding"] = torch.tensor(
                np.asarray(self.intervened_semantic_embeddings[index]), dtype=torch.float32
            )
        return item


def effective_class_weights(labels: torch.Tensor, num_classes: int, beta: float) -> torch.Tensor:
    counts = torch.bincount(labels, minlength=num_classes).float().clamp(min=1)
    weights = (1.0 - beta) / (1.0 - torch.pow(torch.tensor(beta), counts))
    return weights * num_classes / weights.sum()


@torch.no_grad()
def predict_packet_views(
    model,
    loader,
    device,
    include_masked=False,
    include_identifiability=False,
    return_gate_diagnostics=False,
    return_gate_values=False,
    ablate_channel="none",
    ablate_intervention_view="none",
):
    if return_gate_values and not return_gate_diagnostics:
        raise ValueError("return_gate_values requires return_gate_diagnostics")
    model.eval()
    ys, raw_probabilities, masked_probabilities, identifiability = [], [], [], []
    content_group_ids = []
    channel_gates, intervention_gates = [], []
    for batch in tqdm(loader, desc="eval byte transformer", leave=False):
        payload_tokens = batch["payload_tokens"].to(device) if model.use_payload_channel else None
        payload_lengths = batch["payload_length"].to(device) if model.use_payload_channel else None
        field_ids = batch["field_ids"].to(device) if model.use_protocol_fields else None
        semantic_features = batch["semantic_embedding"].to(device) if model.semantic_dim > 0 else None
        intervened_semantic_features = (
            batch["intervened_semantic_embedding"].to(device)
            if model.use_intervention_views else None
        )
        model_inputs = (
            batch["tokens"].to(device),
            batch["length"].to(device),
            batch["meta"].to(device),
            payload_tokens,
            payload_lengths,
            field_ids,
            semantic_features,
            intervened_semantic_features,
        )
        if return_gate_diagnostics:
            raw_logits, _, channel_gate, intervention_gate = (
                model.forward_with_gate_diagnostics(
                    *model_inputs,
                    ablate_channel=ablate_channel,
                    ablate_intervention_view=ablate_intervention_view,
                )
            )
            channel_gates.append(channel_gate.detach().float().cpu())
            if intervention_gate is not None:
                intervention_gates.append(intervention_gate.detach().float().cpu())
        else:
            raw_logits, _, _ = model(*model_inputs)
        ys.extend(batch["label"].tolist())
        content_group_ids.extend(batch["content_group_id"].tolist())
        raw_probabilities.append(torch.softmax(raw_logits.float(), dim=-1).cpu().numpy())
        if include_masked:
            if include_identifiability:
                masked_logits, _, _, reliability_logit = model.forward_with_identifiability(
                    batch["masked_tokens"].to(device),
                    batch["length"].to(device),
                    batch["masked_meta"].to(device),
                    payload_tokens,
                    payload_lengths,
                    field_ids,
                    semantic_features,
                    intervened_semantic_features,
                )
                identifiability.append(torch.sigmoid(reliability_logit.float()).cpu().numpy())
            else:
                masked_logits, _, _ = model(
                    batch["masked_tokens"].to(device),
                    batch["length"].to(device),
                    batch["masked_meta"].to(device),
                    payload_tokens,
                    payload_lengths,
                    field_ids,
                    semantic_features,
                    intervened_semantic_features,
                )
            masked_probabilities.append(
                torch.softmax(masked_logits.float(), dim=-1).cpu().numpy()
            )
    y_true = np.asarray(ys, dtype=np.int64)
    raw = np.concatenate(raw_probabilities)
    masked = np.concatenate(masked_probabilities) if masked_probabilities else None
    reliability = np.concatenate(identifiability) if identifiability else None
    groups = np.asarray(content_group_ids, dtype=np.int64)
    if not return_gate_diagnostics:
        return y_true, raw, masked, reliability, groups
    gate_diagnostics = summarize_packet_gate_diagnostics(
        model, channel_gates, intervention_gates
    )
    if return_gate_values:
        gate_values = packet_effective_gate_values(
            model, channel_gates, intervention_gates
        )
        return (
            y_true,
            raw,
            masked,
            reliability,
            groups,
            gate_diagnostics,
            gate_values,
        )
    return y_true, raw, masked, reliability, groups, gate_diagnostics


def packet_effective_gate_values(model, channel_chunks, intervention_chunks):
    """Return per-packet effective routing weights for mechanism diagnostics."""
    gate_values: dict[str, np.ndarray] = {}
    if channel_chunks:
        values = torch.cat(channel_chunks, dim=0).float()
        fusion = model.shared_packet_fusion
        if getattr(model, "train_fixed_channel_fusion", False):
            effective = values
        elif fusion.base_mode == "semantic_anchor":
            effective = fusion.effective_weights(values)
        else:
            effective = values
        gate_values["packet_channel_gate"] = effective.numpy()
    if intervention_chunks:
        values = torch.cat(intervention_chunks, dim=0).float()
        fusion = getattr(model, "intervention_view_fusion", None)
        if fusion is None and hasattr(model, "shared_packet_encoder"):
            fusion = model.shared_packet_encoder.intervention_view_fusion
        if fusion is None:
            raise ValueError("intervention gate was emitted without its fusion module")
        gate_values["intervention_view_gate"] = (
            fusion.effective_weights(values).numpy()
        )
    return gate_values


def summarize_packet_gate_diagnostics(model, channel_chunks, intervention_chunks):
    diagnostics = {}

    def summarize(values: torch.Tensor) -> dict:
        return {
            "num_samples": int(values.shape[0]),
            "mean": values.mean(dim=0).tolist(),
            "std": values.std(dim=0, unbiased=False).tolist(),
            "p05": torch.quantile(values, 0.05, dim=0).tolist(),
            "p50": torch.quantile(values, 0.50, dim=0).tolist(),
            "p95": torch.quantile(values, 0.95, dim=0).tolist(),
        }

    if channel_chunks:
        values = torch.cat(channel_chunks, dim=0)
        summary = summarize(values)
        fusion = model.shared_packet_fusion
        summary["channel_names"] = list(fusion.channel_names)
        summary["base_mode"] = fusion.base_mode
        summary["weight_semantics"] = (
            "bounded_effective_routing_weights_excluding_interaction_correction"
        )
        residual = float(fusion.interaction_max_weight)
        summary["max_residual_weight"] = residual
        if getattr(model, "train_fixed_channel_fusion", False):
            effective = values
            fixed_weight = 1.0 / len(fusion.channel_names)
            summary["weight_semantics"] = "fixed_equal_normalized_channel_mean"
            summary["max_residual_weight"] = 0.0
            summary["effective_routing_mean"] = effective.mean(dim=0).tolist()
            summary["effective_routing_p05"] = torch.quantile(effective, 0.05, dim=0).tolist()
            summary["effective_routing_p50"] = torch.quantile(effective, 0.50, dim=0).tolist()
            summary["effective_routing_p95"] = torch.quantile(effective, 0.95, dim=0).tolist()
            summary["theoretical_bounds"] = {
                name: [fixed_weight, fixed_weight] for name in fusion.channel_names
            }
            summary["bounds_satisfied"] = bool(
                torch.allclose(effective, torch.full_like(effective, fixed_weight))
            )
        elif fusion.base_mode == "semantic_anchor":
            effective = fusion.effective_weights(values)
            bounds = fusion.effective_weight_bounds()
            summary["effective_routing_mean"] = effective.mean(dim=0).tolist()
            summary["effective_routing_p05"] = torch.quantile(effective, 0.05, dim=0).tolist()
            summary["effective_routing_p50"] = torch.quantile(effective, 0.50, dim=0).tolist()
            summary["effective_routing_p95"] = torch.quantile(effective, 0.95, dim=0).tolist()
            summary["theoretical_bounds"] = {
                name: list(bounds[name]) for name in fusion.channel_names
            }
            lower = torch.tensor(
                [bounds[name][0] for name in fusion.channel_names], dtype=effective.dtype
            )
            upper = torch.tensor(
                [bounds[name][1] for name in fusion.channel_names], dtype=effective.dtype
            )
            summary["bounds_satisfied"] = bool(
                torch.all(effective >= lower - 1e-6)
                and torch.all(effective <= upper + 1e-6)
            )
        diagnostics["packet_channel_gate"] = summary
    if intervention_chunks:
        values = torch.cat(intervention_chunks, dim=0)
        summary = summarize(values)
        fusion = getattr(model, "intervention_view_fusion", None)
        if fusion is None and hasattr(model, "shared_packet_encoder"):
            fusion = model.shared_packet_encoder.intervention_view_fusion
        if fusion is None:
            raise ValueError("intervention gate was emitted without its fusion module")
        summary["base_mode"] = fusion.base_mode
        summary["view_names"] = ["factual", "intervened"]
        summary["weight_semantics"] = (
            "bounded_effective_routing_weights_before_channel_fusion"
        )
        residual = float(fusion.max_residual_weight)
        summary["max_residual_weight"] = residual
        effective = fusion.effective_weights(values)
        bounds = fusion.effective_weight_bounds()
        view_names = ("factual", "intervened")
        lower = torch.tensor(
            [bounds[name][0] for name in view_names], dtype=effective.dtype
        )
        upper = torch.tensor(
            [bounds[name][1] for name in view_names], dtype=effective.dtype
        )
        summary["effective_routing_mean"] = effective.mean(dim=0).tolist()
        summary["effective_routing_p05"] = torch.quantile(effective, 0.05, dim=0).tolist()
        summary["effective_routing_p50"] = torch.quantile(effective, 0.50, dim=0).tolist()
        summary["effective_routing_p95"] = torch.quantile(effective, 0.95, dim=0).tolist()
        summary["theoretical_bounds"] = {
            name: list(bounds[name]) for name in view_names
        }
        summary["bounds_satisfied"] = bool(
            torch.all(effective >= lower - 1e-6)
            and torch.all(effective <= upper + 1e-6)
        )
        diagnostics["intervention_view_gate"] = summary
    return diagnostics


def select_invariant_blend(
    y_true: np.ndarray,
    raw_probabilities: np.ndarray,
    masked_probabilities: np.ndarray,
    num_classes: int,
    label_names: list[str],
    metric: str = "macro_f1",
    grid_size: int = 21,
) -> tuple[float, dict, np.ndarray]:
    if grid_size < 2:
        raise ValueError("invariant blend grid size must be at least 2")
    best = None
    for raw_weight in np.linspace(0.0, 1.0, grid_size):
        probabilities = (
            raw_weight * raw_probabilities + (1.0 - raw_weight) * masked_probabilities
        )
        metrics = packet_classification_metrics(
            y_true, probabilities.argmax(axis=1), num_classes, label_names
        )
        key = (
            float(metrics[metric]),
            float(metrics["macro_f1"]),
            float(metrics["accuracy"]),
            float(raw_weight),
        )
        if best is None or key > best[0]:
            best = (key, float(raw_weight), metrics, probabilities)
    assert best is not None
    return best[1], best[2], best[3]


def select_routed_invariant_blend(
    y_true: np.ndarray,
    raw_probabilities: np.ndarray,
    masked_probabilities: np.ndarray,
    reliability: np.ndarray,
    num_classes: int,
    label_names: list[str],
    metric: str = "macro_f1",
    grid_size: int = 21,
) -> tuple[float, dict, np.ndarray]:
    if grid_size < 2:
        raise ValueError("routed invariant blend grid size must be at least 2")
    reliability = np.asarray(reliability, dtype=np.float32).reshape(-1, 1)
    best = None
    for invariant_scale in np.linspace(0.0, 1.0, grid_size):
        masked_weight = invariant_scale * reliability
        probabilities = (1.0 - masked_weight) * raw_probabilities + masked_weight * masked_probabilities
        metrics = packet_classification_metrics(
            y_true, probabilities.argmax(axis=1), num_classes, label_names
        )
        key = (
            float(metrics[metric]),
            float(metrics["macro_f1"]),
            float(metrics["accuracy"]),
            -float(invariant_scale),
        )
        if best is None or key > best[0]:
            best = (key, float(invariant_scale), metrics, probabilities)
    assert best is not None
    return best[1], best[2], best[3]


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    num_classes,
    label_names,
    return_probabilities=False,
    raw_weight: float = 1.0,
):
    use_masked = raw_weight < 1.0
    y_true, raw, masked, _, content_group_ids = predict_packet_views(
        model, loader, device, include_masked=use_masked
    )
    probabilities = raw if masked is None else raw_weight * raw + (1.0 - raw_weight) * masked
    metrics = packet_classification_metrics(
        y_true, probabilities.argmax(axis=1), num_classes, label_names
    )
    metrics.update(
        packet_content_group_metrics(
            y_true,
            probabilities.argmax(axis=1),
            content_group_ids,
            num_classes,
            label_names,
        )
    )
    return metrics, probabilities if return_probabilities else None, y_true


def save_checkpoint(
    path: Path,
    model,
    config,
    metrics,
    inference_config=None,
    initialization=None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": config,
            "validation_metrics": metrics,
            "inference_config": inference_config or {"raw_weight": 1.0},
            "initialization": initialization or {},
        },
        path,
    )


def initialize_protocol_content_encoder(
    model: PacketByteTransformer,
    packet_config: dict,
    checkpoint_path: str,
) -> dict:
    if not model.use_protocol_fields:
        raise ValueError("native packet-content initialization requires --use_protocol_fields")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    observed_protocol = checkpoint.get("pretraining_protocol")
    if observed_protocol != NATIVE_PACKET_PRETRAINING_PROTOCOL:
        raise ValueError(
            "native checkpoint pretraining protocol mismatch: "
            f"expected={NATIVE_PACKET_PRETRAINING_PROTOCOL!r} "
            f"observed={observed_protocol!r}"
        )
    native_config = checkpoint.get("model_config") or {}
    expected = {
        "max_bytes": int(packet_config["max_bytes"]),
        "hidden_dim": int(packet_config["hidden_dim"]),
        "byte_layers": int(packet_config["num_layers"]),
        "num_heads": int(packet_config["num_heads"]),
        "dropout": float(packet_config["dropout"]),
        "num_field_types": 9,
    }
    observed = {
        key: (float(native_config[key]) if key == "dropout" else int(native_config[key]))
        for key in expected
    }
    differences = {
        key: {"packet": expected[key], "native": observed[key]}
        for key in expected
        if expected[key] != observed[key]
    }
    if differences:
        raise ValueError(
            "native packet-content architecture mismatch: "
            + json.dumps(differences, sort_keys=True)
        )
    prefix = "packet_content_encoder."
    source_state = {
        key[len(prefix) :]: value
        for key, value in checkpoint["state_dict"].items()
        if key.startswith(prefix)
    }
    if not source_state:
        raise ValueError("native checkpoint has no packet_content_encoder parameters")
    model.protocol_content_encoder.load_state_dict(source_state, strict=True)
    source_hash = hashlib.sha256()
    with open(checkpoint_path, "rb") as source_file:
        for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
            source_hash.update(chunk)
    return {
        "protocol_content_pretraining": NATIVE_PACKET_PRETRAINING_PROTOCOL,
        "protocol_content_checkpoint": str(Path(checkpoint_path)),
        "protocol_content_checkpoint_sha256": source_hash.hexdigest(),
        "protocol_content_architecture": expected,
        "strict_state_dict_load": True,
    }


def weighted_soft_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    sample_weights: torch.Tensor,
) -> torch.Tensor:
    losses = -(targets * F.log_softmax(logits.float(), dim=-1)).sum(dim=-1)
    return (losses * sample_weights).sum() / sample_weights.sum().clamp_min(1e-8)


def reliability_weighted_symmetric_kl(
    first_logits: torch.Tensor,
    second_logits: torch.Tensor,
    reliability: torch.Tensor,
) -> torch.Tensor:
    first_log = F.log_softmax(first_logits.float(), dim=-1)
    second_log = F.log_softmax(second_logits.float(), dim=-1)
    first_prob = first_log.exp().detach()
    second_prob = second_log.exp().detach()
    per_sample = 0.5 * (
        F.kl_div(first_log, second_prob, reduction="none").sum(dim=-1)
        + F.kl_div(second_log, first_prob, reduction="none").sum(dim=-1)
    )
    return (per_sample * reliability).sum() / reliability.sum().clamp_min(1e-8)


def interpolate_identifiability_reliability(
    reliability: torch.Tensor, strength: float
) -> torch.Tensor:
    return 1.0 - float(strength) + float(strength) * reliability


def assign_main_protected_gradients(
    main_loss: torch.Tensor,
    auxiliary_loss: torch.Tensor,
    parameters: list[torch.nn.Parameter],
    projection_scope: str = "layerwise",
    eps: float = 1e-12,
) -> dict[str, float | bool]:
    main_grads = torch.autograd.grad(
        main_loss, parameters, retain_graph=True, allow_unused=True
    )
    auxiliary_grads = torch.autograd.grad(
        auxiliary_loss, parameters, allow_unused=True
    )
    if projection_scope not in {"global", "layerwise"}:
        raise ValueError(f"unsupported gradient projection scope: {projection_scope}")
    paired_indices = [
        index
        for index, (main_grad, auxiliary_grad) in enumerate(zip(main_grads, auxiliary_grads))
        if main_grad is not None and auxiliary_grad is not None
    ]
    projections: dict[int, torch.Tensor] = {}
    cosines = []
    conflicts = []
    if projection_scope == "global" and paired_indices:
        dot = torch.stack(
            [(main_grads[i] * auxiliary_grads[i]).sum() for i in paired_indices]
        ).sum()
        main_norm_sq = torch.stack(
            [(main_grads[i] * main_grads[i]).sum() for i in paired_indices]
        ).sum()
        auxiliary_norm_sq = torch.stack(
            [(auxiliary_grads[i] * auxiliary_grads[i]).sum() for i in paired_indices]
        ).sum()
        cosine = dot / (main_norm_sq.sqrt() * auxiliary_norm_sq.sqrt()).clamp_min(eps)
        conflict = bool(dot.detach() < 0)
        projection = dot / main_norm_sq.clamp_min(eps) if conflict else dot.new_zeros(())
        for index in paired_indices:
            projections[index] = projection
        cosines.append(cosine)
        conflicts.append(conflict)
    elif projection_scope == "layerwise":
        for index in paired_indices:
            main_grad = main_grads[index]
            auxiliary_grad = auxiliary_grads[index]
            dot = (main_grad * auxiliary_grad).sum()
            main_norm_sq = (main_grad * main_grad).sum()
            auxiliary_norm_sq = (auxiliary_grad * auxiliary_grad).sum()
            cosine = dot / (main_norm_sq.sqrt() * auxiliary_norm_sq.sqrt()).clamp_min(eps)
            conflict = bool(dot.detach() < 0)
            projections[index] = (
                dot / main_norm_sq.clamp_min(eps) if conflict else dot.new_zeros(())
            )
            cosines.append(cosine)
            conflicts.append(conflict)

    for index, (parameter, main_grad, auxiliary_grad) in enumerate(
        zip(parameters, main_grads, auxiliary_grads)
    ):
        if main_grad is None and auxiliary_grad is None:
            parameter.grad = None
            continue
        protected_auxiliary = auxiliary_grad
        if index in projections and bool(projections[index].detach() < 0):
            protected_auxiliary = auxiliary_grad - projections[index] * main_grad
        if main_grad is None:
            parameter.grad = protected_auxiliary
        elif protected_auxiliary is None:
            parameter.grad = main_grad
        else:
            parameter.grad = main_grad + protected_auxiliary
    conflict_rate = float(np.mean(conflicts)) if conflicts else 0.0
    cosine_mean = (
        float(torch.stack(cosines).mean().detach().cpu()) if cosines else 0.0
    )
    return {
        "conflict": conflict_rate > 0.0,
        "conflict_rate": conflict_rate,
        "cosine": cosine_mean,
    }


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
    ap.add_argument("--use_protocol_fields", action="store_true")
    ap.add_argument("--exact_shared_representation", action="store_true")
    ap.add_argument("--mask_protocol_session_fields", action="store_true")
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
    ap.add_argument("--semantic_embedding_cache", default="")
    ap.add_argument("--semantic_embedding_manifest", default="")
    ap.add_argument("--valid_semantic_embedding_cache", default="")
    ap.add_argument("--valid_semantic_embedding_manifest", default="")
    ap.add_argument("--test_semantic_embedding_cache", default="")
    ap.add_argument("--test_semantic_embedding_manifest", default="")
    ap.add_argument("--required_semantic_header_policy", default="")
    ap.add_argument("--required_semantic_packet_context_policy", default="")
    ap.add_argument("--intervened_semantic_embedding_cache", default="")
    ap.add_argument("--intervened_semantic_embedding_manifest", default="")
    ap.add_argument("--valid_intervened_semantic_embedding_cache", default="")
    ap.add_argument("--valid_intervened_semantic_embedding_manifest", default="")
    ap.add_argument("--test_intervened_semantic_embedding_cache", default="")
    ap.add_argument("--test_intervened_semantic_embedding_manifest", default="")
    ap.add_argument("--required_intervened_semantic_header_policy", default="")
    ap.add_argument(
        "--protocol_content_checkpoint",
        default="",
        help=(
            "Optional NativeFlowEncoder checkpoint used to strictly initialize the "
            "shared current-packet protocol content encoder."
        ),
    )
    ap.add_argument("--intervention_max_residual_weight", type=float, default=0.25)
    ap.add_argument(
        "--intervention_view_base_mode",
        choices=["symmetric_mean", "factual_anchor"],
        default="symmetric_mean",
    )
    ap.add_argument("--channel_fusion_base_mode", choices=["legacy", "semantic_anchor"], default="legacy")
    ap.add_argument(
        "--channel_fusion_max_weight",
        type=float,
        default=0.25,
        help="Hard multiplier on the shared packet-fusion residual path.",
    )
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
    ap.add_argument(
        "--content_group_loss_reduction",
        choices=["none", "group_mean"],
        default="none",
        help="Average the main clean CE inside each pcap_path-derived content group before averaging across groups.",
    )
    ap.add_argument("--mask_probability", type=float, default=0.5)
    ap.add_argument("--masked_ce_weight", type=float, default=0.3)
    ap.add_argument("--consistency_weight", type=float, default=0.1)
    ap.add_argument(
        "--ambiguity_aware_targets",
        action="store_true",
        help="Use session-invariant signature ambiguity when supervising the masked view.",
    )
    ap.add_argument("--ambiguity_min_support", type=int, default=2)
    ap.add_argument(
        "--ambiguity_gate_strength",
        type=float,
        default=1.0,
        help="Interpolation from hard masked supervision (0) to full reliability gating (1).",
    )
    ap.add_argument(
        "--ambiguity_supervision",
        choices=["reliability_gate", "empirical_soft"],
        default="reliability_gate",
        help=(
            "For ambiguous masked packets, either attenuate hard-label supervision by "
            "identifiability reliability or use the empirical label distribution."
        ),
    )
    ap.add_argument(
        "--select_invariant_blend",
        action="store_true",
        help="Select raw/session-masked probability blending on validation only.",
    )
    ap.add_argument("--invariant_blend_grid_size", type=int, default=21)
    ap.add_argument(
        "--invariant_blend_metric",
        choices=["accuracy", "macro_f1"],
        default="macro_f1",
    )
    ap.add_argument("--learned_identifiability_router", action="store_true")
    ap.add_argument("--identifiability_loss_weight", type=float, default=0.1)
    ap.add_argument("--protect_main_gradient", action="store_true")
    ap.add_argument(
        "--gradient_projection_scope",
        choices=["global", "layerwise"],
        default="layerwise",
    )
    ap.add_argument("--contrastive_weight", type=float, default=0.05)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--packets_per_flow", type=int, default=2)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    if not 0.0 <= args.ambiguity_gate_strength <= 1.0:
        ap.error("--ambiguity_gate_strength must be in [0, 1]")
    if args.learned_identifiability_router and not args.ambiguity_aware_targets:
        ap.error("--learned_identifiability_router requires --ambiguity_aware_targets")
    if args.train_ablate_input_channel != "none" and not args.exact_shared_representation:
        ap.error("--train_ablate_input_channel requires --exact_shared_representation")
    if args.train_ablate_intervention_view != "none" and not args.exact_shared_representation:
        ap.error("--train_ablate_intervention_view requires --exact_shared_representation")
    if args.train_fixed_channel_fusion and not args.exact_shared_representation:
        ap.error("--train_fixed_channel_fusion requires --exact_shared_representation")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    label_names = load_label_names(args.label_map)
    num_classes = len(label_names)
    device = torch.device(args.device)
    needs_training_view = (
        args.masked_ce_weight > 0
        or args.consistency_weight > 0
        or args.learned_identifiability_router
    )
    payload_width = args.max_payload_bytes if args.use_payload_channel else 0
    train_dataset = PacketByteDataset(
        args.train_index,
        args.max_bytes,
        include_augmented=needs_training_view,
        max_payload_bytes=payload_width,
        build_ambiguity_targets=args.ambiguity_aware_targets,
        num_classes=num_classes,
        semantic_embedding_cache=args.semantic_embedding_cache,
        semantic_embedding_manifest=args.semantic_embedding_manifest,
        required_header_policy=args.required_semantic_header_policy,
        required_packet_context_policy=args.required_semantic_packet_context_policy,
        intervened_semantic_embedding_cache=args.intervened_semantic_embedding_cache,
        intervened_semantic_embedding_manifest=args.intervened_semantic_embedding_manifest,
        required_intervened_header_policy=args.required_intervened_semantic_header_policy,
    )
    valid_dataset = PacketByteDataset(
        args.valid_index,
        args.max_bytes,
        include_augmented=args.select_invariant_blend or args.learned_identifiability_router,
        max_payload_bytes=payload_width,
        num_classes=num_classes,
        semantic_embedding_cache=args.valid_semantic_embedding_cache,
        semantic_embedding_manifest=args.valid_semantic_embedding_manifest,
        required_header_policy=args.required_semantic_header_policy,
        required_packet_context_policy=args.required_semantic_packet_context_policy,
        intervened_semantic_embedding_cache=args.valid_intervened_semantic_embedding_cache,
        intervened_semantic_embedding_manifest=args.valid_intervened_semantic_embedding_manifest,
        required_intervened_header_policy=args.required_intervened_semantic_header_policy,
    )
    if bool(train_dataset.intervened_semantic_embeddings is None) != bool(
        valid_dataset.intervened_semantic_embeddings is None
    ):
        raise ValueError("train and validation must both provide intervention semantic views")
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
        "use_identifiability_head": args.learned_identifiability_router,
        "use_protocol_fields": args.use_protocol_fields,
        "exact_shared_representation": args.exact_shared_representation,
        "mask_protocol_session_fields": args.mask_protocol_session_fields,
        "train_ablate_input_channel": args.train_ablate_input_channel,
        "train_ablate_intervention_view": args.train_ablate_intervention_view,
        "train_fixed_channel_fusion": args.train_fixed_channel_fusion,
        "semantic_dim": train_dataset.semantic_dim,
        "use_intervention_views": train_dataset.intervened_semantic_embeddings is not None,
        "intervention_max_residual_weight": args.intervention_max_residual_weight,
        "intervention_view_base_mode": args.intervention_view_base_mode,
        "channel_fusion_base_mode": args.channel_fusion_base_mode,
        "channel_fusion_max_weight": float(
            max(0.0, min(1.0, args.channel_fusion_max_weight))
        ),
    }
    model = PacketByteTransformer(**config).to(device)
    initialization = (
        initialize_protocol_content_encoder(model, config, args.protocol_content_checkpoint)
        if args.protocol_content_checkpoint
        else {}
    )
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
        gradient_conflicts = []
        gradient_cosines = []
        for batch in tqdm(train_loader, desc=f"byte transformer epoch {epoch}"):
            clean_tokens = batch["tokens"].to(device)
            lengths = batch["length"].to(device)
            clean_meta = batch["meta"].to(device)
            labels = batch["label"].to(device)
            flow_ids = batch["flow_id"].to(device)
            content_group_ids = batch["content_group_id"].to(device)
            field_ids = batch["field_ids"].to(device) if model.use_protocol_fields else None
            semantic_features = batch["semantic_embedding"].to(device) if model.semantic_dim > 0 else None
            intervened_semantic_features = (
                batch["intervened_semantic_embedding"].to(device)
                if model.use_intervention_views else None
            )
            if needs_training_view:
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
                    clean_tokens, lengths, clean_meta, payload_tokens, payload_lengths,
                    field_ids, semantic_features, intervened_semantic_features
                )
                ce = packet_ce_loss(
                    clean_logits,
                    labels,
                    class_weights,
                    content_group_ids,
                    args.content_group_loss_reduction,
                )
                masked_ce = clean_logits.sum() * 0.0
                consistency = clean_logits.sum() * 0.0
                identifiability_loss = clean_logits.sum() * 0.0
                if needs_training_view:
                    if args.learned_identifiability_router:
                        view_logits, _, _, reliability_logit = model.forward_with_identifiability(
                            view_tokens, lengths, view_meta, payload_tokens, payload_lengths,
                            field_ids, semantic_features, intervened_semantic_features
                        )
                    else:
                        view_logits, _, _ = model(
                            view_tokens, lengths, view_meta, payload_tokens, payload_lengths,
                            field_ids, semantic_features, intervened_semantic_features
                        )
                    if args.ambiguity_aware_targets:
                        hard_targets = F.one_hot(labels, num_classes=num_classes).float()
                        ambiguity_targets = batch["ambiguity_target"].to(device)
                        enough_support = (
                            batch["signature_support"].to(device) >= args.ambiguity_min_support
                        )
                        invariant_reliability = batch["invariant_reliability"].to(device)
                        reliability = torch.where(
                            use_mask & enough_support,
                            invariant_reliability,
                            torch.ones_like(invariant_reliability),
                        )
                        if args.learned_identifiability_router:
                            supervised_router = (use_mask & enough_support).float()
                            per_sample_router_loss = F.smooth_l1_loss(
                                torch.sigmoid(reliability_logit.float()),
                                invariant_reliability,
                                reduction="none",
                            )
                            identifiability_loss = (
                                per_sample_router_loss * supervised_router
                            ).sum() / supervised_router.sum().clamp_min(1.0)
                        if args.ambiguity_supervision == "empirical_soft":
                            masked_targets = torch.where(
                                (use_mask & enough_support)[:, None],
                                ambiguity_targets,
                                hard_targets,
                            )
                            supervision_reliability = torch.ones_like(reliability)
                        else:
                            masked_targets = hard_targets
                            supervision_reliability = interpolate_identifiability_reliability(
                                reliability, args.ambiguity_gate_strength
                            )
                        masked_ce = weighted_soft_cross_entropy(
                            view_logits,
                            masked_targets,
                            class_weights[labels] * supervision_reliability,
                        )
                    else:
                        masked_ce = F.cross_entropy(view_logits, labels, weight=class_weights)
                        reliability = torch.ones_like(labels, dtype=torch.float32)
                    consistency = reliability_weighted_symmetric_kl(
                        clean_logits, view_logits, reliability
                    )
                contrastive = flow_aware_contrastive_loss(
                    clean_z, labels, flow_ids, temperature=args.temperature,
                    same_flow_weight=2.0, same_label_weight=1.0,
                )
                auxiliary_loss = (
                    args.masked_ce_weight * masked_ce
                    + args.consistency_weight * consistency
                    + args.contrastive_weight * contrastive
                    + args.identifiability_loss_weight * identifiability_loss
                )
                loss = ce + auxiliary_loss
            if args.protect_main_gradient:
                diagnostics = assign_main_protected_gradients(
                    ce,
                    auxiliary_loss,
                    list(model.parameters()),
                    projection_scope=args.gradient_projection_scope,
                )
                gradient_conflicts.append(float(diagnostics["conflict_rate"]))
                gradient_cosines.append(float(diagnostics["cosine"]))
            else:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if args.protect_main_gradient:
                optimizer.step()
            else:
                scaler.step(optimizer)
                scaler.update()
            scheduler.step()
            losses.append(float(loss.detach().cpu()))
        inference_config = {"raw_weight": 1.0, "selection_scope": "raw_only"}
        if args.select_invariant_blend or args.learned_identifiability_router:
            y_valid, raw_valid, masked_valid, routed_reliability, valid_content_group_ids = predict_packet_views(
                model,
                valid_loader,
                device,
                include_masked=True,
                include_identifiability=args.learned_identifiability_router,
            )
            assert masked_valid is not None
            if args.learned_identifiability_router:
                assert routed_reliability is not None
                invariant_scale, valid_metrics, selected_valid_probabilities = select_routed_invariant_blend(
                    y_valid,
                    raw_valid,
                    masked_valid,
                    routed_reliability,
                    num_classes,
                    label_names,
                    args.invariant_blend_metric,
                    args.invariant_blend_grid_size,
                )
                raw_weight = 1.0
            else:
                raw_weight, valid_metrics, selected_valid_probabilities = select_invariant_blend(
                    y_valid,
                    raw_valid,
                    masked_valid,
                    num_classes,
                    label_names,
                    args.invariant_blend_metric,
                    args.invariant_blend_grid_size,
                )
                invariant_scale = 0.0
            raw_metrics = packet_classification_metrics(
                y_valid, raw_valid.argmax(axis=1), num_classes, label_names
            )
            raw_metrics.update(
                packet_content_group_metrics(
                    y_valid,
                    raw_valid.argmax(axis=1),
                    valid_content_group_ids,
                    num_classes,
                    label_names,
                )
            )
            masked_metrics = packet_classification_metrics(
                y_valid, masked_valid.argmax(axis=1), num_classes, label_names
            )
            masked_metrics.update(
                packet_content_group_metrics(
                    y_valid,
                    masked_valid.argmax(axis=1),
                    valid_content_group_ids,
                    num_classes,
                    label_names,
                )
            )
            valid_metrics.update(
                packet_content_group_metrics(
                    y_valid,
                    selected_valid_probabilities.argmax(axis=1),
                    valid_content_group_ids,
                    num_classes,
                    label_names,
                )
            )
            inference_config = {
                "raw_weight": raw_weight,
                "router_enabled": args.learned_identifiability_router,
                "invariant_scale": invariant_scale,
                "selection_scope": "validation_only",
                "select_metric": args.invariant_blend_metric,
                "grid_size": args.invariant_blend_grid_size,
                "raw_validation_metrics": raw_metrics,
                "masked_validation_metrics": masked_metrics,
            }
        else:
            valid_metrics, _, _ = evaluate(
                model, valid_loader, device, num_classes, label_names
            )
        record = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "validation_metrics": valid_metrics,
            "inference_config": inference_config,
            "gradient_conflict_rate": (
                float(np.mean(gradient_conflicts)) if gradient_conflicts else None
            ),
            "gradient_cosine_mean": (
                float(np.mean(gradient_cosines)) if gradient_cosines else None
            ),
        }
        history.append(record)
        key = (valid_metrics["macro_f1"], valid_metrics["accuracy"])
        gradient_text = ""
        if record["gradient_conflict_rate"] is not None:
            gradient_text = (
                f" grad_conflict={record['gradient_conflict_rate']:.3f}"
                f" grad_cos={record['gradient_cosine_mean']:.3f}"
            )
        print(
            f"epoch={epoch} loss={record['loss']:.4f} valid_acc={valid_metrics['accuracy']:.4f} "
            f"valid_macro_f1={valid_metrics['macro_f1']:.4f}{gradient_text}",
            flush=True,
        )
        if best_key is None or key > best_key:
            best_key = key
            best_epoch = epoch
            save_checkpoint(
                output_dir / "best.pt",
                model,
                config,
                valid_metrics,
                inference_config,
                initialization,
            )
        if epoch - best_epoch >= args.patience:
            print(f"early stopping after epoch={epoch}; best_epoch={best_epoch}", flush=True)
            break

    checkpoint = torch.load(output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    result = {
        "task": "packet-level-classification",
        "sample_unit": "one_packet",
        "architecture": (
            (
                "shared-protocol-aware-semantic-content-structural-trichannel-single-head"
                if config.get("semantic_dim", 0) > 0
                else "shared-protocol-aware-content-structural-gated"
            )
            if args.use_protocol_fields else (
                "dual-channel-byte-payload-transformer-meta-gated"
                if args.use_payload_channel else "local-byte-transformer-meta-gated"
            )
        ),
        "config": vars(args),
        "model_config": config,
        "best_epoch": best_epoch,
        "validation_metrics": checkpoint["validation_metrics"],
        "inference_config": checkpoint.get("inference_config", {"raw_weight": 1.0}),
        "initialization": checkpoint.get("initialization", {}),
        "history": history,
        "content_group_manifests": {
            "train": train_dataset.content_group_manifest,
            "valid": valid_dataset.content_group_manifest,
        },
    }
    if args.test_index:
        selected_raw_weight = float(result["inference_config"].get("raw_weight", 1.0))
        test_dataset = PacketByteDataset(
            args.test_index,
            args.max_bytes,
            include_augmented=args.select_invariant_blend or args.learned_identifiability_router,
            max_payload_bytes=payload_width,
            num_classes=num_classes,
            semantic_embedding_cache=args.test_semantic_embedding_cache,
            semantic_embedding_manifest=args.test_semantic_embedding_manifest,
            required_header_policy=args.required_semantic_header_policy,
            required_packet_context_policy=args.required_semantic_packet_context_policy,
            intervened_semantic_embedding_cache=args.test_intervened_semantic_embedding_cache,
            intervened_semantic_embedding_manifest=args.test_intervened_semantic_embedding_manifest,
            required_intervened_header_policy=args.required_intervened_semantic_header_policy,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        result["content_group_manifests"]["test"] = test_dataset.content_group_manifest
        router_enabled = bool(result["inference_config"].get("router_enabled", False))
        y_true, raw_test, masked_test, routed_reliability, test_content_group_ids = predict_packet_views(
            model,
            test_loader,
            device,
            include_masked=args.select_invariant_blend or router_enabled,
            include_identifiability=router_enabled,
        )
        if router_enabled:
            assert masked_test is not None and routed_reliability is not None
            invariant_scale = float(result["inference_config"].get("invariant_scale", 0.0))
            masked_weight = invariant_scale * routed_reliability.reshape(-1, 1)
            probabilities = (1.0 - masked_weight) * raw_test + masked_weight * masked_test
        else:
            probabilities = (
                raw_test
                if masked_test is None
                else selected_raw_weight * raw_test + (1.0 - selected_raw_weight) * masked_test
            )
        test_metrics = packet_classification_metrics(
            y_true, probabilities.argmax(axis=1), num_classes, label_names
        )
        test_metrics.update(
            packet_content_group_metrics(
                y_true,
                probabilities.argmax(axis=1),
                test_content_group_ids,
                num_classes,
                label_names,
            )
        )
        result["test_metrics"] = test_metrics
        result["test_view_metrics"] = {
            "raw": packet_classification_metrics(
                y_true, raw_test.argmax(axis=1), num_classes, label_names
            )
        }
        if masked_test is not None:
            result["test_view_metrics"]["session_invariant"] = packet_classification_metrics(
                y_true, masked_test.argmax(axis=1), num_classes, label_names
            )
        if args.output_npz:
            Path(args.output_npz).parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                args.output_npz,
                y_true=y_true,
                probabilities=probabilities,
                content_group_ids=test_content_group_ids,
            )
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"saved {output_path} and {output_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
