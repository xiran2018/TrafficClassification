#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

from train_tower2 import (
    FlowAggregationHead,
    SeqDataset,
    GraphDataset,
    collate_seq,
    apply_hierarchical_logits,
    build_class_to_coarse,
    parse_label_groups,
    file_evidence,
    build_flow_groups,
    aligned_graph_items,
)
from models.flow_transformer import FlowTransformerClassifier
from models.flow_graph_transformer import FlowGraphTransformerClassifier
from models.counterfactual_flow_fusion import CounterfactualFlowFusion
from probability_metrics import calibration_metrics


def checkpoint_flow_stat_meta_dim(ckpt: dict) -> int:
    """Read the persisted statistics contract with legacy compatibility."""
    return int(ckpt.get("flow_stat_meta_dim", ckpt.get("meta_feature_dim", 0)))


def prediction_provenance(
    checkpoint_path: str,
    dataset_path: str,
    paired_dataset_path: str,
    checkpoint: dict,
) -> dict:
    payload = {
        "schema": "tower2_prediction_provenance_v1",
        "checkpoint": file_evidence(checkpoint_path),
        "evaluation_dataset": file_evidence(dataset_path),
        "checkpoint_training_input_evidence": checkpoint.get(
            "training_input_evidence", {}
        ),
        "checkpoint_binds_training_dataset": bool(
            checkpoint.get("training_input_evidence", {}).get("train_dataset")
        ),
    }
    if paired_dataset_path:
        payload["paired_evaluation_dataset"] = file_evidence(paired_dataset_path)
    return payload


def load_model(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device)
    if ckpt["model_type"] == "seq":
        model = FlowTransformerClassifier(
            ckpt["input_dim"], ckpt["num_classes"], ckpt["hidden_dim"],
            ckpt["num_layers"], ckpt["num_heads"], ckpt["dropout"],
            identifiability_feature_index=ckpt.get("identifiability_feature_index", -1),
            identifiability_pooling=ckpt.get("packet_identifiability_pooling", False),
            identifiability_feature_mode=ckpt.get("identifiability_feature_mode", "observed"),
            identifiability_prior_init=ckpt.get("identifiability_prior_init", 0.1),
            identifiability_dual_pooling=ckpt.get(
                "packet_identifiability_dual_pooling", False
            ),
            identifiability_evidence_adapter=ckpt.get(
                "packet_identifiability_evidence_adapter", False
            ),
            identifiability_adapter_max_delta=ckpt.get("identifiability_adapter_max_delta", 0.25),
            identifiability_residual_max_weight=ckpt.get(
                "identifiability_residual_max_weight", 0.0
            ),
            identifiability_residual_init=ckpt.get("identifiability_residual_init", 0.5),
            dual_channel_mode=ckpt.get("dual_channel_mode", "concat"),
            meta_feature_dim=ckpt.get("meta_feature_dim", 14),
            native_structural_dim=ckpt.get("native_structural_dim", 0),
            dual_channel_max_weight=ckpt.get("dual_channel_max_weight", 0.25),
            dual_channel_init=ckpt.get("dual_channel_init", 0.1),
            dual_channel_gate_mode=ckpt.get("dual_channel_gate_mode", "global"),
            channel_fusion_base_mode=ckpt.get("channel_fusion_base_mode", "legacy"),
            use_intervention_views=ckpt.get("use_intervention_views", False),
            intervention_max_residual_weight=ckpt.get(
                "intervention_max_residual_weight", 0.25
            ),
            intervention_view_base_mode=ckpt.get(
                "intervention_view_base_mode", "symmetric_mean"
            ),
            exact_shared_packet_encoder=ckpt.get("exact_shared_packet_encoder", False),
            shared_packet_hidden_dim=ckpt.get("shared_packet_hidden_dim", 128),
            packet_evidence_max_weight=ckpt.get("packet_evidence_max_weight", 0.0),
            train_ablate_input_channel=ckpt.get("train_ablate_input_channel", "none"),
            train_ablate_intervention_view=ckpt.get(
                "train_ablate_intervention_view", "none"
            ),
            train_fixed_channel_fusion=ckpt.get("train_fixed_channel_fusion", False),
        )
    else:
        model = FlowGraphTransformerClassifier(
            ckpt["input_dim"],
            ckpt["num_classes"],
            ckpt["hidden_dim"],
            ckpt["num_layers"],
            ckpt["num_heads"],
            edge_attr_dim=ckpt.get("edge_attr_dim", 4),
            dropout=ckpt["dropout"],
            identifiability_feature_index=ckpt.get("identifiability_feature_index", -1),
            identifiability_pooling=ckpt.get("packet_identifiability_pooling", False),
            identifiability_feature_mode=ckpt.get("identifiability_feature_mode", "observed"),
            identifiability_prior_init=ckpt.get("identifiability_prior_init", 0.1),
            identifiability_dual_pooling=ckpt.get(
                "packet_identifiability_dual_pooling", False
            ),
            identifiability_evidence_adapter=ckpt.get(
                "packet_identifiability_evidence_adapter", False
            ),
            identifiability_adapter_max_delta=ckpt.get("identifiability_adapter_max_delta", 0.25),
            identifiability_residual_max_weight=ckpt.get(
                "identifiability_residual_max_weight", 0.0
            ),
            identifiability_residual_init=ckpt.get("identifiability_residual_init", 0.5),
            dual_channel_mode=ckpt.get("dual_channel_mode", "concat"),
            meta_feature_dim=ckpt.get("meta_feature_dim", 14),
            native_structural_dim=ckpt.get("native_structural_dim", 0),
            channel_fusion_base_mode=ckpt.get(
                "channel_fusion_base_mode", "legacy"
            ),
            dual_channel_max_weight=ckpt.get("dual_channel_max_weight", 0.25),
            use_intervention_views=ckpt.get("use_intervention_views", False),
            intervention_max_residual_weight=ckpt.get(
                "intervention_max_residual_weight", 0.25
            ),
            intervention_view_base_mode=ckpt.get(
                "intervention_view_base_mode", "symmetric_mean"
            ),
            exact_shared_packet_encoder=ckpt.get(
                "exact_shared_packet_encoder", False
            ),
            shared_packet_hidden_dim=ckpt.get("shared_packet_hidden_dim", 128),
            train_ablate_input_channel=ckpt.get(
                "train_ablate_input_channel", "none"
            ),
            train_ablate_intervention_view=ckpt.get(
                "train_ablate_intervention_view", "none"
            ),
            train_fixed_channel_fusion=ckpt.get(
                "train_fixed_channel_fusion", False
            ),
        )
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    flow_head = None
    if "flow_head_state" in ckpt:
        hierarchical_mode = ckpt.get("hierarchical_mode", "logit")
        class_groups = ckpt.get("class_groups")
        if class_groups is None and hierarchical_mode == "expert":
            class_groups = parse_label_groups(ckpt.get("coarse_groups", "vpn_app"), ckpt["num_classes"])
        flow_head = FlowAggregationHead(
            ckpt["hidden_dim"],
            ckpt["num_classes"],
            pooling=ckpt.get("flow_pooling", "attention"),
            num_coarse_classes=ckpt.get("num_coarse_classes", 0),
            class_groups=class_groups if hierarchical_mode == "expert" else None,
            hierarchical_mode=hierarchical_mode,
            flow_transformer_layers=ckpt.get("flow_transformer_layers", 1),
            flow_transformer_heads=ckpt.get("flow_transformer_heads", 4),
            dropout=ckpt.get("dropout", 0.1),
            # New checkpoints persist the exact explicit-field contract. The
            # fallback preserves legacy checkpoints whose statistics head was
            # trained over the full trailing structural block.
            flow_stat_meta_dim=checkpoint_flow_stat_meta_dim(ckpt),
            flow_stat_expert_weight=ckpt.get("flow_stat_expert_weight", 0.0),
            flow_stat_aux_weight=ckpt.get("flow_stat_aux_weight", 0.0),
            identifiability_attention_prior=ckpt.get(
                "identifiability_attention_prior", False
            ),
            identifiability_prior_init=ckpt.get("identifiability_prior_init", 0.1),
        )
        flow_head.load_state_dict(ckpt["flow_head_state"])
        flow_head.to(device).eval()
    return model, ckpt, flow_head


def load_counterfactual_fusion(ckpt: dict, device: str):
    has_state = "counterfactual_fusion_state" in ckpt
    state = ckpt.get("counterfactual_fusion_state", {})
    mode = ckpt.get("counterfactual_fusion", "none")
    if not has_state or mode == "none":
        return None
    head = CounterfactualFlowFusion(
        ckpt["hidden_dim"],
        ckpt["num_classes"],
        mode=mode,
        base_mode=ckpt.get("counterfactual_base_mode", "mean"),
        max_residual_weight=ckpt.get("counterfactual_max_residual_weight", 0.25),
        initial_residual_fraction=ckpt.get(
            "counterfactual_initial_residual_fraction", 0.1
        ),
        dropout=ckpt.get("dropout", 0.1),
    )
    head.load_state_dict(state)
    return head.to(device).eval()


@torch.no_grad()
def ablate_seq_input_channel(
    x: torch.Tensor,
    model,
    channel: str,
) -> torch.Tensor:
    """Zero one packet channel for inference-only contribution diagnostics."""
    if channel == "none":
        return x
    if getattr(model, "dual_channel_mode", "concat") != "residual":
        raise ValueError("input-channel ablation requires a residual dual-channel checkpoint")
    embedding_dim = int(getattr(model, "embedding_feature_dim", 0))
    native_dim = int(getattr(model, "native_structural_dim", 0))
    meta_dim = int(getattr(model, "meta_feature_dim", 0))
    if embedding_dim + meta_dim != x.shape[-1]:
        raise ValueError(
            "checkpoint/input dimensions disagree for channel ablation: "
            f"embedding={embedding_dim} meta={meta_dim} input={x.shape[-1]}"
        )
    ranges = {
        "semantic": (0, embedding_dim),
        "content": (embedding_dim, embedding_dim + native_dim),
        "structural": (embedding_dim + native_dim, embedding_dim + meta_dim),
    }
    start, end = ranges[channel]
    if end <= start:
        raise ValueError(f"checkpoint has no {channel} channel to ablate")
    x = x.clone()
    x[..., start:end] = 0
    return x


def ablate_intervention_inputs(
    factual_x: torch.Tensor,
    intervened_x: torch.Tensor,
    mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Duplicate one aligned view without changing the trained fusion module."""
    if mode == "none":
        return factual_x, intervened_x
    if mode == "factual_only":
        return factual_x, factual_x
    if mode == "intervened_only":
        return intervened_x, intervened_x
    raise ValueError(f"unknown intervention-view ablation: {mode}")


@torch.no_grad()
def predict_seq(
    model,
    dataset_path: str,
    device: str,
    batch_size: int,
    intervened_dataset_path: str = "",
    ablate_input_channel: str = "none",
    ablate_intervention_view: str = "none",
):
    ds = SeqDataset(dataset_path)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_seq)
    intervened_dl = None
    if getattr(model, "use_intervention_views", False):
        if not intervened_dataset_path:
            raise ValueError("intervention-aware checkpoint requires a paired dataset")
        intervened_ds = SeqDataset(intervened_dataset_path)
        if len(intervened_ds) != len(ds):
            raise ValueError("factual/intervened test dataset lengths differ")
        intervened_dl = DataLoader(
            intervened_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_seq
        )
    y_true, y_pred, flow_ids, logits_all, emb_all, x_all, window_ranges = [], [], [], [], [], [], []
    gate_values = {
        "intervention_view_gate": [],
        "dual_channel_gate": [],
        "packet_evidence_gate": [],
    }
    batches = zip(dl, intervened_dl) if intervened_dl is not None else ((batch, None) for batch in dl)
    for batch, intervened_batch in batches:
        factual_x = ablate_seq_input_channel(
            batch["x"].to(device), model, ablate_input_channel
        )
        intervened_x = None
        if intervened_batch is not None:
            if (
                intervened_batch["flow_id"] != batch["flow_id"]
                or not torch.equal(intervened_batch["label"], batch["label"])
                or intervened_batch["x"].shape != batch["x"].shape
            ):
                raise ValueError("factual/intervened test windows are not aligned")
            intervened_x = ablate_seq_input_channel(
                intervened_batch["x"].to(device), model, ablate_input_channel
            )
            factual_x, intervened_x = ablate_intervention_inputs(
                factual_x, intervened_x, ablate_intervention_view
            )
        out = model(
            factual_x,
            batch["mask"].to(device),
            intervened_x=intervened_x,
        )
        valid_mask = batch["mask"].bool()
        for gate_name in gate_values:
            gate = out.get(gate_name)
            if gate is None:
                continue
            gate = gate.detach().cpu()
            if gate.ndim == 3 and gate.shape[:2] == valid_mask.shape:
                gate = gate[valid_mask]
            else:
                gate = gate.reshape(-1, gate.shape[-1])
            gate_values[gate_name].append(gate)
        logits = out["logits"].cpu()
        emb = out["embedding"].cpu()
        x_cpu = factual_x.cpu()
        labels = batch["label"]
        for i in range(logits.size(0)):
            if int(labels[i]) < 0:
                continue
            y_true.append(int(labels[i]))
            y_pred.append(int(logits[i].argmax()))
            flow_ids.append(batch["flow_id"][i])
            logits_all.append(logits[i].numpy())
            emb_all.append(emb[i].numpy())
            x_all.append(x_cpu[i][batch["mask"][i]].numpy())
            window_ranges.append(batch["window"][i])
    gate_diagnostics = {}
    for gate_name, chunks in gate_values.items():
        if not chunks:
            continue
        values = torch.cat(chunks, dim=0).float()
        gate_diagnostics[gate_name] = {
            "num_samples": int(values.shape[0]),
            "mean": values.mean(dim=0).tolist(),
            "std": values.std(dim=0, unbiased=False).tolist(),
            "p05": torch.quantile(values, 0.05, dim=0).tolist(),
            "p50": torch.quantile(values, 0.50, dim=0).tolist(),
            "p95": torch.quantile(values, 0.95, dim=0).tolist(),
        }
        if gate_name == "intervention_view_gate":
            fusion = getattr(model, "intervention_view_fusion", None)
            if fusion is None and hasattr(model, "shared_packet_encoder"):
                fusion = model.shared_packet_encoder.intervention_view_fusion
            if fusion is None:
                raise ValueError("intervention gate was emitted without its fusion module")
            max_residual = float(getattr(fusion, "max_residual_weight", 1.0))
            base_mode = getattr(fusion, "base_mode", "symmetric_mean")
            effective = fusion.effective_weights(values)
            named_bounds = fusion.effective_weight_bounds()
            view_names = ("factual", "intervened")
            lower = torch.tensor(
                [named_bounds[name][0] for name in view_names], dtype=effective.dtype
            )
            upper = torch.tensor(
                [named_bounds[name][1] for name in view_names], dtype=effective.dtype
            )
            gate_diagnostics[gate_name]["base_mode"] = base_mode
            gate_diagnostics[gate_name]["view_names"] = list(view_names)
            gate_diagnostics[gate_name]["weight_semantics"] = (
                "bounded_effective_routing_weights_before_channel_fusion"
            )
            gate_diagnostics[gate_name]["max_residual_weight"] = max_residual
            gate_diagnostics[gate_name]["effective_routing_mean"] = effective.mean(dim=0).tolist()
            gate_diagnostics[gate_name]["effective_routing_p05"] = torch.quantile(effective, 0.05, dim=0).tolist()
            gate_diagnostics[gate_name]["effective_routing_p50"] = torch.quantile(effective, 0.50, dim=0).tolist()
            gate_diagnostics[gate_name]["effective_routing_p95"] = torch.quantile(effective, 0.95, dim=0).tolist()
            gate_diagnostics[gate_name]["theoretical_bounds"] = {
                name: list(named_bounds[name]) for name in view_names
            }
            gate_diagnostics[gate_name]["bounds_satisfied"] = bool(
                torch.all(effective >= lower - 1e-6)
                and torch.all(effective <= upper + 1e-6)
            )
        elif gate_name == "packet_evidence_gate":
            max_weight = float(getattr(model, "packet_evidence_max_weight", 0.0))
            gate_diagnostics[gate_name]["weight_semantics"] = (
                "learned_convex_weight_of_reused_packet_classifier_evidence"
            )
            gate_diagnostics[gate_name]["max_weight"] = max_weight
            gate_diagnostics[gate_name]["bounds_satisfied"] = bool(
                torch.all(values >= -1e-6)
                and torch.all(values <= max_weight + 1e-6)
            )
        elif (
            gate_name == "dual_channel_gate"
            and getattr(model, "channel_fusion_base_mode", "legacy")
            == "semantic_anchor"
        ):
            fusion = model.shared_packet_fusion
            max_residual = float(getattr(model, "dual_channel_max_weight", 1.0))
            fixed_fusion = bool(getattr(model, "train_fixed_channel_fusion", False))
            if fixed_fusion:
                effective = values
                fixed_weight = 1.0 / len(fusion.channel_names)
                named_bounds = {
                    name: (fixed_weight, fixed_weight)
                    for name in fusion.channel_names
                }
            else:
                effective = fusion.effective_weights(values)
                named_bounds = fusion.effective_weight_bounds()
            lower = torch.tensor(
                [named_bounds[name][0] for name in fusion.channel_names],
                dtype=effective.dtype,
            )
            upper = torch.tensor(
                [named_bounds[name][1] for name in fusion.channel_names],
                dtype=effective.dtype,
            )
            gate_diagnostics[gate_name]["base_mode"] = "semantic_anchor"
            gate_diagnostics[gate_name]["channel_names"] = list(fusion.channel_names)
            gate_diagnostics[gate_name]["weight_semantics"] = (
                "fixed_equal_normalized_channel_mean"
                if fixed_fusion
                else "bounded_effective_routing_weights_excluding_interaction_correction"
            )
            gate_diagnostics[gate_name]["max_residual_weight"] = (
                0.0 if fixed_fusion else max_residual
            )
            gate_diagnostics[gate_name]["effective_routing_mean"] = effective.mean(dim=0).tolist()
            gate_diagnostics[gate_name]["effective_routing_p05"] = torch.quantile(effective, 0.05, dim=0).tolist()
            gate_diagnostics[gate_name]["effective_routing_p50"] = torch.quantile(effective, 0.50, dim=0).tolist()
            gate_diagnostics[gate_name]["effective_routing_p95"] = torch.quantile(effective, 0.95, dim=0).tolist()
            gate_diagnostics[gate_name]["theoretical_bounds"] = {
                name: list(named_bounds[name]) for name in fusion.channel_names
            }
            gate_diagnostics[gate_name]["bounds_satisfied"] = bool(
                torch.all(effective >= lower - 1e-6)
                and torch.all(effective <= upper + 1e-6)
            )
    return y_true, y_pred, flow_ids, logits_all, emb_all, x_all, window_ranges, gate_diagnostics


def summarize_shared_graph_gate(model, gate_name: str, values: torch.Tensor) -> dict:
    """Report raw router outputs and the effective bounded graph weights."""
    summary = {
        "num_samples": int(values.shape[0]),
        "mean": values.mean(dim=0).tolist(),
        "std": values.std(dim=0, unbiased=False).tolist(),
        "p05": torch.quantile(values, 0.05, dim=0).tolist(),
        "p50": torch.quantile(values, 0.50, dim=0).tolist(),
        "p95": torch.quantile(values, 0.95, dim=0).tolist(),
    }
    if gate_name == "intervention_view_gate":
        fusion = model.shared_packet_encoder.intervention_view_fusion
        names = ("factual", "intervened")
        bounds = fusion.effective_weight_bounds()
        effective = fusion.effective_weights(values)
        summary.update(
            {
                "base_mode": fusion.base_mode,
                "view_names": list(names),
                "weight_semantics": (
                    "bounded_effective_routing_weights_before_channel_fusion"
                ),
                "max_residual_weight": float(fusion.max_residual_weight),
            }
        )
    elif (
        gate_name == "dual_channel_gate"
        and model.channel_fusion_base_mode == "semantic_anchor"
    ):
        fusion = model.shared_packet_fusion
        names = fusion.channel_names
        fixed_fusion = bool(model.train_fixed_channel_fusion)
        if fixed_fusion:
            effective = values
            fixed_weight = 1.0 / len(names)
            bounds = {name: (fixed_weight, fixed_weight) for name in names}
        else:
            effective = fusion.effective_weights(values)
            bounds = fusion.effective_weight_bounds()
        summary.update(
            {
                "base_mode": "semantic_anchor",
                "channel_names": list(names),
                "weight_semantics": (
                    "fixed_equal_normalized_channel_mean"
                    if fixed_fusion
                    else "bounded_effective_routing_weights_excluding_interaction_correction"
                ),
                "max_residual_weight": (
                    0.0 if fixed_fusion else model.dual_channel_max_weight
                ),
            }
        )
    else:
        return summary
    lower = torch.tensor([bounds[name][0] for name in names], dtype=effective.dtype)
    upper = torch.tensor([bounds[name][1] for name in names], dtype=effective.dtype)
    summary.update(
        {
            "effective_routing_mean": effective.mean(dim=0).tolist(),
            "effective_routing_p05": torch.quantile(effective, 0.05, dim=0).tolist(),
            "effective_routing_p50": torch.quantile(effective, 0.50, dim=0).tolist(),
            "effective_routing_p95": torch.quantile(effective, 0.95, dim=0).tolist(),
            "theoretical_bounds": {name: list(bounds[name]) for name in names},
            "bounds_satisfied": bool(
                torch.all(effective >= lower - 1e-6)
                and torch.all(effective <= upper + 1e-6)
            ),
        }
    )
    return summary


@torch.no_grad()
def predict_graph(
    model,
    dataset_path: str,
    device: str,
    intervened_dataset_path: str = "",
    ablate_input_channel: str = "none",
    ablate_intervention_view: str = "none",
):
    ds = GraphDataset(dataset_path)
    factual_groups = build_flow_groups(ds)
    paired_groups = {}
    if getattr(model, "use_intervention_views", False):
        if not intervened_dataset_path:
            raise ValueError("intervention-aware graph checkpoint requires a paired dataset")
        paired_groups = {
            str(group["flow_id"]): group
            for group in build_flow_groups(GraphDataset(intervened_dataset_path))
        }
    y_true, y_pred, flow_ids, logits_all, emb_all, x_all, window_ranges = [], [], [], [], [], [], []
    gate_values = {"intervention_view_gate": [], "dual_channel_gate": []}
    for group in factual_groups:
        if getattr(model, "use_intervention_views", False):
            item_pairs = aligned_graph_items(
                group, paired_groups.get(str(group["flow_id"]))
            )
        else:
            item_pairs = [(item, None) for item in group["items"]]
        for item, paired_item in item_pairs:
            label = int(item.get("label", -1))
            if label < 0:
                continue
            factual_x = ablate_seq_input_channel(
                item["x"].to(device), model, ablate_input_channel
            )
            intervened_x = None
            if paired_item is not None:
                intervened_x = ablate_seq_input_channel(
                    paired_item["x"].to(device), model, ablate_input_channel
                )
                factual_x, intervened_x = ablate_intervention_inputs(
                    factual_x, intervened_x, ablate_intervention_view
                )
            out = model(
                factual_x,
                item["edge_index"].to(device),
                item["edge_attr"].to(device),
                intervened_x=intervened_x,
            )
            for gate_name in gate_values:
                gate = out.get(gate_name)
                if gate is not None:
                    gate_values[gate_name].append(
                        gate.detach().cpu().reshape(-1, gate.shape[-1])
                    )
            logits = out["logits"].squeeze(0).cpu().numpy()
            y_true.append(label)
            y_pred.append(int(logits.argmax()))
            flow_ids.append(str(item.get("flow_id", "")))
            logits_all.append(logits)
            emb_all.append(out["embedding"].cpu().numpy())
            x_all.append(factual_x.cpu().numpy())
            window_ranges.append(item.get("window"))
    gate_diagnostics = {}
    for gate_name, chunks in gate_values.items():
        if not chunks:
            continue
        values = torch.cat(chunks, dim=0).float()
        gate_diagnostics[gate_name] = summarize_shared_graph_gate(
            model, gate_name, values
        )
    return y_true, y_pred, flow_ids, logits_all, emb_all, x_all, window_ranges, gate_diagnostics


def aggregate_by_flow(
    y_true,
    flow_ids,
    logits_all,
    emb_all=None,
    x_all=None,
    window_ranges=None,
    flow_head=None,
    device: str = "cpu",
    class_to_coarse=None,
    hierarchical_logit_weight: float = 0.0,
    hierarchical_mode: str = "logit",
    flow_eval_pooling: str = "checkpoint",
    flow_eval_topk: int = 3,
    return_embeddings: bool = False,
):
    buckets = defaultdict(list)
    emb_buckets = defaultdict(list)
    x_buckets = defaultdict(list)
    range_buckets = defaultdict(list)
    labels = {}
    for i, (y, fid, logit) in enumerate(zip(y_true, flow_ids, logits_all)):
        buckets[fid].append(logit)
        if emb_all is not None:
            emb_buckets[fid].append(emb_all[i])
        if x_all is not None:
            x_buckets[fid].append(x_all[i])
        if window_ranges is not None:
            range_buckets[fid].append(window_ranges[i])
        labels[fid] = y
    flow_true, flow_pred, out_flow_ids, flow_logits_all = [], [], [], []
    multi_view_gates = []
    stat_gates = []
    flow_embeddings = []
    for fid, arrs in buckets.items():
        if flow_head is None or flow_eval_pooling != "checkpoint":
            pooling = "mean_logits" if flow_eval_pooling == "checkpoint" else flow_eval_pooling
            logits = pool_window_logits(arrs, pooling, flow_eval_topk)
            flow_embedding = (
                np.stack(emb_buckets[fid], axis=0).mean(axis=0)
                if emb_buckets.get(fid)
                else None
            )
        else:
            emb = torch.tensor(np.stack(emb_buckets[fid], axis=0), dtype=torch.float32, device=device)
            win_logits = torch.tensor(np.stack(arrs, axis=0), dtype=torch.float32, device=device)
            win_x = [torch.tensor(x, dtype=torch.float32, device=device) for x in x_buckets.get(fid, [])]
            window_reliability = None
            if flow_head.identifiability_attention_prior and win_x:
                reliability_index = -(flow_head.flow_stat_meta_dim + 2)
                reliability_values = []
                for x in win_x:
                    valid = x.abs().sum(dim=-1) > 0
                    packet_values = x[valid, reliability_index]
                    reliability_values.append(
                        packet_values.mean() if packet_values.numel() else x.new_tensor(1.0)
                    )
                window_reliability = torch.stack(reliability_values)
            pooled = flow_head(
                emb,
                window_logits=win_logits,
                window_x=win_x,
                window_ranges=(
                    range_buckets[fid]
                    if range_buckets.get(fid)
                    and all(item is not None for item in range_buckets[fid])
                    else None
                ),
                window_reliability=window_reliability,
            )
            flow_embedding = pooled["embedding"].detach().cpu().numpy()
            if pooled.get("multi_view_gate") is not None:
                multi_view_gates.append(pooled["multi_view_gate"].detach().cpu().numpy())
            if pooled.get("stat_gate") is not None:
                stat_gates.append(float(pooled["stat_gate"].detach().cpu()))
            logits = pooled["logits"]
            if hierarchical_mode != "expert":
                logits = apply_hierarchical_logits(
                    logits,
                    pooled.get("coarse_logits"),
                    class_to_coarse,
                    hierarchical_logit_weight,
                )
            logits = logits.detach().cpu().numpy()
        flow_true.append(labels[fid])
        flow_pred.append(int(logits.argmax()))
        out_flow_ids.append(fid)
        flow_logits_all.append(np.asarray(logits, dtype=np.float32))
        flow_embeddings.append(
            None if flow_embedding is None else np.asarray(flow_embedding, dtype=np.float32)
        )
    gate_summary = None
    if multi_view_gates:
        gate_arr = np.stack(multi_view_gates, axis=0)
        gate_safe = np.clip(gate_arr, 1e-12, 1.0)
        entropy = -(gate_safe * np.log(gate_safe)).sum(axis=1)
        num_branches = gate_arr.shape[1]
        gate_summary = {
            "branches": ["mean", "max", "std", "attention"],
            "mean": gate_arr.mean(axis=0).astype(float).tolist(),
            "std": gate_arr.std(axis=0).astype(float).tolist(),
            "entropy_mean": float(entropy.mean()),
            "normalized_entropy_mean": float(entropy.mean() / max(np.log(num_branches), 1e-12)),
            "effective_branches_mean": float(np.exp(entropy).mean()),
            "num_flows": int(gate_arr.shape[0]),
        }
    stat_gate_summary = None
    if stat_gates:
        stat_arr = np.asarray(stat_gates, dtype=np.float64)
        stat_gate_summary = {
            "mean": float(stat_arr.mean()),
            "std": float(stat_arr.std()),
            "min": float(stat_arr.min()),
            "max": float(stat_arr.max()),
            "num_flows": int(stat_arr.size),
        }
    result = (
        flow_true,
        flow_pred,
        out_flow_ids,
        flow_logits_all,
        gate_summary,
        stat_gate_summary,
    )
    return result + (flow_embeddings,) if return_embeddings else result


def fuse_counterfactual_flows(
    fusion_head,
    clean_records,
    paired_records,
    device: str,
):
    clean_true, _, clean_ids, clean_logits, _, _, clean_embeddings = clean_records
    paired_true, _, paired_ids, paired_logits, _, _, paired_embeddings = paired_records
    paired_index = {flow_id: idx for idx, flow_id in enumerate(paired_ids)}
    missing = [flow_id for flow_id in clean_ids if flow_id not in paired_index]
    if missing:
        raise ValueError(
            f"paired test dataset is missing {len(missing)} clean flow_ids; "
            f"examples={missing[:3]}"
        )
    ordered = [paired_index[flow_id] for flow_id in clean_ids]
    aligned_true = [paired_true[idx] for idx in ordered]
    if list(clean_true) != aligned_true:
        raise ValueError("clean and paired test labels disagree after flow_id alignment")
    if any(embedding is None for embedding in clean_embeddings) or any(
        paired_embeddings[idx] is None for idx in ordered
    ):
        raise ValueError("counterfactual evaluation requires flow embeddings")

    clean_embedding_t = torch.tensor(
        np.stack(clean_embeddings), dtype=torch.float32, device=device
    )
    paired_embedding_t = torch.tensor(
        np.stack([paired_embeddings[idx] for idx in ordered]),
        dtype=torch.float32,
        device=device,
    )
    clean_logits_t = torch.tensor(
        np.stack(clean_logits), dtype=torch.float32, device=device
    )
    paired_logits_t = torch.tensor(
        np.stack([paired_logits[idx] for idx in ordered]),
        dtype=torch.float32,
        device=device,
    )
    with torch.no_grad():
        output = fusion_head(
            clean_embedding_t,
            paired_embedding_t,
            clean_logits_t,
            paired_logits_t,
        )
    logits = output["logits"].detach().cpu().numpy()
    gates = output["residual_gate"].detach().cpu().numpy().reshape(-1)
    diagnostics = {
        "mode": fusion_head.mode,
        "paired_coverage": 1.0,
        "gate_mean": float(gates.mean()) if gates.size else 0.0,
        "gate_std": float(gates.std()) if gates.size else 0.0,
        "gate_min": float(gates.min()) if gates.size else 0.0,
        "gate_max": float(gates.max()) if gates.size else 0.0,
    }
    return logits, diagnostics


def softmax_np(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=-1, keepdims=True).clip(min=1e-12)


def pool_window_logits(arrs, mode: str, topk: int) -> np.ndarray:
    logits = np.stack(arrs, axis=0)
    if mode == "mean_logits":
        return logits.mean(axis=0)

    probs = softmax_np(logits)
    if mode == "mean_probs":
        return probs.mean(axis=0)

    conf = probs.max(axis=1)
    if mode == "max_conf":
        return logits[int(conf.argmax())]
    if mode == "topk_logits":
        k = min(max(1, int(topk)), logits.shape[0])
        idx = np.argsort(conf)[-k:]
        return logits[idx].mean(axis=0)
    if mode == "vote":
        votes = logits.argmax(axis=1)
        counts = np.bincount(votes, minlength=logits.shape[1]).astype(np.float32)
        return counts + 1e-6 * probs.mean(axis=0)
    raise ValueError(f"Unknown --flow_eval_pooling: {mode}")


def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred) if y_true else 0.0
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    p_weight, r_weight, f_weight, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    return {
        "accuracy": acc,
        "macro_precision": p_macro,
        "macro_recall": r_macro,
        "macro_f1": f_macro,
        "weighted_precision": p_weight,
        "weighted_recall": r_weight,
        "weighted_f1": f_weight,
    }


def load_label_names(path: str):
    if not path:
        return None, None
    with open(path, "r", encoding="utf-8") as f:
        label_map = json.load(f)
    if not label_map:
        return None, label_map
    max_id = max(int(v) for v in label_map.values())
    label_names = [str(i) for i in range(max_id + 1)]
    for name, idx in label_map.items():
        label_names[int(idx)] = name
    return label_names, label_map


def print_report(title: str, y_true, y_pred, label_names):
    print(f"\n{title}:")
    if label_names:
        labels = list(range(len(label_names)))
        print(classification_report(y_true, y_pred, labels=labels, target_names=label_names, zero_division=0))
    else:
        print(classification_report(y_true, y_pred, zero_division=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--paired_view_dataset", default="", help="Header-intervened counterpart required by counterfactual checkpoints.")
    ap.add_argument("--label_map", default="", help="Optional JSON mapping label name to label id, used for readable reports.")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--output_json", default="")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no_report", action="store_true", help="Only print the metrics JSON, without classification reports.")
    ap.add_argument(
        "--ablate_input_channel",
        choices=["none", "semantic", "content", "structural"],
        default="none",
        help="Inference-only diagnostic: zero one aligned packet channel in both factual and intervened inputs.",
    )
    ap.add_argument(
        "--ablate_intervention_view",
        choices=["none", "factual_only", "intervened_only"],
        default="none",
        help=(
            "Inference-only diagnostic for shared intervention checkpoints: replace "
            "both semantic views with the factual or intervened view while preserving "
            "the trained architecture."
        ),
    )
    ap.add_argument(
        "--flow_eval_pooling",
        default="checkpoint",
        choices=["checkpoint", "mean_logits", "mean_probs", "max_conf", "topk_logits", "vote"],
        help="Flow-level aggregation at eval time. 'checkpoint' uses the trained flow head when available.",
    )
    ap.add_argument("--flow_eval_topk", type=int, default=3, help="Top-k windows for --flow_eval_pooling topk_logits.")
    args = ap.parse_args()
    model, ckpt, flow_head = load_model(args.checkpoint, args.device)
    fusion_head = load_counterfactual_fusion(ckpt, args.device)
    if fusion_head is not None and not args.paired_view_dataset:
        ap.error("this checkpoint requires --paired_view_dataset for counterfactual evaluation")
    if ckpt.get("use_intervention_views", False) and not args.paired_view_dataset:
        ap.error("this checkpoint requires --paired_view_dataset for shared intervention views")
    if args.ablate_intervention_view != "none" and not ckpt.get(
        "use_intervention_views", False
    ):
        ap.error("--ablate_intervention_view requires a shared intervention-view checkpoint")
    label_names, label_map = load_label_names(args.label_map)
    class_to_coarse = None
    if ckpt.get("num_coarse_classes", 0) > 0:
        class_to_coarse, _ = build_class_to_coarse(ckpt.get("coarse_groups", "vpn_app"), ckpt["num_classes"], args.device)
    if ckpt["model_type"] == "seq":
        y_true, y_pred, flow_ids, logits_all, emb_all, x_all, window_ranges, gate_diagnostics = predict_seq(
            model,
            args.dataset,
            args.device,
            args.batch_size,
            intervened_dataset_path=(
                args.paired_view_dataset if ckpt.get("use_intervention_views", False) else ""
            ),
            ablate_input_channel=args.ablate_input_channel,
            ablate_intervention_view=args.ablate_intervention_view,
        )
    else:
        y_true, y_pred, flow_ids, logits_all, emb_all, x_all, window_ranges, gate_diagnostics = predict_graph(
            model,
            args.dataset,
            args.device,
            intervened_dataset_path=(
                args.paired_view_dataset
                if ckpt.get("use_intervention_views", False)
                else ""
            ),
            ablate_input_channel=args.ablate_input_channel,
            ablate_intervention_view=args.ablate_intervention_view,
        )

    window_prob = softmax_np(np.stack(logits_all, axis=0)) if logits_all else np.zeros((0, ckpt["num_classes"]))
    window_metrics = compute_metrics(y_true, y_pred)
    window_metrics["calibration"] = calibration_metrics(y_true, window_prob)
    clean_records = aggregate_by_flow(
        y_true,
        flow_ids,
        logits_all,
        emb_all=emb_all,
        x_all=x_all,
        window_ranges=window_ranges,
        flow_head=flow_head,
        device=args.device,
        class_to_coarse=class_to_coarse,
        hierarchical_logit_weight=ckpt.get("hierarchical_logit_weight", 0.0),
        hierarchical_mode=ckpt.get("hierarchical_mode", "logit"),
        flow_eval_pooling=args.flow_eval_pooling,
        flow_eval_topk=args.flow_eval_topk,
        return_embeddings=fusion_head is not None,
    )
    if fusion_head is None:
        flow_true, flow_pred, out_flow_ids, flow_logits_all, multi_view_gate_summary, stat_gate_summary = clean_records
        counterfactual_summary = None
    else:
        if ckpt["model_type"] == "seq":
            paired_outputs = predict_seq(
                model, args.paired_view_dataset, args.device, args.batch_size
            )
        else:
            paired_outputs = predict_graph(model, args.paired_view_dataset, args.device)
        paired_records = aggregate_by_flow(
            paired_outputs[0],
            paired_outputs[2],
            paired_outputs[3],
            emb_all=paired_outputs[4],
            x_all=paired_outputs[5],
            window_ranges=paired_outputs[6],
            flow_head=flow_head,
            device=args.device,
            class_to_coarse=class_to_coarse,
            hierarchical_logit_weight=ckpt.get("hierarchical_logit_weight", 0.0),
            hierarchical_mode=ckpt.get("hierarchical_mode", "logit"),
            flow_eval_pooling=args.flow_eval_pooling,
            flow_eval_topk=args.flow_eval_topk,
            return_embeddings=True,
        )
        flow_true, _, out_flow_ids, _, multi_view_gate_summary, stat_gate_summary, _ = clean_records
        fused_logits, counterfactual_summary = fuse_counterfactual_flows(
            fusion_head, clean_records, paired_records, args.device
        )
        flow_logits_all = list(fused_logits)
        flow_pred = fused_logits.argmax(axis=-1).astype(int).tolist()
    flow_prob = softmax_np(np.stack(flow_logits_all, axis=0)) if flow_logits_all else np.zeros((0, ckpt["num_classes"]))
    flow_metrics = compute_metrics(flow_true, flow_pred)
    flow_metrics["calibration"] = calibration_metrics(flow_true, flow_prob)
    metrics = {
        "window_level": window_metrics,
        "flow_level": flow_metrics,
        "eval_config": {
            "flow_eval_pooling": args.flow_eval_pooling,
            "flow_eval_topk": args.flow_eval_topk,
            "ablate_input_channel": args.ablate_input_channel,
            "ablate_intervention_view": args.ablate_intervention_view,
            "trained_ablate_input_channel": ckpt.get(
                "train_ablate_input_channel", "none"
            ),
            "trained_ablate_intervention_view": ckpt.get(
                "train_ablate_intervention_view", "none"
            ),
            "trained_fixed_channel_fusion": ckpt.get(
                "train_fixed_channel_fusion", False
            ),
            "mechanism_sensitivity_scope": (
                "retrained_ablation"
                if ckpt.get("train_ablate_input_channel", "none") != "none"
                or ckpt.get("train_ablate_intervention_view", "none") != "none"
                or ckpt.get("train_fixed_channel_fusion", False)
                else "inference_only_not_retrained_ablation"
            ),
            "multi_view_gate": multi_view_gate_summary,
            "flow_stat_gate": stat_gate_summary,
            "counterfactual_fusion": counterfactual_summary,
            "learned_gate_diagnostics": gate_diagnostics,
        },
    }
    print(json.dumps(metrics, indent=2))
    if not args.no_report:
        print_report("Window-level report", y_true, y_pred, label_names)
        print_report("Flow-level report", flow_true, flow_pred, label_names)
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "metrics": metrics,
                    "label_map": label_map,
                    "window_y_true": y_true,
                    "window_y_pred": y_pred,
                    "window_prob": window_prob.tolist() if logits_all else [],
                    "window_flow_ids": flow_ids,
                    "flow_y_true": flow_true,
                    "flow_y_pred": flow_pred,
                    "flow_prob": flow_prob.tolist() if flow_logits_all else [],
                    "flow_ids": out_flow_ids,
                    "provenance": prediction_provenance(
                        args.checkpoint,
                        args.dataset,
                        args.paired_view_dataset,
                        ckpt,
                    ),
                },
                f,
                indent=2,
            )


if __name__ == "__main__":
    main()
