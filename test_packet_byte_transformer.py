#!/usr/bin/env python3
"""Evaluate a saved strict-one-packet byte Transformer checkpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.packet_byte_transformer import PacketByteTransformer
from packet_eval_utils import packet_classification_metrics
from train_packet_byte_transformer import PacketByteDataset, predict_packet_views, sha256_file
from train_packet_byte_transformer import packet_content_group_metrics
from train_tower1_multitask import load_label_names


def load_packet_uids(packet_index: str | Path) -> np.ndarray:
    packet_uids: list[str] = []
    seen: set[str] = set()
    with Path(packet_index).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            packet_uid = str(row.get("packet_uid") or "")
            if not packet_uid:
                raise ValueError(f"{packet_index}:{line_number}: missing packet_uid")
            if packet_uid in seen:
                raise ValueError(f"{packet_index}:{line_number}: duplicate packet_uid")
            seen.add(packet_uid)
            packet_uids.append(packet_uid)
    if not packet_uids:
        raise ValueError(f"{packet_index}: packet index is empty")
    return np.asarray(packet_uids, dtype=np.str_)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--packet_index", required=True)
    ap.add_argument("--label_map", required=True)
    ap.add_argument("--output_json", required=True)
    ap.add_argument("--output_npz", default="")
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--semantic_embedding_cache", default="")
    ap.add_argument("--semantic_embedding_manifest", default="")
    ap.add_argument("--required_semantic_header_policy", default="")
    ap.add_argument("--required_semantic_packet_context_policy", default="")
    ap.add_argument("--intervened_semantic_embedding_cache", default="")
    ap.add_argument("--intervened_semantic_embedding_manifest", default="")
    ap.add_argument("--required_intervened_semantic_header_policy", default="")
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument(
        "--ablate_input_channel",
        choices=["none", "semantic", "content", "structural"],
        default="none",
        help="Inference-only sensitivity diagnostic; this is not a retrained ablation.",
    )
    ap.add_argument(
        "--ablate_intervention_view",
        choices=["none", "factual_only", "intervened_only"],
        default="none",
        help="Inference-only view sensitivity diagnostic; this is not a retrained ablation.",
    )
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint["config"]
    inference_config = checkpoint.get("inference_config", {"raw_weight": 1.0})
    raw_weight = float(inference_config.get("raw_weight", 1.0))
    router_enabled = bool(inference_config.get("router_enabled", False))
    evaluate_invariant_view = (
        inference_config.get("selection_scope") == "validation_only"
        or raw_weight < 1.0
        or router_enabled
    )
    model = PacketByteTransformer(**config).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    if args.ablate_input_channel != "none" and not model.exact_shared_representation:
        ap.error("--ablate_input_channel requires an exact shared-representation checkpoint")
    if args.ablate_intervention_view != "none" and not model.use_intervention_views:
        ap.error("--ablate_intervention_view requires an intervention-view checkpoint")
    dataset = PacketByteDataset(
        args.packet_index,
        int(config["max_bytes"]),
        include_augmented=evaluate_invariant_view,
        max_payload_bytes=(
            int(config.get("max_payload_bytes", 0))
            if bool(config.get("use_payload_channel", False)) else 0
        ),
        semantic_embedding_cache=args.semantic_embedding_cache,
        semantic_embedding_manifest=args.semantic_embedding_manifest,
        required_header_policy=args.required_semantic_header_policy,
        required_packet_context_policy=args.required_semantic_packet_context_policy,
        intervened_semantic_embedding_cache=args.intervened_semantic_embedding_cache,
        intervened_semantic_embedding_manifest=args.intervened_semantic_embedding_manifest,
        required_intervened_header_policy=args.required_intervened_semantic_header_policy,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    label_names = load_label_names(args.label_map)
    prediction_outputs = predict_packet_views(
        model,
        loader,
        device,
        include_masked=evaluate_invariant_view,
        include_identifiability=router_enabled,
        return_gate_diagnostics=True,
        return_gate_values=bool(args.output_npz),
        ablate_channel=args.ablate_input_channel,
        ablate_intervention_view=args.ablate_intervention_view,
    )
    if args.output_npz:
        (
            y_true,
            raw_probabilities,
            masked_probabilities,
            routed_reliability,
            content_group_ids,
            gate_diagnostics,
            gate_values,
        ) = prediction_outputs
    else:
        (
            y_true,
            raw_probabilities,
            masked_probabilities,
            routed_reliability,
            content_group_ids,
            gate_diagnostics,
        ) = prediction_outputs
        gate_values = {}
    if router_enabled:
        assert masked_probabilities is not None and routed_reliability is not None
        invariant_scale = float(inference_config.get("invariant_scale", 0.0))
        masked_weight = invariant_scale * routed_reliability.reshape(-1, 1)
        probabilities = (
            (1.0 - masked_weight) * raw_probabilities
            + masked_weight * masked_probabilities
        )
    else:
        probabilities = (
            raw_probabilities
            if masked_probabilities is None
            else raw_weight * raw_probabilities + (1.0 - raw_weight) * masked_probabilities
        )
    metrics = packet_classification_metrics(
        y_true, probabilities.argmax(axis=1), len(label_names), label_names
    )
    metrics.update(
        packet_content_group_metrics(
            y_true,
            probabilities.argmax(axis=1),
            content_group_ids,
            len(label_names),
            label_names,
        )
    )
    packet_uids = load_packet_uids(args.packet_index)
    if len(packet_uids) != len(y_true):
        raise ValueError(
            f"packet index/prediction length mismatch: {len(packet_uids)} != {len(y_true)}"
        )
    artifact_hash_cache: dict[str, str] = {}
    payload = {
        "task": "packet-level-classification",
        "sample_unit": "one_packet",
        "architecture": (
            (
                "shared-protocol-aware-semantic-content-structural-trichannel-single-head"
                if int(config.get("semantic_dim", 0)) > 0
                else "shared-protocol-aware-content-structural-gated"
            )
            if bool(config.get("use_protocol_fields", False)) else (
                "dual-channel-byte-payload-transformer-meta-gated"
                if bool(config.get("use_payload_channel", False))
                else "local-byte-transformer-meta-gated"
            )
        ),
        "checkpoint": args.checkpoint,
        "packet_index": args.packet_index,
        "provenance": {
            "checkpoint": {
                "path": str(Path(args.checkpoint).expanduser().resolve()),
                "sha256": sha256_file(args.checkpoint, artifact_hash_cache),
            },
            "packet_index": {
                "path": str(Path(args.packet_index).expanduser().resolve()),
                "sha256": sha256_file(args.packet_index, artifact_hash_cache),
            },
            "label_map": {
                "path": str(Path(args.label_map).expanduser().resolve()),
                "sha256": sha256_file(args.label_map, artifact_hash_cache),
            },
            "checkpoint_training_input_evidence": checkpoint.get(
                "training_input_evidence"
            ),
            "prediction_npz": None,
        },
        "inference_config": inference_config,
        "mechanism_sensitivity_config": {
            "scope": (
                "retrained_ablation"
                if config.get("train_ablate_input_channel", "none") != "none"
                or config.get("train_ablate_intervention_view", "none") != "none"
                or config.get("train_fixed_channel_fusion", False)
                else "inference_only_not_retrained_ablation"
            ),
            "ablate_input_channel": args.ablate_input_channel,
            "ablate_intervention_view": args.ablate_intervention_view,
            "trained_ablate_input_channel": config.get(
                "train_ablate_input_channel", "none"
            ),
            "trained_ablate_intervention_view": config.get(
                "train_ablate_intervention_view", "none"
            ),
            "trained_fixed_channel_fusion": config.get(
                "train_fixed_channel_fusion", False
            ),
        },
        "learned_gate_diagnostics": gate_diagnostics,
        "metrics": metrics,
        "view_metrics": {
            "raw": packet_classification_metrics(
                y_true,
                raw_probabilities.argmax(axis=1),
                len(label_names),
                label_names,
            )
        },
    }
    payload["view_metrics"]["raw"].update(
        packet_content_group_metrics(
            y_true,
            raw_probabilities.argmax(axis=1),
            content_group_ids,
            len(label_names),
            label_names,
        )
    )
    if masked_probabilities is not None:
        payload["view_metrics"]["session_invariant"] = packet_classification_metrics(
            y_true,
            masked_probabilities.argmax(axis=1),
            len(label_names),
            label_names,
        )
        payload["view_metrics"]["session_invariant"].update(
            packet_content_group_metrics(
                y_true,
                masked_probabilities.argmax(axis=1),
                content_group_ids,
                len(label_names),
                label_names,
            )
        )
    if args.output_npz:
        output_npz = Path(args.output_npz)
        output_npz.parent.mkdir(parents=True, exist_ok=True)
        prediction_arrays = {
            "y_true": y_true,
            "probabilities": probabilities,
            "content_group_ids": content_group_ids,
            "flow_ids": dataset.flow_ids.cpu().numpy(),
            "packet_uids": packet_uids,
        }
        prediction_arrays.update(
            {
                f"effective_{name}": values
                for name, values in gate_values.items()
            }
        )
        np.savez_compressed(
            output_npz,
            **prediction_arrays,
        )
        payload["provenance"]["prediction_npz"] = {
            "path": str(output_npz.expanduser().resolve()),
            "sha256": sha256_file(output_npz, artifact_hash_cache),
            "num_packets": int(len(packet_uids)),
            "packet_uid_alignment": "exact_packet_index_row_order",
            "effective_gate_arrays": sorted(gate_values),
        }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"accuracy={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f}")
    print(f"saved {output_json}" + (f" and {args.output_npz}" if args.output_npz else ""))


if __name__ == "__main__":
    main()
