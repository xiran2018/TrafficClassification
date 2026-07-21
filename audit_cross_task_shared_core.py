#!/usr/bin/env python3
"""Verify one packet/flow fold implements the exact shared-core contract."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch

from audit_shared_packet_core import audit_shared_core, load_group_reduction
from models.native_flow_encoder import NATIVE_PACKET_PRETRAINING_PROTOCOL
from unified_framework_spec import (
    tower1_hyperparameter_differences,
    tower1_shared_protocol_signature,
)


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def framework_notes(manifest: dict[str, Any]) -> dict[str, Any]:
    framework = manifest.get("framework") or {}
    notes = framework.get("notes") or {}
    if not isinstance(notes, dict):
        raise ValueError("framework notes must be a JSON object")
    return notes


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def native_contract(checkpoint: dict[str, Any]) -> dict[str, Any]:
    config = checkpoint.get("model_config") or {}
    return {
        "protocol": checkpoint.get("pretraining_protocol"),
        "max_bytes": int(config.get("max_bytes", -1)),
        "hidden_dim": int(config.get("hidden_dim", -1)),
        "byte_layers": int(config.get("byte_layers", -1)),
        "num_heads": int(config.get("num_heads", -1)),
        "dropout": float(config.get("dropout", -1.0)),
        "num_field_types": int(config.get("num_field_types", 9)),
    }


NATIVE_TRAINING_CONTRACT_KEYS = (
    "max_packets",
    "epochs",
    "batch_size",
    "eval_batch_size",
    "num_workers",
    "learning_rate",
    "weight_decay",
    "field_mask_probability",
    "payload_dropout_probability",
    "session_mask_probability",
    "masked_byte_weight",
    "relative_order_weight",
    "same_flow_weight",
    "next_length_weight",
    "next_iat_weight",
    "direction_weight",
    "packet_consistency_weight",
    "flow_contrastive_weight",
    "temperature",
    "patience",
    "seed",
)


def native_training_contract(checkpoint: dict[str, Any]) -> dict[str, Any]:
    config = checkpoint.get("pretraining_config") or {}
    return {key: config.get(key) for key in NATIVE_TRAINING_CONTRACT_KEYS}


def native_training_protocol_signature(
    checkpoint: dict[str, Any],
) -> dict[str, Any]:
    """Describe the native objective graph without optimizer-scale numerics."""
    config = checkpoint.get("pretraining_config") or {}
    required = (
        "field_mask_probability",
        "payload_dropout_probability",
        "session_mask_probability",
        "masked_byte_weight",
        "relative_order_weight",
        "same_flow_weight",
        "next_length_weight",
        "next_iat_weight",
        "direction_weight",
        "packet_consistency_weight",
        "flow_contrastive_weight",
    )
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(
            f"native pretraining config is missing protocol fields: {missing}"
        )
    return {
        "protocol": checkpoint.get("pretraining_protocol"),
        "interventions": {
            "field_mask": float(config["field_mask_probability"]) > 0.0,
            "payload_dropout": float(config["payload_dropout_probability"]) > 0.0,
            "session_mask": float(config["session_mask_probability"]) > 0.0,
        },
        "objectives": {
            key.removesuffix("_weight"): float(config[key]) > 0.0
            for key in required
            if key.endswith("_weight")
        },
    }


def native_training_hyperparameter_differences(
    packet_contract: dict[str, Any], flow_contract: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    return {
        key: {"packet": packet_contract.get(key), "flow": flow_contract.get(key)}
        for key in sorted(set(packet_contract) | set(flow_contract))
        if packet_contract.get(key) != flow_contract.get(key)
    }


def _finite_vector(value: Any, width: int) -> bool:
    return (
        isinstance(value, list)
        and len(value) == width
        and all(isinstance(item, (int, float)) and math.isfinite(float(item)) for item in value)
    )


def _audit_gate_summary(
    summary: Any,
    *,
    expected_names: tuple[str, ...],
    names_key: str,
) -> dict[str, Any]:
    if not isinstance(summary, dict):
        return {"status": "not_ready", "reason": "missing_gate_summary"}
    names = summary.get(names_key)
    mean = summary.get("effective_routing_mean")
    bounds = summary.get("theoretical_bounds")
    width = len(expected_names)
    names_match = names == list(expected_names)
    finite_mean = _finite_vector(mean, width)
    finite_raw_mean = _finite_vector(summary.get("mean"), width)
    finite_std = _finite_vector(summary.get("std"), width)
    normalized = bool(finite_mean and abs(sum(float(item) for item in mean) - 1.0) <= 1e-5)
    bounds_complete = isinstance(bounds, dict) and all(
        name in bounds and _finite_vector(bounds[name], 2) for name in expected_names
    )
    mean_inside_bounds = bool(
        finite_mean
        and bounds_complete
        and all(
            float(bounds[name][0]) - 1e-6
            <= float(mean[index])
            <= float(bounds[name][1]) + 1e-6
            for index, name in enumerate(expected_names)
        )
    )
    sample_count = int(summary.get("num_samples", 0) or 0)
    bounds_satisfied = summary.get("bounds_satisfied") is True
    passed = all(
        (
            names_match,
            finite_mean,
            finite_raw_mean,
            finite_std,
            normalized,
            bounds_complete,
            mean_inside_bounds,
            bounds_satisfied,
            sample_count > 0,
        )
    )
    return {
        "status": "pass" if passed else "not_ready",
        "names": names,
        "sample_count": sample_count,
        "base_mode": summary.get("base_mode"),
        "weight_semantics": summary.get("weight_semantics"),
        "effective_routing_mean": mean,
        "theoretical_bounds": bounds,
        "names_match": names_match,
        "finite_statistics": finite_mean and finite_raw_mean and finite_std,
        "weights_normalized": normalized,
        "mean_inside_bounds": mean_inside_bounds,
        "bounds_satisfied": bounds_satisfied,
        "observed_gate_variation": bool(
            finite_std and any(float(item) > 1e-8 for item in summary["std"])
        ),
    }


def learned_gate_diagnostics(result: dict[str, Any], task: str) -> dict[str, Any]:
    if task == "packet":
        diagnostics = result.get("learned_gate_diagnostics")
        channel_key = "packet_channel_gate"
    elif task == "flow":
        diagnostics = (
            result.get("metrics", {})
            .get("eval_config", {})
            .get("learned_gate_diagnostics")
        )
        channel_key = "dual_channel_gate"
    else:
        raise ValueError(f"unknown mechanism-evidence task: {task}")
    if not isinstance(diagnostics, dict):
        return {"status": "not_ready", "reason": "missing_learned_gate_diagnostics"}
    channel = _audit_gate_summary(
        diagnostics.get(channel_key),
        expected_names=("semantic", "content", "structural"),
        names_key="channel_names",
    )
    intervention = _audit_gate_summary(
        diagnostics.get("intervention_view_gate"),
        expected_names=("factual", "intervened"),
        names_key="view_names",
    )
    passed = channel["status"] == "pass" and intervention["status"] == "pass"
    return {
        "status": "pass" if passed else "not_ready",
        "channel_router": channel,
        "intervention_router": intervention,
    }


def audit_runtime_mechanism_evidence(
    packet_result: dict[str, Any] | None,
    flow_result: dict[str, Any] | None,
) -> dict[str, Any]:
    if packet_result is None or flow_result is None:
        return {
            "status": "not_ready",
            "reason": "packet_and_flow_fixed_test_results_are_required",
        }
    packet = learned_gate_diagnostics(packet_result, "packet")
    flow = learned_gate_diagnostics(flow_result, "flow")
    schema_match = bool(
        packet.get("channel_router", {}).get("names")
        == flow.get("channel_router", {}).get("names")
        and packet.get("intervention_router", {}).get("names")
        == flow.get("intervention_router", {}).get("names")
        and packet.get("channel_router", {}).get("base_mode")
        == flow.get("channel_router", {}).get("base_mode")
        and packet.get("intervention_router", {}).get("base_mode")
        == flow.get("intervention_router", {}).get("base_mode")
    )
    passed = packet["status"] == "pass" and flow["status"] == "pass" and schema_match
    return {
        "status": "pass" if passed else "not_ready",
        "claim": "same_bounded_learned_router_schema_executes_in_both_tasks",
        "schema_match": schema_match,
        "learned_values_expected_to_differ_by_task_and_fold": True,
        "packet": packet,
        "flow": flow,
    }


def audit_flow_native_extraction(
    manifests: dict[str, dict[str, Any]] | None,
    *,
    expected_checkpoint: str,
    expected_sha256: str,
    expected_session_mask_probability: float,
) -> dict[str, Any]:
    if not manifests or set(manifests) != {"train", "valid", "test"}:
        return {
            "status": "not_ready",
            "reason": "train_valid_test_native_extraction_manifests_are_required",
        }
    expected_path = str(Path(expected_checkpoint).resolve())
    splits: dict[str, Any] = {}
    passed = True
    for split, manifest in manifests.items():
        observed_path = str(Path(str(manifest.get("checkpoint", ""))).resolve())
        split_pass = bool(
            observed_path == expected_path
            and manifest.get("checkpoint_sha256") == expected_sha256
            and float(manifest.get("session_mask_probability", -1.0))
            == float(expected_session_mask_probability)
            and manifest.get("alignment") == "flow_id_and_packet_id"
            and manifest.get("packet_representation_scope") == "strict_current_packet"
            and manifest.get("packet_representation_name")
            == "native_protocol_current_packet_content"
            and manifest.get("contextual_flow_representations_exported_separately")
            is True
            and int(manifest.get("flow_count", 0)) > 0
        )
        passed = passed and split_pass
        splits[split] = {
            "status": "pass" if split_pass else "not_ready",
            "checkpoint": observed_path,
            "checkpoint_sha256": manifest.get("checkpoint_sha256"),
            "session_mask_probability": manifest.get("session_mask_probability"),
            "flow_count": manifest.get("flow_count"),
            "alignment": manifest.get("alignment"),
            "packet_representation_scope": manifest.get(
                "packet_representation_scope"
            ),
            "packet_representation_name": manifest.get(
                "packet_representation_name"
            ),
            "contextual_flow_representations_exported_separately": manifest.get(
                "contextual_flow_representations_exported_separately"
            ),
        }
    return {
        "status": "pass" if passed else "not_ready",
        "claim": "all_flow_native_embeddings_derive_from_the_audited_checkpoint",
        "expected_checkpoint": expected_path,
        "expected_checkpoint_sha256": expected_sha256,
        "splits": splits,
    }


def resolve_fixed_test_result(manifest: dict[str, Any], task: str) -> str:
    paths = framework_notes(manifest).get("result_paths") or []
    if not isinstance(paths, list):
        return ""
    candidates = [str(path) for path in paths if str(path).endswith(".json")]
    if task == "packet":
        candidates = [path for path in candidates if "test_unified_packet_single_head" in path]
    else:
        candidates = [path for path in candidates if "test_seq_metrics" in path]
    return candidates[0] if len(candidates) == 1 else ""


def audit_cross_task_fold(
    packet_manifest: dict[str, Any],
    flow_manifest: dict[str, Any],
    *,
    packet_checkpoint: dict[str, Any],
    packet_native_checkpoint: dict[str, Any],
    flow_checkpoint: dict[str, Any],
    flow_native_checkpoint: dict[str, Any],
    packet_native_sha256: str,
    packet_result: dict[str, Any] | None = None,
    flow_result: dict[str, Any] | None = None,
    flow_native_extraction_manifests: dict[str, dict[str, Any]] | None = None,
    flow_native_checkpoint_path: str = "",
    flow_native_sha256: str = "",
    require_mechanism_evidence: bool = False,
) -> dict[str, Any]:
    packet_notes = framework_notes(packet_manifest)
    flow_notes = framework_notes(flow_manifest)
    packet_source_evidence = packet_notes.get("algorithm_source_evidence") or {}
    flow_source_evidence = flow_notes.get("algorithm_source_evidence") or {}
    packet_source_fingerprint = packet_source_evidence.get("launch_fingerprint")
    flow_source_fingerprint = flow_source_evidence.get("launch_fingerprint")
    algorithm_source_verified = bool(
        packet_source_evidence.get("schema")
        == "algorithm_source_stability_evidence_v1"
        and flow_source_evidence.get("schema")
        == "algorithm_source_stability_evidence_v1"
        and packet_source_evidence.get("status") == "pass"
        and flow_source_evidence.get("status") == "pass"
        and packet_source_evidence.get("scope")
        == "entrypoint_dependency_closure_v1"
        and flow_source_evidence.get("scope")
        == "entrypoint_dependency_closure_v1"
        and isinstance(packet_source_fingerprint, str)
        and len(packet_source_fingerprint) == 64
        and packet_source_fingerprint == flow_source_fingerprint
        and packet_source_evidence.get("completion_fingerprint")
        == packet_source_fingerprint
        and flow_source_evidence.get("completion_fingerprint")
        == flow_source_fingerprint
        and packet_source_evidence.get("changed_paths") == []
        and flow_source_evidence.get("changed_paths") == []
    )
    packet_framework = packet_manifest.get("framework") or {}
    flow_framework = flow_manifest.get("framework") or {}
    packet_dataset = packet_framework.get("dataset") or packet_manifest.get("dataset")
    flow_dataset = flow_framework.get("dataset") or flow_manifest.get("dataset")
    packet_fold = packet_manifest.get("fold", packet_notes.get("fold"))
    flow_fold = flow_manifest.get("fold", flow_notes.get("fold"))
    packet_fingerprint = packet_notes.get("shared_core_config_sha256")
    flow_fingerprint = flow_notes.get("shared_core_config_sha256")
    packet_method_fingerprint = (
        packet_notes.get("shared_core_method_sha256") or packet_fingerprint
    )
    flow_method_fingerprint = (
        flow_notes.get("shared_core_method_sha256") or flow_fingerprint
    )
    packet_tower1_contract = packet_notes.get("tower1_training_contract")
    flow_tower1_contract = flow_notes.get("tower1_training_contract")
    packet_tower1_execution = packet_notes.get("tower1_execution_evidence") or {}
    flow_tower1_execution = flow_notes.get("tower1_execution_evidence") or {}
    try:
        packet_tower1_signature = tower1_shared_protocol_signature(
            packet_tower1_contract or {}
        )
        flow_tower1_signature = tower1_shared_protocol_signature(
            flow_tower1_contract or {}
        )
        tower1_contract_match = packet_tower1_signature == flow_tower1_signature
        tower1_hyperparameter_delta = tower1_hyperparameter_differences(
            packet_tower1_contract, flow_tower1_contract
        )
    except (TypeError, ValueError):
        packet_tower1_signature = None
        flow_tower1_signature = None
        tower1_contract_match = False
        tower1_hyperparameter_delta = {}
    tower1_execution_verified = bool(
        packet_tower1_execution.get("verified") is True
        and flow_tower1_execution.get("verified") is True
        and packet_tower1_execution.get("declared_contract_match") is True
        and flow_tower1_execution.get("declared_contract_match") is True
        and packet_tower1_execution.get("trainer_source_sha256")
        and packet_tower1_execution.get("trainer_source_sha256")
        == flow_tower1_execution.get("trainer_source_sha256")
        and packet_tower1_execution.get("trainer_source_stable_through_completion")
        is True
        and flow_tower1_execution.get("trainer_source_stable_through_completion")
        is True
    )
    semantic_packet_context_match = bool(
        packet_notes.get("semantic_packet_context_policy") == "single_packet"
        and flow_notes.get("semantic_packet_context_policy") == "single_packet"
        and isinstance(packet_tower1_contract, dict)
        and packet_tower1_contract.get("packet_context_policy") == "single_packet"
        and flow_tower1_contract.get("packet_context_policy") == "single_packet"
    )
    packet_semantic_policy_evidence = packet_notes.get(
        "semantic_embedding_policy_evidence"
    ) or {}
    flow_factual_policy_evidence = flow_notes.get(
        "embedding_header_policy_evidence"
    ) or {}
    flow_intervention_policy_evidence = flow_notes.get(
        "intervention_header_policy_evidence"
    ) or {}
    semantic_execution_policy_verified = bool(
        packet_semantic_policy_evidence.get("verified") is True
        and packet_semantic_policy_evidence.get("embedding_audits_verified") is True
        and packet_semantic_policy_evidence.get("expected_packet_context_policy")
        == "single_packet"
        and flow_factual_policy_evidence.get("verified") is True
        and flow_factual_policy_evidence.get("embedding_audits_verified") is True
        and flow_factual_policy_evidence.get("expected_context") == "single_packet"
        and flow_intervention_policy_evidence.get("verified") is True
        and flow_intervention_policy_evidence.get("embedding_audits_verified") is True
        and flow_intervention_policy_evidence.get("expected_context")
        == "single_packet"
    )
    task_local_packet_training = bool(
        packet_notes.get("packet_module_training_source")
        == "packet_task_train_split_packets"
        and flow_notes.get("packet_module_training_source")
        == "flow_task_train_split_packets"
        and packet_notes.get("cross_task_trained_weights_reused") is False
        and flow_notes.get("cross_task_trained_weights_reused") is False
    )
    identity_match = (
        packet_dataset == flow_dataset
        and int(packet_fold) == int(flow_fold)
        and bool(packet_method_fingerprint)
        and packet_method_fingerprint == flow_method_fingerprint
    )
    effective_config_match = bool(
        packet_fingerprint and packet_fingerprint == flow_fingerprint
    )
    packet_native = native_contract(packet_native_checkpoint)
    flow_native = native_contract(flow_native_checkpoint)
    packet_native_training = native_training_contract(packet_native_checkpoint)
    flow_native_training = native_training_contract(flow_native_checkpoint)
    native_training_contract_complete = all(
        value is not None for value in packet_native_training.values()
    ) and all(value is not None for value in flow_native_training.values())
    try:
        packet_native_training_signature = native_training_protocol_signature(
            packet_native_checkpoint
        )
        flow_native_training_signature = native_training_protocol_signature(
            flow_native_checkpoint
        )
        native_training_contract_match = bool(
            native_training_contract_complete
            and packet_native_training_signature == flow_native_training_signature
        )
    except (TypeError, ValueError):
        packet_native_training_signature = None
        flow_native_training_signature = None
        native_training_contract_match = False
    native_hyperparameter_delta = native_training_hyperparameter_differences(
        packet_native_training, flow_native_training
    )
    native_protocol_match = (
        packet_native == flow_native
        and packet_native["protocol"] == NATIVE_PACKET_PRETRAINING_PROTOCOL
    )
    core = audit_shared_core(
        packet_checkpoint,
        packet_native_checkpoint,
        flow_checkpoint=flow_checkpoint,
        packet_group_reduction=packet_notes.get("byte_content_group_loss_reduction"),
        flow_group_reduction=flow_notes.get("content_group_loss_reduction"),
        native_checkpoint_sha256=packet_native_sha256,
    )
    mechanism = audit_runtime_mechanism_evidence(packet_result, flow_result)
    extraction = audit_flow_native_extraction(
        flow_native_extraction_manifests,
        expected_checkpoint=flow_native_checkpoint_path,
        expected_sha256=flow_native_sha256,
        expected_session_mask_probability=float(
            flow_native_training.get("session_mask_probability", -1.0)
        ),
    )
    passed = (
        identity_match
        and native_protocol_match
        and native_training_contract_match
        and tower1_contract_match
        and tower1_execution_verified
        and semantic_packet_context_match
        and semantic_execution_policy_verified
        and task_local_packet_training
        and core["status"] == "pass"
        and (
            not require_mechanism_evidence
            or (
                mechanism["status"] == "pass"
                and extraction["status"] == "pass"
                and algorithm_source_verified
            )
        )
    )
    return {
        "claim": "exact_shared_packet_representation_reused_before_flow_aggregation",
        "status": "pass" if passed else "not_ready",
        "dataset": packet_dataset if packet_dataset == flow_dataset else None,
        "fold": int(packet_fold) if packet_fold == flow_fold else None,
        "identity_match": identity_match,
        "shared_core_method_sha256": (
            packet_method_fingerprint
            if packet_method_fingerprint == flow_method_fingerprint
            else None
        ),
        "effective_shared_core_config_match": effective_config_match,
        "packet_effective_shared_core_config_sha256": packet_fingerprint,
        "flow_effective_shared_core_config_sha256": flow_fingerprint,
        "shared_core_config_sha256": (
            packet_fingerprint if effective_config_match else None
        ),
        "native_pretraining_contract_match": native_protocol_match,
        "native_training_contract_match": native_training_contract_match,
        "packet_native_training_protocol_signature": packet_native_training_signature,
        "flow_native_training_protocol_signature": flow_native_training_signature,
        "allowed_native_hyperparameter_differences": native_hyperparameter_delta,
        "tower1_training_contract_match": tower1_contract_match,
        "tower1_shared_protocol_signature_match": tower1_contract_match,
        "packet_tower1_shared_protocol_signature": packet_tower1_signature,
        "flow_tower1_shared_protocol_signature": flow_tower1_signature,
        "allowed_tower1_hyperparameter_differences": tower1_hyperparameter_delta,
        "tower1_execution_contract_verified": tower1_execution_verified,
        "semantic_packet_context_match": semantic_packet_context_match,
        "semantic_execution_policy_verified": semantic_execution_policy_verified,
        "packet_semantic_embedding_policy_evidence": packet_semantic_policy_evidence,
        "flow_factual_embedding_policy_evidence": flow_factual_policy_evidence,
        "flow_intervention_embedding_policy_evidence": flow_intervention_policy_evidence,
        "packet_semantic_context_policy": packet_notes.get(
            "semantic_packet_context_policy"
        ),
        "flow_semantic_context_policy": flow_notes.get(
            "semantic_packet_context_policy"
        ),
        "task_local_packet_module_training": task_local_packet_training,
        "packet_module_training_source": packet_notes.get(
            "packet_module_training_source"
        ),
        "flow_packet_module_training_source": flow_notes.get(
            "packet_module_training_source"
        ),
        "cross_task_trained_weights_reused": {
            "packet": packet_notes.get("cross_task_trained_weights_reused"),
            "flow": flow_notes.get("cross_task_trained_weights_reused"),
        },
        "packet_tower1_training_contract": packet_tower1_contract,
        "flow_tower1_training_contract": flow_tower1_contract,
        "packet_tower1_execution_evidence": packet_tower1_execution,
        "flow_tower1_execution_evidence": flow_tower1_execution,
        "packet_native_contract": packet_native,
        "flow_native_contract": flow_native,
        "packet_native_training_contract": packet_native_training,
        "flow_native_training_contract": flow_native_training,
        "core_audit": core,
        "runtime_mechanism_evidence_required": require_mechanism_evidence,
        "flow_native_extraction_evidence_required": require_mechanism_evidence,
        "algorithm_source_evidence_required": require_mechanism_evidence,
        "algorithm_source_evidence_verified": algorithm_source_verified,
        "algorithm_source_fingerprint": (
            packet_source_fingerprint if algorithm_source_verified else None
        ),
        "packet_algorithm_source_evidence": packet_source_evidence,
        "flow_algorithm_source_evidence": flow_source_evidence,
        "runtime_mechanism_evidence": mechanism,
        "flow_native_extraction_evidence": extraction,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet_manifest", required=True)
    parser.add_argument("--flow_manifest", required=True)
    parser.add_argument("--packet_checkpoint", default="")
    parser.add_argument("--packet_native_checkpoint", default="")
    parser.add_argument("--flow_checkpoint", default="")
    parser.add_argument("--flow_native_checkpoint", default="")
    parser.add_argument("--packet_result", default="")
    parser.add_argument("--flow_result", default="")
    parser.add_argument("--require_mechanism_evidence", action="store_true")
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    packet_manifest = load_json(args.packet_manifest)
    flow_manifest = load_json(args.flow_manifest)
    packet_notes = framework_notes(packet_manifest)
    flow_notes = framework_notes(flow_manifest)
    flow_checkpoints = flow_notes.get("tower2_checkpoints") or {}
    paths = {
        "packet_checkpoint": args.packet_checkpoint
        or packet_notes.get("packet_model_checkpoint", ""),
        "packet_native_checkpoint": args.packet_native_checkpoint
        or packet_notes.get("resolved_protocol_content_checkpoint", ""),
        "flow_checkpoint": args.flow_checkpoint or flow_checkpoints.get("seq", ""),
        "flow_native_checkpoint": args.flow_native_checkpoint
        or flow_notes.get("native_checkpoint", ""),
        "packet_result": args.packet_result
        or resolve_fixed_test_result(packet_manifest, "packet"),
        "flow_result": args.flow_result
        or resolve_fixed_test_result(flow_manifest, "flow"),
    }
    native_suffix = str(
        flow_manifest.get("native_structural_suffix")
        or flow_notes.get("native_structural_suffix")
        or ""
    )
    flow_dataset_name = (
        (flow_manifest.get("framework") or {}).get("dataset")
        or flow_manifest.get("dataset")
    )
    extraction_paths = {
        split: (
            f"reasoningDataset/{flow_dataset_name}/{split}_native_structural_"
            f"{native_suffix}/manifest.json"
        )
        for split in ("train", "valid", "test")
    }
    required_paths = {
        name: path
        for name, path in paths.items()
        if name.endswith("checkpoint")
        or (args.require_mechanism_evidence and name.endswith("result"))
    }
    if args.require_mechanism_evidence:
        required_paths.update(
            {f"flow_native_{split}_manifest": path for split, path in extraction_paths.items()}
        )
    missing = [
        name for name, path in required_paths.items() if not path or not Path(path).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"missing checkpoint inputs: {missing}; resolved={paths}")
    packet_checkpoint = torch.load(
        paths["packet_checkpoint"], map_location="cpu", weights_only=False
    )
    packet_native_checkpoint = torch.load(
        paths["packet_native_checkpoint"], map_location="cpu", weights_only=False
    )
    flow_checkpoint = torch.load(
        paths["flow_checkpoint"], map_location="cpu", weights_only=False
    )
    flow_native_checkpoint = torch.load(
        paths["flow_native_checkpoint"], map_location="cpu", weights_only=False
    )
    packet_result = load_json(paths["packet_result"]) if paths["packet_result"] else None
    flow_result = load_json(paths["flow_result"]) if paths["flow_result"] else None
    extraction_manifests = (
        {split: load_json(path) for split, path in extraction_paths.items()}
        if args.require_mechanism_evidence
        else None
    )
    flow_native_hash = sha256_file(paths["flow_native_checkpoint"])
    report = audit_cross_task_fold(
        packet_manifest,
        flow_manifest,
        packet_checkpoint=packet_checkpoint,
        packet_native_checkpoint=packet_native_checkpoint,
        flow_checkpoint=flow_checkpoint,
        flow_native_checkpoint=flow_native_checkpoint,
        packet_native_sha256=sha256_file(paths["packet_native_checkpoint"]),
        packet_result=packet_result,
        flow_result=flow_result,
        flow_native_extraction_manifests=extraction_manifests,
        flow_native_checkpoint_path=paths["flow_native_checkpoint"],
        flow_native_sha256=flow_native_hash,
        require_mechanism_evidence=args.require_mechanism_evidence,
    )
    report["inputs"] = {
        "packet_manifest": args.packet_manifest,
        "flow_manifest": args.flow_manifest,
        **paths,
        "flow_native_extraction_manifests": extraction_paths,
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "identity_match": report["identity_match"],
                "native_pretraining_contract_match": report[
                    "native_pretraining_contract_match"
                ],
                "native_training_contract_match": report[
                    "native_training_contract_match"
                ],
                "tower1_training_contract_match": report[
                    "tower1_training_contract_match"
                ],
                "core_status": report["core_audit"]["status"],
                "runtime_mechanism_status": report[
                    "runtime_mechanism_evidence"
                ]["status"],
                "flow_native_extraction_status": report[
                    "flow_native_extraction_evidence"
                ]["status"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
