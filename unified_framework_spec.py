#!/usr/bin/env python3
"""Paper-facing contract for the unified packet-to-flow framework.

The project intentionally keeps several candidate experts available, but the
paper-facing claim must stay unified: packet-level and flow-level classification
share the same shortcut-resistant packet representation modules. Task-specific
parts are limited to the final packet head or the flow/window aggregator.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple


MODEL_SHARED_CORE_MODULES: Tuple[str, ...] = (
    "label_free_protocol_content_pretraining",
    "field_aware_header_intervention",
    "semantic_tower1_channel",
    "current_packet_structural_encoder",
    "shared_intervention_view_fusion",
    "bounded_tri_channel_router",
)

SHARED_PROTOCOL_GUARDS: Tuple[str, ...] = (
    "per_flow_split_guard",
    "content_group_empirical_risk",
    "validation_only_selection",
    "fixed_cross_fold_consensus",
)

SHARED_CORE_MODULES: Tuple[str, ...] = (
    *MODEL_SHARED_CORE_MODULES,
    *SHARED_PROTOCOL_GUARDS,
)

FLOW_ONLY_MODULES: Tuple[str, ...] = (
    "packet_to_window_flow_aggregator",
    "flow_level_classifier",
)

PACKET_ONLY_MODULES: Tuple[str, ...] = (
    "strict_current_packet_protocol",
    "packet_level_classifier",
)

PAPER_MAIN_MODULES: Tuple[str, ...] = (
    *SHARED_CORE_MODULES,
    *FLOW_ONLY_MODULES,
    *PACKET_ONLY_MODULES,
)

UNIFIED_CANDIDATE_EXPERTS: Tuple[str, ...] = (
    "packet_tree_feature_expert",
    "packet_probability_fusion_expert",
    "graph_flow_expert",
    "multi_view_flow_expert",
    "flow_statistics_expert",
    "label_free_prior_calibration_expert",
    "embedding_space_experts",
    "paired_view_expert",
    "cross_fold_consensus_expert",
)

ABLATION_ONLY_MODULES: Tuple[str, ...] = (
    "confidence_penalty",
    "vpn_specific_hierarchical_coarse_head",
    "vpn_specific_confusion_groups",
    "unconstrained_probability_stacker",
    "unsafe_target_tuned_prior",
    "manual_single_split_threshold_routing",
    "residual_fusion_grid_search",
    "slot_stacker_expert",
    "validation_probability_selector",
    "label_prior_residual_expert",
)

FRAMEWORK_PROFILES: Dict[str, Dict[str, Any]] = {
    "legacy": {
        "description": "Do not override existing runner defaults.",
    },
    "paper_unified": {
        "description": (
            "CCF-A paper profile: both tasks use field-aware shortcut "
            "intervention, label-free protocol content, semantic and strict "
            "current-packet structural channels, one bounded tri-channel router, "
            "content-group risk, and fixed cross-fold consensus. Flow classification "
            "uses one sequence backbone with mean aggregation; optional graph, "
            "stacker, and prior experts are excluded from the main path."
        ),
        "shared_module_status": {
            "label_free_protocol_content_pretraining": "native_flow_multitask_v1",
            "field_aware_header_intervention": "factual_full_plus_mask_ip_port_intervention",
            "semantic_tower1_channel": "present",
            "current_packet_structural_encoder": "strict_current_packet_13d",
            "shared_intervention_view_fusion": "required_representation_level",
            "bounded_tri_channel_router": "semantic_anchor_residual_0_25",
            "per_flow_split_guard": "enforced",
            "content_group_empirical_risk": "group_mean",
            "validation_only_selection": "enforced",
            "fixed_cross_fold_consensus": "equal_log_mean",
        },
        "packet_overrides": {
            "packet_context_policy": "single_packet",
            "embedding_header_policy": "full",
            "intervened_embedding_header_policy": "mask_ip_port",
            "intervention_max_residual_weight": 0.25,
            "use_intervention_views": True,
            "max_packet_length": 1024,
            "cls_weight": 1.0,
            "contrastive_weight": 0.1,
            "same_flow_positive_weight": 1.0,
            "same_label_positive_weight": 1.0,
            "temperature": 0.07,
            "packet_batch_size": 16,
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "dtype": "float16",
            "seed": 42,
            "tower1_checkpoint_policy": "validation_best_macro_f1",
            "valid_packets_per_flow": 2,
            "epochs": 8,
            "lr": 1e-5,
            "head_lr": 1e-4,
            "class_weighting": "effective",
            "gradient_accumulation_steps": 1,
            "byte_use_payload_channel": False,
            "byte_use_protocol_fields": True,
            "channel_fusion_base_mode": "semantic_anchor",
            "byte_max_bytes": 128,
            "byte_max_payload_bytes": 128,
        },
        "flow_overrides": {
            "packet_context_policy": "single_packet",
            "embedding_header_policy": "full",
            "intervened_embedding_header_policy": "mask_ip_port",
            "intervention_max_residual_weight": 0.25,
            "use_intervention_views": True,
            "max_packet_length": 1024,
            "cls_weight": 1.0,
            "contrastive_weight": 0.1,
            "same_flow_positive_weight": 1.0,
            "same_label_positive_weight": 1.0,
            "temperature": 0.07,
            "packet_batch_size": 16,
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "dtype": "float16",
            "seed": 42,
            "tower1_checkpoint_policy": "validation_best_macro_f1",
            "tower1_valid_packets_per_flow": 2,
            "tower1_epochs": 8,
            "tower1_lr": 1e-5,
            "tower1_head_lr": 1e-4,
            "tower1_class_weighting": "effective",
            "tower1_disable_packet_information_weights": True,
            "tower1_gradient_accumulation_steps": 1,
            "tower1_use_sft": False,
            "embedding_resume_existing": False,
            "embedding_resume_shards": False,
            "flow_pooling": "mean",
            "model_types": "seq",
            "native_structural_suffix": "shared_content",
            "native_structural_dim": 128,
            "meta_feature_dim": 13,
            "dual_channel_mode": "residual",
            "dual_channel_gate_mode": "adaptive",
            "channel_fusion_base_mode": "semantic_anchor",
            "dual_channel_max_weight": 0.25,
            "flow_stat_expert_weight": 0.0,
            "flow_stat_aux_weight": 0.0,
            "hierarchical_weight": 0.0,
            "hierarchical_logit_weight": 0.0,
            "coarse_groups": "none",
            "contrastive_mode": "standard",
            "confusion_groups": "none",
            "meta_dropout_prob": 0.0,
            "edge_attr_dropout_prob": 0.0,
            "content_group_guard": True,
            "tower2_select_metric": "content_group_macro_f1",
            "split_group_key": "content_group_id",
            "content_group_unique_batches": True,
            "content_group_loss_reduction": "group_mean",
        },
    },
}

PAPER_UNIFIED_SHARED_STATUS_ALIASES: Dict[str, Tuple[Any, ...]] = {
    "label_free_protocol_content_pretraining": ("native_flow_multitask_v1",),
    "field_aware_header_intervention": ("factual_full_plus_mask_ip_port_intervention",),
    "semantic_tower1_channel": (
        "present_or_available",
        "present",
        "present_or_available_in_source_inputs",
    ),
    "current_packet_structural_encoder": ("strict_current_packet_13d",),
    "shared_intervention_view_fusion": ("required_representation_level",),
    "bounded_tri_channel_router": ("semantic_anchor_residual_0_25",),
    "per_flow_split_guard": ("enforced",),
    "content_group_empirical_risk": ("group_mean",),
    "validation_only_selection": (
        "enforced",
        "enforced_by_source_results",
    ),
    "fixed_cross_fold_consensus": (
        "equal_log_mean",
        "equal_log_mean_bound",
    ),
}


@dataclass(frozen=True)
class ResultSpec:
    dataset: str
    task: str
    path: str
    target_accuracy: float | None = None
    target_macro_f1: float | None = None
    framework_profile: str = "paper_unified"
    framework_manifest_glob: str = ""
    note: str = ""


FLOW_LEVEL_RESULTS: Dict[str, ResultSpec] = {
    "vpn-app": ResultSpec(
        dataset="vpn-app",
        task="flow-level",
        path="reasoningDataset/vpn-app/test_crossfold_consensus_auto_confidence.json",
        target_accuracy=0.75,
        target_macro_f1=0.65,
        framework_manifest_glob="reasoningDataset/vpn-app/stage8_flowaware_manifest_*paper_unified*.json",
        note="VPN flow-level paper default; Per-flow Split only.",
    ),
    "tls-120": ResultSpec(
        dataset="tls-120",
        task="flow-level",
        path="reasoningDataset/tls-120/test_crossfold_consensus_auto_confidence.json",
        target_accuracy=0.78,
        target_macro_f1=0.70,
        framework_manifest_glob="reasoningDataset/tls-120/stage8_flowaware_manifest_*paper_unified*.json",
        note="TLS-120 flow-level paper default; Per-flow Split only.",
    ),
}


PACKET_LEVEL_RESULTS: Dict[str, ResultSpec] = {
    "vpn-app": ResultSpec(
        dataset="vpn-app",
        task="packet-level",
        path="reasoningDataset/packet-level/vpn-app/paper_default_result.json",
        target_accuracy=0.90,
        target_macro_f1=0.76,
        framework_manifest_glob="reasoningDataset/packet-level/vpn-app/*/packet_framework_manifest.json",
        note="Strict one-current-packet input.",
    ),
    "tls-120": ResultSpec(
        dataset="tls-120",
        task="packet-level",
        path="reasoningDataset/packet-level/tls-120/paper_default_result.json",
        target_accuracy=0.85,
        target_macro_f1=0.78,
        framework_manifest_glob="reasoningDataset/packet-level/tls-120/*/packet_framework_manifest.json",
        note="Strict one-current-packet input.",
    ),
    "vpn-service": ResultSpec(
        dataset="vpn-service",
        task="packet-level",
        path="reasoningDataset/packet-level/vpn-service/paper_default_result.json",
        framework_manifest_glob="reasoningDataset/packet-level/vpn-service/*/packet_framework_manifest.json",
        note="Strict one-current-packet input.",
    ),
    "vpn-binary": ResultSpec(
        dataset="vpn-binary",
        task="packet-level",
        path="reasoningDataset/packet-level/vpn-binary/paper_default_result.json",
        framework_manifest_glob="reasoningDataset/packet-level/vpn-binary/*/packet_framework_manifest.json",
        note="Strict one-current-packet input.",
    ),
    "ustc-app": ResultSpec(
        dataset="ustc-app",
        task="packet-level",
        path="reasoningDataset/packet-level/ustc-app/paper_default_result.json",
        framework_manifest_glob="reasoningDataset/packet-level/ustc-app/*/packet_framework_manifest.json",
        note="Packet-level only; excluded from current flow-level claims.",
    ),
    "ustc-binary": ResultSpec(
        dataset="ustc-binary",
        task="packet-level",
        path="reasoningDataset/packet-level/ustc-binary/paper_default_result.json",
        framework_manifest_glob="reasoningDataset/packet-level/ustc-binary/*/packet_framework_manifest.json",
        note="All three folds are exact; fold0 path is used as an audit anchor.",
    ),
}


def task_modules(task: str) -> Tuple[str, ...]:
    if task == "flow-level":
        return SHARED_CORE_MODULES + FLOW_ONLY_MODULES
    if task == "packet-level":
        return SHARED_CORE_MODULES + PACKET_ONLY_MODULES
    raise ValueError(f"unknown task: {task}")


def shared_module_overlap() -> Tuple[str, ...]:
    flow_modules = set(task_modules("flow-level"))
    packet_modules = set(task_modules("packet-level"))
    return tuple(name for name in SHARED_CORE_MODULES if name in flow_modules and name in packet_modules)


def profile_task_overrides(profile_name: str, task: str) -> Dict[str, Any]:
    profile = FRAMEWORK_PROFILES.get(profile_name, FRAMEWORK_PROFILES["legacy"])
    if profile_name == "legacy":
        return {}
    key = "packet_overrides" if task == "packet-level" else "flow_overrides"
    return dict(profile.get(key, {}))


def framework_profile_contract(profile_name: str, task: str) -> Dict[str, Any]:
    profile = FRAMEWORK_PROFILES.get(profile_name, FRAMEWORK_PROFILES["legacy"])
    return {
        "framework_profile": profile_name,
        "task": task,
        "description": profile.get("description", ""),
        "shared_core_modules": list(SHARED_CORE_MODULES),
        "model_shared_core_modules": list(MODEL_SHARED_CORE_MODULES),
        "shared_protocol_guards": list(SHARED_PROTOCOL_GUARDS),
        "paper_main_modules": list(PAPER_MAIN_MODULES),
        "unified_candidate_experts": list(UNIFIED_CANDIDATE_EXPERTS),
        "ablation_only_modules": list(ABLATION_ONLY_MODULES),
        "shared_module_status": dict(profile.get("shared_module_status", {})),
        "task_overrides": profile_task_overrides(profile_name, task),
    }


def framework_profile_fingerprint(profile_name: str, task: str) -> str:
    payload = json.dumps(
        framework_profile_contract(profile_name, task),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_framework_manifest(
    *,
    task: str,
    dataset: str,
    input_unit: str,
    stage: str,
    shared_module_status: Dict[str, Any] | None = None,
    task_module_status: Dict[str, Any] | None = None,
    selection_protocol: str = "validation_only",
    calibration_protocol: str = "label_free_or_validation_only",
    notes: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Return a machine-readable contract for one planned run.

    Runners write this into their manifests so audits can tell whether a flow
    or packet experiment still follows the same paper-facing module contract.
    """
    modules = task_modules(task)
    shared_status = {name: "present" for name in SHARED_CORE_MODULES}
    if shared_module_status:
        shared_status.update(shared_module_status)
    task_specific = tuple(name for name in modules if name not in SHARED_CORE_MODULES)
    specific_status = {name: "present" for name in task_specific}
    if task_module_status:
        specific_status.update(task_module_status)
    profile_name = notes.get("framework_profile") if notes else None
    profile_contract = framework_profile_contract(profile_name, task) if profile_name else None
    return {
        "framework_name": "unified_shortcut_resistant_packet_to_flow",
        "framework_profile": profile_name,
        "framework_profile_fingerprint": (
            framework_profile_fingerprint(profile_name, task) if profile_name else None
        ),
        "framework_profile_contract": profile_contract,
        "task": task,
        "dataset": dataset,
        "stage": stage,
        "input_unit": input_unit,
        "paper_main_modules": list(PAPER_MAIN_MODULES),
        "shared_core_modules": list(SHARED_CORE_MODULES),
        "model_shared_core_modules": list(MODEL_SHARED_CORE_MODULES),
        "shared_protocol_guards": list(SHARED_PROTOCOL_GUARDS),
        "task_modules": list(modules),
        "unified_candidate_experts": list(UNIFIED_CANDIDATE_EXPERTS),
        "ablation_only_modules": list(ABLATION_ONLY_MODULES),
        "shared_module_status": shared_status,
        "task_module_status": specific_status,
        "selection_protocol": selection_protocol,
        "calibration_protocol": calibration_protocol,
        "notes": notes or {},
    }


def apply_framework_profile(args: Any, task: str) -> Dict[str, Any]:
    profile_name = getattr(args, "framework_profile", "legacy")
    if profile_name not in FRAMEWORK_PROFILES:
        raise ValueError(f"unknown framework profile: {profile_name}")
    profile = FRAMEWORK_PROFILES[profile_name]
    if profile_name == "legacy":
        return {}
    applied: Dict[str, Any] = {}
    for attr, value in profile_task_overrides(profile_name, task).items():
        if hasattr(args, attr):
            old = getattr(args, attr)
            setattr(args, attr, value)
            applied[attr] = {"old": old, "new": value}
    return applied


def tower1_training_contract(args: Any, task: str) -> Dict[str, Any]:
    """Executed Tower1 settings; numeric optimization values may differ by run."""
    if task == "packet-level":
        value = lambda packet_name, flow_name=None: getattr(args, packet_name)
        use_sft = False
        disable_information_weights = True
        flow_balanced_batches = True
        packets_per_flow = 2
        max_steps = 0
        gradient_checkpointing = True
        init_checkpoint_dir = str(args.init_checkpoint_dir)
        init_adapter_only = bool(args.init_adapter_only)
    elif task == "flow-level":
        value = lambda packet_name, flow_name=None: getattr(
            args, flow_name or packet_name
        )
        use_sft = bool(args.tower1_use_sft)
        disable_information_weights = bool(
            args.tower1_disable_packet_information_weights
        )
        flow_balanced_batches = bool(args.flow_balanced_packet_batches)
        packets_per_flow = int(args.packets_per_flow)
        max_steps = int(args.tower1_max_steps)
        gradient_checkpointing = bool(args.gradient_checkpointing)
        init_checkpoint_dir = str(args.tower1_init_checkpoint_dir)
        init_adapter_only = False
    else:
        raise ValueError(f"unknown Tower1 contract task: {task}")
    return {
        "trainer": "train_tower1_multitask.py",
        "packet_context_policy": str(value("packet_context_policy")),
        "base_model": str(value("base_model")),
        "epochs": int(value("epochs", "tower1_epochs")),
        "max_steps": max_steps,
        "packet_batch_size": int(value("packet_batch_size")),
        "gradient_accumulation_steps": int(
            value("gradient_accumulation_steps", "tower1_gradient_accumulation_steps")
        ),
        "gradient_checkpointing": gradient_checkpointing,
        "max_packet_length": int(value("max_packet_length")),
        "projection_dim": 256,
        "cls_weight": float(value("cls_weight")),
        "contrastive_weight": float(value("contrastive_weight")),
        "same_flow_positive_weight": float(value("same_flow_positive_weight")),
        "same_label_positive_weight": float(value("same_label_positive_weight")),
        "identity_safe_contrastive": bool(
            getattr(args, "identity_safe_contrastive", False)
        ),
        "flow_proto_weight": float(value("flow_proto_weight")),
        "flow_proto_positive": str(value("flow_proto_positive")),
        "flow_proto_context": str(value("flow_proto_context")),
        "temperature": float(value("temperature")),
        "learning_rate": float(value("lr", "tower1_lr")),
        "head_learning_rate": float(value("head_lr", "tower1_head_lr")),
        "weight_decay": 0.01,
        "lora_r": int(value("lora_r")),
        "lora_alpha": int(value("lora_alpha")),
        "lora_dropout": float(value("lora_dropout")),
        "dtype": str(value("dtype")),
        "seed": int(value("seed")),
        "class_weighting": str(value("class_weighting", "tower1_class_weighting")),
        "class_weight_beta": 0.9999,
        "class_weight_basis": str(
            value("class_weight_basis", "tower1_class_weight_basis")
        ),
        "class_weight_strength": float(
            value("class_weight_strength", "tower1_class_weight_strength")
        ),
        "paired_consistency_weight": float(
            value(
                "tower1_paired_consistency_weight",
                "tower1_paired_consistency_weight",
            )
        ),
        "paired_cls_weight": float(
            value("tower1_paired_cls_weight", "tower1_paired_cls_weight")
        ),
        "paired_logit_kl_weight": float(
            value("tower1_paired_logit_kl_weight", "tower1_paired_logit_kl_weight")
        ),
        "paired_raw_consistency_weight": float(
            value(
                "tower1_paired_raw_consistency_weight",
                "tower1_paired_raw_consistency_weight",
            )
        ),
        "cross_scale_weight": float(
            getattr(args, "tower1_cross_scale_weight", 0.0)
        ),
        "cross_scale_temperature": float(
            getattr(args, "tower1_cross_scale_temperature", 0.07)
        ),
        "use_sft": use_sft,
        "disable_packet_information_weights": disable_information_weights,
        "flow_balanced_packet_batches": flow_balanced_batches,
        "packets_per_flow": packets_per_flow,
        "packet_batch_scheduler": "epoch_resampled_dataloader_v1",
        "select_metric": "macro_f1",
        "early_stop_patience": int(
            value("tower1_early_stop_patience", "tower1_early_stop_patience")
        ),
        "init_checkpoint_dir": init_checkpoint_dir,
        "init_adapter_only": init_adapter_only,
    }


TOWER1_SHARED_PROTOCOL_FIELDS: Tuple[str, ...] = (
    "trainer",
    "packet_context_policy",
    "base_model",
    "max_packet_length",
    "projection_dim",
    "lora_r",
    "use_sft",
    "disable_packet_information_weights",
    "flow_balanced_packet_batches",
    "packet_batch_scheduler",
    "class_weighting",
    "class_weight_basis",
    "select_metric",
)


def tower1_shared_protocol_signature(contract: Dict[str, Any]) -> Dict[str, Any]:
    """Return architecture/algorithm invariants, excluding numeric hyperparameters.

    Numeric loss weights may vary by dataset/task, but crossing zero changes the
    objective graph and therefore changes the method.  Record objective presence
    explicitly so a disabled loss cannot masquerade as ordinary tuning.
    """
    missing = [field for field in TOWER1_SHARED_PROTOCOL_FIELDS if field not in contract]
    if missing:
        raise ValueError(f"Tower1 contract is missing shared protocol fields: {missing}")
    signature = {
        field: contract[field] for field in TOWER1_SHARED_PROTOCOL_FIELDS
    }
    signature["identity_safe_contrastive"] = bool(
        contract.get("identity_safe_contrastive", False)
    )
    paired_enabled = float(contract.get("paired_consistency_weight", 0.0)) > 0.0
    signature["objectives"] = {
        "packet_classification": float(contract.get("cls_weight", 0.0)) > 0.0,
        "supervised_contrastive": float(
            contract.get("contrastive_weight", 0.0)
        )
        > 0.0,
        "same_flow_positive": float(
            contract.get("same_flow_positive_weight", 0.0)
        )
        > 0.0,
        "same_label_positive": float(
            contract.get("same_label_positive_weight", 0.0)
        )
        > 0.0,
        "flow_prototype": float(contract.get("flow_proto_weight", 0.0)) > 0.0,
        "paired_consistency": paired_enabled,
        "paired_classification": paired_enabled
        and float(contract.get("paired_cls_weight", 0.0)) > 0.0,
        "paired_logit_consistency": paired_enabled
        and float(contract.get("paired_logit_kl_weight", 0.0)) > 0.0,
        "paired_raw_consistency": paired_enabled
        and float(contract.get("paired_raw_consistency_weight", 0.0)) > 0.0,
        "availability_aware_cross_scale": float(
            contract.get("cross_scale_weight", 0.0)
        )
        > 0.0,
        "class_balancing": (
            str(contract.get("class_weighting", "none")) != "none"
            and str(contract.get("class_weight_basis", "")) == "flow"
        ),
    }
    signature["flow_prototype_mode"] = (
        {
            "positive": str(contract.get("flow_proto_positive", "")),
            "context": str(contract.get("flow_proto_context", "")),
        }
        if signature["objectives"]["flow_prototype"]
        else None
    )
    signature["cross_scale_policy"] = (
        {
            "packet_identity": "exact",
            "own_context": "leave_one_out",
            "singleton_context": "masked",
            "views": "factual_to_intervened_bidirectional",
            "same_class_other_flow": "excluded_from_negatives",
        }
        if signature["objectives"]["availability_aware_cross_scale"]
        else None
    )
    signature["initialization"] = {
        "base_model_only": not bool(contract.get("init_checkpoint_dir")),
        "adapter_only": bool(contract.get("init_adapter_only")),
    }
    return signature


def tower1_hyperparameter_differences(
    packet_contract: Dict[str, Any], flow_contract: Dict[str, Any]
) -> Dict[str, Dict[str, Any]]:
    shared_fields = set(TOWER1_SHARED_PROTOCOL_FIELDS) | {
        "identity_safe_contrastive"
    }
    keys = sorted((set(packet_contract) | set(flow_contract)) - shared_fields)
    return {
        key: {"packet": packet_contract.get(key), "flow": flow_contract.get(key)}
        for key in keys
        if packet_contract.get(key) != flow_contract.get(key)
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tower1_execution_evidence(
    output_dir: str | Path,
    declared_contract: Dict[str, Any],
) -> Dict[str, Any]:
    """Bind a runner declaration to the trainer's completed on-disk contract."""
    output_dir = Path(output_dir)
    contract_path = output_dir / "tower1_training_contract.json"
    evidence: Dict[str, Any] = {
        "schema": "tower1_execution_evidence_v1",
        "contract_path": str(contract_path.resolve()),
        "verified": False,
        "declared_contract_match": False,
    }
    if not contract_path.is_file():
        evidence["reason"] = "missing_training_contract"
        return evidence
    try:
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        config = contract.get("training_config") or {}
        completed = contract.get("completed_artifacts") or {}
        final_path = output_dir / "final" / "tower1_heads.pt"
        history_path = output_dir / "packet_validation_history.jsonl"
        final_sha = _file_sha256(final_path) if final_path.is_file() else None
        history_sha = _file_sha256(history_path) if history_path.is_file() else None
        artifacts_verified = bool(
            final_sha
            and history_sha
            and completed.get("final_heads", {}).get("sha256") == final_sha
            and completed.get("validation_history", {}).get("sha256") == history_sha
        )
        method_config = {
            "trainer": "train_tower1_multitask.py",
            "base_model": str(config.get("base_model")),
            "epochs": int(config.get("epochs", -1)),
            "max_steps": int(config.get("max_steps", -1)),
            "packet_batch_size": int(config.get("packet_batch_size", -1)),
            "gradient_accumulation_steps": int(
                config.get("gradient_accumulation_steps", -1)
            ),
            "gradient_checkpointing": bool(config.get("gradient_checkpointing")),
            "max_packet_length": int(config.get("max_packet_length", -1)),
            "projection_dim": int(config.get("projection_dim", -1)),
            "cls_weight": float(config.get("cls_weight", float("nan"))),
            "contrastive_weight": float(
                config.get("contrastive_weight", float("nan"))
            ),
            "same_flow_positive_weight": float(
                config.get("same_flow_positive_weight", float("nan"))
            ),
            "same_label_positive_weight": float(
                config.get("same_label_positive_weight", float("nan"))
            ),
            "identity_safe_contrastive": bool(
                config.get("identity_safe_contrastive", False)
            ),
            "flow_proto_weight": float(
                config.get("flow_proto_weight", float("nan"))
            ),
            "flow_proto_positive": str(config.get("flow_proto_positive")),
            "flow_proto_context": str(config.get("flow_proto_context")),
            "temperature": float(config.get("temperature", float("nan"))),
            "learning_rate": float(config.get("lr", float("nan"))),
            "head_learning_rate": float(config.get("head_lr", float("nan"))),
            "weight_decay": float(config.get("weight_decay", float("nan"))),
            "lora_r": int(config.get("lora_r", -1)),
            "lora_alpha": int(config.get("lora_alpha", -1)),
            "lora_dropout": float(config.get("lora_dropout", float("nan"))),
            "dtype": str(config.get("dtype")),
            "seed": int(config.get("seed", -1)),
            "class_weighting": str(config.get("class_weighting")),
            "class_weight_beta": float(
                config.get("class_weight_beta", float("nan"))
            ),
            "class_weight_basis": str(config.get("class_weight_basis")),
            "class_weight_strength": float(
                config.get("class_weight_strength", float("nan"))
            ),
            "paired_consistency_weight": float(
                config.get("paired_consistency_weight", float("nan"))
            ),
            "paired_cls_weight": float(
                config.get("paired_cls_weight", float("nan"))
            ),
            "paired_logit_kl_weight": float(
                config.get("paired_logit_kl_weight", float("nan"))
            ),
            "paired_raw_consistency_weight": float(
                config.get("paired_raw_consistency_weight", float("nan"))
            ),
            "cross_scale_weight": float(config.get("cross_scale_weight", 0.0)),
            "cross_scale_temperature": float(
                config.get("cross_scale_temperature", 0.07)
            ),
            "use_sft": not bool(config.get("no_sft")),
            "disable_packet_information_weights": bool(
                config.get("disable_packet_information_weights")
            ),
            "flow_balanced_packet_batches": bool(
                config.get("flow_balanced_packet_batches")
            ),
            "packets_per_flow": int(config.get("packets_per_flow", -1)),
            "packet_batch_scheduler": str(config.get("packet_batch_scheduler")),
            "select_metric": str(config.get("select_metric")),
            "early_stop_patience": int(config.get("early_stop_patience", -1)),
            "init_checkpoint_dir": str(config.get("init_checkpoint_dir")),
            "init_adapter_only": bool(config.get("init_adapter_only")),
        }
        comparable_declaration = {
            key: value
            for key, value in declared_contract.items()
            if key != "packet_context_policy"
        }
        comparable_declaration.setdefault("identity_safe_contrastive", False)
        comparable_declaration.setdefault("cross_scale_weight", 0.0)
        comparable_declaration.setdefault("cross_scale_temperature", 0.07)
        declaration_match = method_config == comparable_declaration
        executed_contract = {
            "packet_context_policy": declared_contract.get(
                "packet_context_policy", ""
            ),
            **method_config,
        }
        launch_source = contract.get("trainer_source") or {}
        completion_source = contract.get("completion_observed_trainer_source") or {}
        source_stable = bool(
            launch_source.get("sha256")
            and launch_source.get("sha256") == completion_source.get("sha256")
        )
        evidence.update(
            {
                "contract_sha256": _file_sha256(contract_path),
                "contract_status": contract.get("status"),
                "method_config": method_config,
                "shared_protocol_signature": tower1_shared_protocol_signature(
                    executed_contract
                ),
                "declared_contract_match": declaration_match,
                "artifacts_verified": artifacts_verified,
                "final_heads_sha256": final_sha,
                "validation_history_sha256": history_sha,
                "trainer_source_sha256": launch_source.get("sha256"),
                "trainer_source_stable_through_completion": source_stable,
                "verified": bool(
                    contract.get("schema") == "tower1_training_contract_v1"
                    and contract.get("status") == "complete"
                    and declaration_match
                    and artifacts_verified
                    and source_stable
                ),
            }
        )
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        evidence["reason"] = f"invalid_training_contract:{type(exc).__name__}"
    return evidence


def profile_shared_status(profile_name: str) -> Dict[str, Any]:
    profile = FRAMEWORK_PROFILES.get(profile_name, FRAMEWORK_PROFILES["legacy"])
    return dict(profile.get("shared_module_status", {}))
