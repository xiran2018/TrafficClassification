"""Validated runtime application of an immutable exact shared-core v2 config."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from freeze_shared_core_v2_config import REQUIRED_DATASETS, canonical_sha256


SCHEMA = "exact_shared_packet_core_v2"
STATUS = "frozen_from_cross_dataset_validation"


# These values control optimization, stochastic regularization, or compute
# budget. They may be selected independently for each dataset/task without
# changing the shared method. Architecture choices and categorical algorithms
# intentionally stay outside these allowlists.
INDEPENDENT_TRAINING_HYPERPARAMETERS = {
    "packet-level": frozenset(
        {
            "epochs",
            "packet_batch_size",
            "gradient_accumulation_steps",
            "tower1_early_stop_patience",
            "semantic_embedding_batch_size",
            "semantic_embedding_flow_batch_packets",
            "cls_weight",
            "contrastive_weight",
            "same_flow_positive_weight",
            "same_label_positive_weight",
            "flow_proto_weight",
            "temperature",
            "lr",
            "head_lr",
            "lora_alpha",
            "lora_dropout",
            "dtype",
            "seed",
            "class_weight_strength",
            "tower1_paired_consistency_weight",
            "tower1_paired_cls_weight",
            "tower1_paired_logit_kl_weight",
            "tower1_paired_raw_consistency_weight",
            "tower1_cross_scale_weight",
            "tower1_cross_scale_temperature",
            "protocol_pretrain_max_packets",
            "protocol_pretrain_epochs",
            "protocol_pretrain_batch_size",
            "protocol_pretrain_eval_batch_size",
            "protocol_pretrain_num_workers",
            "protocol_pretrain_learning_rate",
            "protocol_pretrain_weight_decay",
            "protocol_pretrain_field_mask_probability",
            "protocol_pretrain_payload_dropout_probability",
            "protocol_pretrain_session_mask_probability",
            "protocol_pretrain_masked_byte_weight",
            "protocol_pretrain_relative_order_weight",
            "protocol_pretrain_same_flow_weight",
            "protocol_pretrain_next_length_weight",
            "protocol_pretrain_next_iat_weight",
            "protocol_pretrain_direction_weight",
            "protocol_pretrain_packet_consistency_weight",
            "protocol_pretrain_flow_contrastive_weight",
            "protocol_pretrain_temperature",
            "protocol_pretrain_patience",
            "protocol_pretrain_seed",
        }
    ),
    "flow-level": frozenset(
        {
            "tower1_epochs",
            "tower1_max_steps",
            "packet_batch_size",
            "tower1_gradient_accumulation_steps",
            "gradient_checkpointing",
            "tower1_early_stop_patience",
            "embedding_batch_size",
            "embedding_flow_batch_packets",
            "cls_weight",
            "contrastive_weight",
            "same_flow_positive_weight",
            "same_label_positive_weight",
            "flow_proto_weight",
            "temperature",
            "tower1_lr",
            "tower1_head_lr",
            "lora_alpha",
            "lora_dropout",
            "dtype",
            "seed",
            "tower1_class_weight_strength",
            "tower1_paired_consistency_weight",
            "tower1_paired_cls_weight",
            "tower1_paired_logit_kl_weight",
            "tower1_paired_raw_consistency_weight",
            "tower1_cross_scale_weight",
            "tower1_cross_scale_temperature",
            "native_max_packets",
            "native_epochs",
            "native_batch_size",
            "native_eval_batch_size",
            "native_num_workers",
            "native_learning_rate",
            "native_weight_decay",
            "native_field_mask_probability",
            "native_payload_dropout_probability",
            "native_session_mask_probability",
            "native_masked_byte_weight",
            "native_relative_order_weight",
            "native_same_flow_weight",
            "native_next_length_weight",
            "native_next_iat_weight",
            "native_direction_weight",
            "native_packet_consistency_weight",
            "native_flow_contrastive_weight",
            "native_temperature",
            "native_patience",
        }
    ),
}


_ACTIVATION_GUARDED_FIELDS = {
    "packet-level": {
        "cls_weight": ("tower1", "cls_weight"),
        "contrastive_weight": ("tower1", "contrastive_weight"),
        "same_flow_positive_weight": ("tower1", "same_flow_positive_weight"),
        "same_label_positive_weight": ("tower1", "same_label_positive_weight"),
        "flow_proto_weight": ("tower1", "flow_proto_weight"),
        "tower1_paired_consistency_weight": ("tower1", "paired_consistency_weight"),
        "tower1_paired_cls_weight": ("tower1", "paired_cls_weight"),
        "tower1_paired_logit_kl_weight": ("tower1", "paired_logit_kl_weight"),
        "tower1_paired_raw_consistency_weight": ("tower1", "paired_raw_consistency_weight"),
        "tower1_cross_scale_weight": ("tower1", "cross_scale_weight"),
        "protocol_pretrain_field_mask_probability": ("native_pretraining", "field_mask_probability"),
        "protocol_pretrain_payload_dropout_probability": ("native_pretraining", "payload_dropout_probability"),
        "protocol_pretrain_session_mask_probability": ("native_pretraining", "session_mask_probability"),
        "protocol_pretrain_masked_byte_weight": ("native_pretraining", "masked_byte_weight"),
        "protocol_pretrain_relative_order_weight": ("native_pretraining", "relative_order_weight"),
        "protocol_pretrain_same_flow_weight": ("native_pretraining", "same_flow_weight"),
        "protocol_pretrain_next_length_weight": ("native_pretraining", "next_length_weight"),
        "protocol_pretrain_next_iat_weight": ("native_pretraining", "next_iat_weight"),
        "protocol_pretrain_direction_weight": ("native_pretraining", "direction_weight"),
        "protocol_pretrain_packet_consistency_weight": ("native_pretraining", "packet_consistency_weight"),
        "protocol_pretrain_flow_contrastive_weight": ("native_pretraining", "flow_contrastive_weight"),
    },
    "flow-level": {
        "cls_weight": ("tower1", "cls_weight"),
        "contrastive_weight": ("tower1", "contrastive_weight"),
        "same_flow_positive_weight": ("tower1", "same_flow_positive_weight"),
        "same_label_positive_weight": ("tower1", "same_label_positive_weight"),
        "flow_proto_weight": ("tower1", "flow_proto_weight"),
        "tower1_paired_consistency_weight": ("tower1", "paired_consistency_weight"),
        "tower1_paired_cls_weight": ("tower1", "paired_cls_weight"),
        "tower1_paired_logit_kl_weight": ("tower1", "paired_logit_kl_weight"),
        "tower1_paired_raw_consistency_weight": ("tower1", "paired_raw_consistency_weight"),
        "tower1_cross_scale_weight": ("tower1", "cross_scale_weight"),
        "native_field_mask_probability": ("native_pretraining", "field_mask_probability"),
        "native_payload_dropout_probability": ("native_pretraining", "payload_dropout_probability"),
        "native_session_mask_probability": ("native_pretraining", "session_mask_probability"),
        "native_masked_byte_weight": ("native_pretraining", "masked_byte_weight"),
        "native_relative_order_weight": ("native_pretraining", "relative_order_weight"),
        "native_same_flow_weight": ("native_pretraining", "same_flow_weight"),
        "native_next_length_weight": ("native_pretraining", "next_length_weight"),
        "native_next_iat_weight": ("native_pretraining", "next_iat_weight"),
        "native_direction_weight": ("native_pretraining", "direction_weight"),
        "native_packet_consistency_weight": ("native_pretraining", "packet_consistency_weight"),
        "native_flow_contrastive_weight": ("native_pretraining", "flow_contrastive_weight"),
    },
}


def capture_training_hyperparameter_overrides(
    args, task: str, names_csv: str
) -> dict[str, Any]:
    """Capture explicitly named values before framework defaults are applied."""
    if task not in INDEPENDENT_TRAINING_HYPERPARAMETERS:
        raise ValueError(f"unsupported shared-core task: {task}")
    names = [item.strip() for item in names_csv.split(",") if item.strip()]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"duplicate training hyperparameter overrides: {duplicates}")
    unsupported = sorted(set(names) - INDEPENDENT_TRAINING_HYPERPARAMETERS[task])
    if unsupported:
        raise ValueError(
            f"unsupported independent training hyperparameters for {task}: {unsupported}"
        )
    missing = sorted(name for name in names if not hasattr(args, name))
    if missing:
        raise ValueError(f"runner does not expose training hyperparameters: {missing}")
    return {name: getattr(args, name) for name in names}


def restore_profile_training_hyperparameters(
    args,
    task: str,
    values: dict[str, Any],
) -> dict[str, Any]:
    """Restore explicit numeric values after a non-frozen method profile.

    The post-profile values define the objective activation topology. Numeric
    magnitudes may differ, but an override cannot turn a shared objective on or
    off. Architecture and categorical algorithm fields never enter the
    allowlist in the first place.
    """
    if task not in INDEPENDENT_TRAINING_HYPERPARAMETERS:
        raise ValueError(f"unsupported shared-core task: {task}")
    unsupported = sorted(
        set(values) - INDEPENDENT_TRAINING_HYPERPARAMETERS[task]
    )
    if unsupported:
        raise ValueError(
            f"unsupported independent training hyperparameters for {task}: {unsupported}"
        )
    inactive_paired_children = {
        "tower1_paired_cls_weight",
        "tower1_paired_logit_kl_weight",
        "tower1_paired_raw_consistency_weight",
    }
    for name in _ACTIVATION_GUARDED_FIELDS[task]:
        if name not in values:
            continue
        if (
            name in inactive_paired_children
            and float(getattr(args, "tower1_paired_consistency_weight")) <= 0.0
        ):
            continue
        expected_enabled = float(getattr(args, name)) > 0.0
        actual_enabled = float(values[name]) > 0.0
        if actual_enabled != expected_enabled:
            raise ValueError(
                f"{name} may change magnitude but cannot change shared-method "
                f"activation ({expected_enabled} -> {actual_enabled})"
            )
    overrides: dict[str, Any] = {}
    for name, value in values.items():
        _set(args, name, value, overrides)
    return overrides


def effective_shared_core_sha256(
    payload: dict[str, Any],
    task: str,
    training_hyperparameter_overrides: dict[str, Any] | None = None,
) -> str:
    """Fingerprint the executed config while retaining one method identity."""
    independent = dict(training_hyperparameter_overrides or {})
    if not independent:
        return str(payload["config_sha256"])
    return canonical_sha256(
        {
            "shared_method_sha256": payload["config_sha256"],
            "task": task,
            "training_hyperparameter_overrides": independent,
        }
    )


def resolve_dataset_training_hyperparameters(
    payload: dict[str, Any],
    task: str,
    dataset: str,
    explicit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge validation-frozen numeric defaults with explicit run overrides."""
    task_key = {
        "packet-level": "packet-level-classification",
        "flow-level": "flow-level-classification",
    }.get(task)
    if task_key is None:
        raise ValueError(f"unsupported shared-core task: {task}")
    stored = (
        (payload.get("dataset_numeric_hyperparameter_overrides") or {})
        .get(task_key, {})
        .get(dataset, {})
    )
    name_map = {
        "packet-level": {"class_weight_strength": "class_weight_strength"},
        "flow-level": {"class_weight_strength": "tower1_class_weight_strength"},
    }
    resolved = {
        name_map[task][name]: value for name, value in stored.items()
    }
    resolved.update(explicit or {})
    unsupported = sorted(set(resolved) - INDEPENDENT_TRAINING_HYPERPARAMETERS[task])
    if unsupported:
        raise ValueError(
            f"unsupported independent training hyperparameters for {task}: {unsupported}"
        )
    return resolved


def _validate_activation_topology(
    task: str, payload: dict[str, Any], values: dict[str, Any]
) -> None:
    inactive_paired_children = {
        "tower1_paired_cls_weight",
        "tower1_paired_logit_kl_weight",
        "tower1_paired_raw_consistency_weight",
    }
    for name, (section, key) in _ACTIVATION_GUARDED_FIELDS[task].items():
        if name not in values:
            continue
        if (
            name in inactive_paired_children
            and float(payload["tower1"]["paired_consistency_weight"]) <= 0.0
        ):
            continue
        expected_enabled = float(payload[section][key]) > 0.0
        actual_enabled = float(values[name]) > 0.0
        if actual_enabled != expected_enabled:
            raise ValueError(
                f"{name} may change magnitude but cannot change shared-method "
                f"activation ({expected_enabled} -> {actual_enabled})"
            )


def load_frozen_shared_core(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    fingerprint = payload.get("config_sha256")
    unsigned = {key: value for key, value in payload.items() if key != "config_sha256"}
    if not fingerprint or fingerprint != canonical_sha256(unsigned):
        raise ValueError("frozen shared-core config fingerprint mismatch")
    if payload.get("schema") != SCHEMA or payload.get("status") != STATUS:
        raise ValueError("unsupported or non-frozen shared-core config")
    if set(payload.get("datasets") or []) != REQUIRED_DATASETS:
        raise ValueError("shared-core v2 must be frozen jointly for VPN and TLS-120")
    if (payload.get("selection_protocol") or {}).get("test_labels_used") is not False:
        raise ValueError("shared-core v2 selection must explicitly exclude test labels")
    contract = payload.get("task_contract") or {}
    if contract.get("dataset_specific_manual_options") is not False:
        raise ValueError("shared-core v2 cannot contain dataset-specific manual options")
    if contract.get("shared_packet_module_reuse") != (
        "architecture_and_representation_contract_only"
    ):
        raise ValueError("shared-core v2 reuses the Packet module architecture, not supervised weights")
    if contract.get("packet_task_training_source") != "packet_task_train_split_packets":
        raise ValueError("shared-core v2 Packet task must use its own training split")
    if contract.get("flow_task_training_source") != "flow_task_train_split_packets":
        raise ValueError("shared-core v2 Flow task must train its Packet module from Flow packets")
    if contract.get("cross_task_supervised_weights_reused") is not False:
        raise ValueError("shared-core v2 prohibits cross-task supervised weight reuse")
    core = payload.get("packet_core") or {}
    if core.get("encoder_class") != "ProtocolAwarePacketContentEncoder":
        raise ValueError("shared-core v2 requires ProtocolAwarePacketContentEncoder")
    if core.get("representation_encoder") != "SharedPacketRepresentationEncoder":
        raise ValueError("shared-core v2 requires SharedPacketRepresentationEncoder")
    if core.get("semantic_packet_context_policy") != "single_packet":
        raise ValueError(
            "shared-core v2 requires strict current-packet semantic prompts"
        )
    if core.get("mask_protocol_session_fields") is not True:
        raise ValueError("shared-core v2 requires deterministic protocol session-field masking")
    if int(core.get("structural_dim", -1)) != 13:
        raise ValueError("shared-core v2 requires the fixed 13-dimensional packet-local structure")
    if core.get("channel_fusion_base_mode") != "semantic_anchor":
        raise ValueError("shared-core v2 requires semantic-anchor channel fusion")
    if core.get("intervention_view_base_mode") != "symmetric_mean":
        raise ValueError("shared-core v2 requires symmetric intervention fusion")
    if int(core.get("num_field_types", -1)) != 9:
        raise ValueError("shared-core v2 requires the fixed nine-field protocol schema")
    native = payload.get("native_pretraining") or {}
    if native.get("protocol") != "native_flow_multitask_v1":
        raise ValueError("shared-core v2 requires native_flow_multitask_v1")
    required_native = {
        "max_packets",
        "flow_layers",
        "projection_dim",
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
    }
    missing_native = sorted(required_native - set(native))
    if missing_native:
        raise ValueError(
            f"shared-core v2 native pretraining contract is incomplete: {missing_native}"
        )
    risk = payload.get("empirical_risk") or {}
    if risk.get("content_group_loss_reduction") != "group_mean":
        raise ValueError("shared-core v2 requires content-group group_mean risk")
    tower1 = payload.get("tower1") or {}
    # Older A/B/C frozen configs predate the preregistered D1/D2 screen. Treat
    # their absent fields as the explicitly disabled control topology.
    tower1.setdefault("identity_safe_contrastive", False)
    tower1.setdefault("cross_scale_weight", 0.0)
    tower1.setdefault("cross_scale_temperature", 0.07)
    required_tower1 = {
        "base_model",
        "epochs",
        "max_steps",
        "packet_batch_size",
        "gradient_accumulation_steps",
        "gradient_checkpointing",
        "max_packet_length",
        "projection_dim",
        "cls_weight",
        "contrastive_weight",
        "same_flow_positive_weight",
        "same_label_positive_weight",
        "flow_proto_weight",
        "flow_proto_positive",
        "flow_proto_context",
        "temperature",
        "learning_rate",
        "head_learning_rate",
        "weight_decay",
        "lora_r",
        "lora_alpha",
        "lora_dropout",
        "dtype",
        "seed",
        "use_sft",
        "disable_packet_information_weights",
        "flow_balanced_packet_batches",
        "packets_per_flow",
        "packet_batch_scheduler",
        "class_weighting",
        "class_weight_beta",
        "class_weight_basis",
        "class_weight_strength",
        "paired_consistency_weight",
        "paired_cls_weight",
        "paired_logit_kl_weight",
        "paired_raw_consistency_weight",
        "identity_safe_contrastive",
        "cross_scale_weight",
        "cross_scale_temperature",
        "early_stop_patience",
        "init_checkpoint_dir",
        "init_adapter_only",
    }
    missing_tower1 = sorted(required_tower1 - set(tower1))
    if missing_tower1:
        raise ValueError(f"shared-core v2 Tower1 contract is incomplete: {missing_tower1}")
    if tower1["use_sft"] is not False:
        raise ValueError("shared-core v2 fixes Tower1 to packet objectives without SFT")
    if (
        int(tower1["epochs"]) != 8
        or int(tower1["max_steps"]) != 0
        or int(tower1["early_stop_patience"]) != 0
    ):
        raise ValueError("shared-core v2 requires the complete fixed eight-epoch schedule")
    if tower1["gradient_checkpointing"] is not True:
        raise ValueError("shared-core v2 requires gradient checkpointing")
    if int(tower1["projection_dim"]) != 256:
        raise ValueError("shared-core v2 requires the fixed 256-dimensional Tower1 projection")
    if tower1["init_checkpoint_dir"] or tower1["init_adapter_only"] is not False:
        raise ValueError("shared-core v2 Tower1 must initialize independently from the base model")
    if tower1["disable_packet_information_weights"] is not True:
        raise ValueError("shared-core v2 disables packet shortcut information weights")
    if tower1["flow_balanced_packet_batches"] is not True or int(
        tower1["packets_per_flow"]
    ) != 2:
        raise ValueError("shared-core v2 requires two-packet flow-balanced batches")
    if tower1["packet_batch_scheduler"] != "epoch_resampled_dataloader_v1":
        raise ValueError("shared-core v2 requires fresh deterministic sampling each epoch")
    if float(tower1["weight_decay"]) != 0.01:
        raise ValueError("shared-core v2 requires the fixed Tower1 weight decay")
    if float(tower1["class_weight_beta"]) != 0.9999:
        raise ValueError("shared-core v2 requires the fixed effective-number beta")
    numeric = payload.get("dataset_numeric_hyperparameter_overrides") or {}
    if numeric:
        expected_tasks = {
            "packet-level-classification",
            "flow-level-classification",
        }
        if set(numeric) != expected_tasks or tower1["class_weight_basis"] != "flow":
            raise ValueError("invalid dataset numeric hierarchy override topology")
        for task_name, datasets in numeric.items():
            if set(datasets) != REQUIRED_DATASETS:
                raise ValueError(f"{task_name} numeric overrides do not cover VPN/TLS")
            for dataset, values in datasets.items():
                if set(values) != {"class_weight_strength"}:
                    raise ValueError(
                        f"unsupported numeric hierarchy fields for {task_name} {dataset}"
                    )
                strength = float(values["class_weight_strength"])
                if not 0.0 <= strength <= 1.0:
                    raise ValueError(
                        f"class-weight strength outside [0,1] for {task_name} {dataset}"
                    )
    extraction = payload.get("embedding_extraction") or {}
    if extraction.get("scheduler") != "cross_flow_length_bucketed_v1":
        raise ValueError("shared-core v2 requires the frozen length-bucketed scheduler")
    if extraction.get("embedding_mode") != "concat":
        raise ValueError("shared-core v2 requires raw/projected concatenated embeddings")
    if int(extraction.get("batch_size", 0)) <= 0:
        raise ValueError("shared-core v2 requires a positive embedding batch size")
    if int(extraction.get("flow_batch_packets", 0)) <= 0:
        raise ValueError("shared-core v2 requires positive cross-flow embedding buffering")
    return payload


def _set(args, name: str, value: Any, overrides: dict[str, Any]) -> None:
    if not hasattr(args, name):
        raise ValueError(f"runner does not expose required shared-core setting: {name}")
    old = getattr(args, name)
    setattr(args, name, value)
    overrides[name] = {"old": old, "new": value}


def apply_frozen_shared_core(
    args,
    task: str,
    payload: dict[str, Any],
    *,
    training_hyperparameter_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if task not in {"packet-level", "flow-level"}:
        raise ValueError(f"unsupported shared-core task: {task}")
    core = payload["packet_core"]
    native = payload["native_pretraining"]
    tower1 = payload["tower1"]
    extraction = payload["embedding_extraction"]
    risk = payload["empirical_risk"]
    overrides: dict[str, Any] = {}

    if task == "packet-level":
        mappings = {
            "packet_context_policy": core["semantic_packet_context_policy"],
            "base_model": tower1["base_model"],
            "epochs": tower1["epochs"],
            "packet_batch_size": tower1["packet_batch_size"],
            "gradient_accumulation_steps": tower1["gradient_accumulation_steps"],
            "tower1_early_stop_patience": tower1["early_stop_patience"],
            "init_checkpoint_dir": tower1["init_checkpoint_dir"],
            "init_adapter_only": tower1["init_adapter_only"],
            "max_packet_length": tower1["max_packet_length"],
            "semantic_embedding_mode": extraction["embedding_mode"],
            "semantic_embedding_batch_size": extraction["batch_size"],
            "semantic_embedding_flow_batch_packets": extraction[
                "flow_batch_packets"
            ],
            "cls_weight": tower1["cls_weight"],
            "contrastive_weight": tower1["contrastive_weight"],
            "same_flow_positive_weight": tower1["same_flow_positive_weight"],
            "same_label_positive_weight": tower1["same_label_positive_weight"],
            "flow_proto_weight": tower1["flow_proto_weight"],
            "flow_proto_positive": tower1["flow_proto_positive"],
            "flow_proto_context": tower1["flow_proto_context"],
            "temperature": tower1["temperature"],
            "lr": tower1["learning_rate"],
            "head_lr": tower1["head_learning_rate"],
            "lora_r": tower1["lora_r"],
            "lora_alpha": tower1["lora_alpha"],
            "lora_dropout": tower1["lora_dropout"],
            "dtype": tower1["dtype"],
            "seed": tower1["seed"],
            "byte_max_bytes": core["max_bytes"],
            "byte_hidden_dim": core["hidden_dim"],
            "byte_num_layers": core["num_layers"],
            "byte_num_heads": core["num_heads"],
            "byte_dropout": core["dropout"],
            "byte_exact_shared_representation": True,
            "byte_mask_protocol_session_fields": True,
            "channel_fusion_base_mode": core["channel_fusion_base_mode"],
            "channel_fusion_max_weight": core["channel_fusion_max_weight"],
            "intervention_view_base_mode": core["intervention_view_base_mode"],
            "intervention_max_residual_weight": core[
                "intervention_max_residual_weight"
            ],
            "pretrain_protocol_content": True,
            "protocol_pretrain_max_packets": native["max_packets"],
            "protocol_pretrain_flow_layers": native["flow_layers"],
            "protocol_pretrain_projection_dim": native["projection_dim"],
            "protocol_pretrain_epochs": native["epochs"],
            "protocol_pretrain_batch_size": native["batch_size"],
            "protocol_pretrain_eval_batch_size": native["eval_batch_size"],
            "protocol_pretrain_num_workers": native["num_workers"],
            "protocol_pretrain_learning_rate": native["learning_rate"],
            "protocol_pretrain_weight_decay": native["weight_decay"],
            "protocol_pretrain_field_mask_probability": native["field_mask_probability"],
            "protocol_pretrain_payload_dropout_probability": native[
                "payload_dropout_probability"
            ],
            "protocol_pretrain_session_mask_probability": native["session_mask_probability"],
            "protocol_pretrain_masked_byte_weight": native["masked_byte_weight"],
            "protocol_pretrain_relative_order_weight": native["relative_order_weight"],
            "protocol_pretrain_same_flow_weight": native["same_flow_weight"],
            "protocol_pretrain_next_length_weight": native["next_length_weight"],
            "protocol_pretrain_next_iat_weight": native["next_iat_weight"],
            "protocol_pretrain_direction_weight": native["direction_weight"],
            "protocol_pretrain_packet_consistency_weight": native[
                "packet_consistency_weight"
            ],
            "protocol_pretrain_flow_contrastive_weight": native[
                "flow_contrastive_weight"
            ],
            "protocol_pretrain_temperature": native["temperature"],
            "protocol_pretrain_patience": native["patience"],
            "protocol_pretrain_seed": native["seed"],
            "byte_content_group_loss_reduction": risk["content_group_loss_reduction"],
            "class_weighting": tower1["class_weighting"],
            "class_weight_basis": tower1["class_weight_basis"],
            "class_weight_strength": tower1["class_weight_strength"],
            "tower1_paired_consistency_weight": tower1["paired_consistency_weight"],
            "tower1_paired_cls_weight": tower1["paired_cls_weight"],
            "tower1_paired_logit_kl_weight": tower1["paired_logit_kl_weight"],
            "tower1_paired_raw_consistency_weight": tower1[
                "paired_raw_consistency_weight"
            ],
            "identity_safe_contrastive": tower1["identity_safe_contrastive"],
            "tower1_cross_scale_weight": tower1["cross_scale_weight"],
            "tower1_cross_scale_temperature": tower1[
                "cross_scale_temperature"
            ],
        }
    else:
        mappings = {
            "packet_context_policy": core["semantic_packet_context_policy"],
            "base_model": tower1["base_model"],
            "tower1_epochs": tower1["epochs"],
            "tower1_max_steps": tower1["max_steps"],
            "packet_batch_size": tower1["packet_batch_size"],
            "tower1_gradient_accumulation_steps": tower1[
                "gradient_accumulation_steps"
            ],
            "gradient_checkpointing": tower1["gradient_checkpointing"],
            "tower1_early_stop_patience": tower1["early_stop_patience"],
            "tower1_init_checkpoint_dir": tower1["init_checkpoint_dir"],
            "max_packet_length": tower1["max_packet_length"],
            "embedding_mode": extraction["embedding_mode"],
            "embedding_batch_size": extraction["batch_size"],
            "embedding_flow_batch_packets": extraction["flow_batch_packets"],
            "cls_weight": tower1["cls_weight"],
            "contrastive_weight": tower1["contrastive_weight"],
            "same_flow_positive_weight": tower1["same_flow_positive_weight"],
            "same_label_positive_weight": tower1["same_label_positive_weight"],
            "flow_proto_weight": tower1["flow_proto_weight"],
            "flow_proto_positive": tower1["flow_proto_positive"],
            "flow_proto_context": tower1["flow_proto_context"],
            "temperature": tower1["temperature"],
            "tower1_lr": tower1["learning_rate"],
            "tower1_head_lr": tower1["head_learning_rate"],
            "lora_r": tower1["lora_r"],
            "lora_alpha": tower1["lora_alpha"],
            "lora_dropout": tower1["lora_dropout"],
            "dtype": tower1["dtype"],
            "tower1_use_sft": tower1["use_sft"],
            "tower1_disable_packet_information_weights": tower1[
                "disable_packet_information_weights"
            ],
            "flow_balanced_packet_batches": tower1[
                "flow_balanced_packet_batches"
            ],
            "packets_per_flow": tower1["packets_per_flow"],
            "model_types": "seq",
            "exact_shared_packet_encoder": True,
            "shared_packet_hidden_dim": core["hidden_dim"],
            "meta_feature_dim": core["structural_dim"],
            "native_structural_dim": core["hidden_dim"],
            "dual_channel_mode": "residual",
            "channel_fusion_base_mode": core["channel_fusion_base_mode"],
            "dual_channel_max_weight": core["channel_fusion_max_weight"],
            "use_intervention_views": True,
            "intervention_view_base_mode": core["intervention_view_base_mode"],
            "intervention_max_residual_weight": core[
                "intervention_max_residual_weight"
            ],
            "native_max_bytes": core["max_bytes"],
            "native_hidden_dim": core["hidden_dim"],
            "native_byte_layers": core["num_layers"],
            "native_num_heads": core["num_heads"],
            "native_dropout": core["dropout"],
            "native_max_packets": native["max_packets"],
            "native_flow_layers": native["flow_layers"],
            "native_projection_dim": native["projection_dim"],
            "native_epochs": native["epochs"],
            "native_batch_size": native["batch_size"],
            "native_eval_batch_size": native["eval_batch_size"],
            "native_num_workers": native["num_workers"],
            "native_learning_rate": native["learning_rate"],
            "native_weight_decay": native["weight_decay"],
            "native_field_mask_probability": native["field_mask_probability"],
            "native_payload_dropout_probability": native["payload_dropout_probability"],
            "native_session_mask_probability": native["session_mask_probability"],
            "native_masked_byte_weight": native["masked_byte_weight"],
            "native_relative_order_weight": native["relative_order_weight"],
            "native_same_flow_weight": native["same_flow_weight"],
            "native_next_length_weight": native["next_length_weight"],
            "native_next_iat_weight": native["next_iat_weight"],
            "native_direction_weight": native["direction_weight"],
            "native_packet_consistency_weight": native["packet_consistency_weight"],
            "native_flow_contrastive_weight": native["flow_contrastive_weight"],
            "native_temperature": native["temperature"],
            "native_patience": native["patience"],
            "seed": native["seed"],
            "content_group_loss_reduction": risk["content_group_loss_reduction"],
            "tower1_class_weighting": tower1["class_weighting"],
            "tower1_class_weight_basis": tower1["class_weight_basis"],
            "tower1_class_weight_strength": tower1["class_weight_strength"],
            "tower1_paired_consistency_weight": tower1["paired_consistency_weight"],
            "tower1_paired_cls_weight": tower1["paired_cls_weight"],
            "tower1_paired_logit_kl_weight": tower1["paired_logit_kl_weight"],
            "tower1_paired_raw_consistency_weight": tower1[
                "paired_raw_consistency_weight"
            ],
            "identity_safe_contrastive": tower1["identity_safe_contrastive"],
            "tower1_cross_scale_weight": tower1["cross_scale_weight"],
            "tower1_cross_scale_temperature": tower1[
                "cross_scale_temperature"
            ],
        }
    for name, value in mappings.items():
        _set(args, name, value, overrides)
    independent = dict(training_hyperparameter_overrides or {})
    unsupported = sorted(
        set(independent) - INDEPENDENT_TRAINING_HYPERPARAMETERS[task]
    )
    if unsupported:
        raise ValueError(
            f"unsupported independent training hyperparameters for {task}: {unsupported}"
        )
    _validate_activation_topology(task, payload, independent)
    for name, value in independent.items():
        _set(args, name, value, overrides)
    return overrides
