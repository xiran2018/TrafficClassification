import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from freeze_shared_core_v2_config import canonical_sha256
from audit_flow_embeddings import sha256_file
from run_packet_level_pipeline import (
    semantic_cache_policy_evidence,
    verify_reused_protocol_content_checkpoint,
)
from shared_core_v2 import (
    apply_frozen_shared_core,
    capture_training_hyperparameter_overrides,
    effective_shared_core_sha256,
    load_frozen_shared_core,
    restore_profile_training_hyperparameters,
)


ROOT = Path(__file__).resolve().parents[1]


def test_packet_semantic_cache_policy_evidence_requires_every_split_and_view(tmp_path):
    args = SimpleNamespace(
        packet_context_policy="single_packet",
        semantic_embedding_mode="concat",
        semantic_embedding_batch_size=8,
        semantic_embedding_flow_batch_packets=128,
    )
    for split in ("train", "valid", "test"):
        for suffix, header in (("_full", "full"), ("_mask_ip_port", "mask_ip_port")):
            path = tmp_path / f"{split}_semantic_embeddings{suffix}_manifest.json"
            path.write_text(
                json.dumps(
                    {
                        "header_policy": header,
                        "packet_context_policy": "single_packet",
                        "embedding_mode": "concat",
                        "embedding_scheduler": "cross_flow_length_bucketed_v1",
                        "embedding_batch_size": 8,
                        "embedding_flow_batch_packets": 128,
                    }
                ),
                encoding="utf-8",
            )

    evidence = semantic_cache_policy_evidence(args, tmp_path)
    assert evidence["verified"] is False
    assert evidence["embedding_audits_verified"] is False

    audit_template = {
        "schema": "flow_embedding_audit_v1",
        "status": "pass",
        "require_finite": True,
        "counts": {
            "input_flows": 1,
            "input_packets": 2,
            "output_rows": 1,
            "unique_output_flows": 1,
            "output_packets": 2,
        },
        "integrity": {"missing_flows": 0, "bad_embeddings": 0},
    }
    for split in ("train", "valid", "test"):
        for policy in ("full", "mask_ip_port"):
            audit_dir = tmp_path / f"{split}_semantic_flow_embeddings_{policy}"
            audit_dir.mkdir()
            packet_index = audit_dir / "packet_index.jsonl"
            embedding_index = audit_dir / "flow_embedding_index.jsonl"
            embedding_config = audit_dir / "embedding_config.json"
            packet_index.write_text("packet\n", encoding="utf-8")
            embedding_index.write_text("embedding\n", encoding="utf-8")
            embedding_config.write_text("{}\n", encoding="utf-8")
            audit_payload = {
                **audit_template,
                "inputs": {
                    "packet_index": str(packet_index),
                    "packet_index_sha256": sha256_file(packet_index),
                    "embedding_index": str(embedding_index),
                    "embedding_index_sha256": sha256_file(embedding_index),
                    "embedding_config": str(embedding_config),
                    "embedding_config_sha256": sha256_file(embedding_config),
                },
            }
            (audit_dir / "embedding_audit.json").write_text(
                json.dumps(audit_payload), encoding="utf-8"
            )

    evidence = semantic_cache_policy_evidence(args, tmp_path)
    assert evidence["verified"] is True
    assert evidence["embedding_audits_verified"] is True
    assert evidence["splits"]["valid"]["intervened"]["verified"] is True

    drifted = tmp_path / "test_semantic_embeddings_mask_ip_port_manifest.json"
    payload = json.loads(drifted.read_text(encoding="utf-8"))
    payload["embedding_scheduler"] = "legacy_per_flow_v1"
    drifted.write_text(json.dumps(payload), encoding="utf-8")
    evidence = semantic_cache_policy_evidence(args, tmp_path)
    assert evidence["verified"] is False
    assert evidence["splits"]["test"]["intervened"]["verified"] is False

    payload["embedding_scheduler"] = "cross_flow_length_bucketed_v1"
    payload["packet_context_policy"] = "flow_context"
    drifted.write_text(json.dumps(payload), encoding="utf-8")
    evidence = semantic_cache_policy_evidence(args, tmp_path)
    assert evidence["verified"] is False
    assert evidence["splits"]["test"]["intervened"]["verified"] is False


def frozen_payload():
    payload = {
        "schema": "exact_shared_packet_core_v2",
        "status": "frozen_from_cross_dataset_validation",
        "datasets": ["tls-120", "vpn-app"],
        "selection_protocol": {"test_labels_used": False},
        "packet_core": {
            "encoder_class": "ProtocolAwarePacketContentEncoder",
            "max_bytes": 128,
            "hidden_dim": 128,
            "num_layers": 2,
            "num_heads": 4,
            "dropout": 0.1,
            "num_field_types": 9,
            "representation_encoder": "SharedPacketRepresentationEncoder",
            "semantic_packet_context_policy": "single_packet",
            "mask_protocol_session_fields": True,
            "structural_dim": 13,
            "channel_fusion_base_mode": "semantic_anchor",
            "channel_fusion_max_weight": 0.25,
            "intervention_view_base_mode": "symmetric_mean",
            "intervention_max_residual_weight": 0.25,
        },
        "native_pretraining": {
            "protocol": "native_flow_multitask_v1",
            "max_packets": 64,
            "flow_layers": 2,
            "projection_dim": 128,
            "epochs": 20,
            "batch_size": 8,
            "eval_batch_size": 16,
            "num_workers": 0,
            "learning_rate": 0.0003,
            "weight_decay": 0.01,
            "field_mask_probability": 0.2,
            "payload_dropout_probability": 0.5,
            "session_mask_probability": 1.0,
            "masked_byte_weight": 1.0,
            "relative_order_weight": 0.25,
            "same_flow_weight": 0.25,
            "next_length_weight": 0.2,
            "next_iat_weight": 0.2,
            "direction_weight": 0.1,
            "packet_consistency_weight": 0.25,
            "flow_contrastive_weight": 0.25,
            "temperature": 0.1,
            "patience": 4,
            "seed": 42,
        },
        "empirical_risk": {"content_group_loss_reduction": "group_mean"},
        "tower1": {
            "base_model": "Qwen/Qwen2.5-7B-Instruct",
            "epochs": 8,
            "max_steps": 0,
            "packet_batch_size": 16,
            "gradient_accumulation_steps": 1,
            "gradient_checkpointing": True,
            "max_packet_length": 1024,
            "projection_dim": 256,
            "cls_weight": 1.0,
            "contrastive_weight": 0.1,
            "same_flow_positive_weight": 1.0,
            "same_label_positive_weight": 1.0,
            "flow_proto_weight": 0.0,
            "flow_proto_positive": "same_class",
            "flow_proto_context": "inclusive",
            "temperature": 0.07,
            "learning_rate": 1e-5,
            "head_learning_rate": 1e-4,
            "weight_decay": 0.01,
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "dtype": "float16",
            "seed": 42,
            "use_sft": False,
            "disable_packet_information_weights": True,
            "flow_balanced_packet_batches": True,
            "packets_per_flow": 2,
            "packet_batch_scheduler": "epoch_resampled_dataloader_v1",
            "class_weighting": "effective",
            "class_weight_beta": 0.9999,
            "class_weight_basis": "flow",
            "class_weight_strength": 0.5,
            "paired_consistency_weight": 0.05,
            "paired_cls_weight": 0.2,
            "paired_logit_kl_weight": 0.5,
            "paired_raw_consistency_weight": 1.0,
            "paired_validation_selection": "worst_view_macro_f1",
            "early_stop_patience": 0,
            "init_checkpoint_dir": "",
            "init_adapter_only": False,
        },
        "embedding_extraction": {
            "scheduler": "cross_flow_length_bucketed_v1",
            "embedding_mode": "concat",
            "batch_size": 8,
            "flow_batch_packets": 128,
        },
        "task_contract": {
            "shared_packet_module_reuse": "architecture_and_representation_contract_only",
            "packet_task_training_source": "packet_task_train_split_packets",
            "flow_task_training_source": "flow_task_train_split_packets",
            "cross_task_supervised_weights_reused": False,
            "dataset_specific_manual_options": False,
        },
    }
    payload["config_sha256"] = canonical_sha256(payload)
    return payload


def write_payload(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_frozen_core_rejects_cross_task_supervised_weight_reuse(tmp_path):
    payload = frozen_payload()
    payload["task_contract"]["cross_task_supervised_weights_reused"] = True
    unsigned = {key: value for key, value in payload.items() if key != "config_sha256"}
    payload["config_sha256"] = canonical_sha256(unsigned)

    with pytest.raises(ValueError, match="cross-task supervised weight reuse"):
        load_frozen_shared_core(write_payload(tmp_path / "invalid.json", payload))


def packet_args():
    names = {
        "packet_context_policy": "auto",
        "byte_max_bytes": 1,
        "byte_hidden_dim": 1,
        "byte_num_layers": 1,
        "byte_num_heads": 1,
        "byte_dropout": 0.0,
        "byte_exact_shared_representation": False,
        "byte_mask_protocol_session_fields": False,
        "channel_fusion_base_mode": "legacy",
        "channel_fusion_max_weight": 1.0,
        "intervention_view_base_mode": "factual_anchor",
        "intervention_max_residual_weight": 1.0,
        "pretrain_protocol_content": False,
        "protocol_pretrain_max_packets": 1,
        "protocol_pretrain_flow_layers": 1,
        "protocol_pretrain_projection_dim": 1,
        "protocol_pretrain_epochs": 1,
        "protocol_pretrain_batch_size": 1,
        "protocol_pretrain_eval_batch_size": 1,
        "protocol_pretrain_num_workers": 1,
        "protocol_pretrain_learning_rate": 0.1,
        "protocol_pretrain_weight_decay": 0.1,
        "protocol_pretrain_field_mask_probability": 0.1,
        "protocol_pretrain_payload_dropout_probability": 0.1,
        "protocol_pretrain_session_mask_probability": 0.1,
        "protocol_pretrain_masked_byte_weight": 0.1,
        "protocol_pretrain_relative_order_weight": 0.1,
        "protocol_pretrain_same_flow_weight": 0.1,
        "protocol_pretrain_next_length_weight": 0.1,
        "protocol_pretrain_next_iat_weight": 0.1,
        "protocol_pretrain_direction_weight": 0.1,
        "protocol_pretrain_packet_consistency_weight": 0.1,
        "protocol_pretrain_flow_contrastive_weight": 0.1,
        "protocol_pretrain_temperature": 0.2,
        "protocol_pretrain_patience": 1,
        "protocol_pretrain_seed": 1,
        "byte_content_group_loss_reduction": "none",
        "class_weighting": "none",
        "class_weight_basis": "packet",
        "class_weight_strength": 1.0,
        "tower1_paired_consistency_weight": 0.0,
        "tower1_paired_cls_weight": 0.0,
        "tower1_paired_logit_kl_weight": 0.0,
        "tower1_paired_raw_consistency_weight": 0.0,
        "tower1_paired_validation_selection": "disabled",
        "base_model": "other",
        "epochs": 1,
        "packet_batch_size": 1,
        "gradient_accumulation_steps": 2,
        "max_packet_length": 1,
        "semantic_embedding_mode": "raw",
        "semantic_embedding_batch_size": 1,
        "semantic_embedding_flow_batch_packets": 1,
        "cls_weight": 0.0,
        "contrastive_weight": 0.0,
        "same_flow_positive_weight": 0.0,
        "same_label_positive_weight": 0.0,
        "flow_proto_weight": 1.0,
        "flow_proto_positive": "own_flow",
        "flow_proto_context": "leave_one_out",
        "temperature": 1.0,
        "lr": 0.1,
        "head_lr": 0.1,
        "lora_r": 1,
        "lora_alpha": 1,
        "lora_dropout": 0.0,
        "dtype": "float32",
        "seed": 1,
        "tower1_early_stop_patience": 3,
        "init_checkpoint_dir": "old-checkpoint",
        "init_adapter_only": True,
        "packet_batch_scheduler": "epoch_resampled_dataloader_v1",
    }
    return SimpleNamespace(**names)


def test_load_rejects_tampered_frozen_config(tmp_path):
    payload = frozen_payload()
    payload["packet_core"]["num_layers"] = 3
    with pytest.raises(ValueError, match="fingerprint"):
        load_frozen_shared_core(write_payload(tmp_path / "tampered.json", payload))


def test_development_config_covers_six_packet_and_two_flow_datasets(tmp_path):
    payload = frozen_payload()
    packet_datasets = {
        "vpn-app",
        "vpn-binary",
        "vpn-service",
        "tls-120",
        "ustc-app",
        "ustc-binary",
    }
    flow_datasets = {"vpn-app", "tls-120"}
    payload["status"] = "frozen_for_development_milestone"
    payload["datasets"] = sorted(packet_datasets | flow_datasets)
    payload["task_datasets"] = {
        "packet-level-classification": sorted(packet_datasets),
        "flow-level-classification": sorted(flow_datasets),
    }
    payload["selection_protocol"].update(
        {
            "selection_datasets": sorted(flow_datasets),
            "application_datasets_by_task": payload["task_datasets"],
        }
    )
    unsigned = {key: value for key, value in payload.items() if key != "config_sha256"}
    payload["config_sha256"] = canonical_sha256(unsigned)

    loaded = load_frozen_shared_core(
        write_payload(tmp_path / "development.json", payload)
    )

    assert loaded["status"] == "frozen_for_development_milestone"
    assert set(loaded["task_datasets"]["packet-level-classification"]) == (
        packet_datasets
    )


def test_load_rejects_non_reference_epoch_budget_even_with_valid_hash(tmp_path):
    payload = frozen_payload()
    payload["tower1"]["epochs"] = 12
    unsigned = {key: value for key, value in payload.items() if key != "config_sha256"}
    payload["config_sha256"] = canonical_sha256(unsigned)
    with pytest.raises(ValueError, match="fixed eight-epoch schedule"):
        load_frozen_shared_core(write_payload(tmp_path / "epochs.json", payload))


def test_packet_runtime_receives_the_entire_frozen_shared_core(tmp_path):
    payload = frozen_payload()
    loaded = load_frozen_shared_core(write_payload(tmp_path / "frozen.json", payload))
    args = packet_args()
    apply_frozen_shared_core(args, "packet-level", loaded)
    assert args.byte_max_bytes == 128
    assert args.packet_context_policy == "single_packet"
    assert args.byte_num_layers == 2
    assert args.byte_dropout == 0.1
    assert args.pretrain_protocol_content is True
    assert args.byte_exact_shared_representation is True
    assert args.byte_mask_protocol_session_fields is True
    assert args.channel_fusion_base_mode == "semantic_anchor"
    assert args.intervention_view_base_mode == "symmetric_mean"
    assert args.byte_content_group_loss_reduction == "group_mean"
    assert args.class_weight_basis == "flow"
    assert args.tower1_paired_consistency_weight == 0.05
    assert args.tower1_paired_validation_selection == "worst_view_macro_f1"
    assert args.semantic_embedding_mode == "concat"
    assert args.semantic_embedding_batch_size == 8
    assert args.semantic_embedding_flow_batch_packets == 128


def test_packet_runtime_allows_explicit_independent_optimization_values():
    payload = frozen_payload()
    args = packet_args()
    args.epochs = 12
    args.lr = 2e-5
    args.contrastive_weight = 0.2
    selected = capture_training_hyperparameter_overrides(
        args, "packet-level", "epochs,lr,contrastive_weight"
    )

    apply_frozen_shared_core(
        args,
        "packet-level",
        payload,
        training_hyperparameter_overrides=selected,
    )

    assert args.epochs == 12
    assert args.lr == 2e-5
    assert args.contrastive_weight == 0.2
    assert args.lora_r == payload["tower1"]["lora_r"]
    assert args.packet_context_policy == "single_packet"


def test_independent_hyperparameters_cannot_disable_shared_objective():
    payload = frozen_payload()
    args = packet_args()
    args.contrastive_weight = 0.0
    selected = capture_training_hyperparameter_overrides(
        args, "packet-level", "contrastive_weight"
    )
    with pytest.raises(ValueError, match="cannot change shared-method activation"):
        apply_frozen_shared_core(
            args,
            "packet-level",
            payload,
            training_hyperparameter_overrides=selected,
        )


def test_independent_hyperparameter_allowlist_rejects_architecture_switch():
    args = packet_args()
    with pytest.raises(ValueError, match="unsupported independent"):
        capture_training_hyperparameter_overrides(
            args, "packet-level", "lora_r"
        )


def test_flow_runtime_uses_the_same_core_values():
    payload = frozen_payload()
    packet = packet_args()
    apply_frozen_shared_core(packet, "packet-level", payload)
    flow = SimpleNamespace(
        packet_context_policy="auto",
        base_model="other",
        tower1_epochs=1,
        tower1_max_steps=5,
        packet_batch_size=1,
        tower1_gradient_accumulation_steps=2,
        gradient_checkpointing=False,
        tower1_early_stop_patience=3,
        tower1_init_checkpoint_dir="old-checkpoint",
        max_packet_length=1,
        embedding_mode="raw",
        embedding_batch_size=1,
        embedding_flow_batch_packets=1,
        cls_weight=0.0,
        contrastive_weight=0.0,
        same_flow_positive_weight=0.0,
        same_label_positive_weight=0.0,
        flow_proto_weight=1.0,
        flow_proto_positive="own_flow",
        flow_proto_context="leave_one_out",
        temperature=1.0,
        tower1_lr=0.1,
        tower1_head_lr=0.1,
        lora_r=1,
        lora_alpha=1,
        lora_dropout=0.0,
        dtype="float32",
        tower1_use_sft=True,
        tower1_disable_packet_information_weights=False,
        flow_balanced_packet_batches=False,
        packets_per_flow=1,
        packet_batch_scheduler="epoch_resampled_dataloader_v1",
        model_types="graph,seq",
        exact_shared_packet_encoder=False,
        shared_packet_hidden_dim=1,
        meta_feature_dim=14,
        native_structural_dim=0,
        dual_channel_mode="concat",
        channel_fusion_base_mode="legacy",
        dual_channel_max_weight=1.0,
        use_intervention_views=False,
        intervention_view_base_mode="factual_anchor",
        intervention_max_residual_weight=1.0,
        native_max_bytes=1,
        native_hidden_dim=1,
        native_byte_layers=1,
        native_num_heads=1,
        native_dropout=0.0,
        native_max_packets=1,
        native_flow_layers=1,
        native_projection_dim=1,
        native_epochs=1,
        native_batch_size=1,
        native_eval_batch_size=1,
        native_num_workers=1,
        native_learning_rate=0.1,
        native_weight_decay=0.1,
        native_field_mask_probability=0.1,
        native_payload_dropout_probability=0.1,
        native_session_mask_probability=0.1,
        native_masked_byte_weight=0.1,
        native_relative_order_weight=0.1,
        native_same_flow_weight=0.1,
        native_next_length_weight=0.1,
        native_next_iat_weight=0.1,
        native_direction_weight=0.1,
        native_packet_consistency_weight=0.1,
        native_flow_contrastive_weight=0.1,
        native_temperature=0.2,
        native_patience=1,
        seed=1,
        content_group_loss_reduction="none",
        tower1_class_weighting="none",
        tower1_class_weight_basis="packet",
        tower1_class_weight_strength=1.0,
        tower1_paired_consistency_weight=0.0,
        tower1_paired_cls_weight=0.0,
        tower1_paired_logit_kl_weight=0.0,
        tower1_paired_raw_consistency_weight=0.0,
        tower1_paired_validation_selection="disabled",
    )
    apply_frozen_shared_core(flow, "flow-level", payload)
    assert flow.model_types == "seq"
    assert flow.packet_context_policy == packet.packet_context_policy == "single_packet"
    assert flow.exact_shared_packet_encoder is True
    assert flow.shared_packet_hidden_dim == packet.byte_hidden_dim
    assert flow.meta_feature_dim == 13
    assert flow.native_structural_dim == 128
    assert flow.dual_channel_mode == "residual"
    assert flow.channel_fusion_base_mode == "semantic_anchor"
    assert flow.use_intervention_views is True
    assert flow.native_max_bytes == packet.byte_max_bytes
    assert flow.native_hidden_dim == packet.byte_hidden_dim
    assert flow.native_byte_layers == packet.byte_num_layers
    assert flow.native_num_heads == packet.byte_num_heads
    assert flow.native_dropout == packet.byte_dropout
    assert flow.content_group_loss_reduction == packet.byte_content_group_loss_reduction
    assert flow.tower1_class_weight_basis == packet.class_weight_basis
    assert flow.base_model == packet.base_model
    assert flow.tower1_epochs == packet.epochs
    assert flow.packet_batch_size == packet.packet_batch_size
    assert flow.same_label_positive_weight == packet.same_label_positive_weight
    assert flow.lora_r == packet.lora_r
    assert flow.tower1_use_sft is False
    assert flow.flow_balanced_packet_batches is True
    assert flow.tower1_paired_validation_selection == (
        packet.tower1_paired_validation_selection
    )
    assert flow.tower1_paired_validation_selection == "worst_view_macro_f1"
    assert flow.embedding_mode == packet.semantic_embedding_mode == "concat"
    assert flow.embedding_batch_size == packet.semantic_embedding_batch_size == 8
    assert flow.embedding_flow_batch_packets == 128


def test_numeric_training_hyperparameters_can_differ_without_changing_core():
    payload = frozen_payload()
    packet = packet_args()
    packet.epochs = 12
    packet.lr = 2e-5
    packet.packet_batch_size = 32
    packet.class_weight_strength = 0.75
    independent = capture_training_hyperparameter_overrides(
        packet,
        "packet-level",
        "epochs,lr,packet_batch_size,class_weight_strength",
    )
    apply_frozen_shared_core(
        packet,
        "packet-level",
        payload,
        training_hyperparameter_overrides=independent,
    )
    assert packet.epochs == 12
    assert packet.lr == 2e-5
    assert packet.packet_batch_size == 32
    assert packet.class_weight_strength == 0.75
    assert packet.packet_context_policy == "single_packet"
    assert packet.lora_r == 16
    assert packet.byte_exact_shared_representation is True


def test_numeric_override_cannot_disable_objective_or_change_architecture():
    payload = frozen_payload()
    packet = packet_args()
    packet.contrastive_weight = 0.0
    independent = capture_training_hyperparameter_overrides(
        packet, "packet-level", "contrastive_weight"
    )
    with pytest.raises(ValueError, match="cannot change shared-method activation"):
        apply_frozen_shared_core(
            packet,
            "packet-level",
            payload,
            training_hyperparameter_overrides=independent,
        )

    with pytest.raises(ValueError, match="unsupported independent"):
        capture_training_hyperparameter_overrides(
            packet, "packet-level", "lora_r"
        )


def test_non_frozen_profile_restores_numeric_values_with_activation_guard():
    packet = packet_args()
    packet.lr = 2e-5
    packet.epochs = 12
    packet.contrastive_weight = 0.2
    values = capture_training_hyperparameter_overrides(
        packet, "packet-level", "lr,epochs,contrastive_weight"
    )
    packet.lr = 1e-5
    packet.epochs = 8
    packet.contrastive_weight = 0.1
    restored = restore_profile_training_hyperparameters(
        packet, "packet-level", values
    )
    assert packet.lr == 2e-5
    assert packet.epochs == 12
    assert packet.contrastive_weight == 0.2
    assert restored["lr"] == {"old": 1e-5, "new": 2e-5}

    packet.contrastive_weight = 0.1
    with pytest.raises(ValueError, match="cannot change shared-method activation"):
        restore_profile_training_hyperparameters(
            packet, "packet-level", {"contrastive_weight": 0.0}
        )

    packet.tower1_paired_consistency_weight = 0.0
    packet.tower1_paired_logit_kl_weight = 0.5
    restore_profile_training_hyperparameters(
        packet, "packet-level", {"tower1_paired_logit_kl_weight": 0.0}
    )
    assert packet.tower1_paired_logit_kl_weight == 0.0


def test_effective_config_fingerprint_separates_numeric_overrides_from_method():
    payload = frozen_payload()
    method_sha = payload["config_sha256"]
    assert effective_shared_core_sha256(payload, "packet-level") == method_sha
    packet_sha = effective_shared_core_sha256(
        payload, "packet-level", {"epochs": 12, "lr": 2e-5}
    )
    flow_sha = effective_shared_core_sha256(
        payload, "flow-level", {"tower1_epochs": 12, "tower1_lr": 2e-5}
    )
    assert packet_sha != method_sha
    assert flow_sha != method_sha
    assert packet_sha != flow_sha


def test_packet_runner_common_reference_keeps_method_and_effective_hash_equal(tmp_path):
    config = write_payload(tmp_path / "frozen.json", frozen_payload())
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "run_packet_level_pipeline.py"),
            "--dataset",
            "vpn-app",
            "--fold",
            "0",
            "--stage",
            "paper_unified",
            "--dry_run",
            "--artifact_root",
            str(tmp_path / "packet_artifacts"),
            "--checkpoint_root",
            str(tmp_path / "packet_checkpoints"),
            "--shared_core_config",
            str(config),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    pretrain = next(
        line for line in result.stdout.splitlines() if "pretrain_native_flow_encoder.py" in line
    )
    packet = next(
        line for line in result.stdout.splitlines() if "train_packet_byte_transformer.py" in line
    )
    tower1 = next(
        line for line in result.stdout.splitlines() if "train_tower1_multitask.py" in line
    )
    structural = next(
        line
        for line in result.stdout.splitlines()
        if "train_packet_feature_expert.py" in line
    )
    packet_test = next(
        line for line in result.stdout.splitlines() if "test_packet_byte_transformer.py" in line
    )
    semantic_extractions = [
        line
        for line in result.stdout.splitlines()
        if "extract_packet_embeddings_qwen.py" in line
    ]
    assert len(semantic_extractions) == 6
    assert all("--resume_existing" in line for line in semantic_extractions)
    assert all("--embedding_mode concat" in line for line in semantic_extractions)
    assert all("--flow_batch_packets 128" in line for line in semantic_extractions)
    for option in (
        "--max_bytes 128",
        "--hidden_dim 128",
        "--byte_layers 2",
        "--num_heads 4",
        "--dropout 0.1",
        "--learning_rate 0.0003",
        "--weight_decay 0.01",
        "--temperature 0.1",
        "--seed 42",
    ):
        assert option in pretrain
    assert "--content_group_loss_reduction group_mean" in packet
    assert "--required_semantic_packet_context_policy single_packet" in packet
    assert "--required_semantic_packet_context_policy single_packet" in packet_test
    assert "--mask_session_fields" in structural
    assert "--byte_prefix_len 64" in structural
    assert "--byte_prefix_len 32 64 128" not in structural
    assert "--min_samples_leaf 1" in structural
    assert "--min_samples_leaf 1 2" not in structural
    assert "--estimator_types random_forest" in structural
    assert "extra_trees" not in structural
    assert "--n_estimators 200" in structural
    manifest = json.loads(
        next((tmp_path / "packet_artifacts").glob("vpn-app/fold0/packet_framework_manifest.json")).read_text(
            encoding="utf-8"
        )
    )
    notes = manifest["framework"]["notes"]
    assert notes["shared_core_method_sha256"] == frozen_payload()["config_sha256"]
    assert notes["shared_core_config_sha256"] == notes["shared_core_method_sha256"]


def test_packet_runner_can_screen_valid_without_test_evaluation(tmp_path):
    config = write_payload(tmp_path / "frozen.json", frozen_payload())
    artifact_root = tmp_path / "packet_artifacts"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "run_packet_level_pipeline.py"),
            "--dataset",
            "vpn-app",
            "--fold",
            "0",
            "--stage",
            "paper_unified",
            "--dry_run",
            "--eval_splits",
            "valid",
            "--prepared_splits",
            "train,valid",
            "--artifact_root",
            str(artifact_root),
            "--checkpoint_root",
            str(tmp_path / "packet_checkpoints"),
            "--shared_core_config",
            str(config),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    evaluation_commands = [
        line
        for line in result.stdout.splitlines()
        if "test_packet_byte_transformer.py" in line
    ]
    assert len(evaluation_commands) == 1
    assert "valid/packet_index.jsonl" in evaluation_commands[0]
    assert "test/packet_index.jsonl" not in evaluation_commands[0]
    assert "/vpn-app/test" not in result.stdout
    semantic_extractions = [
        line
        for line in result.stdout.splitlines()
        if "extract_packet_embeddings_qwen.py" in line
    ]
    assert len(semantic_extractions) == 4
    audit_command = next(
        line for line in result.stdout.splitlines() if "audit_packet_flow_split.py" in line
    )
    assert "--test" not in audit_command

    manifest = json.loads(
        next(artifact_root.glob("vpn-app/fold0/packet_framework_manifest.json")).read_text(
            encoding="utf-8"
        )
    )
    notes = manifest["framework"]["notes"]
    assert notes["eval_splits"] == ["valid"]
    assert notes["prepared_splits"] == ["train", "valid"]
    assert notes["result_paths"] == [
        str(
            artifact_root
            / "vpn-app/fold0/valid_unified_packet_single_head.json"
        )
    ]


def test_packet_runner_can_reuse_verified_label_free_native_checkpoint(tmp_path):
    config = write_payload(tmp_path / "frozen.json", frozen_payload())
    reused = tmp_path / "native" / "best.pt"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "run_packet_level_pipeline.py"),
            "--dataset",
            "vpn-app",
            "--fold",
            "0",
            "--stage",
            "paper_unified",
            "--dry_run",
            "--prepared_splits",
            "train,valid",
            "--eval_splits",
            "valid",
            "--reuse_protocol_content_checkpoint",
            str(reused),
            "--artifact_root",
            str(tmp_path / "packet_artifacts"),
            "--checkpoint_root",
            str(tmp_path / "packet_checkpoints"),
            "--shared_core_config",
            str(config),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "pretrain_native_flow_encoder.py" not in result.stdout
    byte_train = next(
        line for line in result.stdout.splitlines() if "train_packet_byte_transformer.py" in line
    )
    assert f"--protocol_content_checkpoint {reused}" in byte_train
    manifest = json.loads(
        next((tmp_path / "packet_artifacts").glob("vpn-app/fold0/packet_framework_manifest.json")).read_text(
            encoding="utf-8"
        )
    )
    execution = manifest["framework"]["notes"]["protocol_content_pretraining_execution"]
    assert execution["status"] == "pending_dry_run_verification"
    assert execution["checkpoint"] == str(reused)


def test_reused_native_checkpoint_requires_exact_contract_and_input_hashes(tmp_path):
    source_train = tmp_path / "source_train.jsonl"
    source_valid = tmp_path / "source_valid.jsonl"
    current_train = tmp_path / "current_train.jsonl"
    current_valid = tmp_path / "current_valid.jsonl"
    for source, current, text in (
        (source_train, current_train, "train\n"),
        (source_valid, current_valid, "valid\n"),
    ):
        source.write_text(text, encoding="utf-8")
        current.write_text(text, encoding="utf-8")
    checkpoint_dir = tmp_path / "native"
    checkpoint_dir.mkdir()
    checkpoint = checkpoint_dir / "best.pt"
    checkpoint.write_bytes(b"checkpoint")
    args = SimpleNamespace(
        byte_max_bytes=128,
        protocol_pretrain_max_packets=64,
        byte_hidden_dim=128,
        byte_num_layers=2,
        protocol_pretrain_flow_layers=2,
        byte_num_heads=4,
        byte_dropout=0.1,
        protocol_pretrain_projection_dim=128,
        protocol_pretrain_epochs=20,
        protocol_pretrain_batch_size=8,
        protocol_pretrain_eval_batch_size=16,
        protocol_pretrain_learning_rate=3e-4,
        protocol_pretrain_weight_decay=0.01,
        protocol_pretrain_field_mask_probability=0.2,
        protocol_pretrain_payload_dropout_probability=0.5,
        protocol_pretrain_session_mask_probability=1.0,
        protocol_pretrain_masked_byte_weight=1.0,
        protocol_pretrain_relative_order_weight=0.25,
        protocol_pretrain_same_flow_weight=0.25,
        protocol_pretrain_next_length_weight=0.2,
        protocol_pretrain_next_iat_weight=0.2,
        protocol_pretrain_direction_weight=0.1,
        protocol_pretrain_packet_consistency_weight=0.25,
        protocol_pretrain_flow_contrastive_weight=0.25,
        protocol_pretrain_temperature=0.1,
        protocol_pretrain_patience=4,
        protocol_pretrain_seed=42,
    )
    metrics = {
        "uses_downstream_labels": False,
        "model_config": {
            "max_bytes": 128,
            "max_packets": 64,
            "hidden_dim": 128,
            "byte_layers": 2,
            "flow_layers": 2,
            "num_heads": 4,
            "dropout": 0.1,
            "projection_dim": 128,
        },
        "pretraining_config": {
            "train_index": str(source_train),
            "valid_index": str(source_valid),
            "epochs": 20,
            "batch_size": 8,
            "eval_batch_size": 16,
            "learning_rate": 3e-4,
            "weight_decay": 0.01,
            "field_mask_probability": 0.2,
            "payload_dropout_probability": 0.5,
            "session_mask_probability": 1.0,
            "masked_byte_weight": 1.0,
            "relative_order_weight": 0.25,
            "same_flow_weight": 0.25,
            "next_length_weight": 0.2,
            "next_iat_weight": 0.2,
            "direction_weight": 0.1,
            "packet_consistency_weight": 0.25,
            "flow_contrastive_weight": 0.25,
            "temperature": 0.1,
            "patience": 4,
            "seed": 42,
        },
    }
    (checkpoint_dir / "pretraining_metrics.json").write_text(
        json.dumps(metrics), encoding="utf-8"
    )
    evidence = verify_reused_protocol_content_checkpoint(
        args, checkpoint, current_train, current_valid
    )
    assert evidence["status"] == "pass"
    assert evidence["uses_downstream_labels"] is False

    current_valid.write_text("drifted\n", encoding="utf-8")
    with pytest.raises(ValueError, match="input mismatch"):
        verify_reused_protocol_content_checkpoint(
            args, checkpoint, current_train, current_valid
        )


def test_packet_runner_preserves_declared_numeric_overrides(tmp_path):
    payload = frozen_payload()
    config = write_payload(tmp_path / "frozen.json", payload)
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "run_packet_level_pipeline.py"),
            "--dataset",
            "vpn-app",
            "--fold",
            "0",
            "--stage",
            "paper_unified",
            "--dry_run",
            "--artifact_root",
            str(tmp_path / "packet_artifacts"),
            "--checkpoint_root",
            str(tmp_path / "packet_checkpoints"),
            "--shared_core_config",
            str(config),
            "--epochs",
            "12",
            "--lr",
            "2e-5",
            "--packet_batch_size",
            "32",
            "--training_hyperparameter_overrides",
            "epochs,lr,packet_batch_size",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    tower1 = next(
        line
        for line in result.stdout.splitlines()
        if "train_tower1_multitask.py" in line
    )
    assert "--epochs 12" in tower1
    assert "--lr 2e-05" in tower1
    assert "--packet_batch_size 32" in tower1
    manifest_path = next(
        (tmp_path / "packet_artifacts").glob(
            "vpn-app/fold0/packet_framework_manifest.json"
        )
    )
    notes = json.loads(manifest_path.read_text(encoding="utf-8"))["framework"][
        "notes"
    ]
    assert notes["shared_core_method_sha256"] == payload["config_sha256"]
    assert notes["shared_core_config_sha256"] != notes["shared_core_method_sha256"]


def test_flow_runner_consumes_same_config_and_preprocesses_paired_view_first(tmp_path):
    config = write_payload(tmp_path / "frozen.json", frozen_payload())
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "run_stage8_flowaware_pipeline.py"),
            "--dataset",
            "vpn-app",
            "--fold",
            "0",
            "--stage",
            "all",
            "--splits",
            "train,valid",
            "--dry_run",
            "--output_suffix",
            "shared_core_v2_test",
            "--embedding_suffix",
            "shared_core_v2_test",
            "--embedding_num_shards",
            "1",
            "--model_types",
            "seq",
            "--shared_core_config",
            str(config),
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    native = next(line for line in lines if "pretrain_native_flow_encoder.py" in line)
    tower1_index = next(i for i, line in enumerate(lines) if "train_tower1_multitask.py" in line)
    intervened_preprocess = next(
        i
        for i, line in enumerate(lines)
        if "preprocess_tower1.py" in line and "_intervention" in line
    )
    tower1 = lines[tower1_index]
    factual_preprocess = next(
        line
        for line in lines
        if "preprocess_tower1.py" in line and "_intervention" not in line
    )
    intervened_preprocess_line = lines[intervened_preprocess]
    assert "--packet_context_policy single_packet" in factual_preprocess
    assert "--packet_context_policy single_packet" in intervened_preprocess_line
    assert (
        "--label_map_in reasoningDataset/vpn-app/"
        "train_tower1_shared_core_v2_test/label_map.json"
    ) in intervened_preprocess_line
    assert intervened_preprocess < tower1_index
    assert (
        "--label_map reasoningDataset/vpn-app/"
        "train_tower1_shared_core_v2_test/label_map.json"
    ) in tower1
    assert (
        "--packet_aux_jsonl reasoningDataset/vpn-app/"
        "train_tower1_shared_core_v2_test/packet_auxiliary.jsonl"
    ) in tower1
    assert (
        "--valid_packet_aux_jsonl reasoningDataset/vpn-app/"
        "valid_tower1_shared_core_v2_test/packet_auxiliary.jsonl"
    ) in tower1
    assert (
        "--output_dir checkpoints/"
        "tower1_qwen_multitask_vpn_app_shared_core_v2_test_fold0"
    ) in tower1
    assert "--paired_packet_aux_jsonl" in tower1
    assert "--paired_consistency_weight 0.05" in tower1
    assert "--valid_paired_packet_aux_jsonl" in tower1
    assert "--paired_validation_selection worst_view_macro_f1" in tower1
    for option in (
        "--max_bytes 128",
        "--hidden_dim 128",
        "--byte_layers 2",
        "--num_heads 4",
        "--dropout 0.1",
        "--learning_rate 0.0003",
        "--weight_decay 0.01",
        "--temperature 0.1",
        "--seed 42",
    ):
        assert option in native

    manifest = json.loads(
        next(
            (tmp_path / "reasoningDataset" / "vpn-app").glob(
                "stage8_flowaware_manifest_*.json"
            )
        ).read_text(encoding="utf-8")
    )
    notes = manifest["framework"]["notes"]
    assert notes["shared_core_method_sha256"] == frozen_payload()["config_sha256"]
    assert notes["shared_core_config_sha256"] == notes["shared_core_method_sha256"]


def test_flow_runner_preserves_declared_numeric_overrides(tmp_path):
    payload = frozen_payload()
    config = write_payload(tmp_path / "frozen.json", payload)
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "run_stage8_flowaware_pipeline.py"),
            "--dataset",
            "vpn-app",
            "--fold",
            "0",
            "--stage",
            "all",
            "--splits",
            "train",
            "--dry_run",
            "--output_suffix",
            "shared_core_v2_override_test",
            "--embedding_suffix",
            "shared_core_v2_override_test",
            "--embedding_num_shards",
            "1",
            "--model_types",
            "seq",
            "--shared_core_config",
            str(config),
            "--tower1_epochs",
            "12",
            "--tower1_lr",
            "2e-5",
            "--packet_batch_size",
            "32",
            "--training_hyperparameter_overrides",
            "tower1_epochs,tower1_lr,packet_batch_size",
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    tower1 = next(
        line
        for line in result.stdout.splitlines()
        if "train_tower1_multitask.py" in line
    )
    assert "--epochs 12" in tower1
    assert "--lr 2e-05" in tower1
    assert "--packet_batch_size 32" in tower1

    manifest_path = next(
        (tmp_path / "reasoningDataset" / "vpn-app").glob(
            "stage8_flowaware_manifest_*.json"
        )
    )
    notes = json.loads(manifest_path.read_text(encoding="utf-8"))["framework"][
        "notes"
    ]
    assert notes["shared_core_method_sha256"] == payload["config_sha256"]
    assert notes["shared_core_config_sha256"] != notes["shared_core_method_sha256"]
