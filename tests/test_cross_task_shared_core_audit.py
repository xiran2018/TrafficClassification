import torch

from audit_cross_task_shared_core import (
    audit_cross_task_fold,
    audit_runtime_mechanism_evidence,
)
from audit_shared_packet_core import PRETRAINING_PROTOCOL


def content_state(prefix):
    return {
        f"{prefix}layer.weight": torch.zeros(4, 4),
        f"{prefix}layer.bias": torch.zeros(4),
    }


def representation_state(structural=13):
    return {
        "shared_packet_encoder.semantic_proj.0.weight": torch.zeros(4, 6),
        "shared_packet_encoder.content_proj.0.weight": torch.zeros(4, 4),
        "shared_packet_encoder.structural_proj.0.weight": torch.zeros(4, structural),
    }


def packet_checkpoint():
    return {
        "config": {
            "max_bytes": 128,
            "hidden_dim": 4,
            "num_layers": 2,
            "num_heads": 2,
            "dropout": 0.1,
            "meta_dim": 13,
            "semantic_dim": 6,
            "exact_shared_representation": True,
            "mask_protocol_session_fields": True,
        },
        "state_dict": content_state("protocol_content_encoder.")
        | representation_state(),
        "initialization": {
            "protocol_content_pretraining": PRETRAINING_PROTOCOL,
            "protocol_content_checkpoint": "packet-native.pt",
            "protocol_content_checkpoint_sha256": "packet-native-sha",
        },
    }


def native_checkpoint(*, hidden=4, seed=42):
    pretraining_config = {
        "max_packets": 64,
        "epochs": 20,
        "batch_size": 8,
        "eval_batch_size": 16,
        "num_workers": 0,
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
        "seed": seed,
    }
    return {
        "model_config": {
            "max_bytes": 128,
            "hidden_dim": hidden,
            "byte_layers": 2,
            "num_heads": 2,
            "dropout": 0.1,
            "num_field_types": 9,
        },
        "state_dict": content_state("packet_content_encoder."),
        "pretraining_protocol": PRETRAINING_PROTOCOL,
        "pretraining_config": pretraining_config,
    }


def flow_checkpoint():
    return {
        "input_dim": 6 + 4 + 13,
        "meta_feature_dim": 4 + 13,
        "native_structural_dim": 4,
        "shared_packet_hidden_dim": 4,
        "exact_shared_packet_encoder": True,
        "model_state": representation_state()
        | {"packet_to_flow_proj.weight": torch.zeros(8, 4)},
    }


def tower1_contract(**overrides):
    tower1_contract = {
        "trainer": "train_tower1_multitask.py",
        "packet_context_policy": "single_packet",
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
        "class_weighting": "effective",
        "class_weight_beta": 0.9999,
        "class_weight_basis": "flow",
        "class_weight_strength": 0.5,
        "paired_consistency_weight": 0.0,
        "paired_cls_weight": 0.0,
        "paired_logit_kl_weight": 0.5,
        "paired_raw_consistency_weight": 1.0,
        "use_sft": False,
        "disable_packet_information_weights": True,
        "flow_balanced_packet_batches": True,
        "packets_per_flow": 2,
        "packet_batch_scheduler": "epoch_resampled_dataloader_v1",
        "select_metric": "macro_f1",
        "early_stop_patience": 0,
        "init_checkpoint_dir": "",
        "init_adapter_only": False,
    }
    tower1_contract.update(overrides)
    return tower1_contract


def manifest(
    task,
    *,
    dataset="vpn-app",
    fold=0,
    fingerprint="shared-v2",
    method_fingerprint="",
    tower1_overrides=None,
):
    contract = tower1_contract(**(tower1_overrides or {}))
    notes = {
        "fold": fold,
        "shared_core_config_sha256": fingerprint,
        "shared_core_method_sha256": method_fingerprint,
        "content_group_loss_reduction": "group_mean",
        "byte_content_group_loss_reduction": "group_mean",
        "tower1_training_contract": contract,
        "tower1_execution_evidence": {
            "verified": True,
            "declared_contract_match": True,
            "method_config": {
                key: value
                for key, value in contract.items()
                if key != "packet_context_policy"
            },
            "trainer_source_sha256": "a" * 64,
            "trainer_source_stable_through_completion": True,
        },
        "semantic_packet_context_policy": "single_packet",
        "semantic_embedding_policy_evidence": {
            "verified": True,
            "embedding_audits_verified": True,
            "expected_packet_context_policy": "single_packet",
        },
        "embedding_header_policy_evidence": {
            "verified": True,
            "embedding_audits_verified": True,
            "expected_context": "single_packet",
        },
        "intervention_header_policy_evidence": {
            "verified": True,
            "embedding_audits_verified": True,
            "expected_context": "single_packet",
        },
        "packet_module_training_source": (
            "packet_task_train_split_packets"
            if task == "packet-level"
            else "flow_task_train_split_packets"
        ),
        "cross_task_trained_weights_reused": False,
    }
    return {
        "dataset": dataset,
        "fold": fold,
        "framework": {"dataset": dataset, "task": task, "notes": notes},
    }


def run_audit(*, flow_native=None, flow_fingerprint="shared-v2"):
    return audit_cross_task_fold(
        manifest("packet-level"),
        manifest("flow-level", fingerprint=flow_fingerprint),
        packet_checkpoint=packet_checkpoint(),
        packet_native_checkpoint=native_checkpoint(),
        flow_checkpoint=flow_checkpoint(),
        flow_native_checkpoint=flow_native or native_checkpoint(),
        packet_native_sha256="packet-native-sha",
    )


def gate_summary(names, mean, *, names_key):
    return {
        "num_samples": 10,
        names_key: list(names),
        "base_mode": "semantic_anchor" if len(names) == 3 else "symmetric_mean",
        "weight_semantics": "bounded_effective_routing_weights",
        "mean": mean,
        "std": [0.01] * len(names),
        "effective_routing_mean": mean,
        "theoretical_bounds": {
            name: ([0.75, 1.0] if name == "semantic" else [0.0, 0.25])
            if len(names) == 3
            else [0.375, 0.625]
            for name in names
        },
        "bounds_satisfied": True,
    }


def mechanism_result(task):
    diagnostics = {
        ("packet_channel_gate" if task == "packet" else "dual_channel_gate"): gate_summary(
            ("semantic", "content", "structural"),
            [0.8, 0.1, 0.1],
            names_key="channel_names",
        ),
        "intervention_view_gate": gate_summary(
            ("factual", "intervened"),
            [0.55, 0.45],
            names_key="view_names",
        ),
    }
    if task == "packet":
        return {"learned_gate_diagnostics": diagnostics}
    return {"metrics": {"eval_config": {"learned_gate_diagnostics": diagnostics}}}


def extraction_manifests():
    return {
        split: {
            "checkpoint": "flow-native.pt",
            "checkpoint_sha256": "flow-native-sha",
            "session_mask_probability": 1.0,
            "flow_count": 10,
            "alignment": "flow_id_and_packet_id",
            "packet_representation_scope": "strict_current_packet",
            "packet_representation_name": "native_protocol_current_packet_content",
            "contextual_flow_representations_exported_separately": True,
        }
        for split in ("train", "valid", "test")
    }


def test_cross_task_fold_requires_identity_native_contract_and_exact_module():
    report = run_audit()
    assert report["status"] == "pass"
    assert report["identity_match"] is True
    assert report["native_pretraining_contract_match"] is True
    assert report["task_local_packet_module_training"] is True
    assert report["semantic_packet_context_match"] is True
    assert report["semantic_execution_policy_verified"] is True
    assert report["core_audit"]["exact_shared_representation_match"] is True


def test_cross_task_fold_rejects_fingerprint_or_native_architecture_drift():
    assert run_audit(flow_fingerprint="other")["status"] == "not_ready"
    report = run_audit(flow_native=native_checkpoint(hidden=8))
    assert report["status"] == "not_ready"
    assert report["native_pretraining_contract_match"] is False
    report = run_audit(flow_native=native_checkpoint(seed=9))
    assert report["status"] == "pass"
    assert report["native_training_contract_match"] is True
    assert report["allowed_native_hyperparameter_differences"]["seed"] == {
        "packet": 42,
        "flow": 9,
    }


def test_cross_task_fold_rejects_disabled_native_objective():
    flow_native = native_checkpoint()
    flow_native["pretraining_config"]["flow_contrastive_weight"] = 0.0
    report = run_audit(flow_native=flow_native)
    assert report["status"] == "not_ready"
    assert report["native_training_contract_match"] is False


def test_cross_task_fold_allows_numeric_tower1_hyperparameters_to_differ():
    packet_manifest = manifest("packet-level")
    flow_manifest = manifest(
        "flow-level",
        tower1_overrides={
            "epochs": 12,
            "packet_batch_size": 32,
            "learning_rate": 2e-5,
            "weight_decay": 0.02,
            "temperature": 0.09,
            "contrastive_weight": 0.2,
            "class_weight_strength": 0.8,
            "seed": 7,
        },
    )
    report = audit_cross_task_fold(
        packet_manifest,
        flow_manifest,
        packet_checkpoint=packet_checkpoint(),
        packet_native_checkpoint=native_checkpoint(),
        flow_checkpoint=flow_checkpoint(),
        flow_native_checkpoint=native_checkpoint(),
        packet_native_sha256="packet-native-sha",
    )
    assert report["status"] == "pass"
    assert report["tower1_shared_protocol_signature_match"] is True
    assert set(report["allowed_tower1_hyperparameter_differences"]) >= {
        "epochs",
        "packet_batch_size",
        "learning_rate",
        "weight_decay",
        "temperature",
        "contrastive_weight",
        "class_weight_strength",
        "seed",
    }


def test_cross_task_fold_uses_method_identity_not_effective_numeric_fingerprint():
    packet_manifest = manifest(
        "packet-level",
        fingerprint="packet-effective",
        method_fingerprint="shared-method",
    )
    flow_manifest = manifest(
        "flow-level",
        fingerprint="flow-effective",
        method_fingerprint="shared-method",
        tower1_overrides={"learning_rate": 2e-5},
    )
    report = audit_cross_task_fold(
        packet_manifest,
        flow_manifest,
        packet_checkpoint=packet_checkpoint(),
        packet_native_checkpoint=native_checkpoint(),
        flow_checkpoint=flow_checkpoint(),
        flow_native_checkpoint=native_checkpoint(seed=7),
        packet_native_sha256="packet-native-sha",
    )
    assert report["status"] == "pass"
    assert report["shared_core_method_sha256"] == "shared-method"
    assert report["effective_shared_core_config_match"] is False


def test_cross_task_fold_rejects_tower1_objective_or_architecture_drift():
    for overrides in (
        {"contrastive_weight": 0.0},
        {"lora_r": 8},
        {"projection_dim": 128},
        {"class_weight_basis": "packet"},
        {"packet_batch_scheduler": "legacy_cached_cycle"},
    ):
        report = audit_cross_task_fold(
            manifest("packet-level"),
            manifest("flow-level", tower1_overrides=overrides),
            packet_checkpoint=packet_checkpoint(),
            packet_native_checkpoint=native_checkpoint(),
            flow_checkpoint=flow_checkpoint(),
            flow_native_checkpoint=native_checkpoint(),
            packet_native_sha256="packet-native-sha",
        )
        assert report["status"] == "not_ready"
        assert report["tower1_shared_protocol_signature_match"] is False


def test_cross_task_fold_rejects_reusing_packet_task_trained_weights_for_flow():
    flow_manifest = manifest("flow-level")
    flow_manifest["framework"]["notes"]["packet_module_training_source"] = (
        "packet_task_train_split_packets"
    )
    report = audit_cross_task_fold(
        manifest("packet-level"),
        flow_manifest,
        packet_checkpoint=packet_checkpoint(),
        packet_native_checkpoint=native_checkpoint(),
        flow_checkpoint=flow_checkpoint(),
        flow_native_checkpoint=native_checkpoint(),
        packet_native_sha256="packet-native-sha",
    )
    assert report["status"] == "not_ready"
    assert report["task_local_packet_module_training"] is False


def test_cross_task_fold_rejects_unverified_or_mismatched_executed_tower1():
    flow_manifest = manifest("flow-level")
    flow_manifest["framework"]["notes"]["tower1_execution_evidence"][
        "trainer_source_sha256"
    ] = "b" * 64
    report = audit_cross_task_fold(
        manifest("packet-level"),
        flow_manifest,
        packet_checkpoint=packet_checkpoint(),
        packet_native_checkpoint=native_checkpoint(),
        flow_checkpoint=flow_checkpoint(),
        flow_native_checkpoint=native_checkpoint(),
        packet_native_sha256="packet-native-sha",
    )
    assert report["status"] == "not_ready"
    assert report["tower1_execution_contract_verified"] is False


def test_cross_task_fold_rejects_flow_context_inside_shared_packet_semantics():
    flow_manifest = manifest("flow-level")
    flow_notes = flow_manifest["framework"]["notes"]
    flow_notes["semantic_packet_context_policy"] = "flow_context"
    flow_notes["tower1_training_contract"]["packet_context_policy"] = "flow_context"
    report = audit_cross_task_fold(
        manifest("packet-level"),
        flow_manifest,
        packet_checkpoint=packet_checkpoint(),
        packet_native_checkpoint=native_checkpoint(),
        flow_checkpoint=flow_checkpoint(),
        flow_native_checkpoint=native_checkpoint(),
        packet_native_sha256="packet-native-sha",
    )

    assert report["status"] == "not_ready"
    assert report["semantic_packet_context_match"] is False


def test_cross_task_fold_rejects_unverified_executed_semantic_policy():
    packet_manifest = manifest("packet-level")
    packet_manifest["framework"]["notes"]["semantic_embedding_policy_evidence"][
        "verified"
    ] = False
    report = audit_cross_task_fold(
        packet_manifest,
        manifest("flow-level"),
        packet_checkpoint=packet_checkpoint(),
        packet_native_checkpoint=native_checkpoint(),
        flow_checkpoint=flow_checkpoint(),
        flow_native_checkpoint=native_checkpoint(),
        packet_native_sha256="packet-native-sha",
    )

    assert report["status"] == "not_ready"
    assert report["semantic_execution_policy_verified"] is False


def test_runtime_mechanism_evidence_requires_bounded_matching_router_schemas():
    report = audit_runtime_mechanism_evidence(
        mechanism_result("packet"), mechanism_result("flow")
    )
    assert report["status"] == "pass"
    assert report["schema_match"] is True
    assert report["packet"]["channel_router"]["observed_gate_variation"] is True


def test_required_runtime_evidence_controls_cross_task_audit_status():
    kwargs = dict(
        packet_checkpoint=packet_checkpoint(),
        packet_native_checkpoint=native_checkpoint(),
        flow_checkpoint=flow_checkpoint(),
        flow_native_checkpoint=native_checkpoint(),
        packet_native_sha256="packet-native-sha",
        require_mechanism_evidence=True,
    )
    missing = audit_cross_task_fold(
        manifest("packet-level"), manifest("flow-level"), **kwargs
    )
    assert missing["status"] == "not_ready"
    complete = audit_cross_task_fold(
        manifest("packet-level"),
        manifest("flow-level"),
        packet_result=mechanism_result("packet"),
        flow_result=mechanism_result("flow"),
        flow_native_extraction_manifests=extraction_manifests(),
        flow_native_checkpoint_path="flow-native.pt",
        flow_native_sha256="flow-native-sha",
        **kwargs,
    )
    assert complete["status"] == "pass"


def test_cross_task_audit_rejects_contextual_native_packet_extraction():
    manifests = extraction_manifests()
    manifests["valid"]["packet_representation_scope"] = "flow_context"
    report = audit_cross_task_fold(
        manifest("packet-level"),
        manifest("flow-level"),
        packet_checkpoint=packet_checkpoint(),
        packet_native_checkpoint=native_checkpoint(),
        flow_checkpoint=flow_checkpoint(),
        flow_native_checkpoint=native_checkpoint(),
        packet_native_sha256="packet-native-sha",
        packet_result=mechanism_result("packet"),
        flow_result=mechanism_result("flow"),
        flow_native_extraction_manifests=manifests,
        flow_native_checkpoint_path="flow-native.pt",
        flow_native_sha256="flow-native-sha",
        require_mechanism_evidence=True,
    )

    assert report["status"] == "not_ready"
    extraction = report["flow_native_extraction_evidence"]
    assert extraction["status"] == "not_ready"
    assert extraction["splits"]["valid"]["status"] == "not_ready"
    assert (
        extraction["splits"]["train"][
            "contextual_flow_representations_exported_separately"
        ]
        is True
    )
