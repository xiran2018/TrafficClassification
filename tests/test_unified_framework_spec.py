import json
from types import SimpleNamespace

from audit_flow_embeddings import sha256_file
from paper_framework_defaults import (
    DEFAULT_ABLATION_ONLY_MODULES,
    DEFAULT_FLOW_DATASETS,
    DEFAULT_FRAMEWORK_PROFILE,
    DEFAULT_PAPER_MAIN_MODULES,
    DEFAULT_SHARED_CORE_MODULES,
    DEFAULT_UNIFIED_CANDIDATE_EXPERTS,
)
from audit_unified_framework import shared_status_matches
from unified_framework_spec import (
    ABLATION_ONLY_MODULES,
    FLOW_LEVEL_RESULTS,
    PAPER_UNIFIED_SHARED_STATUS_ALIASES,
    FRAMEWORK_PROFILES,
    PAPER_MAIN_MODULES,
    PACKET_LEVEL_RESULTS,
    SHARED_CORE_MODULES,
    UNIFIED_CANDIDATE_EXPERTS,
    apply_framework_profile,
    build_framework_manifest,
    framework_profile_contract,
    framework_profile_fingerprint,
    shared_module_overlap,
    task_modules,
    tower1_hyperparameter_differences,
    tower1_shared_protocol_signature,
    tower1_training_contract,
)
from run_stage8_flowaware_pipeline import (
    embedding_policy_evidence,
    field_aware_policy_status,
    intervention_policy_evidence,
    merge_completed_paper_unified_stages,
    selected_paper_unified_stages,
)


def test_flow_scope_is_vpn_tls_only_for_current_flow_paper_claims():
    assert DEFAULT_FLOW_DATASETS == ("vpn-app", "tls-120")
    assert tuple(FLOW_LEVEL_RESULTS) == DEFAULT_FLOW_DATASETS
    assert "ustc-app" not in FLOW_LEVEL_RESULTS
    assert "ustc-binary" not in FLOW_LEVEL_RESULTS


def test_packet_scope_covers_sweet_packet_level_datasets():
    assert set(PACKET_LEVEL_RESULTS) == {
        "vpn-app",
        "vpn-binary",
        "vpn-service",
        "tls-120",
        "ustc-app",
        "ustc-binary",
    }


def test_flow_manifest_requires_both_factual_and_intervention_policy_evidence(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    args = SimpleNamespace(
        dataset="demo",
        splits="train,valid,test",
        embedding_suffix="semantic",
        embedding_header_policy="full",
        packet_context_policy="single_packet",
        intervened_embedding_header_policy="mask_ip_port",
        use_intervention_views=True,
        framework_profile="paper_unified",
    )
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
        factual = tmp_path / "reasoningDataset" / "demo" / f"{split}_embeddings_semantic"
        intervention = tmp_path / "reasoningDataset" / "demo" / f"{split}_embeddings_semantic_mask_ip_port_intervention"
        factual.mkdir(parents=True)
        intervention.mkdir(parents=True)
        (factual / "embedding_config.json").write_text(
            json.dumps(
                    {
                        "packet_index_header_policy": "full",
                        "packet_index_context_policy": "single_packet",
                        "embedding_mode": "concat",
                        "scheduler": "cross_flow_length_bucketed_v1",
                        "batch_size": 8,
                        "flow_batch_packets": 128,
                    }
            )
        )
        (intervention / "embedding_config.json").write_text(
            json.dumps(
                    {
                        "packet_index_header_policy": "mask_ip_port",
                        "packet_index_context_policy": "single_packet",
                        "embedding_mode": "concat",
                        "scheduler": "cross_flow_length_bucketed_v1",
                        "batch_size": 8,
                        "flow_batch_packets": 128,
                    }
            )
        )
        for directory in (factual, intervention):
            packet_index = directory / "packet_index.jsonl"
            embedding_index = directory / "flow_embedding_index.jsonl"
            packet_index.write_text("packet\n", encoding="utf-8")
            embedding_index.write_text("embedding\n", encoding="utf-8")
            embedding_config = directory / "embedding_config.json"
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
            (directory / "embedding_audit.json").write_text(
                json.dumps(audit_payload), encoding="utf-8"
            )

    factual_evidence = embedding_policy_evidence(args)
    intervention_evidence = intervention_policy_evidence(args)
    assert field_aware_policy_status(
        args, factual_evidence, intervention_evidence
    ) == "factual_full_plus_mask_ip_port_intervention"

    bad_config = (
        tmp_path
        / "reasoningDataset"
        / "demo"
        / "test_embeddings_semantic_mask_ip_port_intervention"
        / "embedding_config.json"
    )
    bad_config.write_text(
        json.dumps(
            {
                "packet_index_header_policy": "full",
                "packet_index_context_policy": "single_packet",
            }
        )
    )
    bad_intervention = intervention_policy_evidence(args)
    assert field_aware_policy_status(
        args, factual_evidence, bad_intervention
    ) == "unverified"


def test_paper_result_specs_require_framework_provenance_globs():
    for spec in list(FLOW_LEVEL_RESULTS.values()) + list(PACKET_LEVEL_RESULTS.values()):
        assert spec.framework_profile == "paper_unified"
        assert spec.framework_manifest_glob
        assert "paper_unified" in spec.framework_manifest_glob or "packet_framework_manifest" in spec.framework_manifest_glob


def test_packet_default_results_are_project_local_canonical_artifacts():
    for dataset, spec in PACKET_LEVEL_RESULTS.items():
        assert spec.path == f"reasoningDataset/packet-level/{dataset}/paper_default_result.json"
        assert not spec.path.startswith("/tmp/")


def test_packet_and_flow_share_the_same_core_modules():
    assert DEFAULT_SHARED_CORE_MODULES == SHARED_CORE_MODULES
    assert shared_module_overlap() == SHARED_CORE_MODULES
    assert set(SHARED_CORE_MODULES).issubset(set(task_modules("flow-level")))
    assert set(SHARED_CORE_MODULES).issubset(set(task_modules("packet-level")))


def test_paper_module_boundaries_are_explicit_and_disjoint():
    assert DEFAULT_PAPER_MAIN_MODULES == PAPER_MAIN_MODULES
    assert DEFAULT_UNIFIED_CANDIDATE_EXPERTS == UNIFIED_CANDIDATE_EXPERTS
    assert DEFAULT_ABLATION_ONLY_MODULES == ABLATION_ONLY_MODULES
    assert set(SHARED_CORE_MODULES).issubset(set(PAPER_MAIN_MODULES))
    assert set(PAPER_MAIN_MODULES).isdisjoint(UNIFIED_CANDIDATE_EXPERTS)
    assert set(PAPER_MAIN_MODULES).isdisjoint(ABLATION_ONLY_MODULES)
    assert set(UNIFIED_CANDIDATE_EXPERTS).isdisjoint(ABLATION_ONLY_MODULES)
    assert "confidence_penalty" in ABLATION_ONLY_MODULES
    assert "vpn_specific_hierarchical_coarse_head" in ABLATION_ONLY_MODULES
    assert "slot_stacker_expert" in ABLATION_ONLY_MODULES
    assert "graph_flow_expert" in UNIFIED_CANDIDATE_EXPERTS


def test_paper_defaults_are_bound_to_unified_profile():
    assert DEFAULT_FRAMEWORK_PROFILE == "paper_unified"
    assert DEFAULT_FRAMEWORK_PROFILE in FRAMEWORK_PROFILES
    profile = FRAMEWORK_PROFILES[DEFAULT_FRAMEWORK_PROFILE]
    for module in SHARED_CORE_MODULES:
        assert module in profile["shared_module_status"]


def test_direct_runners_default_to_paper_unified_profile():
    stage8_source = open("run_stage8_flowaware_pipeline.py", encoding="utf-8").read()
    packet_source = open("run_packet_level_pipeline.py", encoding="utf-8").read()

    assert 'default="paper_unified"' in stage8_source
    assert 'default="paper_unified"' in packet_source
    assert "use legacy only for historical" in stage8_source
    assert "use legacy only for historical" in packet_source
    assert "--content_group_index" in stage8_source
    assert "--content_group_loss_reduction" in stage8_source
    assert "--content_group_loss_reduction" in packet_source
    assert "paper_unified fixes --intervention_view_base_mode symmetric_mean" in stage8_source
    assert "paper_unified fixes --intervention_view_base_mode symmetric_mean" in packet_source
    assert "paper_unified fixes --channel_fusion_max_weight 0.25" in packet_source


def test_stage8_paper_unified_smoke_stage_aliases():
    assert selected_paper_unified_stages("model") == [
        "tower2_preprocess",
        "tower2_train",
        "eval",
    ]
    assert selected_paper_unified_stages("model_fusion") == [
        "tower2_preprocess",
        "tower2_train",
        "eval",
        "fusion",
    ]
    assert "selector" in selected_paper_unified_stages("all")


def test_stage8_manifest_accumulates_completed_paper_unified_stages():
    first = merge_completed_paper_unified_stages(
        None,
        selected_paper_unified_stages("model_fusion"),
        completed=True,
    )
    assert first == [
        "tower2_preprocess",
        "tower2_train",
        "eval",
        "fusion",
    ]

    second = merge_completed_paper_unified_stages(
        {"framework": {"notes": {"completed": True, "paper_unified_stages": first}}},
        selected_paper_unified_stages("stacker,prior,selector"),
        completed=True,
    )
    assert second == [
        "tower2_preprocess",
        "tower2_train",
        "eval",
        "fusion",
        "stacker",
        "prior",
        "selector",
    ]


def test_framework_manifest_preserves_shared_core_contract():
    manifest = build_framework_manifest(
        task="packet-level",
        dataset="vpn-app",
        input_unit="one_current_packet",
        stage="packet_best",
        shared_module_status={"field_aware_header_policy": "mask_ip_port"},
    )
    assert manifest["framework_name"] == "unified_shortcut_resistant_packet_to_flow"
    assert tuple(manifest["shared_core_modules"]) == SHARED_CORE_MODULES
    assert manifest["shared_module_status"]["field_aware_header_policy"] == "mask_ip_port"


def test_framework_manifest_records_profile_contract_fingerprint():
    manifest = build_framework_manifest(
        task="flow-level",
        dataset="vpn-app",
        input_unit="flow_packet_sequence",
        stage="paper_unified",
        notes={"framework_profile": "paper_unified"},
    )

    assert manifest["framework_profile_fingerprint"] == framework_profile_fingerprint("paper_unified", "flow-level")
    assert manifest["framework_profile_contract"] == framework_profile_contract("paper_unified", "flow-level")
    assert manifest["framework_profile_contract"]["task_overrides"]["contrastive_mode"] == "standard"
    assert manifest["framework_profile_contract"]["task_overrides"]["coarse_groups"] == "none"


def test_paper_unified_profile_aligns_packet_and_flow_switches():
    class Args:
        framework_profile = "paper_unified"

    packet_args = Args()
    packet_args.embedding_header_policy = "full"
    packet_args.packet_context_policy = "auto"
    packet_args.intervened_embedding_header_policy = ""
    packet_args.use_intervention_views = False
    packet_args.byte_use_payload_channel = False
    packet_args.byte_use_protocol_fields = False
    packet_args.byte_max_bytes = 32
    packet_args.byte_max_payload_bytes = 64
    packet_args.max_packet_length = 640
    packet_args.cls_weight = 0.1
    packet_args.contrastive_weight = 0.3
    packet_args.same_flow_positive_weight = 2.0

    flow_args = Args()
    flow_args.embedding_header_policy = "full"
    flow_args.packet_context_policy = "auto"
    flow_args.intervened_embedding_header_policy = ""
    flow_args.use_intervention_views = False
    flow_args.flow_pooling = "multi_view"
    flow_args.model_types = "graph,seq"
    flow_args.flow_stat_expert_weight = 0.2
    flow_args.flow_stat_aux_weight = 0.05
    flow_args.meta_dropout_prob = 0.05
    flow_args.edge_attr_dropout_prob = 0.05
    flow_args.confidence_penalty_weight = 0.0
    flow_args.hierarchical_weight = 0.2
    flow_args.hierarchical_logit_weight = 0.5
    flow_args.coarse_groups = "vpn_app"
    flow_args.contrastive_mode = "confusion"
    flow_args.confusion_groups = "vpn_app"
    flow_args.content_group_guard = False
    flow_args.tower2_select_metric = "flow_macro_f1"
    flow_args.split_group_key = "flow_id"
    flow_args.content_group_unique_batches = False
    flow_args.content_group_loss_reduction = "none"
    flow_args.max_packet_length = 640
    flow_args.cls_weight = 0.1
    flow_args.contrastive_weight = 0.3
    flow_args.same_flow_positive_weight = 2.0
    flow_args.tower1_use_sft = True
    flow_args.tower1_lr = 2e-5
    flow_args.tower1_class_weighting = "none"
    flow_args.tower1_disable_packet_information_weights = False
    flow_args.tower1_gradient_accumulation_steps = 1
    flow_args.embedding_resume_existing = True
    flow_args.embedding_resume_shards = True
    flow_args.native_structural_suffix = ""
    flow_args.native_structural_dim = 0
    flow_args.meta_feature_dim = 14
    flow_args.dual_channel_mode = "concat"
    flow_args.dual_channel_gate_mode = "global"
    flow_args.dual_channel_max_weight = 0.1

    packet_overrides = apply_framework_profile(packet_args, "packet-level")
    flow_overrides = apply_framework_profile(flow_args, "flow-level")

    assert packet_args.embedding_header_policy == "full"
    assert packet_args.packet_context_policy == "single_packet"
    assert packet_args.intervened_embedding_header_policy == "mask_ip_port"
    assert packet_args.use_intervention_views is True
    assert packet_args.byte_use_payload_channel is False
    assert packet_args.byte_use_protocol_fields is True
    assert packet_args.byte_max_bytes == 128
    assert packet_args.max_packet_length == 1024
    assert packet_args.cls_weight == 1.0
    assert packet_args.contrastive_weight == 0.1
    assert packet_args.same_flow_positive_weight == 1.0
    assert flow_args.embedding_header_policy == "full"
    assert flow_args.packet_context_policy == "single_packet"
    assert flow_args.intervened_embedding_header_policy == "mask_ip_port"
    assert flow_args.use_intervention_views is True
    assert flow_args.flow_pooling == "mean"
    assert flow_args.model_types == "seq"
    assert flow_args.native_structural_suffix == "shared_content"
    assert flow_args.native_structural_dim == 128
    assert flow_args.meta_feature_dim == 13
    assert flow_args.dual_channel_mode == "residual"
    assert flow_args.dual_channel_gate_mode == "adaptive"
    assert flow_args.tower1_use_sft is False
    assert flow_args.tower1_lr == 1e-5
    assert flow_args.tower1_class_weighting == "effective"
    assert flow_args.tower1_disable_packet_information_weights is True
    assert flow_args.tower1_gradient_accumulation_steps == 1
    assert flow_args.cls_weight == packet_args.cls_weight
    assert flow_args.contrastive_weight == packet_args.contrastive_weight
    assert flow_args.same_flow_positive_weight == packet_args.same_flow_positive_weight
    assert flow_args.flow_stat_expert_weight == 0.0
    assert flow_args.flow_stat_aux_weight == 0.0
    assert flow_args.meta_dropout_prob == 0.0
    assert flow_args.edge_attr_dropout_prob == 0.0
    assert flow_args.confidence_penalty_weight == 0.0
    assert flow_args.hierarchical_weight == 0.0
    assert flow_args.hierarchical_logit_weight == 0.0
    assert flow_args.coarse_groups == "none"
    assert flow_args.contrastive_mode == "standard"
    assert flow_args.confusion_groups == "none"
    assert flow_args.embedding_resume_existing is False
    assert flow_args.embedding_resume_shards is False
    assert flow_args.content_group_guard is True
    assert flow_args.tower2_select_metric == "content_group_macro_f1"
    assert flow_args.split_group_key == "content_group_id"
    assert flow_args.content_group_unique_batches is True
    assert flow_args.content_group_loss_reduction == "group_mean"
    assert packet_overrides["embedding_header_policy"]["old"] == "full"
    assert packet_overrides["packet_context_policy"]["old"] == "auto"
    assert flow_overrides["flow_pooling"]["old"] == "multi_view"


def test_tower1_method_signature_is_shared_while_numeric_hyperparameters_may_differ():
    common = {
        "packet_context_policy": "single_packet",
        "base_model": "Qwen/Qwen2.5-7B-Instruct",
        "packet_batch_size": 16,
        "max_packet_length": 1024,
        "cls_weight": 1.0,
        "contrastive_weight": 0.1,
            "same_flow_positive_weight": 1.0,
            "same_label_positive_weight": 1.0,
            "flow_proto_weight": 0.0,
            "flow_proto_positive": "same_class",
            "flow_proto_context": "inclusive",
            "temperature": 0.07,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "dtype": "float16",
        "seed": 42,
        "tower1_paired_consistency_weight": 0.05,
        "tower1_paired_cls_weight": 0.2,
        "tower1_paired_logit_kl_weight": 0.5,
        "tower1_paired_raw_consistency_weight": 1.0,
        "tower1_early_stop_patience": 0,
    }
    packet = SimpleNamespace(
        **common,
        epochs=8,
        gradient_accumulation_steps=1,
        lr=1e-5,
        head_lr=1e-4,
        class_weighting="effective",
        class_weight_basis="flow",
        class_weight_strength=0.5,
        init_checkpoint_dir="",
        init_adapter_only=False,
    )
    flow = SimpleNamespace(
        **common,
        tower1_epochs=12,
        tower1_max_steps=0,
        tower1_gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        tower1_init_checkpoint_dir="",
        tower1_lr=2e-5,
        tower1_head_lr=1e-4,
        tower1_class_weighting="effective",
        tower1_class_weight_basis="flow",
        tower1_class_weight_strength=0.75,
        tower1_use_sft=False,
        tower1_disable_packet_information_weights=True,
        flow_balanced_packet_batches=True,
        packets_per_flow=2,
    )
    packet_contract = tower1_training_contract(packet, "packet-level")
    flow_contract = tower1_training_contract(flow, "flow-level")
    assert packet_contract != flow_contract
    assert tower1_shared_protocol_signature(packet_contract) == (
        tower1_shared_protocol_signature(flow_contract)
    )
    differences = tower1_hyperparameter_differences(
        packet_contract, flow_contract
    )
    assert differences["epochs"] == {"packet": 8, "flow": 12}
    assert differences["learning_rate"] == {"packet": 1e-5, "flow": 2e-5}
    assert differences["class_weight_strength"] == {
        "packet": 0.5,
        "flow": 0.75,
    }


def test_tower1_method_signature_rejects_disabled_objective_or_architecture_change():
    base = {
        "trainer": "train_tower1_multitask.py",
        "packet_context_policy": "single_packet",
        "base_model": "Qwen/Qwen2.5-7B-Instruct",
        "max_packet_length": 1024,
        "projection_dim": 256,
        "lora_r": 16,
        "use_sft": False,
        "disable_packet_information_weights": True,
        "flow_balanced_packet_batches": True,
        "packet_batch_scheduler": "epoch_resampled_dataloader_v1",
        "class_weighting": "effective",
        "class_weight_basis": "flow",
        "select_metric": "macro_f1",
        "cls_weight": 1.0,
        "contrastive_weight": 0.1,
        "same_flow_positive_weight": 1.0,
        "same_label_positive_weight": 1.0,
        "flow_proto_weight": 0.0,
        "flow_proto_positive": "same_class",
        "flow_proto_context": "inclusive",
        "paired_consistency_weight": 0.05,
        "paired_cls_weight": 0.2,
    }
    disabled = dict(base, contrastive_weight=0.0)
    changed_architecture = dict(base, projection_dim=128)
    changed_identity_policy = dict(base, identity_safe_contrastive=True)
    assert tower1_shared_protocol_signature(base) != (
        tower1_shared_protocol_signature(disabled)
    )
    assert tower1_shared_protocol_signature(base) != (
        tower1_shared_protocol_signature(changed_architecture)
    )
    assert tower1_shared_protocol_signature(base) != (
        tower1_shared_protocol_signature(changed_identity_policy)
    )


def test_paper_unified_shared_status_aliases_cover_runner_and_bound_manifests():
    profile_status = FRAMEWORK_PROFILES["paper_unified"]["shared_module_status"]
    assert shared_status_matches(profile_status)

    legacy_single_view_status = {
        "per_flow_split_guard": "enforced",
        "field_aware_header_policy": "mask_ip_port",
        "current_packet_structural_encoder": "present",
        "current_packet_byte_payload_encoder": "present",
        "semantic_tower1_channel": "present_or_available_in_source_inputs",
        "shared_packet_channel_fusion": "strict_current_packet_tri_channel",
        "semantic_structural_gate": "validation_selected_or_consensus_bound",
        "content_stability_guard": "recorded_from_source_results",
        "cross_fold_stability": "crossfold_consensus_bound",
        "validation_only_selection": "enforced_by_source_results",
        "label_free_calibration_guard": "recorded_from_source_results",
    }

    assert set(PAPER_UNIFIED_SHARED_STATUS_ALIASES) == set(SHARED_CORE_MODULES)
    assert not shared_status_matches(legacy_single_view_status)
    result_bound_status = {
        "label_free_protocol_content_pretraining": "native_flow_multitask_v1",
        "field_aware_header_intervention": "factual_full_plus_mask_ip_port_intervention",
        "semantic_tower1_channel": "present_or_available_in_source_inputs",
        "current_packet_structural_encoder": "strict_current_packet_13d",
        "shared_intervention_view_fusion": "required_representation_level",
        "bounded_tri_channel_router": "semantic_anchor_residual_0_25",
        "per_flow_split_guard": "enforced",
        "content_group_empirical_risk": "group_mean",
        "validation_only_selection": "enforced_by_source_results",
        "fixed_cross_fold_consensus": "equal_log_mean_bound",
    }
    assert shared_status_matches(result_bound_status)


def test_paper_unified_shared_status_rejects_full_header_shortcut():
    bad_status = dict(FRAMEWORK_PROFILES["paper_unified"]["shared_module_status"])
    bad_status["field_aware_header_intervention"] = "full"

    assert not shared_status_matches(bad_status)
import json
from types import SimpleNamespace
