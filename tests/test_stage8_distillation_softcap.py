from pathlib import Path
from types import SimpleNamespace

import pytest

from run_stage8_flowaware_pipeline import (
    commands,
    selected_eval_splits,
    tower2_data_suffix,
    tower2_preprocess_cmd,
    tower2_train_cmd,
)


def test_stage8_accepts_and_forwards_distillation_confidence_soft_cap():
    source = Path("run_stage8_flowaware_pipeline.py").read_text(encoding="utf-8")

    assert source.count("--distill_max_confidence") >= 2
    assert "--distill_min_confidence" in source
    assert "--distill_confidence_power" in source


def test_stage8_native_structural_suffix_creates_separate_tower2_dataset():
    args = SimpleNamespace(
        dataset="vpn-app",
        splits="train,valid,test",
        embedding_suffix="rawproj_flowaware_change_weight_split2_retrain",
        tower2_suffix="",
        native_structural_suffix="native_pilot",
        window_size=32,
        stride=16,
    )

    cmd = tower2_preprocess_cmd(args, "train")

    assert tower2_data_suffix(args) == "rawproj_flowaware_change_weight_split2_retrain_native_native_pilot"
    assert "reasoningDataset/vpn-app/train_tower2_rawproj_flowaware_change_weight_split2_retrain_native_native_pilot" in cmd
    assert "--structural_embedding_index" in cmd
    assert (
        cmd[cmd.index("--structural_embedding_index") + 1]
        == "reasoningDataset/vpn-app/train_native_structural_native_pilot/flow_structural_embedding_index.jsonl"
    )


def _tower2_args(**overrides):
    values = dict(
        dataset="vpn-app",
        splits="train,valid,test",
        embedding_suffix="rawproj_flowaware_change_weight_split2_retrain",
        tower2_suffix="",
        native_structural_suffix="struct_pilot",
        run_tag="unit",
        num_classes=16,
        tower2_epochs=1,
        tower2_batch_size=16,
        hidden_dim=256,
        num_layers=2,
        num_heads=4,
        dropout=0.15,
        tower2_lr=1e-4,
        weight_decay=0.03,
        tower2_select_metric="content_group_macro_f1",
        tower2_early_stop_patience=3,
        flow_pooling="multi_view",
        multi_view_gate_entropy_weight=0.0,
        flow_stat_expert_weight=0.2,
        flow_stat_aux_weight=0.05,
        window_loss_weight=0.3,
        class_weight_strength=0.6,
        label_smoothing=0.05,
        confidence_penalty_weight=0.0,
        hierarchical_weight=0.2,
        hierarchical_logit_weight=0.5,
        coarse_groups="vpn_app",
        samples_per_class=2,
        split_group_key="content_group_id",
        content_group_loss_reduction="group_mean",
        content_group_unique_batches=True,
        contrastive_mode="standard",
        confusion_groups="vpn_app",
        flow_contrastive_weight=0.03,
        flow_temperature=0.07,
        window_contrastive_weight=0.0,
        window_contrastive_temperature=0.07,
        window_contrastive_positive="same_class",
        consistency_weight=0.0,
        meta_dropout_prob=0.05,
        meta_feature_dim=14,
        embedding_dropout_prob=0.0,
        window_dropout_prob=0.0,
        edge_attr_dropout_prob=0.05,
        seed=42,
        paired_embedding_suffix="",
        environment_map_json="",
        distill_targets_json="",
        native_structural_dim=0,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_stage8_does_not_auto_pass_native_dim_in_concat_mode():
    cmd = tower2_train_cmd(_tower2_args(), "seq")

    assert "--native_structural_dim" not in cmd
    assert cmd[cmd.index("--contrastive_mode") + 1] == "standard"

    explicit_cmd = tower2_train_cmd(_tower2_args(native_structural_dim=128), "seq")
    assert explicit_cmd[explicit_cmd.index("--native_structural_dim") + 1] == "128"


def test_stage8_forwards_shared_packet_evidence_candidate_and_existing_pair_data():
    cmd = tower2_train_cmd(
        _tower2_args(
            flow_pooling="late_fusion",
            exact_shared_packet_encoder=True,
            shared_packet_hidden_dim=128,
            packet_evidence_max_weight=0.4,
            native_structural_dim=128,
            paired_embedding_suffix="existing_masked_view",
        ),
        "seq",
    )

    assert cmd[cmd.index("--flow_pooling") + 1] == "late_fusion"
    assert cmd[cmd.index("--packet_evidence_max_weight") + 1] == "0.4"
    assert "--exact_shared_packet_encoder" in cmd
    assert (
        cmd[cmd.index("--paired_view_dataset") + 1]
        == "reasoningDataset/vpn-app/train_tower2_existing_masked_view/seq_dataset.pt"
    )


def test_stage8_candidate_evaluation_can_be_validation_only():
    args = _tower2_args(
        stage="eval",
        eval_splits="valid",
        model_types="seq",
        run_tag="packet_evidence_validation",
        tower1_data_suffix="strict_shared_core_v2_fold0",
        use_intervention_views=False,
    )

    generated = list(commands(args))

    assert len(generated) == 1
    assert "valid_tower2_" in generated[0][generated[0].index("--dataset") + 1]
    assert "valid_seq_metrics" in generated[0][generated[0].index("--output_json") + 1]


def test_stage8_rejects_invalid_or_empty_evaluation_splits():
    with pytest.raises(ValueError, match="Unknown evaluation split"):
        selected_eval_splits("valid,train")
    with pytest.raises(ValueError, match="At least one"):
        selected_eval_splits("")
