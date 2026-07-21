from __future__ import annotations

from types import SimpleNamespace

from run_autonomous_research_loop import VARIANT_SCHEDULES, suite_cmd
from run_recommended_experiment import stage_commands
from run_recommended_suite import dataset_cmd


def _flag_value(cmd: list[str], flag: str) -> str:
    assert flag in cmd
    return cmd[cmd.index(flag) + 1]


def test_recommended_suite_passes_crossview_invariance_weights():
    args = SimpleNamespace(
        run_tag="unit",
        model_types="graph,seq",
        tower2_epochs=1,
        tower2_early_stop_patience=1,
        paired_view_weight=0.2,
        paired_consistency_weight=0.1,
        paired_alignment_weight=0.05,
        paired_crossview_contrastive_weight=0.03,
        paired_crossview_temperature=0.11,
        paired_variance_weight=0.07,
        paired_variance_target=0.02,
        paired_covariance_weight=0.005,
        view_domain_adversarial_weight=0.04,
        consistency_weight=0.05,
        meta_dropout_prob=0.1,
        embedding_dropout_prob=0.05,
        window_dropout_prob=0.1,
        edge_attr_dropout_prob=0.1,
        confidence_penalty_weight=0.0,
        seed=42,
        flow_pooling="mean",
        multi_view_gate_entropy_weight=0.0,
        final_selector_unified_expert_slots="base,paired",
        distill_target=[],
        distill_weight=0.0,
        distill_class_prior_weight=0.0,
        distill_temperature=2.0,
        distill_min_confidence=0.0,
        distill_confidence_power=0.0,
        distill_min_coverage=0.0,
        distill_low_coverage_action="warn",
        enable_slot_stacker=True,
        enable_soft_expert_gate=True,
        execute=False,
        allow_no_cuda=True,
        no_skip_existing=False,
        require_cuda_for_tower2=False,
    )

    cmd = dataset_cmd(args, "vpn-app")

    assert _flag_value(cmd, "--framework_profile") == "paper_unified"
    assert _flag_value(cmd, "--paired_alignment_weight") == "0.05"
    assert _flag_value(cmd, "--paired_crossview_contrastive_weight") == "0.03"
    assert _flag_value(cmd, "--paired_crossview_temperature") == "0.11"
    assert _flag_value(cmd, "--paired_variance_weight") == "0.07"
    assert _flag_value(cmd, "--paired_variance_target") == "0.02"
    assert _flag_value(cmd, "--paired_covariance_weight") == "0.005"
    assert _flag_value(cmd, "--view_domain_adversarial_weight") == "0.04"


def test_autonomous_loop_variant_passes_crossview_invariance_weights():
    args = SimpleNamespace(
        model_types="graph,seq",
        tower2_epochs=1,
        tower2_early_stop_patience=1,
        final_selector_unified_expert_slots="base,paired",
        distill_target=[],
        distill_weight=0.0,
        distill_class_prior_weight=0.0,
        distill_temperature=2.0,
        distill_min_confidence=0.0,
        distill_confidence_power=0.0,
        distill_min_coverage=0.0,
        distill_low_coverage_action="warn",
        enable_slot_stacker=True,
        enable_soft_expert_gate=True,
        execute=False,
        allow_no_cuda=True,
        continue_on_error=False,
        suite_output_dir="reasoningDataset/autonomous_loop",
    )
    variant = {
        "paired_view_weight": 0.1,
        "paired_consistency_weight": 0.15,
        "paired_alignment_weight": 0.05,
        "paired_crossview_contrastive_weight": 0.02,
        "paired_crossview_temperature": 0.09,
        "paired_variance_weight": 0.03,
        "paired_variance_target": 0.02,
        "paired_covariance_weight": 0.004,
        "view_domain_adversarial_weight": 0.01,
        "consistency_weight": 0.1,
        "meta_dropout_prob": 0.2,
        "embedding_dropout_prob": 0.1,
        "window_dropout_prob": 0.15,
        "edge_attr_dropout_prob": 0.15,
        "confidence_penalty_weight": 0.0,
        "seed": 43,
        "flow_pooling": "mean",
        "multi_view_gate_entropy_weight": 0.0,
    }

    cmd = suite_cmd(args, ["vpn-app"], 1, "unit", variant)

    assert _flag_value(cmd, "--framework_profile") == "paper_unified"
    assert _flag_value(cmd, "--paired_alignment_weight") == "0.05"
    assert _flag_value(cmd, "--paired_crossview_contrastive_weight") == "0.02"
    assert _flag_value(cmd, "--paired_crossview_temperature") == "0.09"
    assert _flag_value(cmd, "--paired_variance_weight") == "0.03"
    assert _flag_value(cmd, "--paired_variance_target") == "0.02"
    assert _flag_value(cmd, "--paired_covariance_weight") == "0.004"
    assert _flag_value(cmd, "--view_domain_adversarial_weight") == "0.01"


def test_stage8_balanced_prioritizes_gentle_intervention():
    variant = VARIANT_SCHEDULES["stage8_balanced"][0]

    assert variant["name"] == "gentle_intervention"
    assert variant["paired_view_weight"] == 0.05
    assert variant["paired_alignment_weight"] == 0.005
    assert variant["paired_crossview_contrastive_weight"] == 0.0
    assert variant["view_domain_adversarial_weight"] == 0.0


def test_zero_paired_losses_skip_paired_view_preparation():
    args = SimpleNamespace(
        dataset="tls-120",
        num_classes=120,
        label_map="",
        model_types="graph,seq",
        seed=42,
        flow_pooling="mean",
        multi_view_gate_entropy_weight=0.0,
        no_progress=True,
        tower1_output_dir="checkpoints/tower1",
        embedding_suffix="rawproj_flowaware_change_weight_fold2",
        paired_output_suffix="flowaware_ipport_rand_change_weight_fold2",
        paired_embedding_suffix="rawproj_flowaware_ipport_rand_change_weight_fold2",
        embedding_header_policy="randomize_ip_port",
        run_tag="distill",
        paired_view_weight=0.0,
        paired_consistency_weight=0.0,
        paired_alignment_weight=0.0,
        paired_crossview_contrastive_weight=0.0,
        paired_crossview_temperature=0.07,
        paired_variance_weight=0.0,
        paired_variance_target=0.04,
        paired_covariance_weight=0.0,
        view_domain_adversarial_weight=0.0,
        consistency_weight=0.05,
        meta_dropout_prob=0.1,
        embedding_dropout_prob=0.05,
        window_dropout_prob=0.1,
        edge_attr_dropout_prob=0.1,
        confidence_penalty_weight=0.0,
        tower2_epochs=1,
        tower2_early_stop_patience=1,
        distill_targets_json="teacher.json",
        distill_weight=0.05,
        distill_class_prior_weight=0.01,
        distill_temperature=4.0,
        distill_min_confidence=0.45,
        distill_max_confidence=0.85,
        distill_confidence_power=1.0,
        distill_min_coverage=0.5,
        distill_low_coverage_action="fail",
        require_cuda_for_tower2=False,
        skip_existing=False,
        enable_slot_stacker=False,
        enable_soft_expert_gate=False,
        slot_stacker_output="",
        slot_stacker_inputs=[],
        include_paired_in_slot_stacker=True,
        slot_stacker_c_grid="0.01",
        slot_stacker_class_weight_grid="none",
        slot_stacker_select_metric="macro_f1",
        final_selector_output="",
        final_selector_inputs=[],
        base_selector_input="base.json",
        final_selector_metric="macro_f1",
        final_selector_rank_select_metric="accuracy",
        final_selector_rank_metric="bootstrap_gain_quantile",
        final_selector_rank_bootstrap_samples=10,
        final_selector_rank_candidate_limit=8,
        final_selector_strategies="always",
        final_selector_alpha_grid="0.5",
        final_selector_metric_margin_grid="0",
        final_selector_expert_conf_grid="0.3",
        final_selector_expert_margin_grid="0.05",
        final_selector_base_conf_max_grid="1",
        final_selector_delta_conf_grid="-1",
        final_selector_delta_margin_grid="-1",
        final_selector_reliability_power_grid="4",
        final_selector_confidence_power_grid="1",
        final_selector_reliability_min_weight_grid="0",
        final_selector_reliability_temperature_grid="0.5",
        final_selector_calibration_strength_grid="1.0",
        final_selector_calibration_temperature_grid="1.25",
        final_selector_min_valid_gain_over_base=0.0,
        final_selector_bootstrap_samples=10,
        final_selector_bootstrap_min_win_rate=0.6,
        final_selector_bootstrap_min_gain_quantile=-0.001,
        final_selector_max_prediction_change_rate=0.0,
        final_selector_unified_expert_slots="base,paired",
        extra_recommendation_dataset=[],
    )

    stages = stage_commands(args)
    stage_by_name = {stage["name"]: stage for stage in stages}

    assert stage_by_name["paired_tower1_preprocess"]["skip_if"] is True
    assert stage_by_name["paired_embeddings"]["skip_if"] is True
    assert stage_by_name["paired_tower2_preprocess"]["skip_if"] is True
    assert _flag_value(stage_by_name["paired_tower2_train"]["cmd"], "--framework_profile") == "paper_unified"
    assert _flag_value(stage_by_name["paired_tower2_train"]["cmd"], "--distill_max_confidence") == "0.85"
    assert "--paired_embedding_suffix" not in stage_by_name["paired_tower2_train"]["cmd"]
