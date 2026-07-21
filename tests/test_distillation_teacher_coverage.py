from audit_distillation_teacher_coverage import recommendation
import torch

from train_tower2 import cap_prob_target_confidence


def test_recommendation_allows_flow_distillation_when_all_rows_pass():
    rows = [
        {"passes_min_coverage": True},
        {"passes_min_coverage": True},
    ]
    assert recommendation(rows, "disable_flow") == "flow_id_distillation_safe"


def test_recommendation_disables_flow_kl_when_coverage_is_low():
    rows = [
        {"passes_min_coverage": False},
        {"passes_min_coverage": True},
    ]
    assert recommendation(rows, "disable_flow") == "disable_flow_id_kl_keep_class_prior"


def test_recommendation_can_fail_fast_when_configured():
    rows = [{"passes_min_coverage": False}]
    assert recommendation(rows, "fail") == "fail_before_training"


def test_recommendation_can_gate_on_train_subset_only():
    rows = [
        {"name": "train_seq", "passes_min_coverage": True},
        {"name": "valid_seq", "passes_min_coverage": False},
    ]
    assert recommendation(rows, "disable_flow", {"train_seq"}) == "flow_id_distillation_safe"


def test_distillation_teacher_confidence_soft_cap_preserves_probability_rows():
    prob = torch.tensor(
        [
            [0.99, 0.005, 0.005],
            [0.60, 0.30, 0.10],
        ],
        dtype=torch.float32,
    )

    capped = cap_prob_target_confidence(prob, 0.85)

    torch.testing.assert_close(capped.sum(dim=-1), torch.ones(2))
    assert float(capped[0].max()) <= 0.850001
    torch.testing.assert_close(capped[1], prob[1], atol=1e-6, rtol=1e-6)
