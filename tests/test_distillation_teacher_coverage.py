from audit_distillation_teacher_coverage import recommendation, teacher_contract
import pytest
import torch

from train_tower2 import cap_prob_target_confidence, load_distillation_targets


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


def test_single_teacher_union_fails_multi_teacher_contract():
    teacher = {
        "flow_ids": ["a", "b"],
        "teacher_multiplicity": {
            "flow_ids": ["b", "a"],
            "teacher_counts": [1, 1],
            "oof_exclusion_proven": False,
        },
    }

    contract = teacher_contract(teacher, min_teachers_per_flow=2, require_oof_exclusion_proof=False)

    assert contract["multiplicity_available_and_aligned"] is True
    assert contract["minimum_teacher_count"] == 1
    assert contract["passes"] is False
    assert recommendation(
        [{"passes_min_coverage": True}],
        "disable_flow",
        teacher_gate_passes=contract["passes"],
    ) == "disable_flow_id_kl_keep_class_prior"


def test_multi_teacher_still_fails_when_oof_exclusion_is_unproven():
    teacher = {
        "flow_ids": ["a", "b"],
        "teacher_multiplicity": {
            "flow_ids": ["a", "b"],
            "teacher_counts": [3, 2],
            "oof_exclusion_proven": False,
        },
    }

    permissive = teacher_contract(teacher, 2, require_oof_exclusion_proof=False)
    strict = teacher_contract(teacher, 2, require_oof_exclusion_proof=True)

    assert permissive["passes"] is True
    assert strict["passes_teacher_count"] is True
    assert strict["passes_oof_exclusion"] is False
    assert strict["passes"] is False


def test_oof_boolean_without_aligned_per_flow_counts_fails_strict_contract():
    teacher = {
        "flow_ids": ["a", "b"],
        "teacher_multiplicity": {
            "flow_ids": ["a", "b"],
            "teacher_counts": [2, 2],
            "oof_exclusion_proven": True,
            "oof_multi_teacher_consensus_proven": True,
        },
    }

    contract = teacher_contract(teacher, 2, require_oof_exclusion_proof=True)

    assert contract["oof_counts_available_and_aligned"] is False
    assert contract["passes_oof_exclusion"] is False
    assert contract["passes"] is False


def test_legacy_teacher_without_multiplicity_cannot_pass_teacher_count_gate():
    contract = teacher_contract(
        {"flow_ids": ["a"]}, 1, require_oof_exclusion_proof=False
    )

    assert contract["multiplicity_available_and_aligned"] is False
    assert contract["passes"] is False


def write_teacher(path, multiplicity=None):
    payload = {
        "flow_ids": ["a", "b"],
        "flow_prob": [[0.8, 0.2], [0.3, 0.7]],
    }
    if multiplicity is not None:
        payload["teacher_multiplicity"] = multiplicity
    import json

    path.write_text(json.dumps(payload), encoding="utf-8")


def test_train_loader_keeps_legacy_reproduction_but_rejects_strict_legacy(tmp_path):
    path = tmp_path / "teacher.json"
    write_teacher(path)

    assert set(load_distillation_targets(str(path), 2, "cpu")) == {"a", "b"}
    with pytest.raises(ValueError, match="lacks teacher_multiplicity"):
        load_distillation_targets(str(path), 2, "cpu", min_teachers_per_flow=2)


def test_train_loader_filters_by_aligned_teacher_count(tmp_path):
    path = tmp_path / "teacher.json"
    write_teacher(
        path,
        {
            "flow_ids": ["b", "a"],
            "teacher_counts": [1, 3],
            "oof_exclusion_proven": False,
        },
    )

    targets = load_distillation_targets(
        str(path), 2, "cpu", min_teachers_per_flow=2
    )

    assert set(targets) == {"a"}
    with pytest.raises(ValueError, match="does not prove per-flow OOF"):
        load_distillation_targets(
            str(path),
            2,
            "cpu",
            min_teachers_per_flow=2,
            require_oof_exclusion_proof=True,
        )


def test_train_loader_requires_every_contributing_teacher_to_be_oof(tmp_path):
    path = tmp_path / "teacher.json"
    write_teacher(
        path,
        {
            "flow_ids": ["a", "b"],
            "teacher_counts": [2, 2],
            "oof_teacher_counts": [2, 1],
            "oof_exclusion_proven": True,
            "oof_multi_teacher_consensus_proven": True,
        },
    )

    with pytest.raises(ValueError, match="unproven contributing teachers"):
        load_distillation_targets(
            str(path),
            2,
            "cpu",
            min_teachers_per_flow=2,
            require_oof_exclusion_proof=True,
        )


def test_train_loader_accepts_aligned_multi_teacher_oof_contract(tmp_path):
    path = tmp_path / "teacher.json"
    write_teacher(
        path,
        {
            "flow_ids": ["b", "a"],
            "teacher_counts": [2, 3],
            "oof_teacher_counts": [2, 3],
            "oof_exclusion_proven": True,
            "oof_multi_teacher_consensus_proven": True,
        },
    )

    targets = load_distillation_targets(
        str(path),
        2,
        "cpu",
        min_teachers_per_flow=2,
        require_oof_exclusion_proof=True,
    )

    assert set(targets) == {"a", "b"}


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
