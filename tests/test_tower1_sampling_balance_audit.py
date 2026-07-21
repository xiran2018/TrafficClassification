import json

import pytest

from analyze_tower1_sampling_balance import (
    audit,
    flow_balanced_objective_exposure,
    normalized_class_weights,
    same_flow_identity_collision_exposure,
)


def test_sampling_audit_reports_packet_and_flow_imbalance(tmp_path):
    path = tmp_path / "packet_auxiliary.jsonl"
    rows = [
        {"flow_id": "a", "label_id": 0},
        {"flow_id": "a", "label_id": 0},
        {"flow_id": "b", "label_id": 0},
        {"flow_id": "c", "label_id": 1},
        {"flow_id": "c", "label_id": 1},
        {"flow_id": "c", "label_id": 1},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    report = audit(path, method="effective", beta=0.9999, strengths=[0.5, 1.0])
    assert report["packet_imbalance_ratio"] == 1.0
    assert report["flow_imbalance_ratio"] == 2.0
    assert report["flow_count_weights"]["1.0"][1] > report["flow_count_weights"]["1.0"][0]
    assert report["flow_balanced_objective_exposure"]["unweighted"]["imbalance_ratio"] == 2.0
    assert report["flow_balanced_objective_exposure"]["0.5"]["imbalance_ratio"] < 2.0
    flow_audit = report["flow_length_audit"]
    assert flow_audit["singleton_flows"] == 1
    assert flow_audit["flows_requiring_replacement"] == 1
    assert flow_audit["flow_replacement_rate"] == pytest.approx(1 / 3)
    assert flow_audit["replacement_slots_per_epoch"] == 1
    assert flow_audit["sampled_slots_per_epoch"] == 6
    assert flow_audit["replacement_slot_rate"] == pytest.approx(1 / 6)
    assert flow_audit["class_flows_requiring_replacement"] == {0: 1}
    assert flow_audit["same_flow_positive_pairs_per_epoch"] == 6
    assert flow_audit["expected_duplicate_identity_positive_pairs_per_epoch"] == 2
    assert flow_audit["expected_distinct_identity_positive_pairs_per_epoch"] == 4
    assert flow_audit["expected_same_flow_positive_identity_collision_rate"] == pytest.approx(1 / 3)


def test_weight_strength_interpolates_without_changing_mean():
    counts = {0: 100, 1: 25}
    half = normalized_class_weights(counts, "inverse", beta=0.9999, strength=0.5)
    full = normalized_class_weights(counts, "inverse", beta=0.9999, strength=1.0)
    assert full[1] / full[0] == 4.0
    assert half[1] / half[0] == 2.0
    assert sum(half.values()) / len(half) == 1.0


def test_flow_balanced_exposure_matches_sampler_visits_times_loss_weight():
    exposure = flow_balanced_objective_exposure(
        {0: 100, 1: 25},
        {0: 0.5, 1: 2.0},
    )

    assert exposure["class_weighted_mass"] == {0: 50.0, 1: 50.0}
    assert exposure["normalized_mass"] == {0: 0.5, 1: 0.5}
    assert exposure["imbalance_ratio"] == 1.0


def test_sampling_audit_rejects_nonpositive_packets_per_flow(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="must be positive"):
        audit(path, method="effective", beta=0.9999, strengths=[1.0], packets_per_flow=0)


def test_identity_collision_expectation_handles_general_short_flows():
    exposure = same_flow_identity_collision_exposure(
        {"singleton": 1, "two_packets": 2, "long": 4},
        packets_per_flow=3,
    )

    assert exposure["same_flow_positive_pairs_per_epoch"] == 18
    assert exposure["expected_duplicate_identity_positive_pairs_per_epoch"] == 9
    assert exposure["expected_distinct_identity_positive_pairs_per_epoch"] == 9
    assert exposure["expected_same_flow_positive_identity_collision_rate"] == 0.5


def test_identity_collision_is_zero_without_positive_pairs():
    exposure = same_flow_identity_collision_exposure(
        {"singleton": 1, "long": 4}, packets_per_flow=1
    )
    assert exposure["same_flow_positive_pairs_per_epoch"] == 0
    assert exposure["expected_same_flow_positive_identity_collision_rate"] == 0.0
