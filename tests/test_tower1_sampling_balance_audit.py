import json

from analyze_tower1_sampling_balance import (
    audit,
    flow_balanced_objective_exposure,
    normalized_class_weights,
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
