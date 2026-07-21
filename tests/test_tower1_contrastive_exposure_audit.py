import json

import pytest

from audit_tower1_contrastive_exposure import (
    audit_rows,
    batch_exposure,
    load_rows,
    sampled_batches,
)


def rows():
    return [
        {"flow_id": "a", "packet_uid": "a0", "label_id": 0},
        {"flow_id": "a", "packet_uid": "a1", "label_id": 0},
        {"flow_id": "b", "packet_uid": "b0", "label_id": 0},
        {"flow_id": "c", "packet_uid": "c0", "label_id": 1},
        {"flow_id": "c", "packet_uid": "c1", "label_id": 1},
        {"flow_id": "c", "packet_uid": "c2", "label_id": 1},
    ]


def test_sampled_batches_reproduce_short_flow_replacement_and_epoch_seed():
    first = list(sampled_batches(rows(), 6, 2, seed=42, epoch=0))
    repeated = list(sampled_batches(rows(), 6, 2, seed=42, epoch=0))
    next_epoch = list(sampled_batches(rows(), 6, 2, seed=42, epoch=1))

    assert first == repeated
    assert first != next_epoch
    assert len(first) == 1
    assert len(first[0]) == 6
    assert first[0].count(2) == 2


def test_batch_exposure_deduplicates_alias_from_all_contrastive_roles():
    # Two copies of b0 simulate packets_per_flow=2 on a singleton flow.
    exposure = batch_exposure(
        rows(), [0, 1, 2, 2, 3, 4], same_flow_weight=1.0, same_label_weight=1.0
    )

    assert exposure["sampled_rows"] == 6
    assert exposure["unique_packet_identities"] == 5
    assert exposure["duplicate_rows"] == 1
    assert exposure["naive_denominator_pairs"] == 30
    assert exposure["identity_safe_denominator_pairs"] == 20
    assert exposure["alias_positive_pairs"] == 2
    assert exposure["alias_positive_weight_mass"] == 4.0
    assert exposure["naive_positive_weight_mass"] == 20.0
    assert exposure["identity_safe_positive_weight_mass"] == 12.0


def test_audit_reports_rates_without_using_predictions():
    report = audit_rows(
        rows(),
        batch_size=6,
        packets_per_flow=2,
        epochs=2,
        seed=42,
        same_flow_weight=1.0,
        same_label_weight=1.0,
    )

    aggregate = report["aggregate"]
    assert aggregate["duplicate_row_rate"] == pytest.approx(1 / 6)
    assert aggregate["denominator_pairs_removed_by_identity_dedup_rate"] == pytest.approx(1 / 3)
    assert report["interpretation"]["scope"].endswith("no_validation_or_test_predictions")


def test_load_rows_requires_unique_explicit_packet_identity(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text(
        json.dumps({"flow_id": "a", "packet_uid": "same", "label_id": 0})
        + "\n"
        + json.dumps({"flow_id": "a", "packet_uid": "same", "label_id": 0})
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate source packet_uid"):
        load_rows(path)
