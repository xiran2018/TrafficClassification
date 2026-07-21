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


def test_same_class_pairing_visits_each_flow_once_and_keeps_pairs_together():
    source = [
        {"flow_id": f"a{index}", "packet_uid": f"a{index}", "label_id": 0}
        for index in range(4)
    ] + [
        {"flow_id": f"b{index}", "packet_uid": f"b{index}", "label_id": 1}
        for index in range(4)
    ]
    batches = list(
        sampled_batches(
            source,
            batch_size=4,
            packets_per_flow=1,
            seed=42,
            epoch=0,
            flow_pairing="same_class",
        )
    )

    visited = [source[index]["flow_id"] for batch in batches for index in batch]
    assert sorted(visited) == sorted(row["flow_id"] for row in source)
    for batch in batches:
        labels = [source[index]["label_id"] for index in batch]
        assert all(labels.count(label) >= 2 for label in set(labels))


def test_same_class_pairing_rejects_odd_flow_capacity():
    with pytest.raises(ValueError, match="even number of flows"):
        list(
            sampled_batches(
                rows(),
                batch_size=6,
                packets_per_flow=2,
                seed=42,
                epoch=0,
                flow_pairing="same_class",
            )
        )


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
        batch_size=8,
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


def test_same_class_pairing_restores_identity_safe_positive_coverage():
    report = audit_rows(
        rows(),
        batch_size=8,
        packets_per_flow=2,
        epochs=2,
        seed=42,
        same_flow_weight=1.0,
        same_label_weight=1.0,
        flow_pairing="same_class",
    )

    assert report["flow_pairing"] == "same_class"
    assert report["aggregate"]["identity_safe_valid_anchor_rate"] == 1.0


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
