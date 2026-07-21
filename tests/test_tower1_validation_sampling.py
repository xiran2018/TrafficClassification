from train_tower1_multitask import (
    flow_balanced_validation_rows,
    validation_patience_exhausted,
)


def test_flow_balanced_validation_rows_is_deterministic_and_non_repeating():
    rows = [
        {"flow_id": "b", "packet_uid": f"b-{index}"} for index in range(5)
    ] + [
        {"flow_id": "a", "packet_uid": f"a-{index}"} for index in range(3)
    ] + [
        {"flow_id": "short", "packet_uid": "short-0"}
    ]

    selected = flow_balanced_validation_rows(rows, packets_per_flow=2, seed=42)
    repeated = flow_balanced_validation_rows(list(reversed(rows)), packets_per_flow=2, seed=42)

    assert [row["packet_uid"] for row in selected] == [row["packet_uid"] for row in repeated]
    assert len(selected) == 5
    assert len({row["packet_uid"] for row in selected}) == len(selected)
    assert sum(row["flow_id"] == "a" for row in selected) == 2
    assert sum(row["flow_id"] == "b" for row in selected) == 2
    assert sum(row["flow_id"] == "short" for row in selected) == 1


def test_flow_balanced_validation_rows_zero_keeps_full_dataset():
    rows = [{"flow_id": "a", "packet_uid": str(index)} for index in range(4)]
    assert flow_balanced_validation_rows(rows, packets_per_flow=0, seed=42) == rows


def test_tower1_validation_patience_is_disabled_at_zero_and_exact_at_limit():
    assert not validation_patience_exhausted(100, 0)
    assert not validation_patience_exhausted(2, 3)
    assert validation_patience_exhausted(3, 3)
