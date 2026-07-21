import math

import pytest

from train_tower1_multitask import configure_packet_weights, packet_class_counts


def make_rows():
    return [
        {"flow_id": "a", "label_id": 0, "packet_weight": 0.5},
        {"flow_id": "a", "label_id": 0, "packet_weight": 0.5},
        {"flow_id": "b", "label_id": 0, "packet_weight": 0.5},
        {"flow_id": "c", "label_id": 1, "packet_weight": 0.5},
        {"flow_id": "c", "label_id": 1, "packet_weight": 0.5},
        {"flow_id": "c", "label_id": 1, "packet_weight": 0.5},
    ]


def test_packet_and_flow_class_counts_capture_different_sampling_distributions():
    rows = make_rows()
    assert packet_class_counts(rows, "packet") == {0: 3, 1: 3}
    assert packet_class_counts(rows, "flow") == {0: 2, 1: 1}


def test_flow_count_weights_upweight_classes_with_fewer_flows():
    rows = make_rows()
    weights = configure_packet_weights(
        rows,
        weighting="inverse",
        beta=0.9999,
        disable_information_weights=True,
        count_basis="flow",
        strength=1.0,
    )
    assert weights[1] > weights[0]
    assert math.isclose(sum(weights.values()) / len(weights), 1.0)
    assert rows[0]["packet_weight"] == weights[0]
    assert rows[-1]["packet_weight"] == weights[1]


def test_zero_strength_keeps_information_weights_only():
    rows = make_rows()
    weights = configure_packet_weights(
        rows,
        weighting="inverse",
        beta=0.9999,
        disable_information_weights=False,
        count_basis="flow",
        strength=0.0,
    )
    assert weights == {0: 1.0, 1: 1.0}
    assert all(row["packet_weight"] == 0.5 for row in rows)


def test_flow_counting_rejects_conflicting_flow_labels():
    rows = [
        {"flow_id": "a", "label_id": 0},
        {"flow_id": "a", "label_id": 1},
    ]
    with pytest.raises(ValueError, match="Conflicting labels"):
        packet_class_counts(rows, "flow")


def test_class_weight_strength_is_bounded():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        configure_packet_weights(
            make_rows(),
            weighting="effective",
            beta=0.9999,
            disable_information_weights=True,
            count_basis="flow",
            strength=1.1,
        )
