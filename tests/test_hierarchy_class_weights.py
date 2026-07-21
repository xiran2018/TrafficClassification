import math

import pytest

from hierarchy_class_weights import (
    hierarchy_class_weights,
    normalized_effective_weights,
)


def assert_weights_close(left, right):
    assert set(left) == set(right)
    for label in left:
        assert math.isclose(left[label], right[label], rel_tol=1e-12, abs_tol=1e-12)


def test_formula_recovers_packet_and_flow_corners():
    packet_counts = {0: 100, 1: 50, 2: 25}
    flow_counts = {0: 20, 1: 10, 2: 2}

    packet_corner = hierarchy_class_weights(
        packet_counts, flow_counts, alpha=0.0, gamma=1.0
    )
    flow_corner = hierarchy_class_weights(
        packet_counts, flow_counts, alpha=1.0, gamma=1.0
    )

    assert_weights_close(packet_corner, normalized_effective_weights(packet_counts))
    assert_weights_close(flow_corner, normalized_effective_weights(flow_counts))


def test_balanced_packet_counts_collapse_alpha_gamma_to_eta():
    packet_counts = {0: 100, 1: 100, 2: 100}
    flow_counts = {0: 20, 1: 10, 2: 2}

    alpha_half = hierarchy_class_weights(
        packet_counts, flow_counts, alpha=0.5, gamma=1.0
    )
    gamma_half = hierarchy_class_weights(
        packet_counts, flow_counts, alpha=1.0, gamma=0.5
    )

    assert_weights_close(alpha_half, gamma_half)


def test_alpha_and_gamma_remain_distinct_when_packet_counts_are_not_balanced():
    packet_counts = {0: 100, 1: 50, 2: 25}
    flow_counts = {0: 20, 1: 10, 2: 2}
    alpha_half = hierarchy_class_weights(
        packet_counts, flow_counts, alpha=0.5, gamma=1.0
    )
    gamma_half = hierarchy_class_weights(
        packet_counts, flow_counts, alpha=1.0, gamma=0.5
    )

    assert any(
        not math.isclose(alpha_half[label], gamma_half[label], rel_tol=1e-6)
        for label in alpha_half
    )


@pytest.mark.parametrize(
    ("alpha", "gamma"), [(-0.1, 1.0), (1.1, 1.0), (1.0, -0.1), (1.0, 1.1)]
)
def test_rejects_parameters_outside_unit_interval(alpha, gamma):
    with pytest.raises(ValueError, match="must be in"):
        hierarchy_class_weights(
            {0: 1, 1: 1}, {0: 1, 1: 1}, alpha=alpha, gamma=gamma
        )

