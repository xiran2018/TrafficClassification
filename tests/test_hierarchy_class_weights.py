import math

import pytest

from hierarchy_class_weights import (
    bounded_flow_risk_strength,
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


def test_bounded_strength_saturates_the_requested_weight_ratio():
    flow_counts = {0: 500, 1: 50, 2: 5}
    eta = bounded_flow_risk_strength(flow_counts, max_weight_ratio=4.0)
    unit_packet_counts = {label: 1 for label in flow_counts}
    weights = hierarchy_class_weights(
        unit_packet_counts,
        flow_counts,
        alpha=1.0,
        gamma=eta,
    )

    assert 0.0 < eta < 1.0
    assert max(weights.values()) / min(weights.values()) == pytest.approx(4.0)


def test_bounded_strength_keeps_full_strength_when_risk_is_already_bounded():
    flow_counts = {0: 12, 1: 11, 2: 10}

    assert bounded_flow_risk_strength(
        flow_counts, max_weight_ratio=4.0
    ) == pytest.approx(1.0)
    assert bounded_flow_risk_strength(
        flow_counts, max_weight_ratio=4.0, max_strength=0.6
    ) == pytest.approx(0.6)


def test_bounded_strength_is_invariant_to_normalization():
    flow_counts = {0: 1000, 1: 100, 2: 10}
    eta = bounded_flow_risk_strength(flow_counts, max_weight_ratio=3.0)
    base = normalized_effective_weights(flow_counts)
    powered = {label: weight**eta for label, weight in base.items()}

    assert max(powered.values()) / min(powered.values()) == pytest.approx(3.0)


@pytest.mark.parametrize("ratio", [1.0, 0.0, -2.0, float("inf")])
def test_bounded_strength_rejects_invalid_ratio(ratio):
    with pytest.raises(ValueError, match="max_weight_ratio"):
        bounded_flow_risk_strength({0: 2, 1: 1}, max_weight_ratio=ratio)


@pytest.mark.parametrize("strength", [-0.1, 1.1])
def test_bounded_strength_rejects_invalid_max_strength(strength):
    with pytest.raises(ValueError, match="max_strength"):
        bounded_flow_risk_strength(
            {0: 2, 1: 1}, max_weight_ratio=4.0, max_strength=strength
        )


@pytest.mark.parametrize(
    ("alpha", "gamma"), [(-0.1, 1.0), (1.1, 1.0), (1.0, -0.1), (1.0, 1.1)]
)
def test_rejects_parameters_outside_unit_interval(alpha, gamma):
    with pytest.raises(ValueError, match="must be in"):
        hierarchy_class_weights(
            {0: 1, 1: 1}, {0: 1, 1: 1}, alpha=alpha, gamma=gamma
        )
