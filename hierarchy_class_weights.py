#!/usr/bin/env python3
"""Pure class-risk interpolation used by the conditional hierarchy screen."""
from __future__ import annotations

import math
from collections.abc import Mapping


def normalized_effective_weights(
    counts: Mapping[int, int], *, beta: float = 0.9999
) -> dict[int, float]:
    if not 0.0 < beta < 1.0:
        raise ValueError("beta must be in (0,1)")
    if not counts or any(int(value) <= 0 for value in counts.values()):
        raise ValueError("all class counts must be positive")
    raw = {
        int(label): (1.0 - beta) / max(1.0 - beta ** int(count), 1e-12)
        for label, count in counts.items()
    }
    mean = sum(raw.values()) / len(raw)
    return {label: value / mean for label, value in raw.items()}


def bounded_flow_risk_strength(
    flow_counts: Mapping[int, int],
    *,
    max_weight_ratio: float,
    beta: float = 0.9999,
    max_strength: float = 1.0,
) -> float:
    """Choose the strongest flow-risk exponent that respects a ratio bound.

    Raising positive class risks to ``eta`` raises their max/min ratio to the
    same power.  This gives a closed-form, train-only strength that adapts to
    class geometry without a dataset-specific search rule.
    """
    if not math.isfinite(max_weight_ratio) or max_weight_ratio <= 1.0:
        raise ValueError("max_weight_ratio must be finite and greater than 1")
    if not 0.0 <= max_strength <= 1.0:
        raise ValueError("max_strength must be in [0,1]")
    weights = normalized_effective_weights(flow_counts, beta=beta)
    observed_ratio = max(weights.values()) / min(weights.values())
    if observed_ratio <= 1.0 + 1e-12:
        return max_strength
    bounded = math.log(max_weight_ratio) / math.log(observed_ratio)
    return min(max_strength, max(0.0, bounded))


def hierarchy_class_weights(
    packet_counts: Mapping[int, int],
    flow_counts: Mapping[int, int],
    *,
    alpha: float,
    gamma: float,
    beta: float = 0.9999,
) -> dict[int, float]:
    """Geometrically interpolate packet/flow risks, then apply strength gamma."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0,1]")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0,1]")
    packet = normalized_effective_weights(packet_counts, beta=beta)
    flow = normalized_effective_weights(flow_counts, beta=beta)
    if set(packet) != set(flow):
        raise ValueError("packet and flow count labels must match")
    raw = {
        label: math.exp(
            gamma
            * (
                (1.0 - alpha) * math.log(packet[label])
                + alpha * math.log(flow[label])
            )
        )
        for label in sorted(packet)
    }
    mean = sum(raw.values()) / len(raw)
    return {label: value / mean for label, value in raw.items()}
