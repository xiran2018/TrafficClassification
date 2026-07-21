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

