from __future__ import annotations

import torch
import torch.nn.functional as F


def _validate_inputs(
    factual: torch.Tensor,
    intervened: torch.Tensor,
    labels: torch.Tensor,
    flow_ids: torch.Tensor,
    packet_ids: torch.Tensor,
    intervention_mask: torch.Tensor,
) -> None:
    if factual.ndim != 2 or intervened.shape != factual.shape:
        raise ValueError("factual and intervened representations must have shape [N, D]")
    expected = factual.size(0)
    for name, value in {
        "labels": labels,
        "flow_ids": flow_ids,
        "packet_ids": packet_ids,
        "intervention_mask": intervention_mask,
    }.items():
        if value.ndim != 1 or value.numel() != expected:
            raise ValueError(f"{name} must have shape [N]")


def _identity_means(
    values: torch.Tensor,
    packet_inverse: torch.Tensor,
    num_packets: int,
    row_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    sums = values.new_zeros((num_packets, values.size(-1)))
    counts = values.new_zeros((num_packets,))
    selected = row_mask.nonzero(as_tuple=False).flatten()
    if selected.numel() > 0:
        sums.index_add_(0, packet_inverse[selected], values[selected])
        counts.index_add_(0, packet_inverse[selected], values.new_ones(selected.numel()))
    means = sums / counts.clamp(min=1.0).unsqueeze(1)
    return F.normalize(means, p=2, dim=-1), counts > 0


def _consistent_identity_metadata(
    values: torch.Tensor,
    packet_inverse: torch.Tensor,
    num_packets: int,
    name: str,
) -> torch.Tensor:
    result = values.new_empty((num_packets,))
    for packet_index in range(num_packets):
        packet_values = values[packet_inverse == packet_index]
        if packet_values.numel() == 0:
            raise RuntimeError("internal packet identity compaction error")
        if not torch.all(packet_values == packet_values[0]):
            raise ValueError(f"repeated packet identity has conflicting {name}")
        result[packet_index] = packet_values[0]
    return result


def _cross_view_direction(
    anchors: torch.Tensor,
    anchor_available: torch.Tensor,
    contexts: torch.Tensor,
    context_available: torch.Tensor,
    labels: torch.Tensor,
    flow_inverse: torch.Tensor,
    num_flows: int,
    temperature: float,
) -> torch.Tensor:
    context_sums = contexts.new_zeros((num_flows, contexts.size(-1)))
    context_counts = contexts.new_zeros((num_flows,))
    selected = context_available.nonzero(as_tuple=False).flatten()
    if selected.numel() > 0:
        context_sums.index_add_(0, flow_inverse[selected], contexts[selected])
        context_counts.index_add_(0, flow_inverse[selected], contexts.new_ones(selected.numel()))

    flow_labels = labels.new_empty((num_flows,))
    for flow_index in range(num_flows):
        values = labels[flow_inverse == flow_index]
        if not torch.all(values == values[0]):
            raise ValueError("packets from one flow have conflicting labels")
        flow_labels[flow_index] = values[0]

    losses = []
    scale = max(float(temperature), 1e-6)
    for packet_index in range(anchors.size(0)):
        if not bool(anchor_available[packet_index]):
            continue
        own_flow = int(flow_inverse[packet_index])
        own_count = context_counts[own_flow] - context_available[packet_index].to(context_counts.dtype)
        if float(own_count) <= 0:
            continue

        candidate_sums = context_sums.clone()
        candidate_counts = context_counts.clone()
        if bool(context_available[packet_index]):
            candidate_sums[own_flow] = candidate_sums[own_flow] - contexts[packet_index]
            candidate_counts[own_flow] = candidate_counts[own_flow] - 1.0
        candidate_available = candidate_counts > 0

        # Same-class flows are neither negatives nor extra positives. The objective
        # teaches cross-scale context without separating instances of one class.
        different_class = flow_labels != labels[packet_index]
        denominator_mask = candidate_available & different_class
        denominator_mask[own_flow] = True
        if int(denominator_mask.sum()) <= 1:
            continue

        prototypes = F.normalize(
            candidate_sums / candidate_counts.clamp(min=1.0).unsqueeze(1),
            p=2,
            dim=-1,
        )
        logits = torch.matmul(anchors[packet_index], prototypes.T) / scale
        logits = logits.masked_fill(~denominator_mask, -torch.inf)
        losses.append(torch.logsumexp(logits, dim=0) - logits[own_flow])

    if not losses:
        return anchors.sum() * 0.0
    return torch.stack(losses).mean()


def availability_aware_cross_scale_loss(
    factual: torch.Tensor,
    intervened: torch.Tensor,
    labels: torch.Tensor,
    flow_ids: torch.Tensor,
    packet_ids: torch.Tensor,
    intervention_mask: torch.Tensor | None = None,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Align a packet with real leave-one-out flow context across interventions.

    Repeated sampler aliases are compacted by exact packet identity. A packet is
    eligible only when another distinct packet from its flow supplies context.
    Factual packets retrieve intervened flow context and intervened packets
    retrieve factual context; same-class flows are excluded from the negatives.
    """
    if intervention_mask is None:
        intervention_mask = torch.ones_like(labels, dtype=torch.bool)
    _validate_inputs(
        factual,
        intervened,
        labels,
        flow_ids,
        packet_ids,
        intervention_mask,
    )
    if factual.size(0) <= 1:
        return factual.sum() * 0.0

    factual = F.normalize(factual.float(), p=2, dim=-1)
    intervened = F.normalize(intervened.float(), p=2, dim=-1)
    labels = labels.long()
    flow_ids = flow_ids.long()
    packet_ids = packet_ids.long()
    intervention_mask = intervention_mask.bool()

    _, packet_inverse = torch.unique(packet_ids, sorted=True, return_inverse=True)
    num_packets = int(packet_inverse.max()) + 1
    compact_labels = _consistent_identity_metadata(labels, packet_inverse, num_packets, "labels")
    compact_flows = _consistent_identity_metadata(flow_ids, packet_inverse, num_packets, "flow_ids")
    factual_packets, factual_available = _identity_means(
        factual,
        packet_inverse,
        num_packets,
        torch.ones_like(intervention_mask),
    )
    intervened_packets, intervened_available = _identity_means(
        intervened,
        packet_inverse,
        num_packets,
        intervention_mask,
    )
    _, flow_inverse = torch.unique(compact_flows, sorted=True, return_inverse=True)
    num_flows = int(flow_inverse.max()) + 1

    factual_to_intervened = _cross_view_direction(
        factual_packets,
        factual_available,
        intervened_packets,
        intervened_available,
        compact_labels,
        flow_inverse,
        num_flows,
        temperature,
    )
    intervened_to_factual = _cross_view_direction(
        intervened_packets,
        intervened_available,
        factual_packets,
        factual_available,
        compact_labels,
        flow_inverse,
        num_flows,
        temperature,
    )
    return 0.5 * (factual_to_intervened + intervened_to_factual)
