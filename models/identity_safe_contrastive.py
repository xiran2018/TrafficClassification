import torch
import torch.nn.functional as F


def first_packet_identity_mask(packet_ids: torch.Tensor) -> torch.Tensor:
    """Keep exactly the first batch occurrence of every packet identity."""
    packet_ids = packet_ids.reshape(-1)
    positions = torch.arange(packet_ids.numel(), device=packet_ids.device)
    same_identity = packet_ids[:, None].eq(packet_ids[None, :])
    has_earlier_copy = (same_identity & positions[None, :].lt(positions[:, None])).any(
        dim=1
    )
    return ~has_earlier_copy


def identity_safe_flow_aware_contrastive_loss(
    z: torch.Tensor,
    labels: torch.Tensor,
    flow_ids: torch.Tensor,
    packet_ids: torch.Tensor,
    temperature: float = 0.07,
    same_flow_weight: float = 1.0,
    same_label_weight: float = 1.0,
) -> torch.Tensor:
    """Flow-aware SupCon after removing repeated packet identities.

    Classification may still consume repeated sampler rows. This objective
    restricts every contrastive role to one occurrence per packet identity.
    """
    batch_size = z.size(0)
    if any(value.reshape(-1).numel() != batch_size for value in (labels, flow_ids, packet_ids)):
        raise ValueError("z, labels, flow_ids, and packet_ids must share batch size")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if same_flow_weight < 0.0 or same_label_weight < 0.0:
        raise ValueError("positive weights must be nonnegative")
    if batch_size <= 1:
        return z.sum() * 0.0

    keep = first_packet_identity_mask(packet_ids)
    unique_z = z[keep]
    unique_labels = labels.reshape(-1)[keep]
    unique_flows = flow_ids.reshape(-1)[keep]
    if unique_z.size(0) <= 1:
        return z.sum() * 0.0

    normalized = F.normalize(unique_z.float(), p=2, dim=-1)
    logits = torch.matmul(normalized, normalized.T) / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()

    self_mask = torch.eye(unique_z.size(0), dtype=torch.bool, device=z.device)
    same_label = unique_labels[:, None].eq(unique_labels[None, :]) & ~self_mask
    same_flow = unique_flows[:, None].eq(unique_flows[None, :]) & ~self_mask
    positive_weight = same_label.float() * float(same_label_weight)
    positive_weight += same_flow.float() * float(same_flow_weight)
    valid = positive_weight.sum(dim=1) > 0.0
    if not valid.any():
        return z.sum() * 0.0

    denominator_logits = logits.masked_fill(self_mask, -torch.inf)
    valid_logits = denominator_logits[valid]
    valid_positive_weight = positive_weight[valid]
    log_probability = valid_logits - torch.logsumexp(
        valid_logits, dim=1, keepdim=True
    )
    positive_log_probability = log_probability.masked_fill(
        valid_positive_weight.eq(0.0), 0.0
    )
    mean_log_positive = (
        valid_positive_weight * positive_log_probability
    ).sum(dim=1) / valid_positive_weight.sum(dim=1).clamp(min=1e-12)
    return -mean_log_positive.mean()
