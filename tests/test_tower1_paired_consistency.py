import pytest
import torch

from models.qwen_packet_multitask import (
    active_group_objective,
    grouped_means,
    paired_view_consistency_loss,
)


def test_raw_consistency_covers_concat_embedding_component():
    projected = torch.tensor([[1.0, 0.0]])
    logits = torch.tensor([[2.0, 0.0]])
    raw = torch.tensor([[1.0, 0.0]])
    paired_raw = torch.tensor([[-1.0, 0.0]])
    projected_only = paired_view_consistency_loss(
        projected,
        projected,
        logits,
        logits,
        logit_kl_weight=0.0,
        raw_z=raw,
        paired_raw_z=paired_raw,
        raw_consistency_weight=0.0,
    )
    concat_consistency = paired_view_consistency_loss(
        projected,
        projected,
        logits,
        logits,
        logit_kl_weight=0.0,
        raw_z=raw,
        paired_raw_z=paired_raw,
        raw_consistency_weight=1.0,
    )
    assert projected_only.item() == pytest.approx(0.0)
    assert concat_consistency.item() == pytest.approx(2.0)


def test_raw_consistency_requires_aligned_pair():
    value = torch.tensor([[1.0, 0.0]])
    with pytest.raises(ValueError, match="both raw_z and paired_raw_z"):
        paired_view_consistency_loss(
            value,
            value,
            value,
            value,
            raw_z=value,
            paired_raw_z=None,
        )


def test_raw_consistency_weight_must_be_non_negative():
    value = torch.tensor([[1.0, 0.0]])
    with pytest.raises(ValueError, match="non-negative"):
        paired_view_consistency_loss(
            value,
            value,
            value,
            value,
            raw_z=value,
            paired_raw_z=value,
            raw_consistency_weight=-1.0,
        )


def test_factual_teacher_stops_gradient_only_on_factual_view():
    factual = torch.tensor([[1.0, 0.2]], requires_grad=True)
    intervened = torch.tensor([[0.2, 1.0]], requires_grad=True)
    factual_logits = torch.tensor([[2.0, -1.0]], requires_grad=True)
    intervened_logits = torch.tensor([[-1.0, 2.0]], requires_grad=True)
    loss = paired_view_consistency_loss(
        factual,
        intervened,
        factual_logits,
        intervened_logits,
        consistency_mode="factual_teacher",
    )
    loss.backward()
    assert factual.grad is None
    assert factual_logits.grad is None
    assert intervened.grad.abs().sum() > 0
    assert intervened_logits.grad.abs().sum() > 0


def test_group_objective_uses_group_means_not_group_frequency():
    values = torch.tensor([1.0, 3.0, 10.0])
    groups = torch.tensor([0, 0, 1])
    means, counts = grouped_means(values, groups, num_groups=2)
    objective = active_group_objective(
        means, counts, torch.tensor([0.25, 0.75])
    )
    assert means.tolist() == [2.0, 10.0]
    assert counts.tolist() == [2.0, 1.0]
    assert objective.item() == pytest.approx(8.0)
