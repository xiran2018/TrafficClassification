import pytest
import torch

from models.identity_safe_contrastive import (
    first_packet_identity_mask,
    identity_safe_flow_aware_contrastive_loss,
)


def loss(z, labels, flows, packets):
    return identity_safe_flow_aware_contrastive_loss(
        z,
        torch.tensor(labels),
        torch.tensor(flows),
        torch.tensor(packets),
        temperature=0.2,
        same_flow_weight=1.0,
        same_label_weight=1.0,
    )


def test_first_packet_identity_mask_keeps_one_occurrence_per_identity():
    mask = first_packet_identity_mask(torch.tensor([7, 7, 9, 7, 11, 9]))
    assert mask.tolist() == [True, False, True, False, True, False]


def test_alias_embedding_cannot_change_identity_safe_loss_or_receive_gradient():
    unique = torch.tensor(
        [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]], requires_grad=True
    )
    unique_loss = loss(unique, [0, 0, 1], [10, 11, 12], [100, 200, 300])

    with_alias = torch.tensor(
        [[1.0, 0.0], [-99.0, 42.0], [0.8, 0.2], [0.0, 1.0]],
        requires_grad=True,
    )
    alias_loss = loss(
        with_alias,
        [0, 0, 0, 1],
        [10, 10, 11, 12],
        [100, 100, 200, 300],
    )

    assert alias_loss.detach().item() == pytest.approx(unique_loss.detach().item())
    alias_loss.backward()
    assert with_alias.grad[1].abs().sum().item() == 0.0
    assert with_alias.grad[[0, 2]].abs().sum().item() > 0.0


def test_no_real_positive_returns_differentiable_zero():
    z = torch.randn(3, 4, requires_grad=True)
    value = loss(z, [0, 1, 2], [10, 11, 12], [100, 200, 300])

    assert value.item() == 0.0
    value.backward()
    assert z.grad.abs().sum().item() == 0.0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"temperature": 0.0}, "temperature"),
        ({"same_flow_weight": -1.0}, "weights"),
        ({"same_label_weight": -1.0}, "weights"),
    ],
)
def test_invalid_objective_configuration_is_rejected(kwargs, message):
    z = torch.randn(2, 4)
    with pytest.raises(ValueError, match=message):
        identity_safe_flow_aware_contrastive_loss(
            z,
            torch.tensor([0, 0]),
            torch.tensor([1, 1]),
            torch.tensor([10, 11]),
            **kwargs,
        )


def test_batch_alignment_is_required():
    with pytest.raises(ValueError, match="share batch size"):
        identity_safe_flow_aware_contrastive_loss(
            torch.randn(2, 4),
            torch.tensor([0]),
            torch.tensor([1, 1]),
            torch.tensor([10, 11]),
        )
