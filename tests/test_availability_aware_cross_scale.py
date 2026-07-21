import torch

from models.availability_aware_cross_scale import (
    _cross_view_direction,
    availability_aware_cross_scale_loss,
)


def test_singleton_and_repeated_aliases_do_not_create_context():
    factual = torch.randn(4, 6, requires_grad=True)
    intervened = torch.randn(4, 6, requires_grad=True)
    loss = availability_aware_cross_scale_loss(
        factual,
        intervened,
        labels=torch.tensor([0, 0, 1, 1]),
        flow_ids=torch.tensor([10, 10, 20, 20]),
        packet_ids=torch.tensor([100, 100, 200, 200]),
    )
    assert loss.item() == 0.0
    loss.backward()
    assert factual.grad is not None
    assert torch.count_nonzero(factual.grad) == 0


def test_real_leave_one_out_context_produces_finite_gradients():
    torch.manual_seed(3)
    factual = torch.randn(4, 8, requires_grad=True)
    intervened = torch.randn(4, 8, requires_grad=True)
    loss = availability_aware_cross_scale_loss(
        factual,
        intervened,
        labels=torch.tensor([0, 0, 1, 1]),
        flow_ids=torch.tensor([10, 10, 20, 20]),
        packet_ids=torch.tensor([100, 101, 200, 201]),
    )
    assert torch.isfinite(loss)
    assert loss.item() > 0.0
    loss.backward()
    assert torch.isfinite(factual.grad).all()
    assert torch.count_nonzero(factual.grad) > 0
    assert torch.count_nonzero(intervened.grad) > 0


def test_missing_intervention_context_is_availability_masked():
    torch.manual_seed(5)
    factual = torch.randn(4, 8, requires_grad=True)
    intervened = torch.randn(4, 8, requires_grad=True)
    loss = availability_aware_cross_scale_loss(
        factual,
        intervened,
        labels=torch.tensor([0, 0, 1, 1]),
        flow_ids=torch.tensor([10, 10, 20, 20]),
        packet_ids=torch.tensor([100, 101, 200, 201]),
        intervention_mask=torch.tensor([False, False, False, False]),
    )
    assert loss.item() == 0.0


def test_same_class_flow_is_not_used_as_a_false_negative():
    factual = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [0.9, 0.1], [0.9, 0.1], [-1.0, 0.0], [-1.0, 0.0]]
    )
    intervened = factual.clone()
    labels = torch.tensor([0, 0, 0, 0, 1, 1])
    flow_inverse = torch.tensor([0, 0, 1, 1, 2, 2])
    anchor_available = torch.tensor([True, True, False, False, False, False])
    context_available = torch.ones(6, dtype=torch.bool)
    with_same_class = _cross_view_direction(
        factual,
        anchor_available,
        intervened,
        context_available,
        labels,
        flow_inverse,
        num_flows=3,
        temperature=0.2,
    )
    keep = torch.tensor([0, 1, 4, 5])
    without_same_class = _cross_view_direction(
        factual[keep],
        anchor_available[keep],
        intervened[keep],
        context_available[keep],
        labels[keep],
        torch.tensor([0, 0, 1, 1]),
        num_flows=2,
        temperature=0.2,
    )
    assert torch.allclose(with_same_class, without_same_class, atol=1e-6)


def test_loss_is_permutation_invariant():
    torch.manual_seed(7)
    factual = torch.randn(6, 5)
    intervened = torch.randn(6, 5)
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    flows = torch.tensor([10, 10, 20, 20, 30, 30])
    packets = torch.tensor([100, 101, 200, 201, 300, 301])
    expected = availability_aware_cross_scale_loss(
        factual, intervened, labels, flows, packets
    )
    order = torch.tensor([4, 1, 5, 0, 3, 2])
    actual = availability_aware_cross_scale_loss(
        factual[order],
        intervened[order],
        labels[order],
        flows[order],
        packets[order],
    )
    assert torch.allclose(actual, expected, atol=1e-6)
