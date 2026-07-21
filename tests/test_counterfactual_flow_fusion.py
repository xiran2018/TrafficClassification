import torch

from models.counterfactual_flow_fusion import (
    CounterfactualFlowFusion,
    counterfactual_regularization,
    intervention_routing_loss,
)
from test_tower2 import load_counterfactual_fusion
from train_tower2 import configure_counterfactual_head_only


def test_counterfactual_head_is_exact_mean_control_at_initialization():
    torch.manual_seed(3)
    head = CounterfactualFlowFusion(
        8, 4, base_mode="mean", max_residual_weight=0.25
    )
    clean_embedding = torch.randn(5, 8)
    intervened_embedding = torch.randn(5, 8)
    clean_logits = torch.randn(5, 4)
    intervened_logits = torch.randn(5, 4)

    output = head(
        clean_embedding, intervened_embedding, clean_logits, intervened_logits
    )

    expected = 0.5 * (clean_logits + intervened_logits)
    assert torch.equal(output["logits"], expected)
    assert output["residual_gate"].max().item() <= 0.25


def test_counterfactual_residual_receives_gradient_without_changing_initial_output():
    torch.manual_seed(5)
    head = CounterfactualFlowFusion(8, 3)
    clean_embedding = torch.randn(6, 8, requires_grad=True)
    intervened_embedding = torch.randn(6, 8, requires_grad=True)
    clean_logits = torch.randn(6, 3, requires_grad=True)
    intervened_logits = torch.randn(6, 3, requires_grad=True)

    output = head(
        clean_embedding, intervened_embedding, clean_logits, intervened_logits
    )
    loss = torch.nn.functional.cross_entropy(output["logits"], torch.arange(6) % 3)
    loss.backward()

    assert head.changed_classifier.weight.grad is not None
    assert head.changed_classifier.weight.grad.abs().sum().item() > 0


def test_mean_control_is_order_invariant_and_has_zero_regularization():
    head = CounterfactualFlowFusion(6, 2, mode="mean")
    a = torch.randn(4, 6)
    b = torch.randn(4, 6)
    la = torch.randn(4, 2)
    lb = torch.randn(4, 2)

    forward = head(a, b, la, lb)
    reverse = head(b, a, lb, la)

    assert torch.equal(forward["logits"], reverse["logits"])
    regularization = counterfactual_regularization(
        forward, gate_weight=1.0, orthogonality_weight=1.0
    )
    assert regularization.item() == 0.0


def test_clean_control_and_clean_residual_are_identity_preserving():
    clean_embedding = torch.randn(4, 6)
    intervened_embedding = torch.randn(4, 6)
    clean_logits = torch.randn(4, 3)
    intervened_logits = torch.randn(4, 3)

    for mode in ("clean", "counterfactual"):
        head = CounterfactualFlowFusion(6, 3, mode=mode, base_mode="clean")
        output = head(
            clean_embedding,
            intervened_embedding,
            clean_logits,
            intervened_logits,
        )
        assert torch.equal(output["logits"], clean_logits)


def test_empty_mean_state_dict_still_reconstructs_checkpoint_head():
    checkpoint = {
        "counterfactual_fusion": "mean",
        "counterfactual_fusion_state": {},
        "counterfactual_base_mode": "mean",
        "hidden_dim": 6,
        "num_classes": 2,
        "counterfactual_max_residual_weight": 0.25,
        "counterfactual_initial_residual_fraction": 0.1,
        "dropout": 0.1,
    }
    head = load_counterfactual_fusion(checkpoint, "cpu")
    assert head is not None
    assert head.mode == "mean"


def test_counterfactual_head_only_freezes_backbone_and_flow_head():
    backbone = torch.nn.Linear(5, 6)
    flow_head = torch.nn.Linear(6, 3)
    fusion = CounterfactualFlowFusion(6, 3, mode="counterfactual")

    names = configure_counterfactual_head_only(backbone, flow_head, fusion)

    assert names
    assert not any(parameter.requires_grad for parameter in backbone.parameters())
    assert not any(parameter.requires_grad for parameter in flow_head.parameters())
    assert all(parameter.requires_grad for parameter in fusion.parameters())


def test_router_is_identity_initialized_and_gate_supervision_is_informative():
    head = CounterfactualFlowFusion(6, 3, mode="router", max_residual_weight=1.0)
    clean_embedding = torch.randn(4, 6)
    intervened_embedding = torch.randn(4, 6)
    clean_logits = torch.tensor(
        [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [5.0, 0.0, 0.0], [0.0, 5.0, 0.0]]
    )
    intervened_logits = torch.tensor(
        [[0.0, 5.0, 0.0], [0.0, 5.0, 0.0], [5.0, 0.0, 0.0], [5.0, 0.0, 0.0]]
    )
    labels = torch.tensor([1, 1, 0, 1])
    output = head(
        clean_embedding, intervened_embedding, clean_logits, intervened_logits
    )

    assert torch.equal(output["logits"], clean_logits)
    loss = intervention_routing_loss(
        output, clean_logits, intervened_logits, labels, weight=1.0
    )
    assert loss.item() > 0
    loss.backward()
    assert head.gate[-1].bias.grad is not None
