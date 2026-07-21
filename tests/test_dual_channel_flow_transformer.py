from types import SimpleNamespace

import torch
import pytest

from models.flow_transformer import FlowTransformerClassifier
from test_tower2 import ablate_intervention_inputs, ablate_seq_input_channel
from train_tower2 import (
    configure_dual_channel_train_scope,
    dual_channel_auxiliary_loss,
)


def build_model(mode: str, gate_mode: str = "global") -> FlowTransformerClassifier:
    return FlowTransformerClassifier(
        input_dim=10,
        num_classes=3,
        hidden_dim=8,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        dual_channel_mode=mode,
        meta_feature_dim=4,
        dual_channel_max_weight=0.25,
        dual_channel_gate_mode=gate_mode,
    )


def test_dual_channel_projection_exactly_preserves_concat_path():
    torch.manual_seed(7)
    legacy = build_model("concat").eval()
    dual = build_model("residual").eval()
    dual.import_concat_projection(legacy.proj.weight, legacy.proj.bias)
    legacy_state = legacy.state_dict()
    compatible = {
        key: value
        for key, value in legacy_state.items()
        if not key.startswith("proj.") and key in dual.state_dict()
    }
    dual.load_state_dict(compatible, strict=False)
    x = torch.randn(3, 5, 10)
    mask = torch.tensor(
        [[1, 1, 1, 1, 1], [1, 1, 1, 0, 0], [1, 1, 1, 1, 0]],
        dtype=torch.bool,
    )
    with torch.no_grad():
        legacy_out = legacy(x, mask)
        dual_out = dual(x, mask)
    torch.testing.assert_close(dual_out["embedding"], legacy_out["embedding"])
    torch.testing.assert_close(dual_out["logits"], legacy_out["logits"])
    assert dual_out["dual_channel_gate"].shape == (3, 5, 2)
    torch.testing.assert_close(
        dual_out["dual_channel_gate"].sum(dim=-1),
        torch.ones(3, 5),
    )


def test_zero_initialized_interaction_is_trainable():
    model = build_model("residual")
    x = torch.randn(4, 5, 10)
    out = model(x, torch.ones(4, 5, dtype=torch.bool))
    out["logits"].square().mean().backward()
    final = model.channel_interaction[-1]
    assert final.weight.grad is not None
    assert final.weight.grad.abs().sum().item() > 0


def test_inference_channel_ablation_zeros_only_requested_slice():
    model = FlowTransformerClassifier(
        input_dim=12,
        num_classes=3,
        hidden_dim=8,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        dual_channel_mode="residual",
        meta_feature_dim=6,
        native_structural_dim=4,
    )
    x = torch.ones(2, 3, 12)
    expected_ranges = {
        "semantic": (0, 6),
        "content": (6, 10),
        "structural": (10, 12),
    }
    for channel, (start, end) in expected_ranges.items():
        ablated = ablate_seq_input_channel(x, model, channel)
        assert torch.count_nonzero(ablated[..., start:end]) == 0
        assert torch.count_nonzero(ablated).item() == x.numel() - (end - start) * 6
    torch.testing.assert_close(ablate_seq_input_channel(x, model, "none"), x)


def test_intervention_view_ablation_duplicates_only_the_requested_view():
    factual = torch.tensor([[1.0, 2.0]])
    intervened = torch.tensor([[3.0, 4.0]])

    actual_factual, actual_intervened = ablate_intervention_inputs(
        factual, intervened, "none"
    )
    assert actual_factual is factual
    assert actual_intervened is intervened

    actual_factual, actual_intervened = ablate_intervention_inputs(
        factual, intervened, "factual_only"
    )
    assert actual_factual is factual
    assert actual_intervened is factual

    actual_factual, actual_intervened = ablate_intervention_inputs(
        factual, intervened, "intervened_only"
    )
    assert actual_factual is intervened
    assert actual_intervened is intervened

    with pytest.raises(ValueError, match="unknown intervention-view ablation"):
        ablate_intervention_inputs(factual, intervened, "bad")


def test_dual_channel_auxiliary_loss_supervises_both_channels():
    model = build_model("residual")
    out = model(torch.randn(4, 5, 10), torch.ones(4, 5, dtype=torch.bool))
    args = SimpleNamespace(
        dual_channel_semantic_aux_weight=0.2,
        dual_channel_structural_aux_weight=0.3,
        dual_channel_consistency_weight=0.1,
        consistency_temperature=2.0,
        label_smoothing=0.0,
        focal_gamma=0.0,
        confidence_penalty_weight=0.0,
    )
    loss = dual_channel_auxiliary_loss(
        out, torch.tensor([0, 1, 2, 1]), None, args
    )
    loss.backward()
    assert model.semantic_channel_cls.weight.grad.abs().sum().item() > 0
    assert model.structural_channel_cls.weight.grad.abs().sum().item() > 0


def test_interaction_scope_freezes_the_legacy_path():
    model = build_model("residual")
    trainable = configure_dual_channel_train_scope(model, None, "interaction")
    assert trainable
    assert all(name.startswith("shared_packet_fusion.") for name in trainable)
    assert not model.semantic_proj.weight.requires_grad
    assert not model.encoder.layers[0].linear1.weight.requires_grad


def test_adaptive_gate_is_identity_initialized_and_trainable():
    legacy = build_model("concat").eval()
    model = build_model("residual", gate_mode="adaptive").eval()
    model.import_concat_projection(legacy.proj.weight, legacy.proj.bias)
    compatible = {
        key: value
        for key, value in legacy.state_dict().items()
        if not key.startswith("proj.") and key in model.state_dict()
    }
    model.load_state_dict(compatible, strict=False)
    x = torch.randn(3, 4, 10)
    mask = torch.ones(3, 4, dtype=torch.bool)
    torch.testing.assert_close(model(x, mask)["logits"], legacy(x, mask)["logits"])
    model.train()
    out = model(x, mask)
    out["logits"].square().mean().backward()
    assert model.channel_interaction[-1].weight.grad.abs().sum().item() > 0
