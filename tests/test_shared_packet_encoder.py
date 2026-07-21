import torch
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from build_packet_semantic_cache import main as build_semantic_cache
from models.flow_transformer import FlowTransformerClassifier
from models.native_flow_encoder import NativeFlowEncoder, ProtocolAwarePacketContentEncoder
from models.packet_byte_transformer import PacketByteTransformer
from models.unified_packet_encoder import (
    SharedInterventionViewFusion,
    SharedPacketChannelFusion,
    SharedPacketClassifierHead,
    SharedPacketRepresentationEncoder,
)
from train_packet_byte_transformer import (
    PacketByteDataset,
    summarize_packet_gate_diagnostics,
)
from test_tower2 import predict_seq


def test_packet_and_flow_use_the_same_protocol_content_encoder_class():
    packet = PacketByteTransformer(
        num_classes=3,
        max_bytes=16,
        hidden_dim=16,
        num_layers=1,
        num_heads=4,
        use_protocol_fields=True,
    )
    flow = NativeFlowEncoder(
        max_bytes=16,
        max_packets=4,
        hidden_dim=16,
        byte_layers=1,
        flow_layers=1,
        num_heads=4,
    )
    assert isinstance(packet.protocol_content_encoder, ProtocolAwarePacketContentEncoder)
    assert isinstance(flow.packet_content_encoder, ProtocolAwarePacketContentEncoder)
    assert not hasattr(packet, "byte_embedding")
    assert not hasattr(packet, "transformer")


def test_shared_protocol_content_encoder_accepts_packet_and_flow_shapes():
    encoder = ProtocolAwarePacketContentEncoder(16, 12, 1, 3, 0.0)
    packet_tokens = torch.randint(0, 255, (2, 16))
    packet_fields = torch.ones_like(packet_tokens)
    packet_mask = torch.ones_like(packet_tokens, dtype=torch.bool)
    packet_repr, packet_token_repr = encoder(packet_tokens, packet_fields, packet_mask)
    assert packet_repr.shape == (2, 12)
    assert packet_token_repr.shape == (2, 16, 12)

    flow_repr, flow_token_repr = encoder(
        packet_tokens[:, None].expand(-1, 3, -1),
        packet_fields[:, None].expand(-1, 3, -1),
        packet_mask[:, None].expand(-1, 3, -1),
    )
    assert flow_repr.shape == (2, 3, 12)
    assert flow_token_repr.shape == (2, 3, 16, 12)


def test_exact_packet_and_flow_paths_reuse_identical_representation_module_schema():
    packet = PacketByteTransformer(
        num_classes=3,
        max_bytes=16,
        meta_dim=13,
        hidden_dim=8,
        num_layers=1,
        num_heads=2,
        dropout=0.1,
        use_protocol_fields=True,
        semantic_dim=16,
        use_intervention_views=True,
        channel_fusion_base_mode="semantic_anchor",
        exact_shared_representation=True,
        mask_protocol_session_fields=True,
    )
    flow = FlowTransformerClassifier(
        input_dim=16 + 8 + 13,
        num_classes=3,
        hidden_dim=16,
        num_layers=1,
        num_heads=4,
        dropout=0.1,
        dual_channel_mode="residual",
        meta_feature_dim=8 + 13,
        native_structural_dim=8,
        channel_fusion_base_mode="semantic_anchor",
        use_intervention_views=True,
        exact_shared_packet_encoder=True,
        shared_packet_hidden_dim=8,
    )

    assert isinstance(packet.shared_packet_encoder, SharedPacketRepresentationEncoder)
    assert isinstance(flow.shared_packet_encoder, SharedPacketRepresentationEncoder)
    packet_schema = {
        key: tuple(value.shape)
        for key, value in packet.shared_packet_encoder.state_dict().items()
    }
    flow_schema = {
        key: tuple(value.shape)
        for key, value in flow.shared_packet_encoder.state_dict().items()
    }
    assert packet_schema == flow_schema
    assert flow.packet_to_flow_proj.in_features == 8
    assert flow.packet_to_flow_proj.out_features == 16


def test_exact_shared_packet_module_runs_before_flow_only_aggregation():
    model = FlowTransformerClassifier(
        input_dim=12 + 8 + 13,
        num_classes=4,
        hidden_dim=16,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
        dual_channel_mode="residual",
        meta_feature_dim=8 + 13,
        native_structural_dim=8,
        channel_fusion_base_mode="semantic_anchor",
        use_intervention_views=True,
        exact_shared_packet_encoder=True,
        shared_packet_hidden_dim=8,
    ).eval()
    factual = torch.randn(2, 3, 12 + 8 + 13)
    intervened = factual.clone()
    intervened[..., :12] = torch.randn(2, 3, 12)
    output = model(
        factual,
        mask=torch.ones(2, 3, dtype=torch.bool),
        intervened_x=intervened,
    )
    assert output["logits"].shape == (2, 4)
    assert output["dual_channel_gate"].shape == (2, 3, 3)
    assert output["intervention_view_gate"].shape == (2, 3, 2)


def test_retrained_channel_ablation_is_persistent_for_packet_and_flow():
    packet_full = PacketByteTransformer(
        num_classes=3,
        max_bytes=8,
        meta_dim=13,
        hidden_dim=8,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        use_protocol_fields=True,
        semantic_dim=12,
        use_intervention_views=True,
        exact_shared_representation=True,
    ).eval()
    packet_ablation = PacketByteTransformer(
        num_classes=3,
        max_bytes=8,
        meta_dim=13,
        hidden_dim=8,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        use_protocol_fields=True,
        semantic_dim=12,
        use_intervention_views=True,
        exact_shared_representation=True,
        train_ablate_input_channel="semantic",
    ).eval()
    packet_ablation.load_state_dict(packet_full.state_dict())
    packet_inputs = {
        "byte_tokens": torch.randint(0, 255, (2, 8)),
        "byte_lengths": torch.full((2,), 8),
        "meta_features": torch.randn(2, 13),
        "field_ids": torch.ones(2, 8, dtype=torch.long),
        "semantic_features": torch.randn(2, 12),
        "intervened_semantic_features": torch.randn(2, 12),
    }
    implicit = packet_ablation.forward_with_gate_diagnostics(**packet_inputs)[0]
    explicit = packet_full.forward_with_gate_diagnostics(
        **packet_inputs, ablate_channel="semantic"
    )[0]
    torch.testing.assert_close(implicit, explicit)

    flow_full = FlowTransformerClassifier(
        input_dim=12 + 8 + 13,
        num_classes=3,
        hidden_dim=16,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
        dual_channel_mode="residual",
        meta_feature_dim=8 + 13,
        native_structural_dim=8,
        channel_fusion_base_mode="semantic_anchor",
        use_intervention_views=True,
        exact_shared_packet_encoder=True,
        shared_packet_hidden_dim=8,
        train_ablate_input_channel="semantic",
    ).eval()
    factual = torch.randn(2, 3, 12 + 8 + 13)
    intervened = torch.randn_like(factual)
    output = flow_full(factual, torch.ones(2, 3, dtype=torch.bool), intervened)
    structural_input = factual[..., 12:]
    expected_fused = flow_full.shared_packet_encoder(
        factual[..., :12],
        structural_input[..., :8],
        structural_input[..., 8:],
        intervened[..., :12],
        ablate_channel="semantic",
    )[0]
    implicit_fused = flow_full._project_channels(
        factual, torch.ones(2, 3, dtype=torch.bool), intervened
    )[0]
    torch.testing.assert_close(implicit_fused, expected_fused)
    expected_embedding = flow_full.packet_to_flow_proj(expected_fused)
    assert output["logits"].shape == (2, 3)
    assert expected_embedding.shape == (2, 3, 16)


def test_training_ablation_rejects_non_shared_legacy_models():
    with pytest.raises(ValueError, match="channel ablation"):
        PacketByteTransformer(num_classes=2, train_ablate_input_channel="content")
    with pytest.raises(ValueError, match="channel ablation"):
        FlowTransformerClassifier(
            input_dim=20,
            num_classes=2,
            train_ablate_input_channel="structural",
        )


def test_fixed_shared_fusion_is_an_exact_equal_mean_without_router_dependence():
    fusion = SharedPacketChannelFusion(
        hidden_dim=6,
        channel_names=("semantic", "content", "structural"),
        dropout=0.0,
        interaction_max_weight=0.25,
        base_mode="semantic_anchor",
    ).eval()
    channels = {name: torch.randn(4, 6) for name in fusion.channel_names}
    fused, weights = fusion(channels, fixed_equal=True)
    expected = torch.stack(
        [fusion.channel_norms[name](channels[name]) for name in fusion.channel_names],
        dim=-2,
    ).mean(dim=-2)
    torch.testing.assert_close(fused, expected)
    torch.testing.assert_close(weights, torch.full_like(weights, 1.0 / 3.0))

    with pytest.raises(ValueError, match="fixed channel fusion"):
        PacketByteTransformer(num_classes=2, train_fixed_channel_fusion=True)
    with pytest.raises(ValueError, match="fixed channel fusion"):
        FlowTransformerClassifier(
            input_dim=20,
            num_classes=2,
            train_fixed_channel_fusion=True,
        )


def test_packet_evidence_candidate_reuses_packet_head_with_bounded_learned_gate():
    packet = PacketByteTransformer(
        num_classes=4,
        max_bytes=16,
        meta_dim=13,
        hidden_dim=8,
        num_layers=1,
        num_heads=2,
        use_protocol_fields=True,
        semantic_dim=12,
        use_intervention_views=True,
        exact_shared_representation=True,
    )
    flow = FlowTransformerClassifier(
        input_dim=12 + 8 + 13,
        num_classes=4,
        hidden_dim=16,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
        dual_channel_mode="residual",
        meta_feature_dim=8 + 13,
        native_structural_dim=8,
        channel_fusion_base_mode="semantic_anchor",
        use_intervention_views=True,
        exact_shared_packet_encoder=True,
        shared_packet_hidden_dim=8,
        packet_evidence_max_weight=0.4,
    )
    assert isinstance(packet.classifier, SharedPacketClassifierHead)
    assert isinstance(flow.packet_classifier, SharedPacketClassifierHead)
    factual = torch.randn(2, 5, 12 + 8 + 13)
    intervened = factual.clone()
    intervened[..., :12] = torch.randn(2, 5, 12)
    output = flow(
        factual,
        mask=torch.ones(2, 5, dtype=torch.bool),
        intervened_x=intervened,
    )
    assert output["packet_evidence_logits"].shape == (2, 4)
    assert output["packet_evidence_gate"].shape == (2, 1)
    assert torch.all(output["packet_evidence_gate"] >= 0)
    assert torch.all(output["packet_evidence_gate"] <= 0.4)
    output["logits"].sum().backward()
    assert flow.packet_classifier.weight.grad is not None


def test_packet_evidence_candidate_is_absent_from_frozen_v2_default_path():
    flow = FlowTransformerClassifier(
        input_dim=12 + 8 + 13,
        num_classes=4,
        hidden_dim=16,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
        dual_channel_mode="residual",
        meta_feature_dim=8 + 13,
        native_structural_dim=8,
        channel_fusion_base_mode="semantic_anchor",
        use_intervention_views=True,
        exact_shared_packet_encoder=True,
        shared_packet_hidden_dim=8,
    ).eval()
    factual = torch.randn(2, 3, 12 + 8 + 13)
    output = flow(
        factual,
        mask=torch.ones(2, 3, dtype=torch.bool),
        intervened_x=factual.clone(),
    )
    assert not hasattr(flow, "packet_classifier")
    assert "packet_evidence_logits" not in output
    assert "packet_evidence_gate" not in output


def test_exact_flow_evaluation_reports_intervention_and_packet_evidence_gates(
    tmp_path: Path,
):
    model = FlowTransformerClassifier(
        input_dim=12 + 8 + 13,
        num_classes=4,
        hidden_dim=16,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
        dual_channel_mode="residual",
        meta_feature_dim=8 + 13,
        native_structural_dim=8,
        channel_fusion_base_mode="semantic_anchor",
        use_intervention_views=True,
        exact_shared_packet_encoder=True,
        shared_packet_hidden_dim=8,
        packet_evidence_max_weight=0.4,
    ).eval()
    factual_path = tmp_path / "factual.pt"
    intervened_path = tmp_path / "intervened.pt"
    factual = torch.randn(3, 12 + 8 + 13)
    intervened = factual.clone()
    intervened[:, :12] = torch.randn(3, 12)
    common = {"label": 1, "flow_id": "flow-1", "window": [0, 3]}
    torch.save([{**common, "x": factual}], factual_path)
    torch.save([{**common, "x": intervened}], intervened_path)

    outputs = predict_seq(
        model,
        str(factual_path),
        "cpu",
        1,
        intervened_dataset_path=str(intervened_path),
    )
    diagnostics = outputs[-1]

    assert diagnostics["intervention_view_gate"]["bounds_satisfied"] is True
    assert diagnostics["packet_evidence_gate"]["bounds_satisfied"] is True
    assert diagnostics["packet_evidence_gate"]["max_weight"] == 0.4


def test_shared_tri_channel_gate_is_normalized_and_differentiable():
    fusion = SharedPacketChannelFusion(
        8, ("semantic", "content", "structural"), dropout=0.0
    )
    with torch.no_grad():
        fusion.interaction[-1].weight.normal_(std=0.01)
    channels = {
        name: torch.randn(4, 8, requires_grad=True)
        for name in fusion.channel_names
    }
    fused, gate = fusion(channels)
    assert fused.shape == (4, 8)
    assert gate.shape == (4, 3)
    torch.testing.assert_close(gate.sum(dim=-1), torch.ones(4))
    fused.square().mean().backward()
    assert fusion.gate[-1].weight.grad is not None
    assert fusion.gate[-1].weight.grad.abs().sum().item() > 0
    for value in channels.values():
        assert value.grad is not None and value.grad.abs().sum().item() > 0


def test_shared_gate_starts_at_baseline_but_learns_on_first_step():
    fusion = SharedPacketChannelFusion(
        6, ("semantic", "content", "structural"), dropout=0.0
    )
    channels = {
        name: torch.randn(5, 6, requires_grad=True)
        for name in fusion.channel_names
    }
    normalized = [fusion.channel_norms[name](channels[name]) for name in fusion.channel_names]
    expected = torch.stack(normalized, dim=-2).mean(dim=-2)
    fused, gate = fusion(channels)
    torch.testing.assert_close(fused, expected)
    fused.square().mean().backward()
    assert fusion.gate[-1].weight.grad is not None
    assert fusion.gate[-1].weight.grad.abs().sum().item() > 0
    torch.testing.assert_close(gate, torch.full_like(gate, 1.0 / 3.0))


def test_semantic_anchor_gate_controls_all_nonsemantic_contributions():
    fusion = SharedPacketChannelFusion(
        6,
        ("semantic", "content", "structural"),
        dropout=0.0,
        interaction_max_weight=0.25,
        base_mode="semantic_anchor",
    )
    channels = {name: torch.randn(5, 6) for name in fusion.channel_names}
    normalized = {
        name: fusion.channel_norms[name](value) for name, value in channels.items()
    }
    uniform = torch.stack(list(normalized.values()), dim=-2).mean(dim=-2)
    expected = normalized["semantic"] + 0.25 * (
        uniform - normalized["semantic"]
    )
    fused, gate = fusion(channels)
    torch.testing.assert_close(fused, expected)
    torch.testing.assert_close(gate, torch.full_like(gate, 1.0 / 3.0))


def test_semantic_anchor_rejects_external_base_bypass():
    fusion = SharedPacketChannelFusion(
        4,
        ("semantic", "content", "structural"),
        base_mode="semantic_anchor",
    )
    channels = {name: torch.randn(2, 4) for name in fusion.channel_names}
    with pytest.raises(ValueError, match="computes its own normalized base"):
        fusion(channels, base=torch.randn(2, 4))


def test_intervention_router_starts_at_exact_mean_and_learns_first_step():
    fusion = SharedInterventionViewFusion(
        hidden_dim=8, dropout=0.0, max_residual_weight=0.25
    )
    factual = torch.randn(6, 8, requires_grad=True)
    intervened = torch.randn(6, 8, requires_grad=True)
    expected = 0.5 * (
        fusion.factual_norm(factual) + fusion.intervened_norm(intervened)
    )
    fused, gate = fusion(factual, intervened)
    torch.testing.assert_close(fused, expected)
    torch.testing.assert_close(gate, torch.full_like(gate, 0.5))
    fused.square().mean().backward()
    assert fusion.router[-1].weight.grad is not None
    assert fusion.router[-1].weight.grad.abs().sum().item() > 0
    assert factual.grad is not None and factual.grad.abs().sum().item() > 0
    assert intervened.grad is not None and intervened.grad.abs().sum().item() > 0


def test_symmetric_intervention_effective_weights_are_bounded():
    fusion = SharedInterventionViewFusion(
        hidden_dim=8, dropout=0.0, max_residual_weight=0.25
    )
    router_gate = torch.tensor([[0.0, 1.0], [1.0, 0.0], [0.3, 0.7]])
    effective = fusion.effective_weights(router_gate)

    assert fusion.effective_weight_bounds() == {
        "factual": (0.375, 0.625),
        "intervened": (0.375, 0.625),
    }
    torch.testing.assert_close(effective.sum(dim=-1), torch.ones(3))
    assert torch.all(effective >= 0.375)
    assert torch.all(effective <= 0.625)


def test_intervention_factual_anchor_has_identity_path_and_bounded_residual():
    fusion = SharedInterventionViewFusion(
        hidden_dim=8,
        dropout=0.0,
        max_residual_weight=0.25,
        base_mode="factual_anchor",
    )
    factual = torch.randn(6, 8, requires_grad=True)
    intervened = torch.randn(6, 8, requires_grad=True)
    factual_norm = fusion.factual_norm(factual)
    intervened_norm = fusion.intervened_norm(intervened)

    fused, gate = fusion(factual, intervened)

    torch.testing.assert_close(gate[:, 0], torch.full_like(gate[:, 0], 0.875))
    torch.testing.assert_close(gate[:, 1], torch.full_like(gate[:, 1], 0.125))
    torch.testing.assert_close(
        fused,
        0.875 * factual_norm + 0.125 * intervened_norm,
    )
    assert torch.all(gate[:, 1] <= fusion.max_residual_weight)
    assert fusion.effective_weight_bounds() == {
        "factual": (0.75, 1.0),
        "intervened": (0.0, 0.25),
    }
    fused.square().mean().backward()
    assert fusion.router[-1].weight.grad is not None
    assert fusion.router[-1].weight.grad.abs().sum().item() > 0


def test_semantic_anchor_effective_channel_weights_are_bounded():
    fusion = SharedPacketChannelFusion(
        6,
        ("semantic", "content", "structural"),
        dropout=0.0,
        interaction_max_weight=0.25,
        base_mode="semantic_anchor",
    )
    router_gate = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.2, 0.3, 0.5]]
    )
    effective = fusion.effective_weights(router_gate)

    assert fusion.effective_weight_bounds() == {
        "semantic": (0.75, 1.0),
        "content": (0.0, 0.25),
        "structural": (0.0, 0.25),
    }
    torch.testing.assert_close(effective.sum(dim=-1), torch.ones(3))
    assert torch.all(effective[:, 0] >= 0.75)
    assert torch.all(effective[:, 1:] <= 0.25)


def test_intervention_router_rejects_unknown_base_mode():
    with pytest.raises(ValueError, match="base_mode"):
        SharedInterventionViewFusion(hidden_dim=8, base_mode="unknown")


def test_packet_and_flow_share_the_same_intervention_router_class():
    packet = PacketByteTransformer(
        num_classes=4,
        max_bytes=16,
        hidden_dim=12,
        num_layers=1,
        num_heads=3,
        use_protocol_fields=True,
        semantic_dim=7,
        use_intervention_views=True,
    )
    flow = FlowTransformerClassifier(
        input_dim=7 + 4 + 3,
        num_classes=4,
        hidden_dim=12,
        num_layers=1,
        num_heads=3,
        dropout=0.0,
        dual_channel_mode="residual",
        meta_feature_dim=7,
        native_structural_dim=4,
        use_intervention_views=True,
    )
    assert isinstance(packet.intervention_view_fusion, SharedInterventionViewFusion)
    assert isinstance(flow.intervention_view_fusion, SharedInterventionViewFusion)

    packet_logits, _, packet_gate, packet_intervention_gate = (
        packet.forward_with_gate_diagnostics(
        torch.randint(0, 255, (2, 16)),
        torch.full((2,), 16),
        torch.randn(2, 28),
        field_ids=torch.ones(2, 16, dtype=torch.long),
        semantic_features=torch.randn(2, 7),
        intervened_semantic_features=torch.randn(2, 7),
        )
    )
    flow_out = flow(
        torch.randn(2, 5, 14),
        torch.ones(2, 5, dtype=torch.bool),
        intervened_x=torch.randn(2, 5, 14),
    )
    assert packet_logits.shape == (2, 4)
    assert packet_gate.shape == (2, 3)
    assert packet_intervention_gate.shape == (2, 2)
    assert flow_out["intervention_view_gate"].shape == (2, 5, 2)
    torch.testing.assert_close(
        flow_out["intervention_view_gate"].sum(dim=-1), torch.ones(2, 5)
    )


def test_packet_and_flow_expose_the_same_bounded_channel_residual():
    packet = PacketByteTransformer(
        num_classes=3,
        max_bytes=16,
        hidden_dim=16,
        num_layers=1,
        num_heads=4,
        use_protocol_fields=True,
        semantic_dim=8,
        channel_fusion_base_mode="semantic_anchor",
        channel_fusion_max_weight=0.2,
    )
    flow = FlowTransformerClassifier(
        input_dim=21,
        num_classes=3,
        hidden_dim=16,
        num_layers=1,
        num_heads=4,
        dual_channel_mode="residual",
        meta_feature_dim=13,
        native_structural_dim=4,
        channel_fusion_base_mode="semantic_anchor",
        dual_channel_max_weight=0.2,
    )
    assert packet.shared_packet_fusion.interaction_max_weight == pytest.approx(0.2)
    assert flow.shared_packet_fusion.interaction_max_weight == pytest.approx(0.2)

    clipped = PacketByteTransformer(
        num_classes=2,
        max_bytes=8,
        hidden_dim=8,
        num_layers=1,
        num_heads=2,
        semantic_dim=4,
        channel_fusion_base_mode="semantic_anchor",
        channel_fusion_max_weight=2.0,
    )
    assert clipped.shared_packet_fusion.interaction_max_weight == pytest.approx(1.0)


def test_packet_gate_diagnostics_report_named_effective_routing_weights():
    packet = PacketByteTransformer(
        num_classes=3,
        max_bytes=8,
        hidden_dim=8,
        num_layers=1,
        num_heads=2,
        semantic_dim=4,
        use_intervention_views=True,
        channel_fusion_base_mode="semantic_anchor",
        channel_fusion_max_weight=0.25,
    )
    diagnostics = summarize_packet_gate_diagnostics(
        packet,
        [torch.tensor([[0.6, 0.3, 0.1], [0.4, 0.2, 0.4]])],
        [torch.tensor([[0.8, 0.2], [0.6, 0.4]])],
    )
    channel = diagnostics["packet_channel_gate"]
    assert channel["channel_names"] == ["semantic", "content", "structural"]
    assert channel["max_residual_weight"] == pytest.approx(0.25)
    assert channel["effective_routing_mean"] == pytest.approx(
        [0.875, 0.0625, 0.0625]
    )
    assert channel["theoretical_bounds"] == {
        "semantic": [0.75, 1.0],
        "content": [0.0, 0.25],
        "structural": [0.0, 0.25],
    }
    assert channel["bounds_satisfied"] is True
    intervention = diagnostics["intervention_view_gate"]
    assert intervention["effective_routing_mean"] == pytest.approx([0.55, 0.45])
    assert intervention["theoretical_bounds"] == {
        "factual": [0.375, 0.625],
        "intervened": [0.375, 0.625],
    }
    assert intervention["bounds_satisfied"] is True


def test_exact_packet_gate_diagnostics_find_shared_intervention_router_and_fixed_mean():
    packet = PacketByteTransformer(
        num_classes=3,
        max_bytes=8,
        meta_dim=13,
        hidden_dim=8,
        num_layers=1,
        num_heads=2,
        use_protocol_fields=True,
        semantic_dim=4,
        use_intervention_views=True,
        exact_shared_representation=True,
        channel_fusion_base_mode="semantic_anchor",
        train_fixed_channel_fusion=True,
    )
    diagnostics = summarize_packet_gate_diagnostics(
        packet,
        [torch.full((2, 3), 1.0 / 3.0)],
        [torch.tensor([[0.5, 0.5], [0.5, 0.5]])],
    )
    channel = diagnostics["packet_channel_gate"]
    assert channel["weight_semantics"] == "fixed_equal_normalized_channel_mean"
    assert channel["effective_routing_mean"] == pytest.approx([1 / 3] * 3)
    assert channel["max_residual_weight"] == 0.0
    assert channel["bounds_satisfied"] is True
    assert diagnostics["intervention_view_gate"]["bounds_satisfied"] is True


def test_shared_encoder_applies_named_inference_only_sensitivity_masks():
    encoder = SharedPacketRepresentationEncoder(
        semantic_dim=6,
        content_dim=4,
        structural_dim=3,
        hidden_dim=8,
        dropout=0.0,
        use_intervention_views=True,
    ).eval()
    semantic = torch.randn(5, 6)
    intervened = torch.randn(5, 6)
    content = torch.randn(5, 4)
    structural = torch.randn(5, 3)
    full = encoder(semantic, content, structural, intervened)[0]
    without_content = encoder(
        semantic,
        content,
        structural,
        intervened,
        ablate_channel="content",
    )[0]
    assert not torch.allclose(full, without_content)

    factual_only = encoder(
        semantic,
        content,
        structural,
        intervened,
        ablate_intervention_view="factual_only",
    )[0]
    explicit_factual_pair = encoder(semantic, content, structural, semantic)[0]
    torch.testing.assert_close(factual_only, explicit_factual_pair)

    with pytest.raises(ValueError, match="unknown shared packet channel"):
        encoder(
            semantic,
            content,
            structural,
            intervened,
            ablate_channel="unknown",
        )


def test_flow_tri_channel_gate_uses_strict_content_as_its_own_channel():
    model = FlowTransformerClassifier(
        input_dim=6 + 4 + 3,
        num_classes=5,
        hidden_dim=12,
        num_layers=1,
        num_heads=3,
        dropout=0.0,
        dual_channel_mode="residual",
        dual_channel_gate_mode="adaptive",
        meta_feature_dim=7,
        native_structural_dim=4,
    )
    out = model(torch.randn(2, 5, 13), torch.ones(2, 5, dtype=torch.bool))
    assert model.shared_packet_fusion.channel_names == (
        "semantic",
        "content",
        "structural",
    )
    assert out["dual_channel_gate"].shape == (2, 5, 3)
    torch.testing.assert_close(
        out["dual_channel_gate"].sum(dim=-1), torch.ones(2, 5)
    )


def test_packet_semantic_cache_aligns_by_flow_and_packet_id(tmp_path: Path, monkeypatch):
    pcap = tmp_path / "sample.pcap"
    pcap.write_bytes(b"pcap")
    packet_index = tmp_path / "packet_index.jsonl"
    packet_rows = [
        {
            "flow_id": "flow-a",
            "packet_id": packet_id,
            "label_id": 0,
            "pcap_path": str(pcap),
            "meta": {"packet_id": packet_id, "l3_hex_prefix": "45 00"},
        }
        for packet_id in (1, 0)
    ]
    packet_index.write_text(
        "".join(json.dumps(row) + "\n" for row in packet_rows), encoding="utf-8"
    )
    embeddings = tmp_path / "flow-a.npy"
    np.save(embeddings, np.asarray([[10.0, 11.0], [20.0, 21.0]], dtype=np.float32))
    embedding_index = tmp_path / "flow_embedding_index.jsonl"
    embedding_index.write_text(
        json.dumps(
            {
                "flow_id": "flow-a",
                "embedding_path": str(embeddings),
                "embedding_mode": "concat",
                "packet_metas": [{"packet_id": 0}, {"packet_id": 1}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "embedding_config.json").write_text(
        json.dumps(
            {
                "packet_index_header_policy": "mask_ip_port",
                "packet_index_context_policy": "single_packet",
            }
        ),
        encoding="utf-8",
    )
    cache = tmp_path / "semantic.npy"
    manifest = tmp_path / "semantic.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_packet_semantic_cache.py",
            "--packet_index", str(packet_index),
            "--flow_embedding_index", str(embedding_index),
            "--output_npy", str(cache),
            "--output_json", str(manifest),
            "--no_progress",
        ],
    )
    build_semantic_cache()
    aligned = np.load(cache)
    np.testing.assert_array_equal(aligned, [[20.0, 21.0], [10.0, 11.0]])

    dataset = PacketByteDataset(
        str(packet_index),
        max_bytes=8,
        include_augmented=False,
        semantic_embedding_cache=str(cache),
        semantic_embedding_manifest=str(manifest),
        required_header_policy="mask_ip_port",
        required_packet_context_policy="single_packet",
    )
    assert dataset.semantic_dim == 2
    torch.testing.assert_close(
        dataset[0]["semantic_embedding"], torch.tensor([20.0, 21.0])
    )
    with pytest.raises(ValueError, match="header policy mismatch"):
        PacketByteDataset(
            str(packet_index),
            max_bytes=8,
            include_augmented=False,
            semantic_embedding_cache=str(cache),
            semantic_embedding_manifest=str(manifest),
            required_header_policy="full",
        )
    with pytest.raises(ValueError, match="packet context policy mismatch"):
        PacketByteDataset(
            str(packet_index),
            max_bytes=8,
            include_augmented=False,
            semantic_embedding_cache=str(cache),
            semantic_embedding_manifest=str(manifest),
            required_packet_context_policy="flow_context",
        )

    intervened_cache = tmp_path / "semantic_masked.npy"
    np.save(intervened_cache, aligned + 1.0)
    intervened_manifest = tmp_path / "semantic_masked.json"
    masked_payload = json.loads(manifest.read_text(encoding="utf-8"))
    masked_payload["header_policy"] = "full"
    intervened_manifest.write_text(json.dumps(masked_payload), encoding="utf-8")
    paired_dataset = PacketByteDataset(
        str(packet_index),
        max_bytes=8,
        include_augmented=False,
        semantic_embedding_cache=str(cache),
        semantic_embedding_manifest=str(manifest),
        required_header_policy="mask_ip_port",
        required_packet_context_policy="single_packet",
        intervened_semantic_embedding_cache=str(intervened_cache),
        intervened_semantic_embedding_manifest=str(intervened_manifest),
        required_intervened_header_policy="full",
    )
    torch.testing.assert_close(
        paired_dataset[0]["intervened_semantic_embedding"],
        torch.tensor([21.0, 22.0]),
    )
    with pytest.raises(ValueError, match="same header policy"):
        PacketByteDataset(
            str(packet_index),
            max_bytes=8,
            include_augmented=False,
            semantic_embedding_cache=str(cache),
            semantic_embedding_manifest=str(manifest),
            intervened_semantic_embedding_cache=str(cache),
            intervened_semantic_embedding_manifest=str(manifest),
        )


def test_packet_model_runs_the_same_three_named_channels():
    model = PacketByteTransformer(
        num_classes=4,
        max_bytes=16,
        hidden_dim=12,
        num_layers=1,
        num_heads=3,
        use_protocol_fields=True,
        semantic_dim=7,
    )
    logits, _, gate = model(
        torch.randint(0, 255, (3, 16)),
        torch.full((3,), 16),
        torch.randn(3, 28),
        field_ids=torch.ones(3, 16, dtype=torch.long),
        semantic_features=torch.randn(3, 7),
    )
    assert logits.shape == (3, 4)
    assert model.shared_packet_fusion.channel_names == (
        "semantic",
        "content",
        "structural",
    )
    assert gate.shape == (3, 3)
    torch.testing.assert_close(gate.sum(dim=-1), torch.ones(3))


def test_legacy_packet_encoder_keeps_its_byte_transformer():
    model = PacketByteTransformer(
        num_classes=2,
        max_bytes=16,
        hidden_dim=12,
        num_layers=1,
        num_heads=3,
        use_protocol_fields=False,
    )
    assert hasattr(model, "byte_embedding")
    assert hasattr(model, "transformer")
