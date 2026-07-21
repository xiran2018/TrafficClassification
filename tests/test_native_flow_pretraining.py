import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from models.flow_transformer import FlowTransformerClassifier
from models.native_flow_encoder import (
    MASK_BYTE,
    NativeFlowEncoder,
    nt_xent_loss,
    sample_relative_pairs,
    sample_same_flow_pairs,
)
from native_flow_data import (
    FIELD_CHECKSUM,
    FIELD_ENDPOINT,
    FIELD_PAYLOAD,
    PacketIndexFlowDataset,
    apply_payload_dropout,
    apply_session_invariant_mask,
    mask_protocol_fields,
    protocol_field_ids,
)
from preprocess_tower2 import apply_structural_embedding_reference
from train_tower2 import configure_dual_channel_train_scope, maybe_load_init_checkpoint


def ipv4_tcp_packet(payload: bytes = b"hello") -> bytes:
    ip = bytearray(20)
    ip[0] = 0x45
    ip[2:4] = (40 + len(payload)).to_bytes(2, "big")
    ip[8] = 64
    ip[9] = 6
    ip[10:12] = b"\x12\x34"
    ip[12:16] = bytes([10, 0, 0, 1])
    ip[16:20] = bytes([10, 0, 0, 2])
    tcp = bytearray(20)
    tcp[0:2] = (12345).to_bytes(2, "big")
    tcp[2:4] = (443).to_bytes(2, "big")
    tcp[12] = 0x50
    tcp[13] = 0x18
    tcp[16:18] = b"\xab\xcd"
    return bytes(ip + tcp + payload)


def packet_row(flow_id: str, packet_id: int, direction: str = "C2S") -> dict:
    raw = ipv4_tcp_packet(bytes([packet_id + 1]) * 5)
    return {
        "flow_id": flow_id,
        "label": "demo",
        "label_id": 3,
        "pcap_path": f"/{flow_id}.pcap",
        "packet_id": packet_id,
        "meta": {
            "packet_id": packet_id,
            "l3_hex_prefix": raw.hex(),
            "direction": direction,
            "packet_len": len(raw),
            "payload_len": 5,
            "iat": packet_id * 0.01,
            "tcp_window": 1024,
        },
    }


def test_protocol_field_ids_mark_endpoints_checksums_and_payload():
    raw = ipv4_tcp_packet()
    fields = protocol_field_ids(raw, 64)
    assert np.all(fields[12:20] == FIELD_ENDPOINT)
    assert np.all(fields[20:24] == FIELD_ENDPOINT)
    assert np.all(fields[10:12] == FIELD_CHECKSUM)
    assert np.all(fields[36:38] == FIELD_CHECKSUM)
    assert np.all(fields[40 : len(raw)] == FIELD_PAYLOAD)


def test_packet_index_dataset_preserves_flow_and_packet_alignment(tmp_path: Path):
    path = tmp_path / "packet_index.jsonl"
    rows = [
        packet_row("flow-a", 0),
        packet_row("flow-a", 1, "S2C"),
        packet_row("flow-b", 0),
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    dataset = PacketIndexFlowDataset(path, max_packets=4, max_bytes=64)
    assert len(dataset) == 2
    first = dataset[0]
    second = dataset[1]
    assert first["flow_id"] == "flow-a"
    assert first["packet_mask"].tolist() == [True, True, False, False]
    assert first["packet_ids"].tolist() == [0, 1, -1, -1]
    assert first["directions"].tolist() == [1, 2, 0, 0]
    assert int(first["next_length"][0]) >= 0
    assert second["flow_id"] == "flow-b"


def test_field_masking_masks_only_valid_bytes_and_guarantees_a_target():
    raw = ipv4_tcp_packet()
    tokens = torch.full((1, 1, 64), 256, dtype=torch.long)
    tokens[0, 0, : len(raw)] = torch.tensor(list(raw))
    fields = torch.tensor(protocol_field_ids(raw, 64))[None, None]
    valid = tokens != 256
    masked, prediction_mask = mask_protocol_fields(
        tokens, fields, valid, 0.01, generator=torch.Generator().manual_seed(7)
    )
    assert prediction_mask.any()
    assert not prediction_mask[~valid].any()
    assert torch.all(masked[prediction_mask] == MASK_BYTE)
    assert torch.equal(masked[~prediction_mask], tokens[~prediction_mask])
    assert not prediction_mask[fields == FIELD_PAYLOAD].any()


def test_payload_dropout_never_creates_reconstruction_targets():
    raw = ipv4_tcp_packet()
    tokens = torch.full((1, 1, 64), 256, dtype=torch.long)
    tokens[0, 0, : len(raw)] = torch.tensor(list(raw))
    fields = torch.tensor(protocol_field_ids(raw, 64))[None, None]
    valid = tokens != 256
    masked, payload_mask = apply_payload_dropout(tokens, fields, valid, 1.0)
    expected = (fields == FIELD_PAYLOAD) & valid
    assert torch.equal(payload_mask, expected)
    assert torch.all(masked[payload_mask] == MASK_BYTE)


def test_session_invariant_mask_hides_endpoints_without_making_reconstruction_targets():
    raw = ipv4_tcp_packet()
    tokens = torch.full((1, 1, 64), 256, dtype=torch.long)
    tokens[0, 0, : len(raw)] = torch.tensor(list(raw))
    fields = torch.tensor(protocol_field_ids(raw, 64))[None, None]
    valid = tokens != 256
    random_view, prediction_mask = mask_protocol_fields(
        tokens, fields, valid, 0.2, generator=torch.Generator().manual_seed(3)
    )
    invariant_view, invariant_mask = apply_session_invariant_mask(
        random_view, fields, valid, 1.0
    )
    endpoint_mask = (fields == FIELD_ENDPOINT) & valid
    assert endpoint_mask.any()
    assert torch.all(invariant_view[endpoint_mask] == MASK_BYTE)
    assert torch.all(invariant_mask[endpoint_mask])
    assert not prediction_mask[endpoint_mask].any()


def test_native_encoder_exposes_all_pretraining_heads():
    model = NativeFlowEncoder(
        max_bytes=48,
        max_packets=4,
        hidden_dim=32,
        byte_layers=1,
        flow_layers=1,
        num_heads=4,
        projection_dim=16,
        dropout=0.0,
    )
    tokens = torch.randint(0, 256, (3, 4, 48))
    fields = torch.ones_like(tokens)
    byte_mask = torch.ones_like(tokens, dtype=torch.bool)
    packet_mask = torch.tensor(
        [[True, True, True, True], [True, True, True, False], [True, True, False, False]]
    )
    directions = torch.ones((3, 4), dtype=torch.long)
    meta = torch.zeros((3, 4, 4))
    output = model(tokens, fields, byte_mask, packet_mask, directions, meta)
    assert output["packet_repr"].shape == (3, 4, 32)
    assert output["packet_observation"].shape == (3, 4, 32)
    assert output["flow_projection"].shape == (3, 16)
    assert output["direction_logits"].shape == (3, 4, 2)
    first, second, targets = sample_relative_pairs(output["packet_repr"], packet_mask)
    assert model.relative_order_logits(first, second).shape == (len(targets), 2)
    first, second, targets = sample_same_flow_pairs(output["packet_repr"], packet_mask)
    assert model.same_flow_logits(first, second).shape == (len(targets), 2)
    next_length, next_iat = model.next_packet_logits(output["packet_repr"])
    assert next_length.shape == (3, 4, 8)
    assert next_iat.shape == (3, 4, 8)
    assert torch.isfinite(nt_xent_loss(output["flow_projection"], output["flow_projection"]))


def test_packet_observation_does_not_expose_absolute_packet_position():
    model = NativeFlowEncoder(
        max_bytes=16,
        max_packets=3,
        hidden_dim=16,
        byte_layers=1,
        flow_layers=1,
        num_heads=4,
        dropout=0.0,
    ).eval()
    packet = torch.randint(0, 256, (1, 1, 16))
    tokens = packet.expand(1, 3, 16).clone()
    fields = torch.ones_like(tokens)
    byte_mask = torch.ones_like(tokens, dtype=torch.bool)
    packet_mask = torch.ones((1, 3), dtype=torch.bool)
    directions = torch.ones((1, 3), dtype=torch.long)
    meta = torch.zeros((1, 3, 4))
    output = model(tokens, fields, byte_mask, packet_mask, directions, meta)
    torch.testing.assert_close(
        output["packet_observation"][:, 0], output["packet_observation"][:, 2]
    )
    assert not torch.allclose(output["packet_repr"][:, 0], output["packet_repr"][:, 2])


def test_exported_packet_content_is_independent_of_neighbor_packets():
    torch.manual_seed(17)
    model = NativeFlowEncoder(
        max_bytes=16,
        max_packets=3,
        hidden_dim=16,
        byte_layers=1,
        flow_layers=1,
        num_heads=4,
        dropout=0.0,
    ).eval()
    first_tokens = torch.randint(0, 256, (1, 3, 16))
    second_tokens = first_tokens.clone()
    second_tokens[:, 1:] = torch.randint(0, 256, (1, 2, 16))
    fields = torch.ones_like(first_tokens)
    byte_mask = torch.ones_like(first_tokens, dtype=torch.bool)
    packet_mask = torch.ones((1, 3), dtype=torch.bool)
    directions = torch.ones((1, 3), dtype=torch.long)
    meta = torch.zeros((1, 3, 4))

    with torch.no_grad():
        first = model(first_tokens, fields, byte_mask, packet_mask, directions, meta)
        second = model(second_tokens, fields, byte_mask, packet_mask, directions, meta)

    torch.testing.assert_close(
        first["packet_content"][:, 0], second["packet_content"][:, 0]
    )
    assert not torch.allclose(first["packet_repr"][:, 0], second["packet_repr"][:, 0])


def test_structural_embedding_alignment_rejects_packet_order_mismatch(tmp_path: Path):
    semantic_path = tmp_path / "semantic.npy"
    structural_path = tmp_path / "structural.npy"
    np.save(semantic_path, np.zeros((2, 8), dtype=np.float32))
    np.save(structural_path, np.zeros((2, 4), dtype=np.float32))
    semantic = {
        "flow_id": "flow-a",
        "label_id": 3,
        "embedding_path": str(semantic_path),
        "packet_metas": [{"packet_id": 0}, {"packet_id": 1}],
    }
    structural = {
        "flow_id": "flow-a",
        "label_id": 3,
        "structural_embedding_path": str(structural_path),
        "packet_ids": [1, 0],
    }
    with pytest.raises(ValueError, match="packet_id order mismatch"):
        apply_structural_embedding_reference(semantic, {"flow-a": structural})


def test_expanded_structural_rgssi_preserves_legacy_logits(tmp_path: Path):
    torch.manual_seed(11)
    semantic_dim = 10
    old_meta_dim = 4
    native_dim = 6
    old = FlowTransformerClassifier(
        semantic_dim + old_meta_dim,
        num_classes=3,
        hidden_dim=16,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
        dual_channel_mode="concat",
        meta_feature_dim=old_meta_dim,
    ).eval()
    checkpoint_path = tmp_path / "legacy.pt"
    torch.save(
        {
            "model_state": old.state_dict(),
            "meta_feature_dim": old_meta_dim,
        },
        checkpoint_path,
    )
    expanded = FlowTransformerClassifier(
        semantic_dim + native_dim + old_meta_dim,
        num_classes=3,
        hidden_dim=16,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
        dual_channel_mode="residual",
        meta_feature_dim=native_dim + old_meta_dim,
        native_structural_dim=native_dim,
        dual_channel_gate_mode="adaptive",
    ).eval()
    args = SimpleNamespace(
        init_checkpoint=str(checkpoint_path),
        device="cpu",
        meta_feature_dim=native_dim + old_meta_dim,
        counterfactual_head_only=False,
    )
    maybe_load_init_checkpoint(args, expanded)
    semantic = torch.randn(2, 5, semantic_dim)
    native = torch.randn(2, 5, native_dim)
    meta = torch.randn(2, 5, old_meta_dim)
    mask = torch.ones(2, 5, dtype=torch.bool)
    legacy_logits = old(torch.cat([semantic, meta], dim=-1), mask)["logits"]
    expanded_logits = expanded(torch.cat([semantic, native, meta], dim=-1), mask)["logits"]
    torch.testing.assert_close(expanded_logits, legacy_logits, atol=1e-6, rtol=1e-6)


def test_native_interaction_scope_trains_only_causal_residual_modules():
    model = FlowTransformerClassifier(
        input_dim=10 + 6 + 4,
        num_classes=3,
        hidden_dim=16,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
        dual_channel_mode="residual",
        meta_feature_dim=10,
        native_structural_dim=6,
        dual_channel_gate_mode="adaptive",
    )
    trainable = configure_dual_channel_train_scope(
        model, flow_head=None, scope="native_interaction"
    )
    assert any(name.startswith("native_structural_adapter.") for name in trainable)
    assert "native_structural_raw_gate" in trainable
    assert any(name.startswith("shared_packet_fusion.interaction.") for name in trainable)
    assert any(name.startswith("shared_packet_fusion.gate.") for name in trainable)
    assert not model.semantic_proj.weight.requires_grad
    assert not model.structural_proj.weight.requires_grad
    assert not any(parameter.requires_grad for parameter in model.encoder.parameters())
    assert not any(parameter.requires_grad for parameter in model.cls.parameters())
