import torch

from audit_shared_packet_core import PRETRAINING_PROTOCOL, audit_shared_core


def states(hidden=4, packet_prefix="protocol_content_encoder."):
    return {
        f"{packet_prefix}layer.weight": torch.zeros(hidden, hidden),
        f"{packet_prefix}layer.bias": torch.zeros(hidden),
    }


def shared_representation_states(hidden=4, semantic=6, structural=13):
    return {
        "shared_packet_encoder.semantic_proj.0.weight": torch.zeros(hidden, semantic),
        "shared_packet_encoder.content_proj.0.weight": torch.zeros(hidden, hidden),
        "shared_packet_encoder.structural_proj.0.weight": torch.zeros(hidden, structural),
        "shared_packet_encoder.channel_fusion.gate.1.weight": torch.zeros(hidden, hidden * 7),
    }


def packet_checkpoint(*, layers=2, initialized=True):
    return {
        "config": {
            "max_bytes": 128,
            "hidden_dim": 4,
            "num_layers": layers,
            "num_heads": 2,
            "dropout": 0.1,
            "meta_dim": 13,
            "semantic_dim": 6,
            "exact_shared_representation": True,
            "mask_protocol_session_fields": True,
        },
        "state_dict": states() | shared_representation_states(),
        "initialization": (
            {
                "protocol_content_pretraining": PRETRAINING_PROTOCOL,
                "protocol_content_checkpoint": "native.pt",
                "protocol_content_checkpoint_sha256": "native-sha256",
            }
            if initialized
            else {}
        ),
    }


def native_checkpoint(*, layers=2):
    return {
        "model_config": {
            "max_bytes": 128,
            "hidden_dim": 4,
            "byte_layers": layers,
            "num_heads": 2,
            "dropout": 0.1,
            "num_field_types": 9,
        },
        "state_dict": states(packet_prefix="packet_content_encoder."),
        "pretraining_protocol": PRETRAINING_PROTOCOL,
    }


def flow_checkpoint(*, semantic=6, structural=13):
    return {
        "input_dim": semantic + 4 + structural,
        "meta_feature_dim": 4 + structural,
        "native_structural_dim": 4,
        "shared_packet_hidden_dim": 4,
        "exact_shared_packet_encoder": True,
        "model_state": shared_representation_states(
            semantic=semantic, structural=structural
        )
        | {"packet_to_flow_proj.weight": torch.zeros(8, 4)},
    }


def test_exact_shared_core_requires_architecture_schema_pretraining_and_risk_match():
    report = audit_shared_core(
        packet_checkpoint(),
        native_checkpoint(),
        packet_group_reduction="group_mean",
        flow_group_reduction="group_mean",
        native_checkpoint_sha256="native-sha256",
    )
    assert report["status"] == "pass"
    assert report["architecture_match"] is True
    assert report["parameter_schema_match"] is True


def test_shared_class_name_is_not_enough_for_exact_core_claim():
    report = audit_shared_core(
        packet_checkpoint(layers=3, initialized=False),
        native_checkpoint(layers=2),
        packet_group_reduction="none",
        flow_group_reduction="group_mean",
        native_checkpoint_sha256="native-sha256",
    )
    assert report["status"] == "not_ready"
    assert report["architecture_differences"]["num_layers"] == {
        "packet": 3,
        "flow": 2,
    }
    assert report["shared_pretraining_protocol_match"] is False
    assert report["content_risk_protocol_match"] is False


def test_exact_core_rejects_a_different_native_checkpoint_hash():
    report = audit_shared_core(
        packet_checkpoint(),
        native_checkpoint(),
        packet_group_reduction="group_mean",
        flow_group_reduction="group_mean",
        native_checkpoint_sha256="different-native-checkpoint",
    )
    assert report["status"] == "not_ready"
    assert report["shared_pretraining_protocol_match"] is False


def test_exact_representation_requires_matching_schema_and_flow_boundary():
    report = audit_shared_core(
        packet_checkpoint(),
        native_checkpoint(),
        flow_checkpoint=flow_checkpoint(),
        packet_group_reduction="group_mean",
        flow_group_reduction="group_mean",
        native_checkpoint_sha256="native-sha256",
    )
    assert report["status"] == "pass"
    assert report["exact_shared_representation_match"] is True
    assert report["flow_only_projection_boundary_observed"] is True


def test_exact_representation_rejects_different_structural_contract():
    report = audit_shared_core(
        packet_checkpoint(),
        native_checkpoint(),
        flow_checkpoint=flow_checkpoint(structural=14),
        packet_group_reduction="group_mean",
        flow_group_reduction="group_mean",
        native_checkpoint_sha256="native-sha256",
    )
    assert report["status"] == "not_ready"
    assert report["exact_shared_representation_match"] is False
