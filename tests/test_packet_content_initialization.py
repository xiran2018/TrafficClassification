import torch
import hashlib

from models.native_flow_encoder import NativeFlowEncoder
from models.packet_byte_transformer import PacketByteTransformer
from train_packet_byte_transformer import initialize_protocol_content_encoder


def configs(layers=1):
    native = {
        "max_bytes": 16,
        "max_packets": 4,
        "hidden_dim": 8,
        "byte_layers": layers,
        "flow_layers": 1,
        "num_heads": 2,
        "dropout": 0.1,
        "num_field_types": 9,
        "projection_dim": 8,
    }
    packet = {
        "max_bytes": 16,
        "hidden_dim": 8,
        "num_layers": layers,
        "num_heads": 2,
        "dropout": 0.1,
    }
    return native, packet


def save_native(path, config):
    model = NativeFlowEncoder(**config)
    torch.save(
        {
            "model_config": config,
            "state_dict": model.state_dict(),
            "pretraining_protocol": "native_flow_multitask_v1",
        },
        path,
    )
    return model


def packet_model(config):
    return PacketByteTransformer(
        num_classes=3,
        max_bytes=config["max_bytes"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        dropout=config["dropout"],
        use_protocol_fields=True,
    )


def test_packet_content_encoder_strictly_loads_native_pretraining(tmp_path):
    native_config, packet_config = configs()
    path = tmp_path / "native.pt"
    native = save_native(path, native_config)
    packet = packet_model(packet_config)

    provenance = initialize_protocol_content_encoder(packet, packet_config, str(path))

    assert provenance["strict_state_dict_load"] is True
    assert provenance["protocol_content_checkpoint_sha256"] == hashlib.sha256(
        path.read_bytes()
    ).hexdigest()
    for key, value in native.packet_content_encoder.state_dict().items():
        torch.testing.assert_close(packet.protocol_content_encoder.state_dict()[key], value)


def test_packet_content_initialization_rejects_architecture_mismatch(tmp_path):
    native_config, packet_config = configs(layers=1)
    path = tmp_path / "native.pt"
    save_native(path, native_config)
    packet_config["num_layers"] = 2
    packet = packet_model(packet_config)

    try:
        initialize_protocol_content_encoder(packet, packet_config, str(path))
    except ValueError as error:
        assert "architecture mismatch" in str(error)
    else:
        raise AssertionError("architecture mismatch should fail strict initialization")


def test_packet_content_initialization_rejects_unproven_pretraining_protocol(tmp_path):
    native_config, packet_config = configs()
    path = tmp_path / "native.pt"
    model = NativeFlowEncoder(**native_config)
    torch.save({"model_config": native_config, "state_dict": model.state_dict()}, path)

    try:
        initialize_protocol_content_encoder(packet_model(packet_config), packet_config, str(path))
    except ValueError as error:
        assert "pretraining protocol mismatch" in str(error)
    else:
        raise AssertionError("missing native pretraining provenance should fail")
