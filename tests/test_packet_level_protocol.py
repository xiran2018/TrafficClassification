import json
import socket
import struct
import sys

import numpy as np
import pytest
import torch

from calibrate_prediction_prior import main as calibrate_prior_main
from fuse_packet_crossfold import main as fuse_crossfold_main
from packet_eval_utils import encode_packet_logits_with_backoff, packet_classification_metrics
from fuse_packet_experts import (
    blend,
    confidence_features,
    predict_fused_labels_chunked,
    select_candidate,
    temperature_scale,
)
from models.packet_byte_transformer import PacketByteTransformer
from train_packet_byte_transformer import MASK_TOKEN, extract_packet_payload, mask_session_tokens
from train_packet_feature_expert import packet_features
from traffic_utils import PacketMeta, extract_packet_classification_flows, format_packet_embedding_prompt


def tcp_packet(src_ip, dst_ip, sport, dport, seq=1, ack=1):
    tcp = struct.pack(
        "!HHIIBBHHH",
        sport,
        dport,
        seq,
        ack,
        0x50,
        0x10,
        4096,
        0,
        0,
    )
    total_len = 20 + len(tcp)
    ip = struct.pack(
        "!BBHHHBBH4s4s",
        0x45,
        0,
        total_len,
        seq & 0xFFFF,
        0,
        64,
        6,
        0,
        socket.inet_aton(src_ip),
        socket.inet_aton(dst_ip),
    )
    return ip + tcp


def ipv6_tcp_packet(src_ip, dst_ip, sport, dport):
    tcp = struct.pack("!HHIIBBHHH", sport, dport, 1, 1, 0x50, 0x10, 4096, 0, 0)
    header = struct.pack(
        "!IHBB16s16s",
        6 << 28,
        len(tcp),
        6,
        64,
        socket.inet_pton(socket.AF_INET6, src_ip),
        socket.inet_pton(socket.AF_INET6, dst_ip),
    )
    return header + tcp


def write_raw_pcap(path, records):
    data = bytearray(struct.pack("<IHHIIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, 101))
    for timestamp, packet in records:
        sec = int(timestamp)
        usec = int(round((timestamp - sec) * 1_000_000))
        data.extend(struct.pack("<IIII", sec, usec, len(packet), len(packet)))
        data.extend(packet)
    path.write_bytes(bytes(data))


def test_class_pcap_recovers_real_flows_but_keeps_single_packet_inputs(tmp_path):
    pcap = tmp_path / "chat.pcap"
    write_raw_pcap(
        pcap,
        [
            (1.0, tcp_packet("10.0.0.1", "20.0.0.1", 50000, 443, seq=1)),
            (2.0, tcp_packet("10.0.0.2", "20.0.0.2", 50001, 443, seq=2)),
            (3.0, tcp_packet("20.0.0.1", "10.0.0.1", 443, 50000, seq=3)),
            (4.0, tcp_packet("20.0.0.2", "10.0.0.2", 443, 50001, seq=4)),
            (5.0, tcp_packet("10.0.0.1", "20.0.0.1", 50000, 443, seq=5)),
        ],
    )

    flows = list(extract_packet_classification_flows(pcap, max_packets_per_flow=2))

    assert len(flows) == 2
    assert len({flow_id for flow_id, *_ in flows}) == 2
    packet_counts = sorted(len(metas) for _, metas, _, _ in flows)
    assert packet_counts == [2, 2]
    for _, metas, _, _ in flows:
        assert metas[0].iat == 0.0
        assert metas[1].iat == 0.0
        assert {meta.direction for meta in metas} == {"C2S", "S2C"}


def test_recovered_flow_id_is_stable_across_split_directories(tmp_path):
    train_pcap = tmp_path / "train" / "chat.pcap"
    test_pcap = tmp_path / "test" / "chat.pcap"
    train_pcap.parent.mkdir()
    test_pcap.parent.mkdir()
    records = [(1.0, tcp_packet("10.0.0.1", "20.0.0.1", 50000, 443))]
    write_raw_pcap(train_pcap, records)
    write_raw_pcap(test_pcap, records)

    train_flow_id = next(iter(extract_packet_classification_flows(train_pcap)))[0]
    test_flow_id = next(iter(extract_packet_classification_flows(test_pcap)))[0]

    assert train_flow_id == test_flow_id


def test_class_pcap_keeps_ipv6_packets_as_single_packet_samples(tmp_path):
    pcap = tmp_path / "tls.pcap"
    write_raw_pcap(pcap, [(1.0, ipv6_tcp_packet("2001:db8::1", "2001:db8::2", 50000, 443))])

    flows = list(extract_packet_classification_flows(pcap))

    assert len(flows) == 1
    _, metas, _, prompts = flows[0]
    assert len(metas) == 1
    assert metas[0].l3 == "IPv6"
    assert metas[0].iat == 0.0
    assert "L3: IPv6" in prompts[0]


def test_packet_macro_f1_includes_missing_labels():
    metrics = packet_classification_metrics(
        y_true=[0, 0, 1, 1],
        y_pred=[0, 0, 1, 0],
        num_classes=3,
        label_names=["a", "b", "c"],
    )

    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["per_class"]["c"]["support"] == 0
    assert metrics["per_class"]["c"]["f1"] == 0.0
    assert len(metrics["confusion_matrix"]) == 3


def test_feature_expert_does_not_consume_flow_identity():
    row = {
        "flow_id": "flow-a",
        "meta": {
            "l3_hex_prefix": "45 00 00 28 00 01 00 00 40 06 00 00 0a 00 00 01 14 00 00 01",
            "packet_len": 40,
            "payload_len": 0,
            "l4": "TCP",
            "sport": 50000,
            "dport": 443,
            "tcp_flags": "A",
        },
    }
    other_flow = {**row, "flow_id": "flow-b"}

    assert np.array_equal(packet_features(row, 32, False), packet_features(other_flow, 32, False))


def test_feature_expert_can_mask_endpoint_shortcuts():
    base = {
        "meta": {
            "l3_hex_prefix": tcp_packet("10.0.0.1", "20.0.0.1", 50000, 443).hex(),
            "packet_len": 40,
            "payload_len": 0,
            "l4": "TCP",
            "sport": 50000,
            "dport": 443,
            "tcp_flags": "A",
        }
    }
    changed = {
        "meta": {
            **base["meta"],
            "l3_hex_prefix": tcp_packet("30.0.0.1", "40.0.0.1", 12345, 8443).hex(),
            "sport": 12345,
            "dport": 8443,
        }
    }

    assert np.array_equal(packet_features(base, 32, True, True), packet_features(changed, 32, True, True))


def test_feature_expert_session_mask_removes_mutable_identifiers():
    base_packet = tcp_packet("10.0.0.1", "20.0.0.1", 50000, 443, seq=1, ack=2)
    changed_packet = tcp_packet("30.0.0.1", "40.0.0.1", 12345, 8443, seq=300, ack=400)
    base = {
        "meta": {
            "l3_hex_prefix": base_packet.hex(),
            "packet_len": 40,
            "payload_len": 0,
            "l4": "TCP",
            "ip_proto": 6,
            "ip_ttl": 64,
            "sport": 50000,
            "dport": 443,
            "seq": 1,
            "ack": 2,
            "tcp_flags": "A",
            "direction": "C2S",
        }
    }
    changed = {
        "meta": {
            **base["meta"],
            "l3_hex_prefix": changed_packet.hex(),
            "ip_ttl": 128,
            "sport": 12345,
            "dport": 8443,
            "seq": 300,
            "ack": 400,
            "direction": "S2C",
        }
    }

    assert np.array_equal(
        packet_features(base, 64, False, False, True),
        packet_features(changed, 64, False, False, True),
    )


def test_byte_transformer_session_mask_and_forward():
    raw = bytes.fromhex(
        "45 00 00 3c 12 34 40 00 40 06 ab cd 0a 00 00 01 0a 00 00 02 "
        "30 39 01 bb 01 02 03 04 05 06 07 08 50 18 10 00 ab cd 00 00 "
        "17 03 03 00 03 01 02 03"
    )
    tokens = np.full(64, 256, dtype=np.int64)
    tokens[: len(raw)] = np.frombuffer(raw, dtype=np.uint8)
    masked = mask_session_tokens(tokens, len(raw))
    for start, stop in [(4, 6), (8, 9), (10, 20), (20, 32), (36, 38)]:
        assert np.all(masked[start:stop] == MASK_TOKEN)
    assert np.array_equal(masked[40:len(raw)], tokens[40:len(raw)])

    model = PacketByteTransformer(
        num_classes=3,
        max_bytes=64,
        meta_dim=28,
        hidden_dim=32,
        num_layers=1,
        num_heads=4,
        projection_dim=16,
    )
    logits, projected, gate = model(
        torch.tensor(np.stack([tokens, masked])),
        torch.tensor([len(raw), len(raw)]),
        torch.zeros(2, 28),
    )
    assert logits.shape == (2, 3)
    assert projected.shape == (2, 16)
    assert gate.shape == (2, 32)
    assert torch.isfinite(logits).all()


def test_dual_channel_byte_transformer_extracts_current_packet_payload():
    payload = bytes.fromhex("17 03 03 00 03 01 02 03")
    raw = bytes.fromhex(
        "45 00 00 30 12 34 40 00 40 06 ab cd 0a 00 00 01 0a 00 00 02 "
        "30 39 01 bb 01 02 03 04 05 06 07 08 50 18 10 00 ab cd 00 00"
    ) + payload
    assert extract_packet_payload(raw, {}) == payload

    tokens = np.full(64, 256, dtype=np.int64)
    tokens[: len(raw)] = np.frombuffer(raw, dtype=np.uint8)
    payload_tokens = np.full(16, 256, dtype=np.int64)
    payload_tokens[: len(payload)] = np.frombuffer(payload, dtype=np.uint8)
    model = PacketByteTransformer(
        num_classes=3,
        max_bytes=64,
        max_payload_bytes=16,
        use_payload_channel=True,
        meta_dim=28,
        hidden_dim=32,
        num_layers=1,
        num_heads=4,
        projection_dim=16,
    )
    logits, projected, gate = model(
        torch.tensor(np.stack([tokens, tokens])),
        torch.tensor([len(raw), len(raw)]),
        torch.zeros(2, 28),
        torch.tensor(np.stack([payload_tokens, payload_tokens])),
        torch.tensor([len(payload), len(payload)]),
    )
    assert logits.shape == (2, 3)
    assert projected.shape == (2, 16)
    assert gate.shape == (2, 32)
    assert torch.isfinite(logits).all()


def test_packet_expert_gate_can_disable_either_channel():
    semantic = np.asarray([[0.8, 0.2], [0.1, 0.9]], dtype=np.float64)
    structural = np.asarray([[0.3, 0.7], [0.6, 0.4]], dtype=np.float64)

    assert np.allclose(blend(semantic, structural, 0.0), structural)
    assert np.allclose(blend(semantic, structural, 1.0), semantic)
    assert confidence_features(semantic, structural).shape == (2, 14)


def test_chunked_packet_fusion_matches_direct_global_blend():
    rng = np.random.default_rng(7)
    semantic = rng.dirichlet(np.ones(4), size=13).astype(np.float32)
    structural = rng.dirichlet(np.ones(4), size=13).astype(np.float32)
    expected = blend(
        temperature_scale(semantic, 0.8),
        temperature_scale(structural, 1.2),
        0.35,
    ).argmax(axis=1)
    actual = predict_fused_labels_chunked(
        semantic,
        structural,
        "global",
        0.8,
        1.2,
        0.35,
        None,
        0.0,
        chunk_size=3,
    )
    assert np.array_equal(actual, expected)


def test_probability_temperature_scaling_preserves_distribution_contract():
    probabilities = np.asarray([[0.8, 0.2], [0.25, 0.75]], dtype=np.float64)

    unchanged = temperature_scale(probabilities, 1.0)
    softened = temperature_scale(probabilities, 2.0)

    assert np.allclose(unchanged, probabilities)
    assert np.allclose(softened.sum(axis=1), 1.0)
    assert softened[0, 0] < probabilities[0, 0]


def test_one_se_gate_selection_prefers_stable_simpler_candidate():
    candidates = [
        {
            "method": "global",
            "gate_c": None,
            "gate_strength": 1.0,
            "macro_f1": 0.80,
            "accuracy": 0.81,
            "fold_macro_f1_mean": 0.80,
            "fold_macro_f1_se": 0.01,
        },
        {
            "method": "reliability_gate",
            "gate_c": 10.0,
            "gate_strength": 1.0,
            "macro_f1": 0.805,
            "accuracy": 0.82,
            "fold_macro_f1_mean": 0.805,
            "fold_macro_f1_se": 0.01,
        },
    ]

    selected, threshold = select_candidate(candidates, "one_se")

    assert threshold == pytest.approx(0.795)
    assert selected["method"] == "global"


def test_semantic_session_mask_keeps_behavior_but_removes_identifiers():
    meta = PacketMeta(
        packet_id=0,
        time=0.0,
        direction="C2S",
        packet_len=60,
        l3_captured_len=60,
        full_l3_captured=True,
        payload_len=8,
        payload_prefix_len=8,
        payload_truncated=False,
        payload_entropy=2.0,
        l3="IPv4",
        l4="TCP",
        l3_hex_prefix="",
        src_ip="10.0.0.1",
        dst_ip="20.0.0.1",
        ip_id=123,
        ip_ttl=64,
        ip_total_len=60,
        ip_header_len=20,
        ip_checksum=456,
        sport=50000,
        dport=443,
        seq=789,
        ack=1011,
        tcp_flags="PA",
        tcp_window=4096,
        tcp_data_offset=20,
        l4_checksum=1213,
    )

    prompt = format_packet_embedding_prompt(meta, "de ad be ef", "mask_session_fields")

    assert "10.0.0.1" not in prompt
    assert "50000" not in prompt
    assert "seq=789" not in prompt
    assert "ttl=64" not in prompt
    assert "flags=PA" in prompt
    assert "window=4096" in prompt
    assert "PayloadPrefix: de ad be ef" in prompt


def test_packet_evaluator_recovers_from_large_batch_oom():
    class CapacityLimitedModel:
        def encode_packets(self, input_ids, attention_mask):
            if len(input_ids) > 2:
                raise torch.OutOfMemoryError("synthetic capacity limit")
            logits = torch.stack([input_ids[:, 0].float(), attention_mask[:, 0].float()], dim=1)
            return logits, logits, logits

    input_ids = torch.arange(5, dtype=torch.long).view(5, 1)
    attention_mask = torch.ones_like(input_ids)

    logits = encode_packet_logits_with_backoff(CapacityLimitedModel(), input_ids, attention_mask)

    assert logits.shape == (5, 2)
    assert torch.equal(logits[:, 0], torch.arange(5, dtype=torch.float32))


def test_weighted_packet_probability_fusion(tmp_path, monkeypatch):
    y_true = np.asarray([0, 1], dtype=np.int64)
    first = np.asarray([[0.9, 0.1], [0.4, 0.6]], dtype=np.float32)
    second = np.asarray([[0.5, 0.5], [0.1, 0.9]], dtype=np.float32)
    first_path = tmp_path / "first.npz"
    second_path = tmp_path / "second.npz"
    output_json = tmp_path / "fused.json"
    output_npz = tmp_path / "fused.npz"
    label_map = tmp_path / "label_map.json"
    np.savez_compressed(first_path, y_true=y_true, probabilities=first)
    np.savez_compressed(second_path, y_true=y_true, probabilities=second)
    label_map.write_text(json.dumps({"zero": 0, "one": 1}), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fuse_packet_crossfold.py",
            "--inputs", str(first_path), str(second_path),
            "--weights", "0.25,0.75",
            "--label_map", str(label_map),
            "--output_json", str(output_json),
            "--output_npz", str(output_npz),
        ],
    )

    fuse_crossfold_main()

    fused = np.load(output_npz)["probabilities"]
    np.testing.assert_allclose(fused, 0.25 * first + 0.75 * second)
    assert json.loads(output_json.read_text(encoding="utf-8"))["weights"] == [0.25, 0.75]


def test_packet_prior_calibration_npz_contract(tmp_path, monkeypatch):
    y_valid = np.asarray([0, 0, 1, 1], dtype=np.int64)
    valid_prob = np.asarray(
        [[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]], dtype=np.float32
    )
    y_test = np.asarray([0, 1], dtype=np.int64)
    test_prob = np.asarray([[0.7, 0.3], [0.3, 0.7]], dtype=np.float32)
    valid_path = tmp_path / "valid.npz"
    test_path = tmp_path / "test.npz"
    output_json = tmp_path / "calibrated.json"
    output_npz = tmp_path / "calibrated.npz"
    label_map = tmp_path / "label_map.json"
    np.savez_compressed(valid_path, y_true=y_valid, probabilities=valid_prob)
    np.savez_compressed(test_path, y_true=y_test, probabilities=test_prob)
    label_map.write_text(json.dumps({"zero": 0, "one": 1}), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "calibrate_prediction_prior.py",
            "--valid_npz", str(valid_path),
            "--test_npz", str(test_path),
            "--label_map", str(label_map),
            "--strengths", "0",
            "--selection_scope", "valid_weighted",
            "--output_json", str(output_json),
            "--output_npz", str(output_npz),
        ],
    )

    calibrate_prior_main()

    result = json.loads(output_json.read_text(encoding="utf-8"))
    assert result["task"] == "packet-level-classification"
    assert result["sample_unit"] == "one_packet"
    assert "flow_prob" not in result
    calibrated = np.load(output_npz)
    np.testing.assert_array_equal(calibrated["y_true"], y_test)
    np.testing.assert_allclose(calibrated["probabilities"], test_prob)


def test_packet_weight_selection_exports_validation_blend(tmp_path, monkeypatch):
    y_true = np.asarray([0, 1], dtype=np.int64)
    first = np.asarray([[0.9, 0.1], [0.1, 0.9]], dtype=np.float32)
    second = np.asarray([[0.4, 0.6], [0.6, 0.4]], dtype=np.float32)
    paths = {}
    for split in ("valid", "test"):
        for name, probabilities in (("first", first), ("second", second)):
            path = tmp_path / f"{split}_{name}.npz"
            np.savez_compressed(path, y_true=y_true, probabilities=probabilities)
            paths[f"{split}_{name}"] = path
    label_map = tmp_path / "label_map.json"
    output_json = tmp_path / "selected.json"
    output_npz = tmp_path / "selected_test.npz"
    output_validation_npz = tmp_path / "selected_valid.npz"
    label_map.write_text(json.dumps({"zero": 0, "one": 1}), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fuse_packet_crossfold.py",
            "--inputs", str(paths["test_first"]), str(paths["test_second"]),
            "--validation_inputs", str(paths["valid_first"]), str(paths["valid_second"]),
            "--weight_grid_size", "5",
            "--label_map", str(label_map),
            "--output_json", str(output_json),
            "--output_npz", str(output_npz),
            "--output_validation_npz", str(output_validation_npz),
        ],
    )

    fuse_crossfold_main()

    result = json.loads(output_json.read_text(encoding="utf-8"))
    assert result["weight_selection"]["scope"] == "validation_only"
    assert result["weights"] == [0.25, 0.75]
    np.testing.assert_allclose(
        np.load(output_validation_npz)["probabilities"],
        0.25 * first + 0.75 * second,
    )


def test_flow_prior_calibration_keeps_legacy_json_contract(tmp_path, monkeypatch):
    input_json = tmp_path / "flow_input.json"
    output_json = tmp_path / "flow_output.json"
    input_json.write_text(
        json.dumps(
            {
                "valid_prob": [[0.9, 0.1], [0.1, 0.9]],
                "valid_y_true": [0, 1],
                "valid_flow_ids": ["valid-0", "valid-1"],
                "flow_prob": [[0.8, 0.2], [0.2, 0.8]],
                "flow_y_true": [0, 1],
                "flow_ids": ["test-0", "test-1"],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "calibrate_prediction_prior.py",
            "--input_json", str(input_json),
            "--strengths", "0",
            "--output_json", str(output_json),
        ],
    )

    calibrate_prior_main()

    result = json.loads(output_json.read_text(encoding="utf-8"))
    assert result["task"] == "flow-level-classification"
    assert result["sample_unit"] == "one_flow"
    assert result["flow_ids"] == ["test-0", "test-1"]
    assert result["valid_flow_ids"] == ["valid-0", "valid-1"]
    assert "flow_prob" in result and "valid_prob" in result
