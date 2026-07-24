from copy import deepcopy

import numpy as np
import pytest

from train_flow_stats_classifier import flow_features, protocol_payload_prefix


def flow_row():
    first_header = [0x45] * 40
    second_header = [0x46] * 40
    return {
        "packet_metas": [
            {
                "direction": "C2S",
                "packet_len": 128,
                "payload_len": 80,
                "iat": 0.01,
                "payload_entropy": 4.0,
                "ip_ttl": 64,
                "l4": "TCP",
                "tcp_flags": "PA",
                "full_l3_captured": True,
                "src_ip": "10.0.0.1",
                "dst_ip": "10.0.0.2",
                "sport": 49152,
                "dport": 443,
                "seq": 1234,
                "ack": 5678,
                "ip_header_len": 20,
                "tcp_data_offset": 20,
                "l3_hex_prefix": " ".join(
                    f"{value:02x}" for value in first_header + [1, 2, 3, 4, 5, 6, 7, 8]
                ),
                "payload_prefix_len": 8,
            },
            {
                "direction": "S2C",
                "packet_len": 256,
                "payload_len": 192,
                "iat": 0.03,
                "payload_entropy": 6.0,
                "ip_ttl": 128,
                "l4": "TCP",
                "tcp_flags": "A",
                "full_l3_captured": True,
                "src_ip": "10.0.0.2",
                "dst_ip": "10.0.0.1",
                "sport": 443,
                "dport": 49152,
                "seq": 9876,
                "ack": 5432,
                "ip_header_len": 20,
                "tcp_data_offset": 20,
                "l3_hex_prefix": " ".join(
                    f"{value:02x}" for value in second_header + [9, 10, 11, 12, 13, 14, 15, 16]
                ),
                "payload_prefix_len": 8,
            },
        ]
    }


def protocol_closed(row, version="protocol_closed"):
    return np.asarray(flow_features(row, 64, 32, False, version))


@pytest.mark.parametrize("version", ["protocol_closed", "protocol_closed_payload"])
def test_protocol_closed_flow_features_ignore_session_and_header_fields(version):
    baseline = flow_row()
    intervened = deepcopy(baseline)
    for index, meta in enumerate(intervened["packet_metas"]):
        meta.update(
            {
                "src_ip": f"203.0.113.{index + 1}",
                "dst_ip": f"198.51.100.{index + 1}",
                "sport": 1000 + index,
                "dport": 2000 + index,
                "ip_ttl": 1 + index,
                "seq": 40000 + index,
                "ack": 50000 + index,
            }
        )
        raw = intervened["packet_metas"][index]["l3_hex_prefix"].split()
        intervened["packet_metas"][index]["l3_hex_prefix"] = " ".join(
            ["ff"] * 40 + raw[40:]
        )
    np.testing.assert_array_equal(
        protocol_closed(baseline, version), protocol_closed(intervened, version)
    )


def test_protocol_closed_flow_features_retain_behavioral_structure():
    baseline = flow_row()
    changed = deepcopy(baseline)
    changed["packet_metas"][0]["packet_len"] += 64
    changed["packet_metas"][1]["iat"] += 0.2
    assert not np.array_equal(protocol_closed(baseline), protocol_closed(changed))


def test_protocol_closed_flow_features_retain_header_stripped_payload():
    baseline = flow_row()
    changed = deepcopy(baseline)
    raw = changed["packet_metas"][0]["l3_hex_prefix"].split()
    raw[40] = "fe"
    changed["packet_metas"][0]["l3_hex_prefix"] = " ".join(raw)
    assert np.array_equal(protocol_closed(baseline), protocol_closed(changed))
    assert not np.array_equal(
        protocol_closed(baseline, "protocol_closed_payload"),
        protocol_closed(changed, "protocol_closed_payload"),
    )


def test_protocol_closed_flow_features_reject_ports():
    with pytest.raises(ValueError, match="cannot include ports"):
        flow_features(flow_row(), 64, 32, True, "protocol_closed")


def test_protocol_payload_extraction_fails_closed_without_header_lengths():
    meta = deepcopy(flow_row()["packet_metas"][0])
    meta.pop("ip_header_len")
    assert protocol_payload_prefix(meta) == []
    meta["ip_header_len"] = 20
    meta.pop("tcp_data_offset")
    assert protocol_payload_prefix(meta) == []
