from traffic_utils import (
    PROTOCOL_CLOSED_INTERVENTION_GROUPS,
    PacketMeta,
    format_packet_embedding_prompt,
    resolve_embedding_header_policy,
)


def packet(packet_id: int = 0) -> PacketMeta:
    return PacketMeta(
        packet_id=packet_id,
        time=0.0,
        direction="C2S",
        packet_len=60,
        l3_captured_len=60,
        full_l3_captured=True,
        payload_len=8,
        payload_prefix_len=8,
        payload_truncated=False,
        payload_entropy=1.0,
        l3="IPv4",
        l4="TCP",
        l3_hex_prefix="45 00",
        src_ip="192.0.2.1",
        dst_ip="198.51.100.2",
        ip_id=123,
        ip_ttl=64,
        ip_total_len=60,
        ip_header_len=20,
        ip_checksum=0x1234,
        sport=12345,
        dport=443,
        seq=100,
        ack=200,
        tcp_flags="A",
        tcp_window=4096,
        tcp_data_offset=20,
        l4_checksum=0x5678,
    )


def test_endpoint_closed_masks_protocol_dependencies_but_keeps_behavior():
    prompt = format_packet_embedding_prompt(
        packet(), "aa bb", header_policy="mask_endpoint_closed"
    )
    for leaked in ("192.0.2.1", "198.51.100.2", "12345", "0x1234", "0x5678"):
        assert leaked not in prompt
    assert "[MASK_DIRECTION]" in prompt
    assert prompt.count("[MASK_CHECKSUM]") == 2
    assert "seq=100" in prompt
    assert "window=4096" in prompt


def test_protocol_closed_mixture_is_stable_and_uses_only_registered_groups():
    first = resolve_embedding_header_policy(packet(7), "protocol_closed_mixture", "flow")
    second = resolve_embedding_header_policy(packet(7), "protocol_closed_mixture", "flow")
    assert first == second
    assert first in PROTOCOL_CLOSED_INTERVENTION_GROUPS

