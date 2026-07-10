from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from scapy.all import rdpcap, raw
from scapy.layers.inet import IP, TCP, UDP, ICMP

COMMON_SERVER_PORTS = {80, 443, 22, 21, 23, 3389, 53, 25, 110, 143, 3306, 5432}


def stable_id(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:16]


def iter_labeled_pcaps(root: str | Path) -> Iterable[Tuple[str, Path]]:
    root = Path(root)
    if not root.exists():
        return
    for label_dir in sorted(root.iterdir()):
        if not label_dir.is_dir():
            continue
        label = label_dir.name
        files: List[Path] = []
        for pat in ("*.pcap", "*.pcapng", "*.cap", "*.pcap.gz"):
            files.extend(label_dir.glob(pat))
        for p in sorted(set(files)):
            yield label, p


def entropy_bytes(data: bytes) -> float:
    if not data:
        return 0.0
    counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
    probs = counts[counts > 0] / len(data)
    return float(-(probs * np.log2(probs)).sum())


def hex_bytes(data: bytes, max_bytes: Optional[int] = None) -> str:
    if max_bytes is not None:
        data = data[:max_bytes]
    return " ".join(f"{b:02x}" for b in data)


def parse_hex(s: str) -> bytes:
    s = s.replace(" ", "").replace("\n", "").strip()
    if len(s) % 2:
        s = s[:-1]
    return bytes.fromhex(s) if s else b""


def ipv4_header_checksum(header: bytes) -> int:
    """Compute IPv4 header checksum. header must contain the IP header with checksum bytes already zeroed."""
    if len(header) % 2:
        header += b"\x00"
    total = 0
    for i in range(0, len(header), 2):
        total += (header[i] << 8) + header[i + 1]
        total = (total & 0xFFFF) + (total >> 16)
    return (~total) & 0xFFFF


def get_server_ip(packets: List[Any]) -> Optional[str]:
    for pkt in packets:
        if IP in pkt and (TCP in pkt or UDP in pkt):
            src, dst = pkt[IP].src, pkt[IP].dst
            l4 = pkt.getlayer(TCP) or pkt.getlayer(UDP)
            sport, dport = int(l4.sport), int(l4.dport)
            if dport in COMMON_SERVER_PORTS:
                return dst
            if sport in COMMON_SERVER_PORTS:
                return src
            if sport < dport:
                return src
            if dport < sport:
                return dst
    return None


def get_direction(pkt: Any, server_ip: Optional[str]) -> str:
    if server_ip is None or IP not in pkt:
        return "UNK"
    return "S2C" if pkt[IP].src == server_ip else "C2S"


def is_full_l3_captured(pkt: Any) -> bool:
    if IP not in pkt:
        return False
    try:
        ip_len = int(pkt[IP].len) if pkt[IP].len is not None else -1
        captured = len(raw(pkt[IP]))
        return ip_len > 0 and captured >= ip_len
    except Exception:
        return False


def ip_checksum_valid(pkt: Any) -> Optional[bool]:
    if IP not in pkt:
        return None
    try:
        ihl = int(pkt[IP].ihl) * 4
        buf = bytearray(raw(pkt[IP])[:ihl])
        if len(buf) < ihl or ihl < 20:
            return None
        old = (buf[10] << 8) + buf[11]
        buf[10] = 0
        buf[11] = 0
        return ipv4_header_checksum(bytes(buf)) == old
    except Exception:
        return None


def tcp_udp_checksum_valid(pkt: Any, require_full_l3: bool = True) -> Optional[bool]:
    """Validate TCP/UDP checksum only when the full L3 packet is captured.

    TCP/UDP checksum covers pseudo-header + L4 header + complete payload. If a pcap is
    truncated, a strict answer would be misleading, so this returns None by default.
    """
    if IP not in pkt:
        return None
    if require_full_l3 and not is_full_l3_captured(pkt):
        return None
    try:
        if TCP in pkt:
            old = pkt[TCP].chksum
            cp = pkt.copy()
            del cp[TCP].chksum
            rebuilt = IP(raw(cp[IP]))
            return int(rebuilt[TCP].chksum) == int(old)
        if UDP in pkt:
            old = pkt[UDP].chksum
            if old == 0:
                return True
            cp = pkt.copy()
            del cp[UDP].chksum
            rebuilt = IP(raw(cp[IP]))
            return int(rebuilt[UDP].chksum) == int(old)
    except Exception:
        return None
    return None


@dataclass
class PacketMeta:
    packet_id: int
    time: float
    direction: str
    packet_len: int
    l3_captured_len: int
    full_l3_captured: bool
    payload_len: int
    payload_prefix_len: int
    payload_truncated: bool
    payload_entropy: float
    l3: str
    l4: str
    l3_hex_prefix: str
    src_ip: str = ""
    dst_ip: str = ""
    ip_id: int = -1
    ip_ttl: int = -1
    ip_total_len: int = -1
    ip_header_len: int = -1
    ip_checksum: int = -1
    ip_checksum_valid: Optional[bool] = None
    sport: int = -1
    dport: int = -1
    seq: int = -1
    ack: int = -1
    tcp_flags: str = ""
    tcp_window: int = -1
    tcp_data_offset: int = -1
    l4_checksum: int = -1
    l4_checksum_valid: Optional[bool] = None
    udp_len: int = -1
    iat: float = 0.0


def format_packet_qa_prompt(m: PacketMeta) -> str:
    """Raw-byte prompt for packet Q&A.

    It intentionally avoids fields such as checksum_valid or parsed TTL to prevent
    answer leakage. The model must infer fields from packet bytes.
    """
    return f"""[PacketBytes]
L3PacketPrefixHex: {m.l3_hex_prefix}
CapturedL3Length: {m.l3_captured_len}
FullL3Captured: {m.full_l3_captured}
[EndPacketBytes]""".strip()


def format_packet_embedding_prompt(m: PacketMeta, payload_prefix: str) -> str:
    """Structured packet prompt for embedding extraction.

    It includes raw header fields and payload prefix, but not checksum validity labels.
    """
    if m.l4 == "TCP":
        l4_line = (
            f"TCP: sport={m.sport} dport={m.dport} seq={m.seq} ack={m.ack} "
            f"flags={m.tcp_flags} data_offset={m.tcp_data_offset} window={m.tcp_window} checksum=0x{m.l4_checksum:04x}"
        )
    elif m.l4 == "UDP":
        l4_line = f"UDP: sport={m.sport} dport={m.dport} length={m.udp_len} checksum=0x{m.l4_checksum:04x}"
    else:
        l4_line = f"L4: {m.l4}"
    return f"""[Packet]
Direction: {m.direction}
L3: {m.l3}
IP: src={m.src_ip} dst={m.dst_ip} id={m.ip_id} ttl={m.ip_ttl} proto={m.l4} total_len={m.ip_total_len} ihl={m.ip_header_len} checksum=0x{m.ip_checksum:04x}
{l4_line}
Observed: packet_len={m.packet_len} captured_l3_len={m.l3_captured_len} full_l3_captured={m.full_l3_captured} payload_len={m.payload_len} entropy={m.payload_entropy} iat={m.iat} payload_truncated={m.payload_truncated}
PayloadPrefix: {payload_prefix}
[EndPacket]""".strip()


def extract_flow_packets(
    pcap_path: str | Path,
    max_packets: int = 128,
    payload_prefix_len: int = 128,
    l3_prefix_len: int = 512,
) -> Tuple[List[PacketMeta], List[str], List[str]]:
    """Return packet metadata, raw QA prompts, and structured embedding prompts."""
    packets = rdpcap(str(pcap_path))
    server_ip = get_server_ip(list(packets))
    metas: List[PacketMeta] = []
    qa_prompts: List[str] = []
    embedding_prompts: List[str] = []
    prev_t: Optional[float] = None
    for pkt in packets:
        if len(metas) >= max_packets:
            break
        if IP not in pkt:
            continue
        pid = len(metas)
        t = float(pkt.time)
        iat = 0.0 if prev_t is None else max(0.0, round(t - prev_t, 6))
        prev_t = t
        l3_bytes = raw(pkt[IP])
        payload = b""
        l4 = "OTHER"
        if TCP in pkt:
            l4 = "TCP"
            payload = bytes(pkt[TCP].payload)
        elif UDP in pkt:
            l4 = "UDP"
            payload = bytes(pkt[UDP].payload)
        elif ICMP in pkt:
            l4 = "ICMP"
            payload = bytes(pkt[ICMP].payload)
        payload_prefix = hex_bytes(payload, payload_prefix_len)
        m = PacketMeta(
            packet_id=pid,
            time=t,
            direction=get_direction(pkt, server_ip),
            packet_len=len(pkt),
            l3_captured_len=len(l3_bytes),
            full_l3_captured=is_full_l3_captured(pkt),
            payload_len=len(payload),
            payload_prefix_len=min(len(payload), payload_prefix_len),
            payload_truncated=len(payload) > payload_prefix_len,
            payload_entropy=round(entropy_bytes(payload), 4),
            l3="IPv4",
            l4=l4,
            l3_hex_prefix=hex_bytes(l3_bytes, l3_prefix_len),
            src_ip=pkt[IP].src,
            dst_ip=pkt[IP].dst,
            ip_id=int(pkt[IP].id),
            ip_ttl=int(pkt[IP].ttl),
            ip_total_len=int(pkt[IP].len) if pkt[IP].len is not None else -1,
            ip_header_len=int(pkt[IP].ihl) * 4 if pkt[IP].ihl is not None else -1,
            ip_checksum=int(pkt[IP].chksum) if pkt[IP].chksum is not None else -1,
            ip_checksum_valid=ip_checksum_valid(pkt),
            iat=iat,
        )
        if TCP in pkt:
            m.sport, m.dport = int(pkt[TCP].sport), int(pkt[TCP].dport)
            m.seq, m.ack = int(pkt[TCP].seq), int(pkt[TCP].ack)
            m.tcp_flags = str(pkt[TCP].flags)
            m.tcp_window = int(pkt[TCP].window)
            m.tcp_data_offset = int(pkt[TCP].dataofs) * 4 if pkt[TCP].dataofs is not None else -1
            m.l4_checksum = int(pkt[TCP].chksum) if pkt[TCP].chksum is not None else -1
            m.l4_checksum_valid = tcp_udp_checksum_valid(pkt)
        elif UDP in pkt:
            m.sport, m.dport = int(pkt[UDP].sport), int(pkt[UDP].dport)
            m.udp_len = int(pkt[UDP].len) if pkt[UDP].len is not None else -1
            m.l4_checksum = int(pkt[UDP].chksum) if pkt[UDP].chksum is not None else -1
            m.l4_checksum_valid = tcp_udp_checksum_valid(pkt)
        metas.append(m)
        qa_prompts.append(format_packet_qa_prompt(m))
        embedding_prompts.append(format_packet_embedding_prompt(m, payload_prefix))
    return metas, qa_prompts, embedding_prompts


def corrupt_ipv4_total_len_keep_ip_checksum_valid(l3_hex_prefix: str) -> Optional[str]:
    """Hard negative: corrupt IPv4 total length while recomputing IP checksum.

    This prevents the validity task from being solved by IP checksum alone.
    """
    buf = bytearray(parse_hex(l3_hex_prefix))
    if len(buf) < 20 or (buf[0] >> 4) != 4:
        return None
    ihl = (buf[0] & 0x0F) * 4
    if ihl < 20 or len(buf) < ihl:
        return None
    old_len = (buf[2] << 8) + buf[3]
    if old_len <= ihl + 4:
        new_len = ihl  # still structurally suspicious for most TCP/UDP packets
    else:
        new_len = max(ihl, old_len - 3)
    buf[2] = (new_len >> 8) & 0xFF
    buf[3] = new_len & 0xFF
    buf[10] = 0
    buf[11] = 0
    csum = ipv4_header_checksum(bytes(buf[:ihl]))
    buf[10] = (csum >> 8) & 0xFF
    buf[11] = csum & 0xFF
    return hex_bytes(bytes(buf))


def corrupt_ipv4_checksum_only(l3_hex_prefix: str) -> Optional[str]:
    buf = bytearray(parse_hex(l3_hex_prefix))
    if len(buf) < 20 or (buf[0] >> 4) != 4:
        return None
    buf[10] ^= 0x01
    return hex_bytes(bytes(buf))


def meta_feature_vector(m: Any) -> List[float]:
    dir_v = 1.0 if m.direction == "C2S" else (-1.0 if m.direction == "S2C" else 0.0)
    l4_tcp = 1.0 if m.l4 == "TCP" else 0.0
    l4_udp = 1.0 if m.l4 == "UDP" else 0.0
    flags = m.tcp_flags or ""
    return [
        dir_v,
        math.log1p(max(0, m.packet_len)),
        math.log1p(max(0, m.payload_len)),
        math.log1p(max(0.0, m.iat)),
        float(m.payload_entropy) / 8.0,
        l4_tcp,
        l4_udp,
        1.0 if "S" in flags else 0.0,
        1.0 if "A" in flags else 0.0,
        1.0 if "P" in flags else 0.0,
        1.0 if "F" in flags else 0.0,
        1.0 if "R" in flags else 0.0,
        math.log1p(max(0, getattr(m, "tcp_window", -1))),
        1.0 if getattr(m, "full_l3_captured", False) else 0.0,
    ]


def make_label_map(labels: Iterable[str]) -> Dict[str, int]:
    return {lab: i for i, lab in enumerate(sorted(set(labels)))}


def packet_information_weight(m: Any) -> float:
    """Heuristic weight for weak packet-level classification loss.

    Packet labels are inherited from flow labels, so some packets are weakly informative
    or label-noisy. This weight down-weights generic control packets and gives higher
    weights to payload-bearing packets that are more likely to contain application
    behavior.
    """
    flags = getattr(m, "tcp_flags", "") or ""
    payload_len = int(getattr(m, "payload_len", 0) or 0)
    l4 = getattr(m, "l4", "OTHER")
    entropy = float(getattr(m, "payload_entropy", 0.0) or 0.0)
    full_l3_captured = bool(getattr(m, "full_l3_captured", False))

    if l4 == "TCP" and payload_len == 0:
        if flags == "A":
            return 0.05
        if flags in {"S", "SA", "F", "FA", "R", "RA"}:
            return 0.12
        return 0.2

    if payload_len <= 8:
        weight = 0.25
    elif payload_len < 32:
        weight = 0.45
    elif payload_len < 128:
        weight = 0.7
    else:
        weight = 1.0

    if l4 == "TCP" and "P" in flags and payload_len > 0:
        weight = max(weight, 0.9)
    if l4 == "UDP" and payload_len >= 32:
        weight = max(weight, 0.8)
    if payload_len >= 128 and entropy >= 5.5:
        weight *= 1.1
    if not full_l3_captured:
        weight *= 0.85

    return round(min(1.2, max(0.05, weight)), 3)
