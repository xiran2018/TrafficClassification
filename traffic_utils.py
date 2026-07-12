from __future__ import annotations

import hashlib
import gzip
import math
import socket
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

COMMON_SERVER_PORTS = {80, 443, 22, 21, 23, 3389, 53, 25, 110, 143, 3306, 5432}


def stable_id(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:16]


def pseudo_ipv4(value: str, salt: str) -> str:
    digest = hashlib.sha1(f"{salt}|ip|{value}".encode("utf-8", errors="ignore")).digest()
    return f"10.{digest[0]}.{digest[1]}.{1 + digest[2] % 254}"


def pseudo_port(value: int, salt: str) -> int:
    if value < 0:
        return value
    digest = hashlib.sha1(f"{salt}|port|{value}".encode("utf-8", errors="ignore")).digest()
    return 1024 + (int.from_bytes(digest[:2], "big") % (65535 - 1024))


def iter_labeled_pcaps(root: str | Path) -> Iterable[Tuple[str, Path]]:
    root = Path(root)
    if not root.exists():
        return
    root_files: List[Path] = []
    for pat in ("*.pcap", "*.pcapng", "*.cap", "*.pcap.gz"):
        root_files.extend(root.glob(pat))
    for p in sorted(set(root_files)):
        label = p.name[:-len(".pcap.gz")] if p.name.endswith(".pcap.gz") else p.stem
        yield label, p
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


def internet_checksum(data: bytes) -> int:
    if len(data) % 2:
        data += b"\x00"
    total = 0
    for i in range(0, len(data), 2):
        total += (data[i] << 8) + data[i + 1]
        total = (total & 0xFFFF) + (total >> 16)
    return (~total) & 0xFFFF


@dataclass
class ParsedPacket:
    time: float
    frame_len: int
    l3_bytes: bytes
    src_ip: str
    dst_ip: str
    ip_id: int
    ip_ttl: int
    ip_total_len: int
    ip_header_len: int
    ip_proto: int
    ip_checksum: int
    l4: str
    sport: int = -1
    dport: int = -1
    seq: int = -1
    ack: int = -1
    tcp_flags: str = ""
    tcp_window: int = -1
    tcp_data_offset: int = -1
    udp_len: int = -1
    l4_checksum: int = -1
    payload: bytes = b""


def open_maybe_gzip(path: Path):
    return gzip.open(path, "rb") if path.suffix == ".gz" else open(path, "rb")


def iter_pcap_records(path: str | Path) -> Iterable[Tuple[float, bytes, int]]:
    path = Path(path)
    with open_maybe_gzip(path) as f:
        header = f.read(24)
        if len(header) < 24:
            return
        magic = header[:4]
        if magic == b"\x0a\x0d\x0d\x0a":
            f.seek(0)
            yield from iter_pcapng_records(f)
            return
        if magic == b"\xd4\xc3\xb2\xa1":
            endian, ts_scale = "<", 1_000_000.0
        elif magic == b"\xa1\xb2\xc3\xd4":
            endian, ts_scale = ">", 1_000_000.0
        elif magic == b"\x4d\x3c\xb2\xa1":
            endian, ts_scale = "<", 1_000_000_000.0
        elif magic == b"\xa1\xb2\x3c\x4d":
            endian, ts_scale = ">", 1_000_000_000.0
        else:
            raise ValueError(f"Unsupported pcap format for {path}; pcapng is not supported by the offline parser.")
        _magic, _vmaj, _vmin, _tz, _sig, _snaplen, linktype = struct.unpack(endian + "IHHIIII", header)
        while True:
            rec = f.read(16)
            if len(rec) == 0:
                break
            if len(rec) < 16:
                break
            ts_sec, ts_frac, incl_len, orig_len = struct.unpack(endian + "IIII", rec)
            data = f.read(incl_len)
            if len(data) < incl_len:
                break
            yield ts_sec + ts_frac / ts_scale, data, linktype


def iter_pcapng_records(f) -> Iterable[Tuple[float, bytes, int]]:
    endian = "<"
    linktypes: Dict[int, int] = {}
    ts_scales: Dict[int, float] = {}
    while True:
        head = f.read(8)
        if len(head) == 0:
            break
        if len(head) < 8:
            break
        block_type_le, block_len_le = struct.unpack("<II", head)
        block_type_be, block_len_be = struct.unpack(">II", head)
        if block_type_le == 0x0A0D0D0A:
            block_type, block_len = block_type_le, block_len_le
        else:
            block_type, block_len = (
                (block_type_le, block_len_le) if endian == "<" else (block_type_be, block_len_be)
            )
        if block_len < 12:
            break
        body = f.read(block_len - 12)
        trailer = f.read(4)
        if len(body) < block_len - 12 or len(trailer) < 4:
            break
        if block_type == 0x0A0D0D0A:
            byte_order_magic = body[:4]
            if byte_order_magic == b"\x4d\x3c\x2b\x1a":
                endian = "<"
            elif byte_order_magic == b"\x1a\x2b\x3c\x4d":
                endian = ">"
            else:
                break
            linktypes.clear()
            ts_scales.clear()
            continue
        if block_type == 0x00000001 and len(body) >= 8:
            iface_id = len(linktypes)
            linktype, _reserved, _snaplen = struct.unpack(endian + "HHI", body[:8])
            linktypes[iface_id] = int(linktype)
            ts_scales[iface_id] = pcapng_ts_scale(body[8:], endian)
            continue
        if block_type == 0x00000006 and len(body) >= 20:
            iface_id, ts_high, ts_low, cap_len, _pkt_len = struct.unpack(endian + "IIIII", body[:20])
            padded_len = (cap_len + 3) & ~3
            if len(body) < 20 + padded_len:
                continue
            data = body[20:20 + cap_len]
            linktype = linktypes.get(int(iface_id), 1)
            ts_scale = ts_scales.get(int(iface_id), 1_000_000.0)
            ts = ((int(ts_high) << 32) | int(ts_low)) / ts_scale
            yield ts, data, linktype


def pcapng_ts_scale(options: bytes, endian: str) -> float:
    offset = 0
    while offset + 4 <= len(options):
        code, length = struct.unpack(endian + "HH", options[offset:offset + 4])
        offset += 4
        value = options[offset:offset + length]
        offset += (length + 3) & ~3
        if code == 0:
            break
        if code == 9 and value:
            raw = value[0]
            if raw & 0x80:
                return float(2 ** (raw & 0x7F))
            return float(10 ** raw)
    return 1_000_000.0


def ipv4_offset(frame: bytes, linktype: int) -> Optional[int]:
    if linktype in {101, 228}:  # LINKTYPE_RAW / LINKTYPE_IPV4
        return 0 if frame and (frame[0] >> 4) == 4 else None
    if linktype == 1:  # Ethernet
        if len(frame) < 14:
            return None
        eth_type = int.from_bytes(frame[12:14], "big")
        offset = 14
        while eth_type in {0x8100, 0x88A8} and len(frame) >= offset + 4:
            eth_type = int.from_bytes(frame[offset + 2:offset + 4], "big")
            offset += 4
        return offset if eth_type == 0x0800 else None
    if linktype == 113:  # Linux cooked capture v1
        if len(frame) < 16:
            return None
        proto = int.from_bytes(frame[14:16], "big")
        return 16 if proto == 0x0800 else None
    return None


def tcp_flags_to_str(flags: int) -> str:
    names = [(0x01, "F"), (0x02, "S"), (0x04, "R"), (0x08, "P"), (0x10, "A"), (0x20, "U"), (0x40, "E"), (0x80, "C")]
    return "".join(name for bit, name in names if flags & bit)


def parse_ipv4_packet(ts: float, frame: bytes, linktype: int) -> Optional[ParsedPacket]:
    off = ipv4_offset(frame, linktype)
    if off is None or len(frame) < off + 20:
        return None
    l3 = frame[off:]
    version = l3[0] >> 4
    ihl = (l3[0] & 0x0F) * 4
    if version != 4 or ihl < 20 or len(l3) < ihl:
        return None
    total_len = int.from_bytes(l3[2:4], "big")
    ip_id = int.from_bytes(l3[4:6], "big")
    ttl = l3[8]
    proto = l3[9]
    chksum = int.from_bytes(l3[10:12], "big")
    src_ip = socket.inet_ntoa(l3[12:16])
    dst_ip = socket.inet_ntoa(l3[16:20])
    captured_l3 = l3[: min(len(l3), total_len)] if total_len > 0 else l3
    l4_bytes = captured_l3[ihl:]
    pkt = ParsedPacket(
        time=ts,
        frame_len=len(frame),
        l3_bytes=captured_l3,
        src_ip=src_ip,
        dst_ip=dst_ip,
        ip_id=ip_id,
        ip_ttl=ttl,
        ip_total_len=total_len,
        ip_header_len=ihl,
        ip_proto=proto,
        ip_checksum=chksum,
        l4="OTHER",
    )
    if proto == 6 and len(l4_bytes) >= 20:
        dataofs = (l4_bytes[12] >> 4) * 4
        if dataofs >= 20 and len(l4_bytes) >= min(dataofs, len(l4_bytes)):
            pkt.l4 = "TCP"
            pkt.sport = int.from_bytes(l4_bytes[0:2], "big")
            pkt.dport = int.from_bytes(l4_bytes[2:4], "big")
            pkt.seq = int.from_bytes(l4_bytes[4:8], "big")
            pkt.ack = int.from_bytes(l4_bytes[8:12], "big")
            pkt.tcp_data_offset = dataofs
            pkt.tcp_flags = tcp_flags_to_str(l4_bytes[13])
            pkt.tcp_window = int.from_bytes(l4_bytes[14:16], "big")
            pkt.l4_checksum = int.from_bytes(l4_bytes[16:18], "big")
            pkt.payload = l4_bytes[dataofs:] if len(l4_bytes) >= dataofs else b""
    elif proto == 17 and len(l4_bytes) >= 8:
        pkt.l4 = "UDP"
        pkt.sport = int.from_bytes(l4_bytes[0:2], "big")
        pkt.dport = int.from_bytes(l4_bytes[2:4], "big")
        pkt.udp_len = int.from_bytes(l4_bytes[4:6], "big")
        pkt.l4_checksum = int.from_bytes(l4_bytes[6:8], "big")
        pkt.payload = l4_bytes[8:]
    elif proto == 1 and len(l4_bytes) >= 4:
        pkt.l4 = "ICMP"
        pkt.payload = l4_bytes[4:]
    return pkt


def get_server_ip(packets: List[ParsedPacket]) -> Optional[str]:
    for pkt in packets:
        if pkt.l4 in {"TCP", "UDP"}:
            src, dst = pkt.src_ip, pkt.dst_ip
            sport, dport = pkt.sport, pkt.dport
            if dport in COMMON_SERVER_PORTS:
                return dst
            if sport in COMMON_SERVER_PORTS:
                return src
            if sport < dport:
                return src
            if dport < sport:
                return dst
    return None


def get_direction(pkt: ParsedPacket, server_ip: Optional[str]) -> str:
    if server_ip is None:
        return "UNK"
    return "S2C" if pkt.src_ip == server_ip else "C2S"


def is_full_l3_captured(pkt: ParsedPacket) -> bool:
    return pkt.ip_total_len > 0 and len(pkt.l3_bytes) >= pkt.ip_total_len


def ip_checksum_valid(pkt: ParsedPacket) -> Optional[bool]:
    try:
        ihl = pkt.ip_header_len
        buf = bytearray(pkt.l3_bytes[:ihl])
        if len(buf) < ihl or ihl < 20:
            return None
        old = (buf[10] << 8) + buf[11]
        buf[10] = 0
        buf[11] = 0
        return ipv4_header_checksum(bytes(buf)) == old
    except Exception:
        return None


def tcp_udp_checksum_valid(pkt: ParsedPacket, require_full_l3: bool = True) -> Optional[bool]:
    """Validate TCP/UDP checksum only when the full L3 packet is captured.

    TCP/UDP checksum covers pseudo-header + L4 header + complete payload. If a pcap is
    truncated, a strict answer would be misleading, so this returns None by default.
    """
    if require_full_l3 and not is_full_l3_captured(pkt):
        return None
    try:
        if pkt.l4 not in {"TCP", "UDP"} or pkt.l4_checksum < 0:
            return None
        old = pkt.l4_checksum
        if pkt.l4 == "UDP":
            if old == 0:
                return True
        proto = 6 if pkt.l4 == "TCP" else 17
        l4 = bytearray(pkt.l3_bytes[pkt.ip_header_len:pkt.ip_total_len])
        if len(l4) < (20 if proto == 6 else 8):
            return None
        csum_offset = 16 if proto == 6 else 6
        l4[csum_offset] = 0
        l4[csum_offset + 1] = 0
        pseudo = socket.inet_aton(pkt.src_ip) + socket.inet_aton(pkt.dst_ip) + bytes([0, proto]) + len(l4).to_bytes(2, "big")
        return internet_checksum(pseudo + bytes(l4)) == old
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


def format_packet_embedding_prompt(
    m: PacketMeta,
    payload_prefix: str,
    header_policy: str = "full",
    header_random_salt: str = "",
) -> str:
    """Structured packet prompt for embedding extraction.

    It includes raw header fields and payload prefix, but not checksum validity labels.
    """
    src_ip, dst_ip = m.src_ip, m.dst_ip
    sport, dport = m.sport, m.dport
    if header_policy == "randomize_ip_port":
        salt = header_random_salt or "default"
        src_ip = pseudo_ipv4(m.src_ip, salt)
        dst_ip = pseudo_ipv4(m.dst_ip, salt)
        sport = pseudo_port(m.sport, f"{salt}|src")
        dport = pseudo_port(m.dport, f"{salt}|dst")
    elif header_policy == "mask_ip_port":
        src_ip = dst_ip = "[MASK_IP]"
        sport = dport = "[MASK_PORT]"
    elif header_policy != "full":
        raise ValueError(f"Unknown embedding header policy: {header_policy}")
    if m.l4 == "TCP":
        l4_line = (
            f"TCP: sport={sport} dport={dport} seq={m.seq} ack={m.ack} "
            f"flags={m.tcp_flags} data_offset={m.tcp_data_offset} window={m.tcp_window} checksum=0x{m.l4_checksum:04x}"
        )
    elif m.l4 == "UDP":
        l4_line = f"UDP: sport={sport} dport={dport} length={m.udp_len} checksum=0x{m.l4_checksum:04x}"
    else:
        l4_line = f"L4: {m.l4}"
    return f"""[Packet]
Direction: {m.direction}
L3: {m.l3}
IP: src={src_ip} dst={dst_ip} id={m.ip_id} ttl={m.ip_ttl} proto={m.l4} total_len={m.ip_total_len} ihl={m.ip_header_len} checksum=0x{m.ip_checksum:04x}
{l4_line}
Observed: packet_len={m.packet_len} captured_l3_len={m.l3_captured_len} full_l3_captured={m.full_l3_captured} payload_len={m.payload_len} entropy={m.payload_entropy} iat={m.iat} payload_truncated={m.payload_truncated}
PayloadPrefix: {payload_prefix}
[EndPacket]""".strip()


def extract_flow_packets(
    pcap_path: str | Path,
    max_packets: int = 128,
    payload_prefix_len: int = 128,
    l3_prefix_len: int = 512,
    embedding_header_policy: str = "full",
    header_random_salt: str = "",
) -> Tuple[List[PacketMeta], List[str], List[str]]:
    """Return packet metadata, raw QA prompts, and structured embedding prompts."""
    packets: List[ParsedPacket] = []
    for ts, frame, linktype in iter_pcap_records(pcap_path):
        pkt = parse_ipv4_packet(ts, frame, linktype)
        if pkt is None:
            continue
        packets.append(pkt)
        if len(packets) >= max_packets:
            break
    server_ip = get_server_ip(packets)
    metas: List[PacketMeta] = []
    qa_prompts: List[str] = []
    embedding_prompts: List[str] = []
    prev_t: Optional[float] = None
    for pkt in packets:
        pid = len(metas)
        t = float(pkt.time)
        iat = 0.0 if prev_t is None else max(0.0, round(t - prev_t, 6))
        prev_t = t
        l3_bytes = pkt.l3_bytes
        payload = pkt.payload
        l4 = pkt.l4
        payload_prefix = hex_bytes(payload, payload_prefix_len)
        m = PacketMeta(
            packet_id=pid,
            time=t,
            direction=get_direction(pkt, server_ip),
            packet_len=pkt.frame_len,
            l3_captured_len=len(l3_bytes),
            full_l3_captured=is_full_l3_captured(pkt),
            payload_len=len(payload),
            payload_prefix_len=min(len(payload), payload_prefix_len),
            payload_truncated=len(payload) > payload_prefix_len,
            payload_entropy=round(entropy_bytes(payload), 4),
            l3="IPv4",
            l4=l4,
            l3_hex_prefix=hex_bytes(l3_bytes, l3_prefix_len),
            src_ip=pkt.src_ip,
            dst_ip=pkt.dst_ip,
            ip_id=pkt.ip_id,
            ip_ttl=pkt.ip_ttl,
            ip_total_len=pkt.ip_total_len,
            ip_header_len=pkt.ip_header_len,
            ip_checksum=pkt.ip_checksum,
            ip_checksum_valid=ip_checksum_valid(pkt),
            iat=iat,
        )
        if pkt.l4 == "TCP":
            m.sport, m.dport = pkt.sport, pkt.dport
            m.seq, m.ack = pkt.seq, pkt.ack
            m.tcp_flags = pkt.tcp_flags
            m.tcp_window = pkt.tcp_window
            m.tcp_data_offset = pkt.tcp_data_offset
            m.l4_checksum = pkt.l4_checksum
            m.l4_checksum_valid = tcp_udp_checksum_valid(pkt)
        elif pkt.l4 == "UDP":
            m.sport, m.dport = pkt.sport, pkt.dport
            m.udp_len = pkt.udp_len
            m.l4_checksum = pkt.l4_checksum
            m.l4_checksum_valid = tcp_udp_checksum_valid(pkt)
        metas.append(m)
        qa_prompts.append(format_packet_qa_prompt(m))
        embedding_prompts.append(
            format_packet_embedding_prompt(
                m,
                payload_prefix,
                header_policy=embedding_header_policy,
                header_random_salt=header_random_salt or stable_id(str(Path(pcap_path).resolve())),
            )
        )
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
