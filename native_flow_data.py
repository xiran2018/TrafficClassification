from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from torch.utils.data import Dataset

from models.native_flow_encoder import MASK_BYTE, PAD_BYTE


FIELD_PAD = 0
FIELD_OTHER = 1
FIELD_IP_CONTROL = 2
FIELD_ENDPOINT = 3
FIELD_CHECKSUM = 4
FIELD_TRANSPORT_CONTROL = 5
FIELD_SEQUENCE = 6
FIELD_PAYLOAD = 7
FIELD_SESSION = 8
NUM_FIELD_TYPES = 9


@dataclass(frozen=True)
class FlowOffset:
    flow_id: str
    start: int
    stop: int
    packet_count: int
    label_id: int
    label: str
    pcap_path: str


def protocol_field_ids(raw: bytes, max_bytes: int) -> np.ndarray:
    """Assign protocol-semantic field types to an L3 packet prefix."""
    width = int(max_bytes)
    field_ids = np.full(width, FIELD_PAD, dtype=np.int64)
    length = min(len(raw), width)
    if length == 0:
        return field_ids
    field_ids[:length] = FIELD_OTHER

    def mark(start: int, stop: int, field_type: int) -> None:
        if start < length:
            field_ids[max(0, start) : min(stop, length)] = field_type

    version = raw[0] >> 4
    protocol = -1
    l4_offset = length
    if version == 4:
        ihl = max(20, (raw[0] & 0x0F) * 4)
        l4_offset = min(ihl, length)
        protocol = raw[9] if length > 9 else -1
        mark(0, 10, FIELD_IP_CONTROL)
        mark(4, 6, FIELD_SESSION)
        mark(8, 9, FIELD_SESSION)
        mark(10, 12, FIELD_CHECKSUM)
        mark(12, 20, FIELD_ENDPOINT)
        mark(20, l4_offset, FIELD_IP_CONTROL)
    elif version == 6:
        l4_offset = min(40, length)
        protocol = raw[6] if length > 6 else -1
        mark(0, 8, FIELD_IP_CONTROL)
        mark(1, 4, FIELD_SESSION)
        mark(7, 8, FIELD_SESSION)
        mark(8, 40, FIELD_ENDPOINT)
    else:
        return field_ids

    if protocol == 6 and l4_offset < length:
        header_len = 20
        if length > l4_offset + 12:
            header_len = max(20, (raw[l4_offset + 12] >> 4) * 4)
        payload_offset = min(length, l4_offset + header_len)
        mark(l4_offset, l4_offset + 4, FIELD_ENDPOINT)
        mark(l4_offset + 4, l4_offset + 12, FIELD_SEQUENCE)
        mark(l4_offset + 12, l4_offset + 18, FIELD_TRANSPORT_CONTROL)
        mark(l4_offset + 16, l4_offset + 18, FIELD_CHECKSUM)
        mark(l4_offset + 18, payload_offset, FIELD_TRANSPORT_CONTROL)
        mark(payload_offset, length, FIELD_PAYLOAD)
    elif protocol == 17 and l4_offset < length:
        payload_offset = min(length, l4_offset + 8)
        mark(l4_offset, l4_offset + 4, FIELD_ENDPOINT)
        mark(l4_offset + 4, l4_offset + 6, FIELD_TRANSPORT_CONTROL)
        mark(l4_offset + 6, l4_offset + 8, FIELD_CHECKSUM)
        mark(payload_offset, length, FIELD_PAYLOAD)
    else:
        mark(l4_offset, length, FIELD_PAYLOAD)
    return field_ids


def length_bin(value: int, num_bins: int = 8) -> int:
    boundaries = (64, 128, 256, 384, 512, 768, 1024)
    return min(sum(int(value) >= boundary for boundary in boundaries), num_bins - 1)


def iat_bin(value: float, num_bins: int = 8) -> int:
    boundaries = (1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1.0, 5.0)
    return min(sum(float(value) >= boundary for boundary in boundaries), num_bins - 1)


def normalized_packet_meta(meta: dict[str, Any]) -> np.ndarray:
    packet_len = max(0.0, float(meta.get("packet_len", 0) or 0))
    payload_len = max(0.0, float(meta.get("payload_len", 0) or 0))
    iat = max(0.0, float(meta.get("iat", 0.0) or 0.0))
    tcp_window = max(0.0, float(meta.get("tcp_window", 0) or 0))
    return np.asarray(
        [
            math.log1p(packet_len) / 8.0,
            math.log1p(payload_len) / 8.0,
            math.log1p(iat * 1000.0) / 8.0,
            math.log1p(tcp_window) / 12.0,
        ],
        dtype=np.float32,
    )


def direction_id(meta: dict[str, Any]) -> int:
    direction = str(meta.get("direction", "")).upper()
    if direction == "C2S":
        return 1
    if direction == "S2C":
        return 2
    return 0


def scan_flow_offsets(index_path: str | Path) -> list[FlowOffset]:
    """Scan a flow-grouped JSONL once and retain only byte ranges in memory."""
    offsets: list[FlowOffset] = []
    current: dict[str, Any] | None = None
    with open(index_path, "rb") as handle:
        while True:
            start = handle.tell()
            line = handle.readline()
            if not line:
                break
            if not line.strip():
                continue
            row = json.loads(line)
            flow_id = str(row["flow_id"])
            if current is None:
                current = {
                    "flow_id": flow_id,
                    "start": start,
                    "stop": handle.tell(),
                    "packet_count": 1,
                    "label_id": int(row.get("label_id", -1)),
                    "label": str(row.get("label", "")),
                    "pcap_path": str(row.get("pcap_path", "")),
                }
            elif flow_id == current["flow_id"]:
                current["stop"] = handle.tell()
                current["packet_count"] += 1
            else:
                offsets.append(FlowOffset(**current))
                current = {
                    "flow_id": flow_id,
                    "start": start,
                    "stop": handle.tell(),
                    "packet_count": 1,
                    "label_id": int(row.get("label_id", -1)),
                    "label": str(row.get("label", "")),
                    "pcap_path": str(row.get("pcap_path", "")),
                }
        if current is not None:
            offsets.append(FlowOffset(**current))
    return offsets


class PacketIndexFlowDataset(Dataset):
    """Random-access flow dataset backed by byte offsets in packet_index.jsonl."""

    def __init__(
        self,
        index_path: str | Path,
        max_packets: int = 64,
        max_bytes: int = 128,
        max_flows: int = 0,
    ) -> None:
        self.index_path = str(Path(index_path))
        self.max_packets = int(max_packets)
        self.max_bytes = int(max_bytes)
        self.offsets = scan_flow_offsets(index_path)
        if max_flows > 0:
            self.offsets = self.offsets[: int(max_flows)]
        self._fd: int | None = None

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_fd"] = None
        return state

    def __del__(self) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    def __len__(self) -> int:
        return len(self.offsets)

    def _rows(self, item: FlowOffset) -> list[dict[str, Any]]:
        if self._fd is None:
            self._fd = os.open(self.index_path, os.O_RDONLY)
        payload = os.pread(self._fd, item.stop - item.start, item.start)
        rows = [json.loads(line) for line in payload.splitlines() if line.strip()]
        rows.sort(key=lambda row: int(row.get("packet_id", 0)))
        return rows[: self.max_packets]

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.offsets[index]
        rows = self._rows(item)
        byte_tokens = np.full(
            (self.max_packets, self.max_bytes), PAD_BYTE, dtype=np.int64
        )
        field_ids = np.zeros((self.max_packets, self.max_bytes), dtype=np.int64)
        byte_mask = np.zeros((self.max_packets, self.max_bytes), dtype=np.bool_)
        packet_mask = np.zeros(self.max_packets, dtype=np.bool_)
        directions = np.zeros(self.max_packets, dtype=np.int64)
        packet_meta = np.zeros((self.max_packets, 4), dtype=np.float32)
        next_length = np.full(self.max_packets, -100, dtype=np.int64)
        next_iat = np.full(self.max_packets, -100, dtype=np.int64)
        packet_ids = np.full(self.max_packets, -1, dtype=np.int64)

        for packet_index, row in enumerate(rows):
            meta = row.get("meta", {})
            raw = bytes.fromhex(str(meta.get("l3_hex_prefix", "")).replace(" ", ""))
            raw = raw[: self.max_bytes]
            if raw:
                byte_tokens[packet_index, : len(raw)] = np.frombuffer(
                    raw, dtype=np.uint8
                ).astype(np.int64)
                field_ids[packet_index] = protocol_field_ids(raw, self.max_bytes)
                byte_mask[packet_index, : len(raw)] = True
            packet_mask[packet_index] = True
            directions[packet_index] = direction_id(meta)
            packet_meta[packet_index] = normalized_packet_meta(meta)
            packet_ids[packet_index] = int(row.get("packet_id", packet_index))
            if packet_index + 1 < len(rows):
                next_meta = rows[packet_index + 1].get("meta", {})
                next_length[packet_index] = length_bin(
                    int(next_meta.get("packet_len", 0) or 0)
                )
                next_iat[packet_index] = iat_bin(float(next_meta.get("iat", 0) or 0.0))

        return {
            "flow_id": item.flow_id,
            "label_id": item.label_id,
            "label": item.label,
            "pcap_path": item.pcap_path,
            "packet_ids": torch.from_numpy(packet_ids),
            "byte_tokens": torch.from_numpy(byte_tokens),
            "field_ids": torch.from_numpy(field_ids),
            "byte_mask": torch.from_numpy(byte_mask),
            "packet_mask": torch.from_numpy(packet_mask),
            "directions": torch.from_numpy(directions),
            "packet_meta": torch.from_numpy(packet_meta),
            "next_length": torch.from_numpy(next_length),
            "next_iat": torch.from_numpy(next_iat),
        }


def mask_protocol_fields(
    tokens: torch.Tensor,
    field_ids: torch.Tensor,
    byte_mask: torch.Tensor,
    mask_probability: float,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mask complete protocol-field categories and return the prediction mask."""
    masked = tokens.clone()
    prediction_mask = torch.zeros_like(byte_mask, dtype=torch.bool)
    # Encrypted payload bytes are intentionally excluded: reconstructing them
    # rewards ciphertext memorization rather than protocol understanding.
    eligible_types = (FIELD_OTHER, FIELD_IP_CONTROL, FIELD_TRANSPORT_CONTROL)
    batch_shape = field_ids.shape[:-1]
    for field_type in eligible_types:
        choose = torch.rand(batch_shape, generator=generator, device=field_ids.device)
        choose = choose < float(mask_probability)
        selected = (field_ids == field_type) & choose.unsqueeze(-1) & byte_mask.bool()
        prediction_mask |= selected
    valid_packets = byte_mask.any(dim=-1)
    missing = valid_packets & ~prediction_mask.any(dim=-1)
    if missing.any():
        candidates = byte_mask & torch.stack(
            [field_ids == field_type for field_type in eligible_types], dim=0
        ).any(dim=0)
        for packet_coord in missing.nonzero(as_tuple=False):
            coord = tuple(int(value) for value in packet_coord.tolist())
            positions = candidates[coord].nonzero(as_tuple=False).flatten()
            if len(positions):
                selected_position = positions[
                    torch.randint(len(positions), (1,), generator=generator).item()
                ]
                prediction_mask[coord + (int(selected_position),)] = True
    masked[prediction_mask] = MASK_BYTE
    return masked, prediction_mask


def apply_payload_dropout(
    tokens: torch.Tensor,
    field_ids: torch.Tensor,
    byte_mask: torch.Tensor,
    mask_probability: float,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stochastically erase encrypted payload without reconstructing it."""
    masked = tokens.clone()
    batch_shape = field_ids.shape[:-1]
    choose = torch.rand(batch_shape, generator=generator, device=field_ids.device)
    choose = choose < float(mask_probability)
    payload_mask = (
        (field_ids == FIELD_PAYLOAD) & choose.unsqueeze(-1) & byte_mask.bool()
    )
    masked[payload_mask] = MASK_BYTE
    return masked, payload_mask


def apply_session_invariant_mask(
    tokens: torch.Tensor,
    field_ids: torch.Tensor,
    byte_mask: torch.Tensor,
    mask_probability: float = 1.0,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Hide environment/session identifiers without reconstructing their values."""
    masked = tokens.clone()
    invariant_mask = torch.zeros_like(byte_mask, dtype=torch.bool)
    batch_shape = field_ids.shape[:-1]
    for field_type in (
        FIELD_ENDPOINT,
        FIELD_CHECKSUM,
        FIELD_SEQUENCE,
        FIELD_SESSION,
    ):
        choose = torch.rand(batch_shape, generator=generator, device=field_ids.device)
        choose = choose < float(mask_probability)
        invariant_mask |= (field_ids == field_type) & choose.unsqueeze(-1) & byte_mask.bool()
    masked[invariant_mask] = MASK_BYTE
    return masked, invariant_mask


def iter_packet_index_flows(index_path: str | Path) -> Iterator[tuple[str, list[dict]]]:
    current_flow = None
    rows: list[dict] = []
    with open(index_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            flow_id = str(row["flow_id"])
            if current_flow is not None and flow_id != current_flow:
                yield current_flow, rows
                rows = []
            current_flow = flow_id
            rows.append(row)
    if current_flow is not None:
        yield current_flow, rows
