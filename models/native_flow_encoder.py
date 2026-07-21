from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


PAD_BYTE = 256
MASK_BYTE = 257
NATIVE_PACKET_PRETRAINING_PROTOCOL = "native_flow_multitask_v1"


class ProtocolAwarePacketContentEncoder(nn.Module):
    """Strict current-packet byte/field encoder shared by both task levels."""

    def __init__(
        self,
        max_bytes: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        num_field_types: int = 9,
    ) -> None:
        super().__init__()
        self.max_bytes = int(max_bytes)
        self.hidden_dim = int(hidden_dim)
        self.byte_embedding = nn.Embedding(258, hidden_dim, padding_idx=PAD_BYTE)
        self.field_embedding = nn.Embedding(num_field_types, hidden_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_bytes + 1, hidden_dim)
        self.packet_cls = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.input_norm = nn.LayerNorm(hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(hidden_dim)
        nn.init.normal_(self.packet_cls, std=0.02)

    def forward(
        self,
        byte_tokens: torch.Tensor,
        field_ids: torch.Tensor,
        byte_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if byte_tokens.shape != field_ids.shape or byte_tokens.shape != byte_mask.shape:
            raise ValueError("byte_tokens, field_ids, and byte_mask must share a shape")
        if byte_tokens.dim() < 2:
            raise ValueError("packet byte tensors require at least batch and byte dimensions")
        leading_shape = byte_tokens.shape[:-1]
        byte_count = byte_tokens.shape[-1]
        if byte_count > self.max_bytes:
            raise ValueError("byte sequence exceeds configured max_bytes")
        flat_tokens = byte_tokens.reshape(-1, byte_count)
        flat_fields = field_ids.reshape(-1, byte_count)
        flat_mask = byte_mask.reshape(-1, byte_count).bool()
        positions = torch.arange(1, byte_count + 1, device=byte_tokens.device)
        x = (
            self.byte_embedding(flat_tokens)
            + self.field_embedding(flat_fields)
            + self.position_embedding(positions)[None, :, :]
        )
        cls = self.packet_cls.expand(x.size(0), -1, -1)
        cls = cls + self.position_embedding.weight[0].view(1, 1, -1)
        x = self.input_norm(torch.cat([cls, x], dim=1))
        valid = torch.cat(
            [torch.ones((x.size(0), 1), dtype=torch.bool, device=x.device), flat_mask],
            dim=1,
        )
        x = self.encoder(x, src_key_padding_mask=~valid)
        packet = self.output_norm(x[:, 0]).reshape(*leading_shape, self.hidden_dim)
        tokens = x[:, 1:].reshape(*leading_shape, byte_count, self.hidden_dim)
        return packet, tokens


class NativeFlowEncoder(nn.Module):
    """Protocol-aware byte/field encoder followed by inter-packet attention.

    The encoder is deliberately independent of downstream class labels. Packet
    content, protocol-field identity, direction, relative packet position, and
    continuous length/timing features are represented explicitly.
    """

    def __init__(
        self,
        max_bytes: int = 128,
        max_packets: int = 64,
        hidden_dim: int = 128,
        byte_layers: int = 2,
        flow_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        num_field_types: int = 9,
        projection_dim: int = 128,
        num_length_bins: int = 8,
        num_iat_bins: int = 8,
    ) -> None:
        super().__init__()
        self.max_bytes = int(max_bytes)
        self.max_packets = int(max_packets)
        self.hidden_dim = int(hidden_dim)
        self.num_field_types = int(num_field_types)
        self.num_length_bins = int(num_length_bins)
        self.num_iat_bins = int(num_iat_bins)

        self.packet_content_encoder = ProtocolAwarePacketContentEncoder(
            max_bytes=max_bytes,
            hidden_dim=hidden_dim,
            num_layers=byte_layers,
            num_heads=num_heads,
            dropout=dropout,
            num_field_types=num_field_types,
        )

        self.packet_position_embedding = nn.Embedding(max_packets, hidden_dim)
        self.direction_embedding = nn.Embedding(3, hidden_dim, padding_idx=0)
        self.packet_meta_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        flow_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.flow_encoder = nn.TransformerEncoder(flow_layer, num_layers=flow_layers)
        self.flow_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1)
        )
        self.flow_norm = nn.LayerNorm(hidden_dim)
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, projection_dim),
        )

        self.masked_byte_head = nn.Linear(hidden_dim, 256)
        self.relative_order_head = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 2)
        )
        self.same_flow_head = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 2)
        )
        self.next_length_head = nn.Linear(hidden_dim, num_length_bins)
        self.next_iat_head = nn.Linear(hidden_dim, num_iat_bins)
        self.direction_head = nn.Linear(hidden_dim, 2)

    @staticmethod
    def _pair_features(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        return torch.cat([first, second, first - second, first * second], dim=-1)

    def encode_packets(
        self,
        byte_tokens: torch.Tensor,
        field_ids: torch.Tensor,
        byte_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if byte_tokens.dim() != 3:
            raise ValueError("byte_tokens must have shape [batch, packets, bytes]")
        return self.packet_content_encoder(byte_tokens, field_ids, byte_mask)

    def contextualize_flow(
        self,
        packet_content: torch.Tensor,
        packet_mask: torch.Tensor,
        directions: torch.Tensor,
        packet_meta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        packet_count = packet_content.size(1)
        if packet_count > self.max_packets:
            raise ValueError("packet sequence exceeds configured max_packets")
        positions = torch.arange(packet_count, device=packet_content.device)
        # This representation contains observable packet dynamics but no packet
        # index. It supports temporal pretext tasks without leaking their order
        # target through an absolute position embedding.
        packet_observation = (
            packet_content
            + self.direction_embedding(directions)
            + self.packet_meta_encoder(packet_meta)
        )
        h = packet_observation + self.packet_position_embedding(positions)[None, :, :]
        h = self.flow_encoder(h, src_key_padding_mask=~packet_mask.bool())
        logits = self.flow_attention(h).squeeze(-1).masked_fill(~packet_mask.bool(), -1e4)
        weights = torch.softmax(logits, dim=-1)
        flow_repr = self.flow_norm((h * weights.unsqueeze(-1)).sum(dim=1))
        flow_projection = F.normalize(self.projector(flow_repr).float(), dim=-1)
        return packet_observation, h, flow_repr, flow_projection

    def forward(
        self,
        byte_tokens: torch.Tensor,
        field_ids: torch.Tensor,
        byte_mask: torch.Tensor,
        packet_mask: torch.Tensor,
        directions: torch.Tensor,
        packet_meta: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        packet_content, token_repr = self.encode_packets(byte_tokens, field_ids, byte_mask)
        packet_observation, packet_repr, flow_repr, flow_projection = self.contextualize_flow(
            packet_content, packet_mask, directions, packet_meta
        )
        return {
            "token_repr": token_repr,
            "packet_content": packet_content,
            "packet_observation": packet_observation,
            "packet_repr": packet_repr,
            "flow_repr": flow_repr,
            "flow_projection": flow_projection,
            "direction_logits": self.direction_head(packet_content),
        }

    def masked_byte_logits(
        self, token_repr: torch.Tensor, prediction_mask: torch.Tensor
    ) -> torch.Tensor:
        return self.masked_byte_head(token_repr[prediction_mask.bool()])

    def relative_order_logits(
        self, first: torch.Tensor, second: torch.Tensor
    ) -> torch.Tensor:
        return self.relative_order_head(self._pair_features(first, second))

    def same_flow_logits(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        return self.same_flow_head(self._pair_features(first, second))

    def next_packet_logits(self, packet_repr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.next_length_head(packet_repr), self.next_iat_head(packet_repr)


def nt_xent_loss(first: torch.Tensor, second: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """Symmetric in-batch contrastive loss with one positive view per flow."""
    if first.shape != second.shape:
        raise ValueError("contrastive views must have equal shape")
    if first.size(0) < 2:
        return first.sum() * 0.0
    logits = first @ second.T / max(float(temperature), 1e-6)
    targets = torch.arange(first.size(0), device=first.device)
    return 0.5 * (
        F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets)
    )


def sample_relative_pairs(
    packet_repr: torch.Tensor, packet_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create balanced before/after pairs without exposing absolute indices to the head."""
    first_rows, second_rows, targets = [], [], []
    for batch_index in range(packet_repr.size(0)):
        count = int(packet_mask[batch_index].sum().item())
        if count < 2:
            continue
        left = torch.arange(count - 1, device=packet_repr.device)
        right = left + 1
        first_rows.extend([packet_repr[batch_index, left], packet_repr[batch_index, right]])
        second_rows.extend([packet_repr[batch_index, right], packet_repr[batch_index, left]])
        targets.extend(
            [
                torch.zeros(count - 1, dtype=torch.long, device=packet_repr.device),
                torch.ones(count - 1, dtype=torch.long, device=packet_repr.device),
            ]
        )
    if not first_rows:
        empty = packet_repr.new_zeros((0, packet_repr.size(-1)))
        return empty, empty, torch.zeros(0, dtype=torch.long, device=packet_repr.device)
    return torch.cat(first_rows), torch.cat(second_rows), torch.cat(targets)


def sample_same_flow_pairs(
    packet_repr: torch.Tensor, packet_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build one positive and one cross-flow negative pair per eligible flow."""
    if packet_repr.size(0) < 2:
        empty = packet_repr.new_zeros((0, packet_repr.size(-1)))
        return empty, empty, torch.zeros(0, dtype=torch.long, device=packet_repr.device)
    first_rows, second_rows, targets = [], [], []
    batch_size = packet_repr.size(0)
    for batch_index in range(batch_size):
        count = int(packet_mask[batch_index].sum().item())
        other = (batch_index + 1) % batch_size
        other_count = int(packet_mask[other].sum().item())
        if count < 2 or other_count < 1:
            continue
        first_rows.extend([packet_repr[batch_index, 0:1], packet_repr[batch_index, 0:1]])
        second_rows.extend([packet_repr[batch_index, 1:2], packet_repr[other, 0:1]])
        targets.extend(
            [
                torch.ones(1, dtype=torch.long, device=packet_repr.device),
                torch.zeros(1, dtype=torch.long, device=packet_repr.device),
            ]
        )
    if not first_rows:
        empty = packet_repr.new_zeros((0, packet_repr.size(-1)))
        return empty, empty, torch.zeros(0, dtype=torch.long, device=packet_repr.device)
    return torch.cat(first_rows), torch.cat(second_rows), torch.cat(targets)
