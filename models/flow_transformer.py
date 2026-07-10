from __future__ import annotations

import torch
import torch.nn as nn


class AttentionPooling(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(dim, dim), nn.Tanh(), nn.Linear(dim, 1))

    def forward(self, h: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        logits = self.score(h).squeeze(-1)
        if mask is not None:
            logits = logits.masked_fill(~mask.bool(), -1e9)
        w = torch.softmax(logits, dim=-1)
        return torch.sum(h * w.unsqueeze(-1), dim=1)


class FlowTransformerClassifier(nn.Module):
    """Non-graph Packet Interaction Transformer.

    Input: padded tensor [B, N, D] where D = packet embedding dim + packet meta feature dim.
    """

    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 256, num_layers: int = 2, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.pool = AttentionPooling(hidden_dim)
        self.cls = nn.Linear(hidden_dim, num_classes)
        self.coherence = nn.Linear(hidden_dim, 2)
        self.next_direction = nn.Linear(hidden_dim, 2)
        self.next_length = nn.Linear(hidden_dim, 4)
        self.next_iat = nn.Linear(hidden_dim, 4)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        h = self.proj(x)
        key_padding_mask = None if mask is None else ~mask.bool()
        h = self.encoder(h, src_key_padding_mask=key_padding_mask)
        g = self.pool(h, mask)
        return {
            "logits": self.cls(g),
            "coherence_logits": self.coherence(g),
            "embedding": g,
            "next_direction_logits": self.next_direction(g),
            "next_length_logits": self.next_length(g),
            "next_iat_logits": self.next_iat(g),
        }
