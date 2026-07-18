from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PacketByteTransformer(nn.Module):
    """Strict single-packet encoder with local byte and parsed-meta branches."""

    def __init__(
        self,
        num_classes: int,
        max_bytes: int = 256,
        meta_dim: int = 28,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.15,
        pool_stride: int = 4,
        projection_dim: int = 128,
    ) -> None:
        super().__init__()
        self.max_bytes = int(max_bytes)
        self.pool_stride = int(pool_stride)
        self.byte_embedding = nn.Embedding(258, hidden_dim, padding_idx=256)
        self.local_mixer = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2, groups=hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
            nn.Dropout(dropout),
        )
        pooled_tokens = (self.max_bytes + self.pool_stride - 1) // self.pool_stride
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.position_embedding = nn.Parameter(torch.zeros(1, pooled_tokens + 1, hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.byte_norm = nn.LayerNorm(hidden_dim)
        self.meta_encoder = nn.Sequential(
            nn.Linear(meta_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.fusion_gate = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid())
        self.fusion_norm = nn.LayerNorm(hidden_dim)
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, projection_dim),
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.position_embedding, std=0.02)

    def forward(
        self,
        byte_tokens: torch.Tensor,
        byte_lengths: torch.Tensor,
        meta_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.byte_embedding(byte_tokens)
        local = self.local_mixer(x.transpose(1, 2)).transpose(1, 2)
        x = x + local
        x = F.avg_pool1d(
            x.transpose(1, 2), kernel_size=self.pool_stride, stride=self.pool_stride, ceil_mode=True
        ).transpose(1, 2)
        pooled_lengths = torch.div(
            byte_lengths + self.pool_stride - 1, self.pool_stride, rounding_mode="floor"
        )
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        padding_mask = positions >= pooled_lengths.unsqueeze(1)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)
        cls_mask = torch.zeros((x.size(0), 1), dtype=torch.bool, device=x.device)
        x = x + self.position_embedding[:, : x.size(1)]
        x = self.transformer(x, src_key_padding_mask=torch.cat([cls_mask, padding_mask], dim=1))
        byte_repr = self.byte_norm(x[:, 0])
        meta_repr = self.meta_encoder(meta_features)
        gate = self.fusion_gate(torch.cat([byte_repr, meta_repr], dim=-1))
        fused = self.fusion_norm(gate * byte_repr + (1.0 - gate) * meta_repr)
        projected = F.normalize(self.projector(fused).float(), dim=-1)
        return self.classifier(fused), projected, gate
