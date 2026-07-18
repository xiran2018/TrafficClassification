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
        use_payload_channel: bool = False,
        max_payload_bytes: int = 128,
        use_identifiability_head: bool = False,
    ) -> None:
        super().__init__()
        self.max_bytes = int(max_bytes)
        self.pool_stride = int(pool_stride)
        self.use_payload_channel = bool(use_payload_channel)
        self.max_payload_bytes = int(max_payload_bytes)
        self.use_identifiability_head = bool(use_identifiability_head)
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
        if self.use_payload_channel:
            self.payload_embedding = nn.Embedding(258, hidden_dim, padding_idx=256)
            self.payload_local_mixer = nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2, groups=hidden_dim),
                nn.GELU(),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
                nn.Dropout(dropout),
            )
            payload_tokens = (self.max_payload_bytes + self.pool_stride - 1) // self.pool_stride
            self.payload_cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
            self.payload_position_embedding = nn.Parameter(
                torch.zeros(1, payload_tokens + 1, hidden_dim)
            )
            payload_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.payload_transformer = nn.TransformerEncoder(payload_layer, num_layers=num_layers)
            self.payload_norm = nn.LayerNorm(hidden_dim)
            self.content_gate = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid())
            self.content_norm = nn.LayerNorm(hidden_dim)
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
        if self.use_identifiability_head:
            self.identifiability_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
            )
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.position_embedding, std=0.02)
        if self.use_payload_channel:
            nn.init.normal_(self.payload_cls_token, std=0.02)
            nn.init.normal_(self.payload_position_embedding, std=0.02)

    def _encode_byte_stream(
        self,
        tokens: torch.Tensor,
        lengths: torch.Tensor,
        embedding: nn.Embedding,
        local_mixer: nn.Module,
        cls_token: torch.Tensor,
        position_embedding: torch.Tensor,
        transformer: nn.TransformerEncoder,
        norm: nn.LayerNorm,
    ) -> torch.Tensor:
        x = embedding(tokens)
        local = local_mixer(x.transpose(1, 2)).transpose(1, 2)
        x = x + local
        x = F.avg_pool1d(
            x.transpose(1, 2), kernel_size=self.pool_stride, stride=self.pool_stride, ceil_mode=True
        ).transpose(1, 2)
        pooled_lengths = torch.div(
            lengths + self.pool_stride - 1, self.pool_stride, rounding_mode="floor"
        )
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        padding_mask = positions >= pooled_lengths.unsqueeze(1)
        cls = cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)
        cls_mask = torch.zeros((x.size(0), 1), dtype=torch.bool, device=x.device)
        x = x + position_embedding[:, : x.size(1)]
        x = transformer(x, src_key_padding_mask=torch.cat([cls_mask, padding_mask], dim=1))
        return norm(x[:, 0])

    def _encode_fused(
        self,
        byte_tokens: torch.Tensor,
        byte_lengths: torch.Tensor,
        meta_features: torch.Tensor,
        payload_tokens: torch.Tensor | None = None,
        payload_lengths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        byte_repr = self._encode_byte_stream(
            byte_tokens,
            byte_lengths,
            self.byte_embedding,
            self.local_mixer,
            self.cls_token,
            self.position_embedding,
            self.transformer,
            self.byte_norm,
        )
        if self.use_payload_channel:
            if payload_tokens is None or payload_lengths is None:
                raise ValueError("payload tensors are required when use_payload_channel=True")
            payload_repr = self._encode_byte_stream(
                payload_tokens,
                payload_lengths,
                self.payload_embedding,
                self.payload_local_mixer,
                self.payload_cls_token,
                self.payload_position_embedding,
                self.payload_transformer,
                self.payload_norm,
            )
            content_gate = self.content_gate(torch.cat([byte_repr, payload_repr], dim=-1))
            byte_repr = self.content_norm(
                content_gate * byte_repr + (1.0 - content_gate) * payload_repr
            )
        meta_repr = self.meta_encoder(meta_features)
        gate = self.fusion_gate(torch.cat([byte_repr, meta_repr], dim=-1))
        fused = self.fusion_norm(gate * byte_repr + (1.0 - gate) * meta_repr)
        return fused, gate

    def forward(
        self,
        byte_tokens: torch.Tensor,
        byte_lengths: torch.Tensor,
        meta_features: torch.Tensor,
        payload_tokens: torch.Tensor | None = None,
        payload_lengths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        fused, gate = self._encode_fused(
            byte_tokens,
            byte_lengths,
            meta_features,
            payload_tokens,
            payload_lengths,
        )
        projected = F.normalize(self.projector(fused).float(), dim=-1)
        return self.classifier(fused), projected, gate

    def forward_with_identifiability(
        self,
        byte_tokens: torch.Tensor,
        byte_lengths: torch.Tensor,
        meta_features: torch.Tensor,
        payload_tokens: torch.Tensor | None = None,
        payload_lengths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.use_identifiability_head:
            raise RuntimeError("identifiability head is disabled")
        fused, gate = self._encode_fused(
            byte_tokens,
            byte_lengths,
            meta_features,
            payload_tokens,
            payload_lengths,
        )
        projected = F.normalize(self.projector(fused).float(), dim=-1)
        reliability_logit = self.identifiability_head(fused).squeeze(-1)
        return self.classifier(fused), projected, gate, reliability_logit
