from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.unified_packet_encoder import (
    SharedInterventionViewFusion,
    SharedPacketChannelFusion,
    SharedPacketClassifierHead,
    SharedPacketRepresentationEncoder,
)
from models.native_flow_encoder import ProtocolAwarePacketContentEncoder
from native_flow_data import apply_session_invariant_mask


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
        use_protocol_fields: bool = False,
        semantic_dim: int = 0,
        use_intervention_views: bool = False,
        intervention_max_residual_weight: float = 0.25,
        intervention_view_base_mode: str = "symmetric_mean",
        channel_fusion_base_mode: str = "legacy",
        channel_fusion_max_weight: float = 0.25,
        exact_shared_representation: bool = False,
        mask_protocol_session_fields: bool = False,
        train_ablate_input_channel: str = "none",
        train_ablate_intervention_view: str = "none",
        train_fixed_channel_fusion: bool = False,
    ) -> None:
        super().__init__()
        self.max_bytes = int(max_bytes)
        self.pool_stride = int(pool_stride)
        self.use_payload_channel = bool(use_payload_channel)
        self.max_payload_bytes = int(max_payload_bytes)
        self.use_identifiability_head = bool(use_identifiability_head)
        self.use_protocol_fields = bool(use_protocol_fields)
        self.semantic_dim = int(semantic_dim)
        self.use_intervention_views = bool(use_intervention_views)
        self.channel_fusion_base_mode = str(channel_fusion_base_mode)
        self.channel_fusion_max_weight = float(
            max(0.0, min(1.0, channel_fusion_max_weight))
        )
        self.exact_shared_representation = bool(exact_shared_representation)
        self.mask_protocol_session_fields = bool(mask_protocol_session_fields)
        self.train_ablate_input_channel = str(train_ablate_input_channel)
        self.train_ablate_intervention_view = str(train_ablate_intervention_view)
        self.train_fixed_channel_fusion = bool(train_fixed_channel_fusion)
        if self.train_ablate_input_channel not in {
            "none", "semantic", "content", "structural"
        }:
            raise ValueError("invalid train_ablate_input_channel")
        if self.train_ablate_intervention_view not in {
            "none", "factual_only", "intervened_only"
        }:
            raise ValueError("invalid train_ablate_intervention_view")
        if self.train_ablate_input_channel != "none" and not self.exact_shared_representation:
            raise ValueError("training-time channel ablation requires exact shared representation")
        if self.train_ablate_intervention_view != "none" and (
            not self.exact_shared_representation or not self.use_intervention_views
        ):
            raise ValueError(
                "training-time intervention-view ablation requires exact shared intervention views"
            )
        if self.train_fixed_channel_fusion and not self.exact_shared_representation:
            raise ValueError("fixed channel fusion requires exact shared representation")
        if self.exact_shared_representation and (
            not self.use_protocol_fields
            or self.semantic_dim <= 0
            or self.use_payload_channel
        ):
            raise ValueError(
                "exact shared representation requires protocol fields, semantic input, "
                "and no separate payload encoder"
            )
        if self.use_intervention_views and self.semantic_dim <= 0:
            raise ValueError("intervention views require semantic_dim > 0")
        if self.semantic_dim > 0 and not self.exact_shared_representation:
            self.semantic_proj = nn.Sequential(
                nn.Linear(self.semantic_dim, hidden_dim, bias=False),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            if self.use_intervention_views:
                self.intervention_view_fusion = SharedInterventionViewFusion(
                    hidden_dim,
                    dropout=dropout,
                    max_residual_weight=intervention_max_residual_weight,
                    base_mode=intervention_view_base_mode,
                )
        if self.use_protocol_fields:
            self.protocol_content_encoder = ProtocolAwarePacketContentEncoder(
                max_bytes=max_bytes,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                dropout=dropout,
            )
        else:
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
        if self.exact_shared_representation:
            self.shared_packet_encoder = SharedPacketRepresentationEncoder(
                semantic_dim=self.semantic_dim,
                content_dim=hidden_dim,
                structural_dim=meta_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
                use_intervention_views=self.use_intervention_views,
                intervention_max_residual_weight=intervention_max_residual_weight,
                intervention_view_base_mode=intervention_view_base_mode,
                channel_fusion_base_mode=self.channel_fusion_base_mode,
                channel_fusion_max_weight=self.channel_fusion_max_weight,
            )
            self.shared_packet_fusion = self.shared_packet_encoder.channel_fusion
        else:
            self.meta_encoder = nn.Sequential(
                nn.Linear(meta_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.shared_packet_fusion = SharedPacketChannelFusion(
                hidden_dim,
                channel_names=(
                    ("semantic", "content", "structural")
                    if self.semantic_dim > 0
                    else ("content", "structural")
                ),
                dropout=dropout,
                interaction_max_weight=self.channel_fusion_max_weight,
                base_mode=self.channel_fusion_base_mode,
            )
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, projection_dim),
        )
        self.classifier = SharedPacketClassifierHead(hidden_dim, num_classes)
        if self.use_identifiability_head:
            self.identifiability_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
            )
        if not self.use_protocol_fields:
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
        field_ids: torch.Tensor | None = None,
        semantic_features: torch.Tensor | None = None,
        intervened_semantic_features: torch.Tensor | None = None,
        ablate_channel: str = "none",
        ablate_intervention_view: str = "none",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if self.use_protocol_fields:
            if field_ids is None:
                raise ValueError("field_ids are required when use_protocol_fields=True")
            positions = torch.arange(byte_tokens.size(1), device=byte_tokens.device)
            byte_mask = positions.unsqueeze(0) < byte_lengths.unsqueeze(1)
            if self.mask_protocol_session_fields:
                byte_tokens, _ = apply_session_invariant_mask(
                    byte_tokens, field_ids, byte_mask, 1.0
                )
            byte_repr, _ = self.protocol_content_encoder(byte_tokens, field_ids, byte_mask)
        else:
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
        if self.exact_shared_representation:
            if semantic_features is None:
                raise ValueError("exact shared packet representation requires semantic features")
            effective_channel_ablation = (
                self.train_ablate_input_channel
                if ablate_channel == "none"
                else ablate_channel
            )
            effective_view_ablation = (
                self.train_ablate_intervention_view
                if ablate_intervention_view == "none"
                else ablate_intervention_view
            )
            fused, gate, intervention_gate, _ = self.shared_packet_encoder(
                semantic_features,
                byte_repr,
                meta_features,
                intervened_semantic_features,
                ablate_channel=effective_channel_ablation,
                ablate_intervention_view=effective_view_ablation,
                fixed_channel_fusion=self.train_fixed_channel_fusion,
            )
            return fused, gate, intervention_gate
        meta_repr = self.meta_encoder(meta_features)
        channels = {"content": byte_repr, "structural": meta_repr}
        if self.semantic_dim > 0:
            if semantic_features is None:
                raise ValueError("semantic_features are required when semantic_dim > 0")
            if semantic_features.shape[-1] != self.semantic_dim:
                raise ValueError(
                    f"semantic feature dimension mismatch: got={semantic_features.shape[-1]} "
                    f"expected={self.semantic_dim}"
                )
            semantic = self.semantic_proj(semantic_features)
            intervention_gate = None
            if self.use_intervention_views:
                if intervened_semantic_features is None:
                    raise ValueError(
                        "intervened_semantic_features are required when "
                        "use_intervention_views=True"
                    )
                if intervened_semantic_features.shape != semantic_features.shape:
                    raise ValueError("intervened semantic feature shape mismatch")
                intervened = self.semantic_proj(intervened_semantic_features)
                semantic, intervention_gate = self.intervention_view_fusion(
                    semantic, intervened
                )
            channels["semantic"] = semantic
        else:
            intervention_gate = None
        fused, gate = self.shared_packet_fusion(channels)
        return fused, gate, intervention_gate

    def forward(
        self,
        byte_tokens: torch.Tensor,
        byte_lengths: torch.Tensor,
        meta_features: torch.Tensor,
        payload_tokens: torch.Tensor | None = None,
        payload_lengths: torch.Tensor | None = None,
        field_ids: torch.Tensor | None = None,
        semantic_features: torch.Tensor | None = None,
        intervened_semantic_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, projected, gate, _ = self.forward_with_gate_diagnostics(
            byte_tokens,
            byte_lengths,
            meta_features,
            payload_tokens,
            payload_lengths,
            field_ids,
            semantic_features,
            intervened_semantic_features,
        )
        return logits, projected, gate

    def forward_with_gate_diagnostics(
        self,
        byte_tokens: torch.Tensor,
        byte_lengths: torch.Tensor,
        meta_features: torch.Tensor,
        payload_tokens: torch.Tensor | None = None,
        payload_lengths: torch.Tensor | None = None,
        field_ids: torch.Tensor | None = None,
        semantic_features: torch.Tensor | None = None,
        intervened_semantic_features: torch.Tensor | None = None,
        ablate_channel: str = "none",
        ablate_intervention_view: str = "none",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        fused, gate, intervention_gate = self._encode_fused(
            byte_tokens,
            byte_lengths,
            meta_features,
            payload_tokens,
            payload_lengths,
            field_ids,
            semantic_features,
            intervened_semantic_features,
            ablate_channel,
            ablate_intervention_view,
        )
        projected = F.normalize(self.projector(fused).float(), dim=-1)
        return self.classifier(fused), projected, gate, intervention_gate

    def forward_with_identifiability(
        self,
        byte_tokens: torch.Tensor,
        byte_lengths: torch.Tensor,
        meta_features: torch.Tensor,
        payload_tokens: torch.Tensor | None = None,
        payload_lengths: torch.Tensor | None = None,
        field_ids: torch.Tensor | None = None,
        semantic_features: torch.Tensor | None = None,
        intervened_semantic_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.use_identifiability_head:
            raise RuntimeError("identifiability head is disabled")
        fused, gate, _ = self._encode_fused(
            byte_tokens,
            byte_lengths,
            meta_features,
            payload_tokens,
            payload_lengths,
            field_ids,
            semantic_features,
            intervened_semantic_features,
        )
        projected = F.normalize(self.projector(fused).float(), dim=-1)
        reliability_logit = self.identifiability_head(fused).squeeze(-1)
        return self.classifier(fused), projected, gate, reliability_logit
