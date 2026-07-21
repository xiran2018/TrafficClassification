from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.unified_packet_encoder import (
    SharedInterventionViewFusion,
    SharedPacketChannelFusion,
    SharedPacketClassifierHead,
    SharedPacketRepresentationEncoder,
)


class AttentionPooling(nn.Module):
    def __init__(
        self,
        dim: int,
        reliability_prior: bool = False,
        reliability_prior_init: float = 0.1,
        reliability_dual: bool = False,
        reliability_evidence_adapter: bool = False,
        reliability_adapter_max_delta: float = 0.25,
        reliability_residual_max_weight: float = 0.0,
        reliability_residual_init: float = 0.5,
    ):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(dim, dim), nn.Tanh(), nn.Linear(dim, 1))
        self.reliability_prior = bool(reliability_prior)
        self.reliability_dual = bool(reliability_dual)
        self.reliability_evidence_adapter = bool(reliability_evidence_adapter)
        self.reliability_adapter_max_delta = float(max(0.0, reliability_adapter_max_delta))
        self.reliability_residual_max_weight = float(
            max(0.0, min(1.0, reliability_residual_max_weight))
        )
        self.last_reliability_gate: torch.Tensor | None = None
        self.last_reliability_adapter_norm: torch.Tensor | None = None
        if sum([self.reliability_prior, self.reliability_dual, self.reliability_evidence_adapter]) > 1:
            raise ValueError("reliability pooling modes are mutually exclusive")
        if self.reliability_prior or self.reliability_dual or self.reliability_evidence_adapter:
            initial_scale = max(float(reliability_prior_init), 1e-6)
            self.reliability_prior_raw_scale = nn.Parameter(
                torch.tensor(math.log(math.expm1(initial_scale)))
            )
        if self.reliability_dual or self.reliability_evidence_adapter:
            self.unidentifiable_prior_raw_scale = nn.Parameter(
                torch.tensor(math.log(math.expm1(initial_scale)))
            )
        if self.reliability_dual:
            self.dual_gate = nn.Sequential(
                nn.LayerNorm(dim * 3),
                nn.Linear(dim * 3, dim),
                nn.GELU(),
                nn.Linear(dim, 3),
            )
            if self.reliability_residual_max_weight > 0:
                initial_fraction = max(1e-4, min(1.0 - 1e-4, reliability_residual_init))
                self.reliability_residual_raw_gate = nn.Parameter(
                    torch.tensor(math.log(initial_fraction / (1.0 - initial_fraction)))
                )
        if self.reliability_evidence_adapter:
            self.evidence_adapter = nn.Sequential(
                nn.LayerNorm(dim * 2 + 2),
                nn.Linear(dim * 2 + 2, dim),
                nn.GELU(),
                nn.Linear(dim, dim),
            )
            nn.init.zeros_(self.evidence_adapter[-1].weight)
            nn.init.zeros_(self.evidence_adapter[-1].bias)

    @staticmethod
    def _masked_softmax(logits: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        if mask is not None:
            logits = logits.masked_fill(~mask.bool(), -1e9)
        return torch.softmax(logits, dim=-1)

    def forward(
        self,
        h: torch.Tensor,
        mask: torch.Tensor | None = None,
        reliability: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.last_reliability_gate = None
        self.last_reliability_adapter_norm = None
        logits = self.score(h).squeeze(-1)
        if (self.reliability_prior or self.reliability_dual or self.reliability_evidence_adapter) and reliability is not None:
            reliability = reliability.to(device=h.device, dtype=h.dtype)
            if reliability.shape != logits.shape:
                raise ValueError("reliability must match the attention-logit shape")
        if (self.reliability_dual or self.reliability_evidence_adapter) and reliability is not None:
            base_weights = self._masked_softmax(logits, mask)
            scale = F.softplus(self.reliability_prior_raw_scale)
            low_scale = F.softplus(self.unidentifiable_prior_raw_scale)
            high_weights = self._masked_softmax(
                logits + scale * reliability.clamp(min=0.05, max=1.0).log(), mask
            )
            low_weights = self._masked_softmax(
                logits + low_scale * (1.0 - reliability).clamp(min=0.05, max=1.0).log(),
                mask,
            )
            views = torch.stack(
                [
                    torch.sum(h * base_weights.unsqueeze(-1), dim=1),
                    torch.sum(h * high_weights.unsqueeze(-1), dim=1),
                    torch.sum(h * low_weights.unsqueeze(-1), dim=1),
                ],
                dim=1,
            )
            if self.reliability_evidence_adapter:
                valid = (
                    torch.ones_like(reliability, dtype=h.dtype)
                    if mask is None
                    else mask.to(dtype=h.dtype)
                )
                count = valid.sum(dim=-1, keepdim=True).clamp(min=1.0)
                rel_mean = (reliability * valid).sum(dim=-1, keepdim=True) / count
                rel_var = ((reliability - rel_mean).square() * valid).sum(dim=-1, keepdim=True) / count
                evidence = torch.cat(
                    [
                        views[:, 1] - views[:, 0],
                        views[:, 2] - views[:, 0],
                        rel_mean,
                        torch.sqrt(rel_var + 1e-6),
                    ],
                    dim=-1,
                )
                correction = torch.tanh(self.evidence_adapter(evidence))
                self.last_reliability_adapter_norm = correction.detach().norm(dim=-1)
                return views[:, 0] + self.reliability_adapter_max_delta * correction
            gate = torch.softmax(self.dual_gate(views.flatten(start_dim=1)), dim=-1)
            mixture = torch.sum(views * gate.unsqueeze(-1), dim=1)
            if self.reliability_residual_max_weight > 0:
                residual_weight = self.reliability_residual_max_weight * torch.sigmoid(
                    self.reliability_residual_raw_gate
                )
                effective_gate = residual_weight * gate
                effective_gate = torch.cat(
                    [effective_gate[:, :1] + (1.0 - residual_weight), effective_gate[:, 1:]],
                    dim=-1,
                )
                self.last_reliability_gate = effective_gate.detach()
                return views[:, 0] + residual_weight * (mixture - views[:, 0])
            self.last_reliability_gate = gate.detach()
            return mixture
        if self.reliability_prior and reliability is not None:
            scale = F.softplus(self.reliability_prior_raw_scale)
            logits = logits + scale * reliability.clamp(min=0.05, max=1.0).log()
        w = self._masked_softmax(logits, mask)
        return torch.sum(h * w.unsqueeze(-1), dim=1)


class FlowTransformerClassifier(nn.Module):
    """Non-graph Packet Interaction Transformer.

    Input: padded tensor [B, N, D] where D = packet embedding dim + packet meta feature dim.
    """

    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 256, num_layers: int = 2, num_heads: int = 4, dropout: float = 0.1, identifiability_feature_index: int = -1, identifiability_pooling: bool = False, identifiability_feature_mode: str = "observed", identifiability_prior_init: float = 0.1, identifiability_dual_pooling: bool = False, identifiability_evidence_adapter: bool = False, identifiability_adapter_max_delta: float = 0.25, identifiability_residual_max_weight: float = 0.0, identifiability_residual_init: float = 0.5, dual_channel_mode: str = "concat", meta_feature_dim: int = 14, native_structural_dim: int = 0, dual_channel_max_weight: float = 0.25, dual_channel_init: float = 0.1, dual_channel_gate_mode: str = "global", channel_fusion_base_mode: str = "legacy", use_intervention_views: bool = False, intervention_max_residual_weight: float = 0.25, intervention_view_base_mode: str = "symmetric_mean", exact_shared_packet_encoder: bool = False, shared_packet_hidden_dim: int = 128, packet_evidence_max_weight: float = 0.0, train_ablate_input_channel: str = "none", train_ablate_intervention_view: str = "none", train_fixed_channel_fusion: bool = False):
        super().__init__()
        if dual_channel_mode not in {"concat", "residual"}:
            raise ValueError("dual_channel_mode must be 'concat' or 'residual'")
        self.dual_channel_mode = str(dual_channel_mode)
        if dual_channel_gate_mode not in {"global", "adaptive"}:
            raise ValueError("dual_channel_gate_mode must be 'global' or 'adaptive'")
        self.dual_channel_gate_mode = str(dual_channel_gate_mode)
        self.channel_fusion_base_mode = str(channel_fusion_base_mode)
        self.meta_feature_dim = int(meta_feature_dim)
        self.native_structural_dim = int(native_structural_dim)
        if not 0 <= self.native_structural_dim <= self.meta_feature_dim:
            raise ValueError("native_structural_dim must be within the structural channel")
        self.structural_feature_dim = self.meta_feature_dim - self.native_structural_dim
        self.embedding_feature_dim = int(input_dim) - self.meta_feature_dim
        self.use_intervention_views = bool(use_intervention_views)
        self.exact_shared_packet_encoder = bool(exact_shared_packet_encoder)
        self.shared_packet_hidden_dim = int(shared_packet_hidden_dim)
        self.packet_evidence_max_weight = float(packet_evidence_max_weight)
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
        if self.train_ablate_input_channel != "none" and not self.exact_shared_packet_encoder:
            raise ValueError("training-time channel ablation requires exact shared packet encoder")
        if self.train_ablate_intervention_view != "none" and (
            not self.exact_shared_packet_encoder or not self.use_intervention_views
        ):
            raise ValueError(
                "training-time intervention-view ablation requires exact shared intervention views"
            )
        if self.train_fixed_channel_fusion and not self.exact_shared_packet_encoder:
            raise ValueError("fixed channel fusion requires exact shared packet encoder")
        if not 0.0 <= self.packet_evidence_max_weight <= 1.0:
            raise ValueError("packet_evidence_max_weight must be in [0, 1]")
        if self.packet_evidence_max_weight > 0 and not self.exact_shared_packet_encoder:
            raise ValueError("packet evidence fusion requires exact_shared_packet_encoder")
        if self.dual_channel_mode == "residual" and self.embedding_feature_dim <= 0:
            raise ValueError("dual-channel residual mode requires embedding features")
        if self.dual_channel_mode == "concat":
            if self.exact_shared_packet_encoder:
                raise ValueError("exact shared packet encoder requires residual channel mode")
            self.proj = nn.Linear(input_dim, hidden_dim)
        else:
            self.dual_channel_max_weight = float(
                max(0.0, min(1.0, dual_channel_max_weight))
            )
            if self.structural_feature_dim <= 0:
                raise ValueError(
                    "dual-channel residual mode requires protocol structural features "
                    "in addition to native content features"
                )
            if self.exact_shared_packet_encoder:
                if self.native_structural_dim <= 0 or not self.use_intervention_views:
                    raise ValueError(
                        "exact shared packet encoder requires native content and intervention views"
                    )
                self.shared_packet_encoder = SharedPacketRepresentationEncoder(
                    semantic_dim=self.embedding_feature_dim,
                    content_dim=self.native_structural_dim,
                    structural_dim=self.structural_feature_dim,
                    hidden_dim=self.shared_packet_hidden_dim,
                    dropout=dropout,
                    use_intervention_views=True,
                    intervention_max_residual_weight=intervention_max_residual_weight,
                    intervention_view_base_mode=intervention_view_base_mode,
                    channel_fusion_base_mode=self.channel_fusion_base_mode,
                    channel_fusion_max_weight=self.dual_channel_max_weight,
                )
                self.shared_packet_fusion = self.shared_packet_encoder.channel_fusion
                self.channel_interaction = self.shared_packet_fusion.interaction
                self.packet_to_flow_proj = nn.Linear(
                    self.shared_packet_hidden_dim, hidden_dim
                )
                if self.packet_evidence_max_weight > 0:
                    self.packet_classifier = SharedPacketClassifierHead(
                        self.shared_packet_hidden_dim, num_classes
                    )
                    self.packet_evidence_gate = nn.Sequential(
                        nn.Linear(hidden_dim + num_classes, hidden_dim),
                        nn.GELU(),
                        nn.Dropout(dropout),
                        nn.Linear(hidden_dim, 1),
                    )
                    nn.init.zeros_(self.packet_evidence_gate[-1].weight)
                    nn.init.constant_(
                        self.packet_evidence_gate[-1].bias, -1.0986122886681098
                    )
                self.semantic_channel_cls = nn.Linear(
                    self.shared_packet_hidden_dim, num_classes
                )
                self.structural_channel_cls = nn.Linear(
                    self.shared_packet_hidden_dim, num_classes
                )
            else:
                initial = max(1e-4, min(1.0 - 1e-4, float(dual_channel_init)))
                self.semantic_proj = nn.Linear(
                    self.embedding_feature_dim, hidden_dim, bias=False
                )
                if self.use_intervention_views:
                    self.intervention_view_fusion = SharedInterventionViewFusion(
                        hidden_dim,
                        dropout=dropout,
                        max_residual_weight=intervention_max_residual_weight,
                        base_mode=intervention_view_base_mode,
                    )
                self.structural_proj = nn.Linear(
                    self.structural_feature_dim, hidden_dim, bias=False
                )
                if self.native_structural_dim > 0:
                    self.native_structural_adapter = nn.Linear(
                        self.native_structural_dim, hidden_dim, bias=False
                    )
                    nn.init.zeros_(self.native_structural_adapter.weight)
                    self.native_structural_raw_gate = nn.Parameter(
                        torch.tensor(math.log(initial / (1.0 - initial)))
                    )
                self.fusion_bias = nn.Parameter(torch.zeros(hidden_dim))
                channel_names = (
                    ("semantic", "content", "structural")
                    if self.native_structural_dim > 0
                    else ("semantic", "structural")
                )
                self.shared_packet_fusion = SharedPacketChannelFusion(
                    hidden_dim,
                    channel_names=channel_names,
                    dropout=dropout,
                    interaction_max_weight=self.dual_channel_max_weight,
                    base_mode=self.channel_fusion_base_mode,
                )
                self.channel_interaction = self.shared_packet_fusion.interaction
                self.semantic_channel_cls = nn.Linear(hidden_dim, num_classes)
                self.structural_channel_cls = nn.Linear(hidden_dim, num_classes)
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
        self.identifiability_feature_index = int(identifiability_feature_index)
        self.identifiability_feature_mode = str(identifiability_feature_mode)
        if self.identifiability_feature_mode not in {"observed", "zero"}:
            raise ValueError("identifiability_feature_mode must be 'observed' or 'zero'")
        self.pool = AttentionPooling(
            hidden_dim,
            reliability_prior=bool(identifiability_pooling) and not identifiability_dual_pooling and not identifiability_evidence_adapter,
            reliability_prior_init=identifiability_prior_init,
            reliability_dual=identifiability_dual_pooling,
            reliability_evidence_adapter=identifiability_evidence_adapter,
            reliability_adapter_max_delta=identifiability_adapter_max_delta,
            reliability_residual_max_weight=identifiability_residual_max_weight,
            reliability_residual_init=identifiability_residual_init,
        )
        self.cls = nn.Linear(hidden_dim, num_classes)
        self.coherence = nn.Linear(hidden_dim, 2)
        self.next_direction = nn.Linear(hidden_dim, 2)
        self.next_length = nn.Linear(hidden_dim, 4)
        self.next_iat = nn.Linear(hidden_dim, 4)

    def import_concat_projection(self, weight: torch.Tensor, bias: torch.Tensor) -> None:
        """Convert a trained concat projection into an exact dual-channel base path."""
        if self.dual_channel_mode != "residual":
            raise ValueError("projection import is only valid in residual mode")
        expected = self.embedding_feature_dim + self.meta_feature_dim
        if weight.shape != (self.semantic_proj.out_features, expected):
            raise ValueError(
                f"concat projection shape mismatch: got={tuple(weight.shape)} "
                f"expected={(self.semantic_proj.out_features, expected)}"
            )
        if bias.shape != self.fusion_bias.shape:
            raise ValueError("concat projection bias shape mismatch")
        with torch.no_grad():
            self.semantic_proj.weight.copy_(weight[:, :self.embedding_feature_dim])
            structural_start = self.embedding_feature_dim + self.native_structural_dim
            self.structural_proj.weight.copy_(weight[:, structural_start:])
            self.fusion_bias.copy_(bias)

    @staticmethod
    def _masked_mean(h: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        if mask is None:
            return h.mean(dim=1)
        weight = mask.to(dtype=h.dtype).unsqueeze(-1)
        return (h * weight).sum(dim=1) / weight.sum(dim=1).clamp(min=1.0)

    def _project_channels(self, x: torch.Tensor, mask: torch.Tensor | None, intervened_x: torch.Tensor | None = None):
        if self.exact_shared_packet_encoder:
            if intervened_x is None or intervened_x.shape != x.shape:
                raise ValueError(
                    "exact shared packet encoder requires an aligned intervention view"
                )
            structural_input = x[..., self.embedding_feature_dim:]
            fused, gate, intervention_gate, channels = self.shared_packet_encoder(
                x[..., :self.embedding_feature_dim],
                structural_input[..., :self.native_structural_dim],
                structural_input[..., self.native_structural_dim:],
                intervened_x[..., :self.embedding_feature_dim],
                ablate_channel=self.train_ablate_input_channel,
                ablate_intervention_view=self.train_ablate_intervention_view,
                fixed_channel_fusion=self.train_fixed_channel_fusion,
            )
            semantic_summary = self._masked_mean(channels["semantic"], mask)
            structural_summary = self._masked_mean(channels["structural"], mask)
            packet_evidence_logits = None
            if self.packet_evidence_max_weight > 0:
                packet_logits = self.packet_classifier(fused)
                packet_evidence_logits = self._masked_mean(packet_logits, mask)
            return (
                fused,
                semantic_summary,
                structural_summary,
                gate,
                intervention_gate,
                packet_evidence_logits,
            )
        semantic = self.semantic_proj(x[..., :self.embedding_feature_dim])
        intervention_gate = None
        if self.use_intervention_views:
            if intervened_x is None:
                raise ValueError("intervened_x is required when use_intervention_views=True")
            if intervened_x.shape != x.shape:
                raise ValueError("factual/intervened flow input shape mismatch")
            intervened_semantic = self.semantic_proj(
                intervened_x[..., :self.embedding_feature_dim]
            )
            semantic, intervention_gate = self.intervention_view_fusion(
                semantic, intervened_semantic
            )
        structural_input = x[..., self.embedding_feature_dim:]
        native_input = structural_input[..., :self.native_structural_dim]
        protocol_input = structural_input[..., self.native_structural_dim:]
        structural = self.structural_proj(protocol_input)
        channels = {"semantic": semantic, "structural": structural}
        if self.native_structural_dim > 0:
            content = torch.tanh(self.native_structural_adapter(native_input))
            if self.channel_fusion_base_mode == "legacy":
                native_weight = self.dual_channel_max_weight * torch.sigmoid(
                    self.native_structural_raw_gate
                )
                content = native_weight * content
            channels["content"] = content
        base = (
            None
            if self.channel_fusion_base_mode == "semantic_anchor"
            else sum(channels.values()) + self.fusion_bias
        )
        fused, gate = self.shared_packet_fusion(channels, base=base)
        semantic_summary = self._masked_mean(semantic + self.fusion_bias, mask)
        structural_summary = self._masked_mean(structural + self.fusion_bias, mask)
        return fused, semantic_summary, structural_summary, gate, intervention_gate, None

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None, intervened_x: torch.Tensor | None = None):
        if self.identifiability_feature_mode == "zero" and self.identifiability_feature_index >= 0:
            x = x.clone()
            x[..., self.identifiability_feature_index:self.identifiability_feature_index + 2] = 0
        reliability = (
            x[..., self.identifiability_feature_index]
            if self.identifiability_feature_index >= 0
            else None
        )
        semantic_summary = None
        structural_summary = None
        channel_gate = None
        packet_evidence_logits = None
        if self.dual_channel_mode == "concat":
            h = self.proj(x)
        else:
            (
                h,
                semantic_summary,
                structural_summary,
                channel_gate,
                intervention_gate,
                packet_evidence_logits,
            ) = self._project_channels(x, mask, intervened_x)
            if self.exact_shared_packet_encoder:
                h = self.packet_to_flow_proj(h)
        key_padding_mask = None if mask is None else ~mask.bool()
        h = self.encoder(h, src_key_padding_mask=key_padding_mask)
        g = self.pool(h, mask, reliability=reliability)
        logits = self.cls(g)
        packet_evidence_gate = None
        if packet_evidence_logits is not None and self.packet_evidence_max_weight > 0:
            packet_evidence_gate = self.packet_evidence_max_weight * torch.sigmoid(
                self.packet_evidence_gate(
                    torch.cat([g, packet_evidence_logits], dim=-1)
                )
            )
            logits = (
                (1.0 - packet_evidence_gate) * logits
                + packet_evidence_gate * packet_evidence_logits
            )
        out = {
            "logits": logits,
            "coherence_logits": self.coherence(g),
            "embedding": g,
            "next_direction_logits": self.next_direction(g),
            "next_length_logits": self.next_length(g),
            "next_iat_logits": self.next_iat(g),
        }
        if semantic_summary is not None and structural_summary is not None:
            out["semantic_channel_logits"] = self.semantic_channel_cls(semantic_summary)
            out["structural_channel_logits"] = self.structural_channel_cls(structural_summary)
            out["dual_channel_gate"] = channel_gate
            if intervention_gate is not None:
                out["intervention_view_gate"] = intervention_gate
        if packet_evidence_logits is not None:
            out["packet_evidence_logits"] = packet_evidence_logits
        if packet_evidence_gate is not None:
            out["packet_evidence_gate"] = packet_evidence_gate
        if self.pool.last_reliability_gate is not None:
            out["identifiability_gate"] = self.pool.last_reliability_gate
        if self.pool.last_reliability_adapter_norm is not None:
            out["identifiability_adapter_norm"] = self.pool.last_reliability_adapter_norm
        return out
