from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn


class SharedPacketClassifierHead(nn.Linear):
    """Common fused-packet task head reused by packet and flow models."""

    def __init__(self, hidden_dim: int, num_classes: int):
        super().__init__(hidden_dim, num_classes)


class SharedInterventionViewFusion(nn.Module):
    """Fuse factual and field-intervened views before task aggregation.

    The same module is used by packet- and flow-level models.  Its identity
    point is the symmetric mean, while a bounded, sample-dependent residual
    can learn whether factual or intervened evidence is more reliable.
    """

    def __init__(
        self,
        hidden_dim: int,
        dropout: float = 0.1,
        max_residual_weight: float = 0.25,
        base_mode: str = "symmetric_mean",
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.max_residual_weight = float(max(0.0, min(1.0, max_residual_weight)))
        if base_mode not in {"symmetric_mean", "factual_anchor"}:
            raise ValueError(
                "intervention view base_mode must be symmetric_mean or factual_anchor"
            )
        self.base_mode = str(base_mode)
        evidence_dim = self.hidden_dim * 4
        self.factual_norm = nn.LayerNorm(self.hidden_dim)
        self.intervened_norm = nn.LayerNorm(self.hidden_dim)
        self.router = nn.Sequential(
            nn.LayerNorm(evidence_dim),
            nn.Linear(evidence_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, 1),
        )
        # A zero router gives sigmoid(0)=0.5 and therefore an exact mean.
        nn.init.zeros_(self.router[-1].weight)
        nn.init.zeros_(self.router[-1].bias)

    def effective_weights(self, router_gate: torch.Tensor) -> torch.Tensor:
        """Map router probabilities to the factual/intervened mixture weights."""
        if router_gate.shape[-1] != 2:
            raise ValueError("intervention router gate must have two columns")
        if self.base_mode == "factual_anchor":
            return router_gate
        return 0.5 * (1.0 - self.max_residual_weight) + (
            self.max_residual_weight * router_gate
        )

    def effective_weight_bounds(self) -> dict[str, tuple[float, float]]:
        if self.base_mode == "factual_anchor":
            return {
                "factual": (1.0 - self.max_residual_weight, 1.0),
                "intervened": (0.0, self.max_residual_weight),
            }
        lower = 0.5 * (1.0 - self.max_residual_weight)
        return {
            "factual": (lower, 1.0 - lower),
            "intervened": (lower, 1.0 - lower),
        }

    def forward(
        self,
        factual: torch.Tensor,
        intervened: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if factual.shape != intervened.shape:
            raise ValueError(
                "factual/intervened representation shape mismatch: "
                f"{tuple(factual.shape)} != {tuple(intervened.shape)}"
            )
        if factual.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"intervention view dim mismatch: got={factual.shape[-1]} "
                f"expected={self.hidden_dim}"
            )
        factual = self.factual_norm(factual)
        intervened = self.intervened_norm(intervened)
        shared = 0.5 * (factual + intervened)
        difference = factual - intervened
        evidence = torch.cat(
            [factual, intervened, difference, difference.abs()], dim=-1
        )
        router_weight = torch.sigmoid(self.router(evidence))
        if self.base_mode == "factual_anchor":
            intervened_weight = self.max_residual_weight * router_weight
            fused = factual + intervened_weight * (intervened - factual)
            effective_gate = torch.cat(
                [1.0 - intervened_weight, intervened_weight], dim=-1
            )
            return fused, effective_gate
        centered_weight = 2.0 * router_weight - 1.0
        fused = shared + 0.5 * self.max_residual_weight * centered_weight * difference
        return fused, torch.cat([router_weight, 1.0 - router_weight], dim=-1)


class SharedPacketChannelFusion(nn.Module):
    """Data-dependent fusion shared by packet- and flow-level models.

    Inputs are channel representations already projected to ``hidden_dim``.
    Every configured channel is always evaluated; a learned per-packet gate
    controls its contribution instead of dataset-specific module switches.
    """

    def __init__(
        self,
        hidden_dim: int,
        channel_names: Sequence[str],
        dropout: float = 0.1,
        interaction_max_weight: float = 0.25,
        base_mode: str = "legacy",
    ) -> None:
        super().__init__()
        names = tuple(str(name) for name in channel_names)
        if len(names) < 2:
            raise ValueError("shared packet fusion requires at least two channels")
        if len(set(names)) != len(names):
            raise ValueError("shared packet fusion channel names must be unique")
        self.hidden_dim = int(hidden_dim)
        self.channel_names = names
        if base_mode not in {"legacy", "semantic_anchor"}:
            raise ValueError("shared packet fusion base_mode must be legacy or semantic_anchor")
        if base_mode == "semantic_anchor" and "semantic" not in names:
            raise ValueError("semantic_anchor fusion requires a semantic channel")
        self.base_mode = str(base_mode)
        self.interaction_max_weight = float(
            max(0.0, min(1.0, interaction_max_weight))
        )
        evidence_dim = self.hidden_dim * (len(names) * 2 + 1)
        self.channel_norms = nn.ModuleDict(
            {name: nn.LayerNorm(self.hidden_dim) for name in names}
        )
        self.gate = nn.Sequential(
            nn.LayerNorm(evidence_dim),
            nn.Linear(evidence_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, len(names)),
        )
        self.interaction = nn.Sequential(
            nn.LayerNorm(evidence_dim),
            nn.Linear(evidence_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.output_norm = nn.Identity()
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)
        nn.init.zeros_(self.interaction[-1].weight)
        nn.init.zeros_(self.interaction[-1].bias)

    def effective_weights(self, router_gate: torch.Tensor) -> torch.Tensor:
        """Return channel mixture weights induced by the bounded residual path."""
        if router_gate.shape[-1] != len(self.channel_names):
            raise ValueError("channel router gate width does not match channel names")
        if self.base_mode != "semantic_anchor":
            return router_gate
        effective = self.interaction_max_weight * router_gate
        anchor_index = self.channel_names.index("semantic")
        effective = effective.clone()
        effective[..., anchor_index] += 1.0 - self.interaction_max_weight
        return effective

    def effective_weight_bounds(self) -> dict[str, tuple[float, float]]:
        if self.base_mode != "semantic_anchor":
            return {name: (0.0, 1.0) for name in self.channel_names}
        residual = self.interaction_max_weight
        return {
            name: ((1.0 - residual, 1.0) if name == "semantic" else (0.0, residual))
            for name in self.channel_names
        }

    def forward(
        self,
        channels: dict[str, torch.Tensor],
        base: torch.Tensor | None = None,
        fixed_equal: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        missing = [name for name in self.channel_names if name not in channels]
        extra = sorted(set(channels) - set(self.channel_names))
        if missing or extra:
            raise ValueError(f"channel mismatch: missing={missing}, extra={extra}")
        normalized = [self.channel_norms[name](channels[name]) for name in self.channel_names]
        reference_shape = normalized[0].shape
        if any(value.shape != reference_shape for value in normalized[1:]):
            raise ValueError("all shared packet channels must have the same shape")
        stacked = torch.stack(normalized, dim=-2)
        mean = stacked.mean(dim=-2)
        if fixed_equal:
            weights = torch.full(
                (*mean.shape[:-1], len(self.channel_names)),
                1.0 / len(self.channel_names),
                dtype=mean.dtype,
                device=mean.device,
            )
            return self.output_norm(mean), weights
        deviations = [value - mean for value in normalized]
        gate_evidence = torch.cat([*normalized, *deviations, mean], dim=-1)
        weights = torch.softmax(self.gate(gate_evidence), dim=-1)
        routed = torch.sum(stacked * weights.unsqueeze(-1), dim=-2)
        interaction_evidence = torch.cat([*normalized, *deviations, routed], dim=-1)
        correction = torch.tanh(self.interaction(interaction_evidence))
        if self.base_mode == "semantic_anchor":
            if base is not None:
                raise ValueError("semantic_anchor fusion computes its own normalized base")
            anchor_index = self.channel_names.index("semantic")
            base = normalized[anchor_index]
            routed_residual = routed - base
        else:
            if base is None:
                base = mean
            routed_residual = routed - mean
        if base.shape != reference_shape:
            raise ValueError("shared packet fusion base must match channel shapes")
        # Legacy preserves old checkpoints. Semantic-anchor mode gives the
        # shared semantic representation a stable path while learned gates
        # control every non-anchor channel through a bounded residual.
        fused = self.output_norm(
            base + self.interaction_max_weight * (routed_residual + correction)
        )
        return fused, weights

    def gate_dict(self, weights: torch.Tensor) -> dict[str, torch.Tensor]:
        return {name: weights[..., index] for index, name in enumerate(self.channel_names)}


class SharedPacketRepresentationEncoder(nn.Module):
    """Exact semantic/content/structural packet module reused by both tasks."""

    def __init__(
        self,
        semantic_dim: int,
        content_dim: int,
        structural_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
        use_intervention_views: bool = True,
        intervention_max_residual_weight: float = 0.25,
        intervention_view_base_mode: str = "symmetric_mean",
        channel_fusion_base_mode: str = "semantic_anchor",
        channel_fusion_max_weight: float = 0.25,
    ) -> None:
        super().__init__()
        dimensions = {
            "semantic": int(semantic_dim),
            "content": int(content_dim),
            "structural": int(structural_dim),
        }
        if any(value <= 0 for value in dimensions.values()):
            raise ValueError("all exact shared packet channels require positive dimensions")
        self.semantic_dim = dimensions["semantic"]
        self.content_dim = dimensions["content"]
        self.structural_dim = dimensions["structural"]
        self.hidden_dim = int(hidden_dim)
        self.use_intervention_views = bool(use_intervention_views)

        def projector(input_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(input_dim, self.hidden_dim, bias=False),
                nn.LayerNorm(self.hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )

        self.semantic_proj = projector(self.semantic_dim)
        self.content_proj = projector(self.content_dim)
        self.structural_proj = projector(self.structural_dim)
        if self.use_intervention_views:
            self.intervention_view_fusion = SharedInterventionViewFusion(
                self.hidden_dim,
                dropout=dropout,
                max_residual_weight=intervention_max_residual_weight,
                base_mode=intervention_view_base_mode,
            )
        self.channel_fusion = SharedPacketChannelFusion(
            self.hidden_dim,
            channel_names=("semantic", "content", "structural"),
            dropout=dropout,
            interaction_max_weight=channel_fusion_max_weight,
            base_mode=channel_fusion_base_mode,
        )

    def forward(
        self,
        semantic: torch.Tensor,
        content: torch.Tensor,
        structural: torch.Tensor,
        intervened_semantic: torch.Tensor | None = None,
        ablate_channel: str = "none",
        ablate_intervention_view: str = "none",
        fixed_channel_fusion: bool = False,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        dict[str, torch.Tensor],
    ]:
        expected = {
            "semantic": self.semantic_dim,
            "content": self.content_dim,
            "structural": self.structural_dim,
        }
        observed = {
            "semantic": semantic.shape[-1],
            "content": content.shape[-1],
            "structural": structural.shape[-1],
        }
        if observed != expected:
            raise ValueError(f"shared packet channel dimensions mismatch: {observed} != {expected}")
        projected_semantic = self.semantic_proj(semantic)
        intervention_gate = None
        if self.use_intervention_views:
            if intervened_semantic is None or intervened_semantic.shape != semantic.shape:
                raise ValueError("aligned intervened semantic input is required")
            projected_intervened = self.semantic_proj(intervened_semantic)
            if ablate_intervention_view == "factual_only":
                projected_intervened = projected_semantic
            elif ablate_intervention_view == "intervened_only":
                projected_semantic = projected_intervened
            elif ablate_intervention_view != "none":
                raise ValueError(
                    f"unknown intervention-view ablation: {ablate_intervention_view}"
                )
            projected_semantic, intervention_gate = self.intervention_view_fusion(
                projected_semantic, projected_intervened
            )
        channels = {
            "semantic": projected_semantic,
            "content": self.content_proj(content),
            "structural": self.structural_proj(structural),
        }
        if ablate_channel != "none":
            if ablate_channel not in channels:
                raise ValueError(f"unknown shared packet channel ablation: {ablate_channel}")
            channels[ablate_channel] = torch.zeros_like(channels[ablate_channel])
        fused, channel_gate = self.channel_fusion(
            channels, fixed_equal=fixed_channel_fusion
        )
        return fused, channel_gate, intervention_gate, channels
