from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CounterfactualFlowFusion(nn.Module):
    """Fuse a flow and its deterministic header-intervened counterpart.

    ``mean`` is the compute-matched two-view control. ``counterfactual`` starts
    from the selected base path and can add only a bounded residual extracted
    from features that change under the intervention. ``router`` keeps the
    selected base path as its identity initialization and learns a bounded
    paired-view correction.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_classes: int,
        mode: str = "counterfactual",
        base_mode: str = "clean",
        max_residual_weight: float = 0.25,
        initial_residual_fraction: float = 0.1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if mode not in {"clean", "mean", "counterfactual", "router"}:
            raise ValueError(
                "mode must be 'clean', 'mean', 'counterfactual', or 'router'"
            )
        if base_mode not in {"clean", "mean"}:
            raise ValueError("base_mode must be 'clean' or 'mean'")
        self.mode = mode
        self.base_mode = mode if mode in {"clean", "mean"} else base_mode
        self.hidden_dim = int(hidden_dim)
        self.num_classes = int(num_classes)
        self.max_residual_weight = float(max(0.0, min(1.0, max_residual_weight)))

        if mode in {"counterfactual", "router"}:
            self.shared_norm = nn.LayerNorm(hidden_dim)
            self.changed_encoder = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
            )
            self.changed_classifier = (
                nn.Linear(hidden_dim, num_classes)
                if mode == "counterfactual"
                else None
            )
            self.gate = nn.Sequential(
                nn.LayerNorm(hidden_dim * 2 + 1),
                nn.Linear(hidden_dim * 2 + 1, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
            if self.changed_classifier is not None:
                nn.init.zeros_(self.changed_classifier.weight)
                nn.init.zeros_(self.changed_classifier.bias)
            nn.init.zeros_(self.gate[-1].weight)
            nn.init.zeros_(self.gate[-1].bias)
            if mode == "router":
                self.residual_raw_scale = nn.Parameter(torch.tensor(0.0))
            else:
                fraction = max(1e-4, min(1.0 - 1e-4, initial_residual_fraction))
                self.residual_raw_scale = nn.Parameter(
                    torch.tensor(math.log(fraction / (1.0 - fraction)))
                )

    def forward(
        self,
        clean_embedding: torch.Tensor,
        intervened_embedding: torch.Tensor,
        clean_logits: torch.Tensor,
        intervened_logits: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if clean_embedding.shape != intervened_embedding.shape:
            raise ValueError("paired flow embeddings must have identical shapes")
        if clean_logits.shape != intervened_logits.shape:
            raise ValueError("paired flow logits must have identical shapes")

        base_logits = (
            clean_logits
            if self.base_mode == "clean"
            else 0.5 * (clean_logits + intervened_logits)
        )
        shared = 0.5 * (clean_embedding + intervened_embedding)
        if self.mode in {"clean", "mean"}:
            return {
                "logits": base_logits,
                "base_logits": base_logits,
                "embedding": clean_embedding if self.mode == "clean" else shared,
                "shared_embedding": shared,
                "changed_embedding": torch.zeros_like(shared),
                "residual_gate": shared.new_zeros((shared.size(0), 1)),
                "residual_logits": torch.zeros_like(base_logits),
            }

        shared = self.shared_norm(shared)
        changed = self.changed_encoder((clean_embedding - intervened_embedding).abs())
        cosine = F.cosine_similarity(
            clean_embedding.float(), intervened_embedding.float(), dim=-1
        ).to(dtype=shared.dtype).unsqueeze(-1)
        local_gate = torch.sigmoid(self.gate(torch.cat([shared, changed, cosine], dim=-1)))
        if self.mode == "router":
            global_scale = self.max_residual_weight * torch.tanh(
                self.residual_raw_scale
            )
            residual_logits = intervened_logits - clean_logits
        else:
            global_scale = self.max_residual_weight * torch.sigmoid(
                self.residual_raw_scale
            )
            residual_logits = self.changed_classifier(changed)
        residual_gate = global_scale * local_gate
        logits = base_logits + residual_gate * residual_logits
        embedding = F.layer_norm(shared + residual_gate * changed, (self.hidden_dim,))
        return {
            "logits": logits,
            "base_logits": base_logits,
            "embedding": embedding,
            "shared_embedding": shared,
            "changed_embedding": changed,
            "residual_gate": residual_gate,
            "routing_gate": local_gate,
            "residual_logits": residual_logits,
        }


def counterfactual_regularization(
    output: dict[str, torch.Tensor],
    gate_weight: float = 0.0,
    orthogonality_weight: float = 0.0,
) -> torch.Tensor:
    shared = output["shared_embedding"]
    loss = shared.sum() * 0.0
    if gate_weight > 0:
        loss = loss + float(gate_weight) * output["residual_gate"].abs().mean()
    if orthogonality_weight > 0 and output["changed_embedding"].numel() > 0:
        similarity = F.cosine_similarity(
            output["shared_embedding"].float(),
            output["changed_embedding"].float(),
            dim=-1,
        )
        loss = loss + float(orthogonality_weight) * similarity.square().mean()
    return loss


def intervention_routing_loss(
    output: dict[str, torch.Tensor],
    clean_logits: torch.Tensor,
    intervened_logits: torch.Tensor,
    labels: torch.Tensor,
    weight: float,
) -> torch.Tensor:
    if weight <= 0 or "routing_gate" not in output:
        return clean_logits.sum() * 0.0
    clean_correct = clean_logits.argmax(dim=-1).eq(labels)
    intervened_correct = intervened_logits.argmax(dim=-1).eq(labels)
    informative = clean_correct ^ intervened_correct
    if not informative.any():
        return clean_logits.sum() * 0.0
    target = intervened_correct[informative].to(dtype=clean_logits.dtype)
    gate = output["routing_gate"][informative].squeeze(-1).clamp(1e-6, 1.0 - 1e-6)
    return float(weight) * F.binary_cross_entropy(gate, target)
