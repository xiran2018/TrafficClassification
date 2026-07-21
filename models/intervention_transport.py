from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LowRankInterventionTransport(nn.Module):
    """Map an intervened packet embedding back to the clean representation space."""

    def __init__(self, embedding_dim: int, rank: int = 64, dropout: float = 0.0):
        super().__init__()
        if embedding_dim <= 0 or rank <= 0:
            raise ValueError("embedding_dim and rank must be positive")
        self.embedding_dim = int(embedding_dim)
        self.rank = int(rank)
        self.log_scale = nn.Parameter(torch.zeros(embedding_dim))
        self.bias = nn.Parameter(torch.zeros(embedding_dim))
        self.norm = nn.LayerNorm(embedding_dim)
        self.down = nn.Linear(embedding_dim, rank, bias=False)
        self.up = nn.Linear(rank, embedding_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.log_scale.clamp(min=-2.0, max=2.0).exp()
        residual = self.up(self.dropout(F.gelu(self.down(self.norm(x)))))
        return x * scale + self.bias + residual


def transport_alignment_loss(
    transported: torch.Tensor,
    clean: torch.Tensor,
    cosine_weight: float = 1.0,
    normalized_mse_weight: float = 1.0,
    moment_weight: float = 0.1,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    cosine = (1.0 - F.cosine_similarity(transported.float(), clean.float(), dim=-1)).mean()
    variance = clean.float().var(unbiased=False).detach().clamp(min=1e-4)
    normalized_mse = (transported.float() - clean.float()).square().mean() / variance
    mean_loss = (transported.float().mean(dim=0) - clean.float().mean(dim=0)).square().mean() / variance
    transported_std = transported.float().std(dim=0, unbiased=False)
    clean_std = clean.float().std(dim=0, unbiased=False)
    std_loss = (transported_std - clean_std).square().mean() / variance
    moment = mean_loss + std_loss
    total = (
        float(cosine_weight) * cosine
        + float(normalized_mse_weight) * normalized_mse
        + float(moment_weight) * moment
    )
    return total, {
        "cosine_loss": cosine,
        "normalized_mse": normalized_mse,
        "moment_loss": moment,
    }
