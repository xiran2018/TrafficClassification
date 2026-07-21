from __future__ import annotations

import torch
import torch.nn as nn

from .flow_transformer import AttentionPooling


class EdgeAwareGraphLayer(nn.Module):
    """Edge-aware graph transformer layer without PyG dependency.

    It performs global self-attention with additive edge bias. Edge type and
    continuous edge attributes jointly modulate attention for connected packets.
    """

    def __init__(self, hidden_dim: int, num_heads: int, num_edge_types: int = 7, edge_attr_dim: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads
        self.edge_attr_dim = edge_attr_dim
        assert hidden_dim % num_heads == 0
        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, hidden_dim)
        self.o = nn.Linear(hidden_dim, hidden_dim)
        self.edge_type_bias = nn.Embedding(num_edge_types, num_heads)
        continuous_dim = max(0, edge_attr_dim - 1)
        self.edge_attr_bias = nn.Sequential(
            nn.LayerNorm(continuous_dim),
            nn.Linear(continuous_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_heads),
        ) if continuous_dim > 0 else None
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        n = h.size(0)
        q = self.q(h).view(n, self.num_heads, self.head_dim).transpose(0, 1)
        k = self.k(h).view(n, self.num_heads, self.head_dim).transpose(0, 1)
        v = self.v(h).view(n, self.num_heads, self.head_dim).transpose(0, 1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if edge_index.numel() > 0:
            src, dst = edge_index[0].long(), edge_index[1].long()
            edge_attr = edge_attr.to(device=h.device, dtype=h.dtype)
            et = edge_attr[:, 0].long().clamp(min=0, max=self.edge_type_bias.num_embeddings - 1)
            eb = self.edge_type_bias(et)
            if self.edge_attr_bias is not None:
                eb = eb + self.edge_attr_bias(edge_attr[:, 1:self.edge_attr_dim])
            eb = eb.transpose(0, 1)
            scores[:, src, dst] = scores[:, src, dst] + eb
        attn = torch.softmax(scores, dim=-1)
        ctx = torch.matmul(attn, v).transpose(0, 1).contiguous().view(n, self.hidden_dim)
        h = self.norm1(h + self.dropout(self.o(ctx)))
        h = self.norm2(h + self.dropout(self.ffn(h)))
        return h


class FlowGraphTransformerClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 256, num_layers: int = 2, num_heads: int = 4, num_edge_types: int = 7, edge_attr_dim: int = 4, dropout: float = 0.1, identifiability_feature_index: int = -1, identifiability_pooling: bool = False, identifiability_feature_mode: str = "observed", identifiability_prior_init: float = 0.1, identifiability_dual_pooling: bool = False, identifiability_evidence_adapter: bool = False, identifiability_adapter_max_delta: float = 0.25, identifiability_residual_max_weight: float = 0.0, identifiability_residual_init: float = 0.5):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.edge_attr_dim = edge_attr_dim
        self.identifiability_feature_index = int(identifiability_feature_index)
        self.identifiability_feature_mode = str(identifiability_feature_mode)
        if self.identifiability_feature_mode not in {"observed", "zero"}:
            raise ValueError("identifiability_feature_mode must be 'observed' or 'zero'")
        self.layers = nn.ModuleList([EdgeAwareGraphLayer(hidden_dim, num_heads, num_edge_types, edge_attr_dim, dropout) for _ in range(num_layers)])
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
        self.edge_mlp = nn.Sequential(nn.Linear(hidden_dim * 2 + edge_attr_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 2))
        self.next_direction = nn.Linear(hidden_dim, 2)
        self.next_length = nn.Linear(hidden_dim, 4)
        self.next_iat = nn.Linear(hidden_dim, 4)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor):
        if self.identifiability_feature_mode == "zero" and self.identifiability_feature_index >= 0:
            x = x.clone()
            x[:, self.identifiability_feature_index:self.identifiability_feature_index + 2] = 0
        reliability = (
            x[:, self.identifiability_feature_index]
            if self.identifiability_feature_index >= 0
            else None
        )
        h = self.proj(x)
        if edge_attr.numel() == 0:
            edge_attr = torch.zeros((0, self.edge_attr_dim), dtype=x.dtype, device=x.device)
        else:
            edge_attr = edge_attr.to(device=x.device, dtype=x.dtype)
        for layer in self.layers:
            h = layer(h, edge_index, edge_attr)
        g = self.pool(
            h.unsqueeze(0),
            torch.ones(1, h.size(0), dtype=torch.bool, device=h.device),
            reliability=None if reliability is None else reliability.unsqueeze(0),
        ).squeeze(0)
        edge_logits = None
        if edge_index.numel() > 0:
            src, dst = edge_index[0].long(), edge_index[1].long()
            edge_feat = torch.cat([h[src], h[dst], edge_attr.float()], dim=-1)
            edge_logits = self.edge_mlp(edge_feat)
        out = {
            "logits": self.cls(g.unsqueeze(0)),
            "coherence_logits": self.coherence(g.unsqueeze(0)),
            "embedding": g,
            "edge_logits": edge_logits,
            "next_direction_logits": self.next_direction(g.unsqueeze(0)),
            "next_length_logits": self.next_length(g.unsqueeze(0)),
            "next_iat_logits": self.next_iat(g.unsqueeze(0)),
        }
        if self.pool.last_reliability_gate is not None:
            out["identifiability_gate"] = self.pool.last_reliability_gate.squeeze(0)
        if self.pool.last_reliability_adapter_norm is not None:
            out["identifiability_adapter_norm"] = self.pool.last_reliability_adapter_norm.squeeze(0)
        return out
