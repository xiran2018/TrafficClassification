from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from models.availability_aware_cross_scale import (
    availability_aware_cross_scale_loss,
)
from models.identity_safe_contrastive import (
    identity_safe_flow_aware_contrastive_loss,
)

try:
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
except Exception:  # pragma: no cover - optional import at runtime
    LoraConfig = None
    PeftModel = None
    TaskType = None
    get_peft_model = None


DEFAULT_QWEN_LORA_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


@dataclass
class Tower1LossOutput:
    loss: torch.Tensor
    lm_loss: torch.Tensor
    pkt_cls_loss: torch.Tensor
    supcon_loss: torch.Tensor
    flow_proto_loss: torch.Tensor
    paired_consistency_loss: torch.Tensor
    cross_scale_loss: torch.Tensor
    packet_logits: Optional[torch.Tensor]
    packet_embeddings: Optional[torch.Tensor]
    projected_embeddings: Optional[torch.Tensor]


def last_token_pooling(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Scheme B for decoder-only LLMs: use the last non-padding token hidden state."""
    last_idx = attention_mask.long().sum(dim=1).clamp(min=1) - 1
    batch_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)
    return hidden_states[batch_idx, last_idx]


class MLPProjectionHead(nn.Module):
    def __init__(self, input_dim: int, projection_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, projection_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), p=2, dim=-1)


class QwenPacketMultiTaskModel(nn.Module):
    """Qwen-LoRA Tower-1 model with protocol QA, weak packet classification and SupCon.

    The packet representation is the final-layer hidden state of the last non-padding
    token, matching scheme B. The projection head is used only for contrastive learning;
    the downstream Tower-2 extractor can still use the raw last-token hidden state.
    """

    def __init__(
        self,
        base_model_name_or_path: str,
        num_classes: int,
        torch_dtype: torch.dtype = torch.float16,
        lora_path: str = "",
        create_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: Optional[List[str]] = None,
        projection_dim: int = 256,
        dropout: float = 0.1,
        trust_remote_code: bool = True,
        local_files_only: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.base_model_name_or_path = base_model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_name_or_path,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.backbone = AutoModelForCausalLM.from_pretrained(
            base_model_name_or_path,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )
        hidden_size = int(self.backbone.config.hidden_size)

        if lora_path:
            if PeftModel is None:
                raise RuntimeError("peft is required to load LoRA adapters")
            self.backbone = PeftModel.from_pretrained(self.backbone, lora_path, is_trainable=True, local_files_only=local_files_only)
        elif create_lora:
            if get_peft_model is None or LoraConfig is None:
                raise RuntimeError("peft is required for LoRA training. Please `pip install peft`.")
            target_modules = lora_target_modules or DEFAULT_QWEN_LORA_TARGETS
            cfg = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=target_modules,
                bias="none",
            )
            self.backbone = get_peft_model(self.backbone, cfg)

        self.packet_classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )
        self.projection_head = MLPProjectionHead(hidden_size, projection_dim, dropout=dropout)

    @property
    def hidden_size(self) -> int:
        return int(self.backbone.config.hidden_size)

    def encode_packets(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        last_hidden = out.hidden_states[-1]
        packet_emb = last_token_pooling(last_hidden, attention_mask).float()
        packet_emb_norm = F.normalize(packet_emb, p=2, dim=-1)
        projected = self.projection_head(packet_emb)
        logits = self.packet_classifier(packet_emb)
        return packet_emb_norm, projected, logits

    def forward_multitask(
        self,
        sft_batch: Optional[Dict[str, torch.Tensor]] = None,
        packet_batch: Optional[Dict[str, torch.Tensor]] = None,
        cls_weight: float = 0.1,
        contrastive_weight: float = 0.3,
        temperature: float = 0.07,
        same_flow_positive_weight: float = 0.0,
        same_label_positive_weight: float = 1.0,
        identity_safe_contrastive: bool = False,
        flow_proto_weight: float = 0.0,
        flow_proto_positive: str = "same_class",
        flow_proto_context: str = "inclusive",
        paired_consistency_weight: float = 0.0,
        paired_cls_weight: float = 0.0,
        paired_logit_kl_weight: float = 0.5,
        paired_raw_consistency_weight: float = 1.0,
        cross_scale_weight: float = 0.0,
        cross_scale_temperature: float = 0.07,
    ) -> Tower1LossOutput:
        device = next(self.parameters()).device
        zero = torch.zeros((), device=device)
        lm_loss = zero
        pkt_cls_loss = zero
        supcon_loss = zero
        flow_proto_loss = zero
        paired_consistency_loss = zero
        cross_scale_loss = zero
        packet_logits = None
        packet_embeddings = None
        projected_embeddings = None

        if sft_batch is not None and (sft_batch["labels"] != -100).any():
            out = self.backbone(
                input_ids=sft_batch["input_ids"],
                attention_mask=sft_batch["attention_mask"],
                labels=sft_batch["labels"],
                use_cache=False,
                return_dict=True,
            )
            lm_loss = out.loss

        if packet_batch is not None:
            packet_embeddings, projected_embeddings, packet_logits = self.encode_packets(
                packet_batch["input_ids"], packet_batch["attention_mask"]
            )
            labels = packet_batch["labels"].long()
            ce_each = F.cross_entropy(packet_logits, labels, reduction="none")
            weights = packet_batch.get("weights")
            if weights is not None:
                weights = weights.to(ce_each.device).float()
                pkt_cls_loss = (ce_each * weights).sum() / weights.sum().clamp(min=1.0)
            else:
                pkt_cls_loss = ce_each.mean()
            flow_ids = packet_batch.get("flow_ids")
            if identity_safe_contrastive:
                packet_ids = packet_batch.get("packet_ids")
                if flow_ids is None or packet_ids is None:
                    raise ValueError(
                        "identity-safe contrastive learning requires flow_ids and packet_ids"
                    )
                supcon_loss = identity_safe_flow_aware_contrastive_loss(
                    projected_embeddings,
                    labels,
                    flow_ids.long(),
                    packet_ids.long(),
                    temperature=temperature,
                    same_flow_weight=same_flow_positive_weight,
                    same_label_weight=same_label_positive_weight,
                )
            elif flow_ids is not None and same_flow_positive_weight > 0:
                supcon_loss = flow_aware_contrastive_loss(
                    projected_embeddings,
                    labels,
                    flow_ids.long(),
                    temperature=temperature,
                    same_flow_weight=same_flow_positive_weight,
                    same_label_weight=same_label_positive_weight,
                )
            else:
                supcon_loss = supervised_contrastive_loss(projected_embeddings, labels, temperature=temperature)
            if flow_ids is not None and flow_proto_weight > 0:
                flow_proto_loss = packet_to_flow_prototype_loss(
                    projected_embeddings,
                    labels,
                    flow_ids.long(),
                    temperature=temperature,
                    positive_mode=flow_proto_positive,
                    context_mode=flow_proto_context,
                )
            if (
                paired_consistency_weight > 0 or cross_scale_weight > 0
            ) and "paired_input_ids" in packet_batch:
                paired_mask = packet_batch.get("paired_mask")
                if paired_mask is None:
                    paired_mask = torch.ones_like(labels, dtype=torch.bool)
                else:
                    paired_mask = paired_mask.bool()
                if paired_mask.any():
                    paired_embeddings, paired_projected, paired_logits = self.encode_packets(
                        packet_batch["paired_input_ids"], packet_batch["paired_attention_mask"]
                    )
                    if paired_consistency_weight > 0:
                        paired_consistency_loss = paired_view_consistency_loss(
                            projected_embeddings[paired_mask],
                            paired_projected[paired_mask],
                            packet_logits[paired_mask],
                            paired_logits[paired_mask],
                            logit_kl_weight=paired_logit_kl_weight,
                            raw_z=packet_embeddings[paired_mask],
                            paired_raw_z=paired_embeddings[paired_mask],
                            raw_consistency_weight=paired_raw_consistency_weight,
                        )
                    if cross_scale_weight > 0:
                        packet_ids = packet_batch.get("packet_ids")
                        if flow_ids is None or packet_ids is None:
                            raise ValueError(
                                "cross-scale consistency requires flow_ids and packet_ids"
                            )
                        cross_scale_loss = availability_aware_cross_scale_loss(
                            projected_embeddings,
                            paired_projected,
                            labels,
                            flow_ids.long(),
                            packet_ids.long(),
                            intervention_mask=paired_mask,
                            temperature=cross_scale_temperature,
                        )
                    if paired_cls_weight > 0:
                        paired_ce_each = F.cross_entropy(paired_logits[paired_mask], labels[paired_mask], reduction="none")
                        paired_weights = weights[paired_mask] if weights is not None else None
                        if paired_weights is not None:
                            paired_cls_loss = (paired_ce_each * paired_weights).sum() / paired_weights.sum().clamp(min=1.0)
                        else:
                            paired_cls_loss = paired_ce_each.mean()
                        pkt_cls_loss = pkt_cls_loss + paired_cls_weight * paired_cls_loss

        loss = (
            lm_loss
            + cls_weight * pkt_cls_loss
            + contrastive_weight * supcon_loss
            + flow_proto_weight * flow_proto_loss
            + paired_consistency_weight * paired_consistency_loss
            + cross_scale_weight * cross_scale_loss
        )
        return Tower1LossOutput(
            loss,
            lm_loss,
            pkt_cls_loss,
            supcon_loss,
            flow_proto_loss,
            paired_consistency_loss,
            cross_scale_loss,
            packet_logits,
            packet_embeddings,
            projected_embeddings,
        )

    def save_packet_heads(self, output_dir: str) -> None:
        torch.save(
            {
                "num_classes": self.num_classes,
                "hidden_size": self.hidden_size,
                "packet_classifier": self.packet_classifier.state_dict(),
                "projection_head": self.projection_head.state_dict(),
            },
            f"{output_dir}/tower1_heads.pt",
        )


def supervised_contrastive_loss(z: torch.Tensor, labels: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """Supervised contrastive loss with all same-label samples in a batch as positives.

    If a mini-batch has no positive pairs, returns a differentiable zero.
    """
    if z.size(0) <= 1:
        return z.sum() * 0.0
    z = F.normalize(z.float(), p=2, dim=-1)
    labels = labels.view(-1, 1)
    logits = torch.matmul(z, z.T) / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()

    self_mask = torch.eye(z.size(0), dtype=torch.bool, device=z.device)
    pos_mask = torch.eq(labels, labels.T).to(z.device) & ~self_mask
    valid = pos_mask.sum(dim=1) > 0
    if valid.sum() == 0:
        return z.sum() * 0.0

    logits_masked = logits.masked_fill(self_mask, -1e9)
    log_prob = logits_masked - torch.logsumexp(logits_masked, dim=1, keepdim=True)
    mean_log_prob_pos = (pos_mask.float() * log_prob).sum(dim=1) / pos_mask.sum(dim=1).clamp(min=1)
    return -mean_log_prob_pos[valid].mean()


def flow_aware_contrastive_loss(
    z: torch.Tensor,
    labels: torch.Tensor,
    flow_ids: torch.Tensor,
    temperature: float = 0.07,
    same_flow_weight: float = 2.0,
    same_label_weight: float = 1.0,
) -> torch.Tensor:
    """Weighted SupCon: same-flow positives are stronger than same-label positives."""
    if z.size(0) <= 1:
        return z.sum() * 0.0
    z = F.normalize(z.float(), p=2, dim=-1)
    labels = labels.view(-1, 1)
    flow_ids = flow_ids.view(-1, 1)
    logits = torch.matmul(z, z.T) / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()

    self_mask = torch.eye(z.size(0), dtype=torch.bool, device=z.device)
    same_label = torch.eq(labels, labels.T).to(z.device) & ~self_mask
    same_flow = torch.eq(flow_ids, flow_ids.T).to(z.device) & ~self_mask
    pos_weight = same_label.float() * float(same_label_weight)
    pos_weight = pos_weight + same_flow.float() * float(same_flow_weight)
    valid = pos_weight.sum(dim=1) > 0
    if valid.sum() == 0:
        return z.sum() * 0.0

    logits_masked = logits.masked_fill(self_mask, -1e9)
    log_prob = logits_masked - torch.logsumexp(logits_masked, dim=1, keepdim=True)
    mean_log_prob_pos = (pos_weight * log_prob).sum(dim=1) / pos_weight.sum(dim=1).clamp(min=1e-12)
    return -mean_log_prob_pos[valid].mean()


def packet_to_flow_prototype_loss(
    z: torch.Tensor,
    labels: torch.Tensor,
    flow_ids: torch.Tensor,
    temperature: float = 0.07,
    positive_mode: str = "same_class",
    context_mode: str = "inclusive",
) -> torch.Tensor:
    """Contrast packets against batch-local flow prototypes.

    This is a flow-level companion to packet-packet SupCon: packets are pulled
    toward their own-flow or same-class flow prototypes and pushed away from
    different-class flow prototypes. ``leave_one_out`` removes the anchor
    packet from its own-flow prototype, preventing a self-inclusion shortcut.
    """
    if context_mode not in {"inclusive", "leave_one_out"}:
        raise ValueError("context_mode must be inclusive or leave_one_out")
    if z.size(0) <= 1:
        return z.sum() * 0.0
    z = F.normalize(z.float(), p=2, dim=-1)
    labels = labels.long()
    flow_ids = flow_ids.long()
    unique_flows, inverse = torch.unique(flow_ids, sorted=True, return_inverse=True)
    if unique_flows.numel() <= 1:
        return z.sum() * 0.0

    proto_sum = z.new_zeros((unique_flows.numel(), z.size(-1)))
    proto_sum.index_add_(0, inverse, z)
    counts = torch.bincount(inverse, minlength=unique_flows.numel()).to(z.device, z.dtype).clamp(min=1.0)
    prototypes = F.normalize(proto_sum / counts.unsqueeze(1), p=2, dim=-1)

    proto_labels = labels.new_full((unique_flows.numel(),), -1)
    for proto_idx in range(unique_flows.numel()):
        flow_label_values = labels[inverse == proto_idx]
        proto_labels[proto_idx] = torch.mode(flow_label_values).values

    temperature = max(float(temperature), 1e-6)
    logits = torch.matmul(z, prototypes.T) / temperature
    candidate_mask = torch.ones_like(logits, dtype=torch.bool)
    if context_mode == "leave_one_out":
        own_counts = counts[inverse] - 1.0
        has_own_context = own_counts > 0
        own_sum = proto_sum[inverse] - z
        own_prototypes = F.normalize(
            own_sum / own_counts.clamp(min=1.0).unsqueeze(1), p=2, dim=-1
        )
        own_logits = (z * own_prototypes).sum(dim=-1) / temperature
        own_columns = F.one_hot(inverse, num_classes=unique_flows.numel()).bool()
        logits = torch.where(own_columns, own_logits.unsqueeze(1), logits)
        candidate_mask = candidate_mask & (~own_columns | has_own_context.unsqueeze(1))
        logits = logits.masked_fill(~candidate_mask, -1e9)
    else:
        has_own_context = torch.ones_like(inverse, dtype=torch.bool)
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    proto_ids = torch.arange(unique_flows.numel(), device=z.device)
    own_flow_mask = proto_ids.unsqueeze(0) == inverse.unsqueeze(1)
    if positive_mode == "own_flow":
        pos_mask = own_flow_mask & has_own_context.unsqueeze(1)
    else:
        pos_mask = (
            labels.view(-1, 1).eq(proto_labels.view(1, -1)) | own_flow_mask
        ) & candidate_mask
    valid = pos_mask.sum(dim=1) > 0
    if valid.sum() == 0:
        return z.sum() * 0.0

    pos_logits = logits.masked_fill(~pos_mask, -1e9)
    return -(torch.logsumexp(pos_logits[valid], dim=1) - torch.logsumexp(logits[valid], dim=1)).mean()


def paired_view_consistency_loss(
    z: torch.Tensor,
    paired_z: torch.Tensor,
    logits: torch.Tensor,
    paired_logits: torch.Tensor,
    logit_kl_weight: float = 0.5,
    raw_z: torch.Tensor | None = None,
    paired_raw_z: torch.Tensor | None = None,
    raw_consistency_weight: float = 1.0,
) -> torch.Tensor:
    """Keep full-header and randomized/masked-header packet views close.

    Projected and raw last-token representations are both constrained because
    downstream concat embeddings expose both. The symmetric KL term keeps class
    evidence stable without requiring the paired view to be a separate expert.
    """
    if z.numel() == 0:
        return logits.sum() * 0.0
    z = F.normalize(z.float(), p=2, dim=-1)
    paired_z = F.normalize(paired_z.float(), p=2, dim=-1)
    rep_loss = (1.0 - (z * paired_z).sum(dim=-1)).mean()
    if (raw_z is None) != (paired_raw_z is None):
        raise ValueError("raw paired consistency requires both raw_z and paired_raw_z")
    if raw_consistency_weight < 0:
        raise ValueError("raw_consistency_weight must be non-negative")
    if raw_z is not None and raw_consistency_weight > 0:
        raw_z = F.normalize(raw_z.float(), p=2, dim=-1)
        paired_raw_z = F.normalize(paired_raw_z.float(), p=2, dim=-1)
        raw_loss = (1.0 - (raw_z * paired_raw_z).sum(dim=-1)).mean()
        rep_loss = rep_loss + float(raw_consistency_weight) * raw_loss
    if logit_kl_weight <= 0:
        return rep_loss
    p_log = F.log_softmax(logits.float(), dim=-1)
    q_log = F.log_softmax(paired_logits.float(), dim=-1)
    p = p_log.exp()
    q = q_log.exp()
    kl = 0.5 * (
        F.kl_div(p_log, q, reduction="batchmean")
        + F.kl_div(q_log, p, reduction="batchmean")
    )
    return rep_loss + float(logit_kl_weight) * kl
