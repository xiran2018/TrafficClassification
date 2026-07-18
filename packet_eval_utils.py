from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from tqdm import tqdm


def _is_cuda_oom(error: RuntimeError) -> bool:
    return isinstance(error, torch.OutOfMemoryError) or "out of memory" in str(error).lower()


def encode_packet_logits_with_backoff(model, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Evaluate one batch, recursively reducing it when shared-GPU memory is tight."""
    try:
        _, _, logits = model.encode_packets(input_ids, attention_mask)
        return logits
    except RuntimeError as error:
        if not _is_cuda_oom(error) or len(input_ids) <= 1:
            raise
        if input_ids.is_cuda:
            torch.cuda.empty_cache()
        midpoint = len(input_ids) // 2
        left = encode_packet_logits_with_backoff(
            model, input_ids[:midpoint], attention_mask[:midpoint]
        )
        right = encode_packet_logits_with_backoff(
            model, input_ids[midpoint:], attention_mask[midpoint:]
        )
        return torch.cat([left, right], dim=0)


def packet_classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    num_classes: int,
    label_names: Sequence[str] | None = None,
) -> Dict:
    labels = list(range(num_classes))
    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_pred_arr = np.asarray(y_pred, dtype=np.int64)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_arr, y_pred_arr, labels=labels, zero_division=0
    )
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
        y_true_arr, y_pred_arr, labels=labels, average="macro", zero_division=0
    )
    p_weighted, r_weighted, f_weighted, _ = precision_recall_fscore_support(
        y_true_arr, y_pred_arr, labels=labels, average="weighted", zero_division=0
    )
    names = list(label_names) if label_names is not None else [str(i) for i in labels]
    per_class = {
        names[i]: {
            "label_id": i,
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i in labels
    }
    return {
        "num_samples": int(len(y_true_arr)),
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "macro_precision": float(p_macro),
        "macro_recall": float(r_macro),
        "macro_f1": float(f_macro),
        "weighted_precision": float(p_weighted),
        "weighted_recall": float(r_weighted),
        "weighted_f1": float(f_weighted),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(y_true_arr, y_pred_arr, labels=labels).tolist(),
    }


@torch.no_grad()
def evaluate_packet_model(model, loader, device: torch.device, num_classes: int, label_names: Sequence[str] | None = None, desc: str = "eval packets", return_probabilities: bool = False) -> Dict:
    was_training = model.training
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []
    losses: List[float] = []
    probabilities: List[List[float]] = []
    for batch in tqdm(loader, desc=desc, leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        logits = encode_packet_logits_with_backoff(model, input_ids, attention_mask)
        losses.append(float(F.cross_entropy(logits, labels).detach().cpu()))
        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(logits.argmax(dim=-1).detach().cpu().tolist())
        if return_probabilities:
            probabilities.extend(torch.softmax(logits.float(), dim=-1).detach().cpu().tolist())
    metrics = packet_classification_metrics(y_true, y_pred, num_classes, label_names)
    metrics["loss"] = float(np.mean(losses)) if losses else 0.0
    metrics["y_true"] = y_true
    metrics["y_pred"] = y_pred
    if return_probabilities:
        metrics["probabilities"] = probabilities
    if was_training:
        model.train()
    return metrics
