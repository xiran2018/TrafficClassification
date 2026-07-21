#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from torch import nn

from probability_metrics import calibration_metrics
from validation_gated_selector import apply_unified_expert_slots, parse_name_list

REQUIRED_PROB_FIELDS = ["valid_flow_ids", "valid_y_true", "valid_prob", "flow_ids", "flow_y_true", "flow_prob"]


def compute_metrics(y_true, y_pred):
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    p_weight, r_weight, f_weight, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    return {
        "accuracy": accuracy_score(y_true, y_pred) if len(y_true) else 0.0,
        "macro_precision": p_macro,
        "macro_recall": r_macro,
        "macro_f1": f_macro,
        "weighted_precision": p_weight,
        "weighted_recall": r_weight,
        "weighted_f1": f_weight,
    }


def load_label_names(path: str):
    if not path:
        return None, None
    with open(path, "r", encoding="utf-8") as f:
        label_map = json.load(f)
    if not label_map:
        return None, label_map
    max_id = max(int(v) for v in label_map.values())
    label_names = [str(i) for i in range(max_id + 1)]
    for name, idx in label_map.items():
        label_names[int(idx)] = name
    return label_names, label_map


def load_prob_payload(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    missing = [key for key in REQUIRED_PROB_FIELDS if key not in data]
    if missing:
        raise ValueError(f"{path} is missing required probability fields: {missing}")
    return data


def load_named_prob_payloads(raw_inputs: List[List[str]]) -> Tuple[List[Tuple[str, Dict[str, Any], str]], List[Dict[str, Any]]]:
    named_payloads: List[Tuple[str, Dict[str, Any], str]] = []
    skipped: List[Dict[str, Any]] = []
    for name, path in raw_inputs:
        try:
            named_payloads.append((name, load_prob_payload(path), path))
        except (FileNotFoundError, json.JSONDecodeError, ValueError) as exc:
            skipped.append({"name": name, "path": path, "reason": str(exc)})
            print("skip_expert_gate_input", json.dumps(skipped[-1], sort_keys=True))
    if not named_payloads:
        raise ValueError("No expert-gate inputs contain the required valid/test probability fields.")
    return named_payloads, skipped


def _id_set(data: Dict[str, Any], split: str) -> set[str]:
    key = "valid_flow_ids" if split == "valid" else "flow_ids"
    return set(map(str, data[key]))


def filter_compatible_inputs(
    named_payloads: List[Tuple[str, Dict[str, Any], str]],
) -> Tuple[List[Tuple[str, Dict[str, Any], str]], List[Dict[str, Any]]]:
    if not named_payloads:
        return named_payloads, []
    kept = [named_payloads[0]]
    skipped: List[Dict[str, Any]] = []
    valid_common = _id_set(named_payloads[0][1], "valid")
    test_common = _id_set(named_payloads[0][1], "test")
    for name, data, path in named_payloads[1:]:
        next_valid = valid_common & _id_set(data, "valid")
        next_test = test_common & _id_set(data, "test")
        if next_valid and next_test:
            kept.append((name, data, path))
            valid_common = next_valid
            test_common = next_test
        else:
            reason = (
                f"incompatible flow ids with fallback group: "
                f"valid_overlap={len(next_valid)}, test_overlap={len(next_test)}"
            )
            skipped.append({"name": name, "path": path, "reason": reason})
            print("skip_expert_gate_input", json.dumps(skipped[-1], sort_keys=True))
    return kept, skipped


def align_prob(data: Dict[str, Any], split: str, fids: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    if split == "valid":
        ids = data["valid_flow_ids"]
        labels = data["valid_y_true"]
        probs = data["valid_prob"]
    elif split == "test":
        ids = data["flow_ids"]
        labels = data["flow_y_true"]
        probs = data["flow_prob"]
    else:
        raise ValueError(split)
    idx = {str(fid): i for i, fid in enumerate(ids)}
    y = np.asarray([labels[idx[fid]] for fid in fids], dtype=np.int64)
    p = np.asarray([probs[idx[fid]] for fid in fids], dtype=np.float32)
    p = np.clip(p, 1e-12, None)
    p = p / p.sum(axis=1, keepdims=True).clip(min=1e-12)
    return y, p


def gate_features(prob_list: List[np.ndarray]) -> np.ndarray:
    parts = []
    for prob in prob_list:
        sorted_prob = np.sort(prob, axis=1)
        conf = sorted_prob[:, -1:]
        margin = (sorted_prob[:, -1:] - sorted_prob[:, -2:-1]) if prob.shape[1] > 1 else conf
        entropy = -np.sum(prob * np.log(prob.clip(min=1e-12)), axis=1, keepdims=True)
        entropy = entropy / max(float(np.log(prob.shape[1])), 1e-12)
        parts.append(np.concatenate([conf, margin, entropy], axis=1).astype(np.float32))
    return np.concatenate(parts, axis=1)


class ExpertGate(nn.Module):
    def __init__(self, input_dim: int, num_experts: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_experts),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.net(x), dim=-1)


def fuse_with_gate(model: ExpertGate, x: np.ndarray, probs: List[np.ndarray], device: str) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        weights = model(torch.from_numpy(x).to(device)).cpu().numpy().astype(np.float32)
    stacked = np.stack(probs, axis=1).astype(np.float32)
    fused = (weights[:, :, None] * stacked).sum(axis=1)
    fused = np.clip(fused, 1e-12, None)
    fused = fused / fused.sum(axis=1, keepdims=True).clip(min=1e-12)
    return fused.astype(np.float32), weights


def train_gate(
    x: np.ndarray,
    probs: List[np.ndarray],
    y: np.ndarray,
    hidden_dim: int,
    dropout: float,
    epochs: int,
    lr: float,
    weight_decay: float,
    entropy_weight: float,
    seed: int,
    device: str,
) -> ExpertGate:
    torch.manual_seed(seed)
    model = ExpertGate(x.shape[1], len(probs), hidden_dim, dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    tx = torch.from_numpy(x).to(device)
    ty = torch.from_numpy(y.astype(np.int64)).to(device)
    tprobs = torch.from_numpy(np.stack(probs, axis=1).astype(np.float32)).to(device)
    for _ in range(epochs):
        model.train()
        weights = model(tx)
        fused = (weights.unsqueeze(-1) * tprobs).sum(dim=1).clamp_min(1e-12)
        nll = -torch.log(fused[torch.arange(len(ty), device=device), ty]).mean()
        entropy = -(weights.clamp_min(1e-12) * weights.clamp_min(1e-12).log()).sum(dim=1).mean()
        loss = nll + entropy_weight * entropy
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
    return model


def oof_gate_predictions(
    x: np.ndarray,
    probs: List[np.ndarray],
    y: np.ndarray,
    args,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    num_classes = probs[0].shape[1]
    num_experts = len(probs)
    class_counts = np.bincount(y, minlength=num_classes)
    min_class_count = int(class_counts[class_counts > 0].min()) if np.any(class_counts > 0) else 0
    if min_class_count < 2:
        model = train_gate(
            x,
            probs,
            y,
            args.hidden_dim,
            args.dropout,
            args.epochs,
            args.lr,
            args.weight_decay,
            args.entropy_weight,
            args.seed,
            args.device,
        )
        fused, weights = fuse_with_gate(model, x, probs, args.device)
        return fused, weights, {"mode": "full_train_valid", "n_splits": 1, "reason": "min_class_count_lt_2"}

    n_splits = max(2, min(args.cv_splits, min_class_count))
    valid_prob = np.zeros((len(y), num_classes), dtype=np.float32)
    valid_weights = np.zeros((len(y), num_experts), dtype=np.float32)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
    for fold, (train_idx, val_idx) in enumerate(skf.split(x, y)):
        fold_model = train_gate(
            x[train_idx],
            [p[train_idx] for p in probs],
            y[train_idx],
            args.hidden_dim,
            args.dropout,
            args.epochs,
            args.lr,
            args.weight_decay,
            args.entropy_weight,
            args.seed + fold,
            args.device,
        )
        fold_prob, fold_weights = fuse_with_gate(fold_model, x[val_idx], [p[val_idx] for p in probs], args.device)
        valid_prob[val_idx] = fold_prob
        valid_weights[val_idx] = fold_weights
    return valid_prob, valid_weights, {"mode": "stratified_oof", "n_splits": int(n_splits)}


def gate_summary(names: List[str], weights: np.ndarray) -> Dict[str, Any]:
    mean = weights.mean(axis=0)
    std = weights.std(axis=0)
    entropy = -(weights.clip(min=1e-12) * np.log(weights.clip(min=1e-12))).sum(axis=1)
    norm_entropy = entropy / max(float(np.log(weights.shape[1])), 1e-12)
    return {
        "experts": names,
        "mean": [float(x) for x in mean],
        "std": [float(x) for x in std],
        "dominant_expert": names[int(mean.argmax())] if names else "",
        "dominant_weight": float(mean.max()) if len(mean) else 0.0,
        "normalized_entropy_mean": float(norm_entropy.mean()) if len(norm_entropy) else 0.0,
        "effective_experts_mean": float(np.exp(entropy).mean()) if len(entropy) else 0.0,
        "num_samples": int(weights.shape[0]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a soft expert gate over unified probability expert slots.")
    ap.add_argument("--input", nargs=2, action="append", metavar=("NAME", "JSON"), required=True)
    ap.add_argument("--label_map", default="")
    ap.add_argument("--select_metric", choices=["accuracy", "macro_f1"], default="macro_f1")
    ap.add_argument("--hidden_dim", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--weight_decay", type=float, default=0.001)
    ap.add_argument("--entropy_weight", type=float, default=0.0, help="Positive values encourage sparse expert weights.")
    ap.add_argument("--cv_splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument(
        "--unified_expert_slots",
        default="",
        help="Comma-separated expert slots. Missing slots are filled with the base input as identity experts.",
    )
    ap.add_argument("--output_json", required=True)
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    named_payloads, skipped_inputs = load_named_prob_payloads(args.input)
    unified_expert_slots = parse_name_list(args.unified_expert_slots)
    named_payloads, input_slot_status = apply_unified_expert_slots(named_payloads, unified_expert_slots)
    named_payloads, incompatible_inputs = filter_compatible_inputs(named_payloads)
    skipped_inputs.extend(incompatible_inputs)
    names = [name for name, _, _ in named_payloads]

    valid_common = sorted(set.intersection(*(set(map(str, data["valid_flow_ids"])) for _, data, _ in named_payloads)))
    test_common = sorted(set.intersection(*(set(map(str, data["flow_ids"])) for _, data, _ in named_payloads)))
    if not valid_common or not test_common:
        raise ValueError("No common flow ids across expert-gate inputs.")

    valid_probs = []
    test_probs = []
    y_valid_ref = None
    y_test_ref = None
    for name, data, _ in named_payloads:
        y_valid, p_valid = align_prob(data, "valid", valid_common)
        y_test, p_test = align_prob(data, "test", test_common)
        if y_valid_ref is None:
            y_valid_ref = y_valid
            y_test_ref = y_test
        elif not np.array_equal(y_valid_ref, y_valid) or not np.array_equal(y_test_ref, y_test):
            raise ValueError(f"Labels do not align for input {name}.")
        valid_probs.append(p_valid)
        test_probs.append(p_test)

    x_valid = gate_features(valid_probs)
    x_test = gate_features(test_probs)
    valid_prob, valid_weights, cv_info = oof_gate_predictions(x_valid, valid_probs, y_valid_ref, args)
    valid_pred = valid_prob.argmax(axis=1).astype(np.int64)
    valid_metrics = compute_metrics(y_valid_ref.tolist(), valid_pred.tolist())
    valid_metrics["calibration"] = calibration_metrics(y_valid_ref, valid_prob)

    final_model = train_gate(
        x_valid,
        valid_probs,
        y_valid_ref,
        args.hidden_dim,
        args.dropout,
        args.epochs,
        args.lr,
        args.weight_decay,
        args.entropy_weight,
        args.seed,
        args.device,
    )
    test_prob, test_weights = fuse_with_gate(final_model, x_test, test_probs, args.device)
    test_pred = test_prob.argmax(axis=1).astype(np.int64)
    test_metrics = compute_metrics(y_test_ref.tolist(), test_pred.tolist())
    test_metrics["calibration"] = calibration_metrics(y_test_ref, test_prob)
    selected = {
        "strategy": "soft_expert_gate",
        "config": {
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "entropy_weight": args.entropy_weight,
            "cv": cv_info,
        },
        "metrics": valid_metrics,
        "valid_gate": gate_summary(names, valid_weights),
        "test_gate": gate_summary(names, test_weights),
    }

    print("selected_expert_gate", json.dumps(selected, sort_keys=True))
    print("test_expert_gate", json.dumps(test_metrics, indent=2, sort_keys=True))
    label_names, label_map = load_label_names(args.label_map)
    if label_names:
        print(classification_report(y_test_ref, test_pred, labels=list(range(len(label_names))), target_names=label_names, zero_division=0))
    else:
        print(classification_report(y_test_ref, test_pred, zero_division=0))

    payload = {
        "metrics": {
            "flow_level": test_metrics,
            "eval_config": {"soft_expert_gate": selected["test_gate"]},
        },
        "selected": selected,
        "valid_reports": [selected],
        "inputs": [{"name": name, "path": path} for name, _, path in named_payloads],
        "label_map": label_map,
        "flow_ids": test_common,
        "flow_y_true": y_test_ref.tolist(),
        "flow_y_pred": test_pred.tolist(),
        "flow_prob": test_prob.tolist(),
        "flow_source": test_weights.argmax(axis=1).astype(int).tolist(),
        "flow_gate_weight": test_weights.tolist(),
        "valid_flow_ids": valid_common,
        "valid_y_true": y_valid_ref.tolist(),
        "valid_y_pred": valid_pred.tolist(),
        "valid_prob": valid_prob.tolist(),
        "valid_source": valid_weights.argmax(axis=1).astype(int).tolist(),
        "valid_gate_weight": valid_weights.tolist(),
        "feature_config": {
            "select_metric": args.select_metric,
            "unified_expert_slots": unified_expert_slots,
            "input_slot_status": input_slot_status,
            "skipped_inputs": skipped_inputs,
            "gate_features": "per_expert_confidence_margin_entropy",
            "cv": cv_info,
        },
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()
