#!/usr/bin/env python3
"""Learn one evidence router across validation environments.

Each environment exposes the same two expert slots: a semantic neural expert and
a protocol-structural expert. The router is trained once over all environments
with GroupDRO and is then applied independently before fixed cross-environment
log-mean aggregation. Test labels are used only for the final report.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch import nn

from probability_metrics import calibration_metrics
from calibrate_prediction_prior import estimate_prior_em


EPS = 1e-12


def normalize_prob(values: Any) -> np.ndarray:
    prob = np.asarray(values, dtype=np.float32)
    if prob.ndim != 2:
        raise ValueError("expert probabilities must be a rank-2 array")
    prob = np.clip(prob, EPS, None)
    return prob / np.clip(prob.sum(axis=1, keepdims=True), EPS, None)


def metrics(y_true: np.ndarray, prob: np.ndarray) -> dict[str, Any]:
    pred = prob.argmax(axis=1)
    macro = precision_recall_fscore_support(
        y_true, pred, average="macro", zero_division=0
    )
    weighted = precision_recall_fscore_support(
        y_true, pred, average="weighted", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "macro_precision": float(macro[0]),
        "macro_recall": float(macro[1]),
        "macro_f1": float(macro[2]),
        "weighted_precision": float(weighted[0]),
        "weighted_recall": float(weighted[1]),
        "weighted_f1": float(weighted[2]),
        "calibration": calibration_metrics(y_true, prob),
    }


def uncertainty_features(prob: np.ndarray) -> np.ndarray:
    ordered = np.sort(prob, axis=1)
    confidence = ordered[:, -1:]
    margin = ordered[:, -1:] - ordered[:, -2:-1]
    entropy = -np.sum(prob * np.log(np.clip(prob, EPS, None)), axis=1, keepdims=True)
    entropy /= max(float(np.log(prob.shape[1])), EPS)
    return np.concatenate([confidence, margin, entropy], axis=1)


def router_features(semantic: np.ndarray, structural: np.ndarray) -> np.ndarray:
    """Class-aware evidence and conflict features without dataset-specific ids."""
    semantic = normalize_prob(semantic)
    structural = normalize_prob(structural)
    if semantic.shape != structural.shape:
        raise ValueError("semantic and structural probabilities must have equal shape")
    midpoint = 0.5 * (semantic + structural)
    js = 0.5 * np.sum(
        semantic * (np.log(semantic) - np.log(midpoint))
        + structural * (np.log(structural) - np.log(midpoint)),
        axis=1,
        keepdims=True,
    )
    agreement = (semantic.argmax(axis=1) == structural.argmax(axis=1)).astype(np.float32)[:, None]
    return np.concatenate(
        [
            semantic,
            structural,
            np.abs(semantic - structural),
            semantic * structural,
            uncertainty_features(semantic),
            uncertainty_features(structural),
            agreement,
            js.astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)


def aligned_split(payload: dict[str, Any], split: str, flow_ids: list[str]):
    if split == "valid":
        ids = [str(value) for value in payload["valid_flow_ids"]]
        labels = payload["valid_y_true"]
        probabilities = payload["valid_prob"]
    else:
        ids = [str(value) for value in payload["flow_ids"]]
        labels = payload["flow_y_true"]
        probabilities = payload["flow_prob"]
    index = {flow_id: idx for idx, flow_id in enumerate(ids)}
    if len(index) != len(ids):
        raise ValueError(f"duplicate flow ids in {split} payload")
    missing = [flow_id for flow_id in flow_ids if flow_id not in index]
    if missing:
        raise ValueError(f"{split} payload misses {len(missing)} aligned flows")
    rows = [index[flow_id] for flow_id in flow_ids]
    return (
        np.asarray([labels[row] for row in rows], dtype=np.int64),
        normalize_prob([probabilities[row] for row in rows]),
    )


def direct_split(payload: dict[str, Any]):
    ids = [str(value) for value in payload["flow_ids"]]
    labels = np.asarray(payload["flow_y_true"], dtype=np.int64)
    prob = normalize_prob(payload["flow_prob"])
    if len(ids) != len(set(ids)) or len(ids) != len(labels):
        raise ValueError("direct prediction payload has invalid flow ids")
    return ids, labels, prob


def packet_split(path: str):
    """Load strict one-packet probabilities with an alignment identity."""
    with np.load(path, allow_pickle=False) as payload:
        if "y_true" not in payload or "probabilities" not in payload:
            raise ValueError(f"{path}: packet payload needs y_true and probabilities")
        labels = np.asarray(payload["y_true"], dtype=np.int64)
        prob = normalize_prob(payload["probabilities"])
        if "packet_uids" in payload:
            ids = [str(value) for value in payload["packet_uids"]]
        else:
            ids = [f"packet:{index}" for index in range(len(labels))]
    if len(ids) != len(set(ids)) or len(ids) != len(labels):
        raise ValueError(f"{path}: packet payload has invalid sample alignment")
    return ids, labels, prob


@dataclass
class Environment:
    name: str
    valid_ids: list[str]
    valid_y: np.ndarray
    valid_semantic: np.ndarray
    valid_structural: np.ndarray
    test_y: np.ndarray
    test_ids: list[str]
    test_semantic: np.ndarray
    test_structural: np.ndarray


def load_environment(spec: list[str]) -> Environment:
    name, semantic_valid_path, semantic_test_path, structural_path = spec
    semantic_valid = json.loads(Path(semantic_valid_path).read_text())
    semantic_test = json.loads(Path(semantic_test_path).read_text())
    structural = json.loads(Path(structural_path).read_text())

    valid_ids, valid_y, valid_semantic = direct_split(semantic_valid)
    structural_valid_y, valid_structural = aligned_split(structural, "valid", valid_ids)
    test_ids, test_y, test_semantic = direct_split(semantic_test)
    structural_test_y, test_structural = aligned_split(structural, "test", test_ids)
    if not np.array_equal(valid_y, structural_valid_y):
        raise ValueError(f"{name}: validation labels differ across expert slots")
    if not np.array_equal(test_y, structural_test_y):
        raise ValueError(f"{name}: test labels differ across expert slots")
    return Environment(
        name,
        valid_ids,
        valid_y,
        valid_semantic,
        valid_structural,
        test_y,
        test_ids,
        test_semantic,
        test_structural,
    )


def load_packet_environment(spec: list[str]) -> Environment:
    name, semantic_valid_path, semantic_test_path, structural_valid_path, structural_test_path = spec
    valid_ids, valid_y, valid_semantic = packet_split(semantic_valid_path)
    structural_valid_ids, structural_valid_y, valid_structural = packet_split(
        structural_valid_path
    )
    test_ids, test_y, test_semantic = packet_split(semantic_test_path)
    structural_test_ids, structural_test_y, test_structural = packet_split(
        structural_test_path
    )
    if valid_ids != structural_valid_ids or test_ids != structural_test_ids:
        raise ValueError(f"{name}: packet identities differ across expert slots")
    if not np.array_equal(valid_y, structural_valid_y):
        raise ValueError(f"{name}: validation labels differ across expert slots")
    if not np.array_equal(test_y, structural_test_y):
        raise ValueError(f"{name}: test labels differ across expert slots")
    return Environment(
        name,
        valid_ids,
        valid_y,
        valid_semantic,
        valid_structural,
        test_y,
        test_ids,
        test_semantic,
        test_structural,
    )


class ReliabilityRouter(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(features)).squeeze(-1)


def fit_router(environments: list[Environment], args, seed: int):
    if not environments:
        raise ValueError("at least one training environment is required")
    np.random.seed(seed)
    torch.manual_seed(seed)
    raw_features = [router_features(env.valid_semantic, env.valid_structural) for env in environments]
    joined = np.concatenate(raw_features, axis=0)
    mean = joined.mean(axis=0).astype(np.float32)
    std = joined.std(axis=0).astype(np.float32)
    std[std < 1e-4] = 1.0
    features = [torch.from_numpy((values - mean) / std).to(args.device) for values in raw_features]
    semantic = [torch.from_numpy(env.valid_semantic).to(args.device) for env in environments]
    structural = [torch.from_numpy(env.valid_structural).to(args.device) for env in environments]
    labels = [torch.from_numpy(env.valid_y).to(args.device) for env in environments]

    model = ReliabilityRouter(joined.shape[1], args.hidden_dim, args.dropout).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    group_weights = torch.full(
        (len(environments),), 1.0 / len(environments), device=args.device
    )
    for _ in range(args.epochs):
        losses = []
        gate_entropies = []
        for x, semantic_prob, structural_prob, y in zip(
            features, semantic, structural, labels
        ):
            structural_weight = model(x)
            fused = (
                (1.0 - structural_weight[:, None]) * semantic_prob
                + structural_weight[:, None] * structural_prob
            ).clamp_min(EPS)
            losses.append(-torch.log(fused[torch.arange(len(y), device=args.device), y]).mean())
            gate_entropies.append(
                -(
                    structural_weight.clamp(EPS, 1.0 - EPS) * structural_weight.clamp(EPS, 1.0 - EPS).log()
                    + (1.0 - structural_weight).clamp(EPS, 1.0 - EPS)
                    * (1.0 - structural_weight).clamp(EPS, 1.0 - EPS).log()
                ).mean()
            )
        loss_vector = torch.stack(losses)
        with torch.no_grad():
            group_weights *= torch.exp(args.groupdro_eta * loss_vector.detach())
            group_weights /= group_weights.sum().clamp_min(EPS)
        objective = torch.sum(group_weights * loss_vector)
        if args.gate_entropy_weight:
            objective += args.gate_entropy_weight * torch.stack(gate_entropies).mean()
        optimizer.zero_grad()
        objective.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
    return model, mean, std, group_weights.detach().cpu().numpy()


def route(model, mean, std, semantic, structural, device):
    features = (router_features(semantic, structural) - mean) / std
    model.eval()
    with torch.no_grad():
        weight = model(torch.from_numpy(features).to(device)).cpu().numpy()
    fused = (1.0 - weight[:, None]) * semantic + weight[:, None] * structural
    fused = normalize_prob(fused)
    return fused, weight


def aligned_test_consensus(outputs):
    common = sorted(set.intersection(*(set(ids) for ids, _, _, _ in outputs)))
    if not common:
        raise ValueError("environments have no common test flows")
    labels = None
    log_prob = []
    weights = []
    for ids, y, prob, gate in outputs:
        index = {flow_id: idx for idx, flow_id in enumerate(ids)}
        rows = [index[flow_id] for flow_id in common]
        current_y = y[rows]
        if labels is None:
            labels = current_y
        elif not np.array_equal(labels, current_y):
            raise ValueError("test labels differ across environments")
        log_prob.append(np.log(np.clip(prob[rows], EPS, None)))
        weights.append(gate[rows])
    consensus = np.exp(np.mean(log_prob, axis=0))
    consensus = normalize_prob(consensus)
    return common, labels, consensus, np.stack(weights, axis=1)


def safe_prior_transport(
    valid_y: np.ndarray,
    valid_prob: np.ndarray,
    test_prob: np.ndarray,
    max_strength: float,
    strength_step: float,
    macro_plateau_tolerance: float = 1e-3,
):
    """Select bounded target-prior transport using cross-environment OOF risk."""
    if max_strength <= 0:
        return valid_prob, test_prob, {
            "enabled": False,
            "selected_strength": 0.0,
        }
    num_classes = test_prob.shape[1]
    source_prior = np.bincount(valid_y, minlength=num_classes).astype(np.float64)
    source_prior /= max(float(source_prior.sum()), 1.0)
    target_prior = estimate_prior_em(test_prob, source_prior)
    correction = np.log(target_prior + EPS) - np.log(source_prior + EPS)
    sample_weight = target_prior[valid_y] / np.maximum(source_prior[valid_y], EPS)
    strengths = np.arange(0.0, max_strength + 0.5 * strength_step, strength_step)
    reports = []
    candidates = []
    for strength in strengths:
        adjusted_valid = normalize_prob(
            np.exp(np.log(np.clip(valid_prob, EPS, None)) + strength * correction[None, :])
        )
        adjusted_test = normalize_prob(
            np.exp(np.log(np.clip(test_prob, EPS, None)) + strength * correction[None, :])
        )
        pred = adjusted_valid.argmax(axis=1)
        weighted_accuracy = float(
            np.average(pred == valid_y, weights=sample_weight)
        )
        weighted_macro_f1 = float(
            precision_recall_fscore_support(
                valid_y,
                pred,
                average="macro",
                zero_division=0,
                sample_weight=sample_weight,
            )[2]
        )
        macro_f1 = float(
            precision_recall_fscore_support(
                valid_y,
                pred,
                average="macro",
                zero_division=0,
            )[2]
        )
        reports.append(
            {
                "strength": float(strength),
                "weighted_accuracy": weighted_accuracy,
                "weighted_macro_f1": weighted_macro_f1,
                "macro_f1": macro_f1,
            }
        )
        candidates.append((adjusted_valid, adjusted_test))
    baseline = reports[0]
    best_macro_f1 = max(report["macro_f1"] for report in reports)
    admissible = [
        index
        for index, report in enumerate(reports)
        if report["macro_f1"] >= best_macro_f1 - macro_plateau_tolerance
        and report["weighted_accuracy"] >= baseline["weighted_accuracy"] - 1e-12
    ]
    selected_index = max(
        admissible,
        key=lambda index: (
            reports[index]["weighted_macro_f1"],
            reports[index]["weighted_accuracy"],
            -reports[index]["strength"],
        ),
    )
    return candidates[selected_index][0], candidates[selected_index][1], {
        "enabled": True,
        "selection_scope": "cross_environment_oof_macro_f1_plateau",
        "selected_strength": reports[selected_index]["strength"],
        "max_strength": float(max_strength),
        "strength_step": float(strength_step),
        "macro_plateau_tolerance": float(macro_plateau_tolerance),
        "source_prior": source_prior.tolist(),
        "estimated_target_prior": target_prior.tolist(),
        "reports": reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--environment",
        nargs=4,
        action="append",
        metavar=("NAME", "SEMANTIC_VALID", "SEMANTIC_TEST", "STRUCTURAL_VALID_TEST"),
    )
    parser.add_argument(
        "--packet_environment",
        nargs=5,
        action="append",
        metavar=(
            "NAME",
            "SEMANTIC_VALID_NPZ",
            "SEMANTIC_TEST_NPZ",
            "STRUCTURAL_VALID_NPZ",
            "STRUCTURAL_TEST_NPZ",
        ),
        help="Strict one-packet expert probabilities for one training environment.",
    )
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--groupdro_eta", type=float, default=0.05)
    parser.add_argument("--gate_entropy_weight", type=float, default=0.01)
    parser.add_argument(
        "--prior_max_strength",
        type=float,
        default=0.0,
        help="Maximum bounded EM prior-transfer strength. Selection uses pooled leave-one-environment-out predictions only.",
    )
    parser.add_argument("--prior_strength_step", type=float, default=0.05)
    parser.add_argument(
        "--prior_macro_plateau_tolerance",
        type=float,
        default=1e-3,
        help="Maximum LOEO Macro-F1 gap admitted to the target-weighted selection plateau.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument(
        "--output_npz",
        default="",
        help="Packet probability artifact; defaults to output_json with an .npz suffix.",
    )
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    if bool(args.environment) == bool(args.packet_environment):
        parser.error("provide exactly one of --environment or --packet_environment")
    task_scope = "packet_level" if args.packet_environment else "flow_level"
    environment_specs = args.packet_environment or args.environment
    loader = load_packet_environment if args.packet_environment else load_environment
    environments = [loader(spec) for spec in environment_specs]
    leave_one_environment_out = []
    oof_valid_ids = []
    oof_valid_y = []
    oof_valid_prob = []
    for held_out_index, held_out in enumerate(environments):
        train_environments = [
            env for index, env in enumerate(environments) if index != held_out_index
        ]
        if not train_environments:
            break
        model, mean, std, group_weights = fit_router(
            train_environments, args, args.seed + held_out_index + 1
        )
        prob, gate = route(
            model,
            mean,
            std,
            held_out.valid_semantic,
            held_out.valid_structural,
            args.device,
        )
        leave_one_environment_out.append(
            {
                "environment": held_out.name,
                "metrics": metrics(held_out.valid_y, prob),
                "structural_weight_mean": float(gate.mean()),
                "train_group_weights": group_weights.tolist(),
            }
        )
        oof_valid_ids.extend(
            [f"{held_out.name}::{flow_id}" for flow_id in held_out.valid_ids]
        )
        oof_valid_y.extend(held_out.valid_y.tolist())
        oof_valid_prob.extend(prob.tolist())

    model, mean, std, group_weights = fit_router(environments, args, args.seed)
    test_outputs = []
    per_environment = []
    for env in environments:
        prob, gate = route(
            model, mean, std, env.test_semantic, env.test_structural, args.device
        )
        test_outputs.append((env.test_ids, env.test_y, prob, gate))
        per_environment.append(
            {
                "environment": env.name,
                "metrics": metrics(env.test_y, prob),
                "structural_weight_mean": float(gate.mean()),
                "structural_weight_std": float(gate.std()),
            }
        )
    flow_ids, y_true, probability, gate_weights = aligned_test_consensus(test_outputs)
    base_metrics = metrics(y_true, probability)
    valid_y_array = np.asarray(oof_valid_y, dtype=np.int64)
    valid_prob_array = normalize_prob(oof_valid_prob)
    valid_prob_array, probability, prior_transport = safe_prior_transport(
        valid_y_array,
        valid_prob_array,
        probability,
        args.prior_max_strength,
        args.prior_strength_step,
        args.prior_macro_plateau_tolerance,
    )
    result_metrics = metrics(y_true, probability)
    payload = {
        "metrics": {task_scope: result_metrics},
        "method": "cross_environment_confusion_conditioned_reliability_router",
        "sample_unit": "one_packet" if task_scope == "packet_level" else "one_flow",
        "config": {
            key: getattr(args, key)
            for key in (
                "hidden_dim",
                "dropout",
                "epochs",
                "lr",
                "weight_decay",
                "groupdro_eta",
                "gate_entropy_weight",
                "prior_max_strength",
                "prior_strength_step",
                "prior_macro_plateau_tolerance",
                "seed",
            )
        },
        "environments": [env.name for env in environments],
        "leave_one_environment_out": leave_one_environment_out,
        "per_environment_test": per_environment,
        "final_train_group_weights": group_weights.tolist(),
        "base_metrics_before_prior_transport": base_metrics,
        "prior_transport": prior_transport,
    }
    if task_scope == "flow_level":
        payload.update(
            {
                "flow_ids": flow_ids,
                "flow_y_true": y_true.tolist(),
                "flow_y_pred": probability.argmax(axis=1).tolist(),
                "flow_prob": probability.tolist(),
                "flow_structural_gate": gate_weights.tolist(),
                "valid_flow_ids": oof_valid_ids,
                "valid_y_true": oof_valid_y,
                "valid_y_pred": valid_prob_array.argmax(axis=1).tolist(),
                "valid_prob": valid_prob_array.tolist(),
            }
        )
    else:
        probability_path = Path(args.output_npz) if args.output_npz else Path(args.output_json).with_suffix(".npz")
        probability_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            probability_path,
            y_true=y_true,
            probabilities=probability.astype(np.float32),
            structural_gate=gate_weights.astype(np.float32),
        )
        payload.update(
            {
                "num_packets": int(len(y_true)),
                "probability_path": str(probability_path),
                "alignment": "packet_uid" if not flow_ids[0].startswith("packet:") else "strict_shared_row_order",
            }
        )
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": model.state_dict(),
                "feature_mean": mean,
                "feature_std": std,
                "config": payload["config"],
            },
            checkpoint_path,
        )
    print(json.dumps({"metrics": result_metrics, "per_environment": per_environment, "loeo": leave_one_environment_out}, indent=2))


if __name__ == "__main__":
    main()
