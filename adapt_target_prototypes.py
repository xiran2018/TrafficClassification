#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler, normalize

from train_flow_embedding_classifier import load_split


EPS = 1e-12


def parse_grid(text: str, cast) -> List[Any]:
    return [cast(item.strip()) for item in text.split(",") if item.strip()]


def metrics(y_true: Sequence[int], prob: np.ndarray) -> Dict[str, float]:
    pred = prob.argmax(axis=1)
    precision, recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, pred, average="macro", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(macro_f1),
    }


def align_rows(flow_ids: Sequence[str], payload_ids: Sequence[str], values: Sequence[Sequence[float]]) -> np.ndarray:
    if len(payload_ids) != len(values):
        raise ValueError("Prediction flow IDs and probability rows have different lengths.")
    lookup = {str(fid): np.asarray(row, dtype=np.float32) for fid, row in zip(payload_ids, values)}
    missing = [str(fid) for fid in flow_ids if str(fid) not in lookup]
    if missing:
        raise ValueError(f"Prediction payload is missing {len(missing)} flow IDs; first={missing[0]}")
    return np.stack([lookup[str(fid)] for fid in flow_ids], axis=0)


def temperature_scale(prob: np.ndarray, temperature: float) -> np.ndarray:
    logits = np.log(np.clip(prob, EPS, 1.0)) / max(float(temperature), 1e-6)
    logits -= logits.max(axis=1, keepdims=True)
    out = np.exp(logits)
    return out / np.maximum(out.sum(axis=1, keepdims=True), EPS)


def fit_feature_transform(
    x_source: np.ndarray,
    x_valid: np.ndarray,
    x_test: np.ndarray,
    n_components: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    scaler = StandardScaler()
    source = scaler.fit_transform(x_source)
    valid = scaler.transform(x_valid)
    test = scaler.transform(x_test)
    actual = min(int(n_components), source.shape[0] - 1, source.shape[1])
    pca = PCA(n_components=actual, svd_solver="randomized", random_state=seed)
    source = normalize(pca.fit_transform(source)).astype(np.float32)
    valid = normalize(pca.transform(valid)).astype(np.float32)
    test = normalize(pca.transform(test)).astype(np.float32)
    return source, valid, test, actual


def source_prototypes(features: np.ndarray, labels: np.ndarray, num_classes: int) -> np.ndarray:
    prototypes = np.zeros((num_classes, features.shape[1]), dtype=np.float32)
    for cls in range(num_classes):
        mask = labels == cls
        if not mask.any():
            raise ValueError(f"Source split has no samples for class {cls}.")
        prototypes[cls] = features[mask].mean(axis=0)
    return normalize(prototypes).astype(np.float32)


def selected_soft_weights(prob: np.ndarray, threshold: float, topk: int, power: float) -> np.ndarray:
    weights = np.zeros_like(prob, dtype=np.float32)
    for cls in range(prob.shape[1]):
        candidates = np.flatnonzero(prob[:, cls] >= threshold)
        if topk > 0 and len(candidates) > topk:
            order = np.argsort(prob[candidates, cls])[-topk:]
            candidates = candidates[order]
        weights[candidates, cls] = np.power(prob[candidates, cls], power)
    return weights


def leave_one_out_prototype_prob(
    features: np.ndarray,
    pseudo_prob: np.ndarray,
    source_proto: np.ndarray,
    threshold: float,
    topk: int,
    pseudo_power: float,
    target_weight: float,
    prototype_temperature: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    weights = selected_soft_weights(pseudo_prob, threshold, topk, pseudo_power)
    weighted_sum = weights.T @ features
    weight_sum = weights.sum(axis=0)
    num_samples, num_classes = pseudo_prob.shape
    logits = np.empty((num_samples, num_classes), dtype=np.float32)

    for row in range(num_samples):
        loo_sum = weighted_sum - weights[row, :, None] * features[row][None, :]
        loo_count = weight_sum - weights[row]
        target_proto = loo_sum / np.maximum(loo_count[:, None], EPS)
        available = loo_count > EPS
        target_proto[available] = normalize(target_proto[available])
        blended = source_proto.copy()
        blended[available] = normalize(
            (1.0 - target_weight) * source_proto[available]
            + target_weight * target_proto[available]
        )
        logits[row] = features[row] @ blended.T

    logits /= max(float(prototype_temperature), 1e-6)
    logits -= logits.max(axis=1, keepdims=True)
    prob = np.exp(logits)
    prob /= np.maximum(prob.sum(axis=1, keepdims=True), EPS)
    diagnostics = {
        "selected_per_class": (weights > 0).sum(axis=0).astype(int).tolist(),
        "effective_weight_per_class": weight_sum.astype(float).tolist(),
    }
    return prob, diagnostics


def adapt_probabilities(
    features: np.ndarray,
    base_prob: np.ndarray,
    source_proto: np.ndarray,
    config: Dict[str, Any],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    calibrated_base = temperature_scale(base_prob, config["base_temperature"])
    current = calibrated_base.copy()
    diagnostics: Dict[str, Any] = {}
    for iteration in range(config["iterations"]):
        proto_prob, proto_diag = leave_one_out_prototype_prob(
            features,
            current,
            source_proto,
            config["confidence_threshold"],
            config["topk_per_class"],
            config["pseudo_power"],
            config["target_prototype_weight"],
            config["prototype_temperature"],
        )
        uncertainty = 1.0 - calibrated_base.max(axis=1)
        gate = config["prototype_weight"] * np.power(uncertainty, config["uncertainty_power"])
        gate = np.clip(gate, 0.0, 1.0)[:, None]
        log_prob = (1.0 - gate) * np.log(np.clip(calibrated_base, EPS, 1.0))
        log_prob += gate * np.log(np.clip(proto_prob, EPS, 1.0))
        log_prob -= log_prob.max(axis=1, keepdims=True)
        current = np.exp(log_prob)
        current /= np.maximum(current.sum(axis=1, keepdims=True), EPS)
        diagnostics[f"iteration_{iteration + 1}"] = proto_diag
    diagnostics["prototype_gate_mean"] = float(gate.mean())
    diagnostics["prototype_gate_max"] = float(gate.max())
    return current.astype(np.float32), diagnostics


def candidate_configs(args: argparse.Namespace) -> Iterable[Dict[str, Any]]:
    keys = [
        "base_temperature",
        "confidence_threshold",
        "topk_per_class",
        "pseudo_power",
        "target_prototype_weight",
        "prototype_temperature",
        "prototype_weight",
        "uncertainty_power",
        "iterations",
    ]
    grids = [
        parse_grid(args.base_temperature_grid, float),
        parse_grid(args.confidence_threshold_grid, float),
        parse_grid(args.topk_per_class_grid, int),
        parse_grid(args.pseudo_power_grid, float),
        parse_grid(args.target_prototype_weight_grid, float),
        parse_grid(args.prototype_temperature_grid, float),
        parse_grid(args.prototype_weight_grid, float),
        parse_grid(args.uncertainty_power_grid, float),
        parse_grid(args.iterations_grid, int),
    ]
    for values in itertools.product(*grids):
        yield dict(zip(keys, values))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Validation-selected source-anchored target prototype adaptation without target labels."
    )
    ap.add_argument("--train_index", required=True)
    ap.add_argument("--valid_index", required=True)
    ap.add_argument("--test_index", required=True)
    ap.add_argument("--base_prediction_json", required=True)
    ap.add_argument("--output_json", required=True)
    ap.add_argument("--max_packets", type=int, default=64)
    ap.add_argument("--include_meta", action="store_true")
    ap.add_argument("--meta_prefix_len", type=int, default=32)
    ap.add_argument(
        "--meta_feature_version",
        choices=["basic", "message", "message_header", "message_header_endpoint", "message_header_fullbytes"],
        default="message",
    )
    ap.add_argument("--use_ports", action="store_true")
    ap.add_argument("--components_grid", default="64,128")
    ap.add_argument("--base_temperature_grid", default="1.0")
    ap.add_argument("--confidence_threshold_grid", default="0.6,0.75,0.9")
    ap.add_argument("--topk_per_class_grid", default="16,32")
    ap.add_argument("--pseudo_power_grid", default="1")
    ap.add_argument("--target_prototype_weight_grid", default="0.25,0.5")
    ap.add_argument("--prototype_temperature_grid", default="0.05,0.1,0.2")
    ap.add_argument("--prototype_weight_grid", default="0.25,0.5")
    ap.add_argument("--uncertainty_power_grid", default="0,1")
    ap.add_argument("--iterations_grid", default="1")
    ap.add_argument("--select_metric", choices=["accuracy", "macro_f1"], default="macro_f1")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    x_source, y_source, _ = load_split(
        args.train_index, args.max_packets, args.include_meta, args.meta_prefix_len,
        args.use_ports, args.meta_feature_version
    )
    x_valid, y_valid, valid_ids = load_split(
        args.valid_index, args.max_packets, args.include_meta, args.meta_prefix_len,
        args.use_ports, args.meta_feature_version
    )
    x_test, y_test, test_ids = load_split(
        args.test_index, args.max_packets, args.include_meta, args.meta_prefix_len,
        args.use_ports, args.meta_feature_version
    )
    with open(args.base_prediction_json, "r", encoding="utf-8") as handle:
        base_payload = json.load(handle)
    valid_base = align_rows(valid_ids, base_payload["valid_flow_ids"], base_payload["valid_prob"])
    test_base = align_rows(test_ids, base_payload["flow_ids"], base_payload["flow_prob"])
    num_classes = valid_base.shape[1]
    if test_base.shape[1] != num_classes:
        raise ValueError("Validation and test probability widths differ.")

    base_valid_metrics = metrics(y_valid, valid_base)
    base_test_metrics = metrics(y_test, test_base)
    print("base_valid", json.dumps(base_valid_metrics, sort_keys=True))
    print("base_test", json.dumps(base_test_metrics, sort_keys=True))

    reports: List[Dict[str, Any]] = []
    baseline_key = (
        base_valid_metrics[args.select_metric],
        base_valid_metrics["macro_f1"],
        base_valid_metrics["accuracy"],
    )
    best: Tuple[Tuple[float, float, float], Dict[str, Any]] | None = None
    for requested_components in parse_grid(args.components_grid, int):
        source, valid, test, actual_components = fit_feature_transform(
            x_source, x_valid, x_test, requested_components, args.seed
        )
        prototypes = source_prototypes(source, y_source, num_classes)
        for config in candidate_configs(args):
            valid_prob, diagnostics = adapt_probabilities(valid, valid_base, prototypes, config)
            valid_metrics = metrics(y_valid, valid_prob)
            row = {
                "n_components": actual_components,
                **config,
                "metrics": valid_metrics,
                "diagnostics": diagnostics,
            }
            reports.append(row)
            key = (
                valid_metrics[args.select_metric],
                valid_metrics["macro_f1"],
                valid_metrics["accuracy"],
            )
            if key > baseline_key and (best is None or key > best[0]):
                best = (key, row)

    if best is None:
        selected = {"mode": "identity_no_adaptation", "metrics": base_valid_metrics}
        valid_prob, test_prob = valid_base, test_base
        valid_diagnostics = test_diagnostics = {"reason": "no candidate improved validation selection metric"}
    else:
        selected = best[1]
        source, valid, test, _ = fit_feature_transform(
            x_source, x_valid, x_test, selected["n_components"], args.seed
        )
        prototypes = source_prototypes(source, y_source, num_classes)
        config = {key: selected[key] for key in [
            "base_temperature", "confidence_threshold", "topk_per_class", "pseudo_power",
            "target_prototype_weight", "prototype_temperature", "prototype_weight",
            "uncertainty_power", "iterations"
        ]}
        valid_prob, valid_diagnostics = adapt_probabilities(valid, valid_base, prototypes, config)
        test_prob, test_diagnostics = adapt_probabilities(test, test_base, prototypes, config)
    valid_metrics = metrics(y_valid, valid_prob)
    test_metrics = metrics(y_test, test_prob)
    print("selected", json.dumps(selected, sort_keys=True))
    print("adapted_test", json.dumps(test_metrics, sort_keys=True))

    payload = {
        "method": "source_anchored_leave_one_out_target_prototypes",
        "result_scope": "single_fold_target_adaptation",
        "metrics": {"flow_level": test_metrics},
        "valid_metrics": valid_metrics,
        "base_metrics": {"valid": base_valid_metrics, "test": base_test_metrics},
        "selected": selected,
        "flow_ids": test_ids,
        "flow_y_true": y_test.astype(int).tolist(),
        "flow_y_pred": test_prob.argmax(axis=1).astype(int).tolist(),
        "flow_prob": test_prob.tolist(),
        "valid_flow_ids": valid_ids,
        "valid_y_true": y_valid.astype(int).tolist(),
        "valid_y_pred": valid_prob.argmax(axis=1).astype(int).tolist(),
        "valid_prob": valid_prob.tolist(),
        "diagnostics": {"valid": valid_diagnostics, "test": test_diagnostics},
        "audit": {
            "source_feature_transform_fit_on": "train_only",
            "hyperparameters_selected_with": "validation_labels_only",
            "target_prototypes_use_labels": False,
            "test_labels_used_for_adaptation": False,
            "self_reinforcement_control": "leave_one_out_target_prototype",
        },
        "inputs": {
            "train_index": args.train_index,
            "valid_index": args.valid_index,
            "test_index": args.test_index,
            "base_prediction_json": args.base_prediction_json,
        },
        "feature_config": {
            "max_packets": args.max_packets,
            "include_meta": args.include_meta,
            "meta_prefix_len": args.meta_prefix_len,
            "meta_feature_version": args.meta_feature_version,
            "use_ports": args.use_ports,
        },
        "valid_reports": reports,
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


if __name__ == "__main__":
    main()
