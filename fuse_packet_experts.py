#!/usr/bin/env python3
"""Learn a validation-only reliability gate for strict one-packet experts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from packet_eval_utils import packet_classification_metrics
from train_tower1_multitask import load_label_names


EPS = 1e-8


def load_probabilities(path: str) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    y_true = data["y_true"].astype(np.int64)
    probabilities = data["probabilities"].astype(np.float32)
    if probabilities.ndim != 2 or len(y_true) != len(probabilities):
        raise ValueError(f"invalid probability artifact: {path}")
    probabilities = np.clip(probabilities, EPS, 1.0)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return y_true, probabilities


def align_pair(first_path: str, second_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_first, first = load_probabilities(first_path)
    y_second, second = load_probabilities(second_path)
    if not np.array_equal(y_first, y_second):
        raise ValueError(f"label alignment mismatch: {first_path} vs {second_path}")
    if first.shape != second.shape:
        raise ValueError(f"probability shape mismatch: {first.shape} vs {second.shape}")
    return y_first, first, second


def temperature_scale(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    scaled = np.exp(np.log(np.clip(probabilities, EPS, 1.0)) / temperature)
    return scaled / scaled.sum(axis=1, keepdims=True)


def select_temperature(probabilities: np.ndarray, y_true: np.ndarray) -> tuple[float, np.ndarray]:
    candidates = np.geomspace(0.25, 4.0, 49)
    losses = []
    for temperature in candidates:
        scaled = temperature_scale(probabilities, float(temperature))
        losses.append(float(-np.log(np.clip(scaled[np.arange(len(y_true)), y_true], EPS, 1.0)).mean()))
    best_index = int(np.argmin(losses))
    temperature = float(candidates[best_index])
    return temperature, temperature_scale(probabilities, temperature)


def confidence_features(semantic: np.ndarray, structural: np.ndarray) -> np.ndarray:
    num_classes = semantic.shape[1]

    def channel_features(probabilities: np.ndarray) -> tuple[np.ndarray, ...]:
        ordered = np.sort(probabilities, axis=1)
        top = ordered[:, -1]
        margin = top - ordered[:, -2]
        entropy = -(probabilities * np.log(np.clip(probabilities, EPS, 1.0))).sum(axis=1) / np.log(num_classes)
        prediction = probabilities.argmax(axis=1)
        one_hot = np.eye(num_classes, dtype=np.float32)[prediction]
        return top, margin, entropy, one_hot

    sem_top, sem_margin, sem_entropy, sem_pred = channel_features(semantic)
    str_top, str_margin, str_entropy, str_pred = channel_features(structural)
    agreement = (semantic.argmax(axis=1) == structural.argmax(axis=1)).astype(np.float32)
    return np.column_stack(
        [
            sem_top,
            sem_margin,
            sem_entropy,
            str_top,
            str_margin,
            str_entropy,
            sem_top - str_top,
            sem_margin - str_margin,
            sem_entropy - str_entropy,
            agreement,
            sem_pred,
            str_pred,
        ]
    )


def blend(semantic: np.ndarray, structural: np.ndarray, semantic_weight: np.ndarray | float) -> np.ndarray:
    weight = np.asarray(semantic_weight, dtype=semantic.dtype)
    if weight.ndim == 1:
        weight = weight[:, None]
    fused = weight * semantic + (1.0 - weight) * structural
    return fused / fused.sum(axis=1, keepdims=True)


def predict_fused_probabilities_chunked(
    semantic_raw: np.ndarray,
    structural_raw: np.ndarray,
    method: str,
    semantic_temperature: float,
    structural_temperature: float,
    global_weight: float,
    gate_model,
    gate_strength: float,
    chunk_size: int = 20_000,
) -> np.ndarray:
    """Apply a validation-fitted fusion rule without materializing huge gate features."""
    probabilities_out = np.empty_like(semantic_raw, dtype=np.float32)
    for start in range(0, len(semantic_raw), chunk_size):
        stop = min(start + chunk_size, len(semantic_raw))
        semantic = temperature_scale(semantic_raw[start:stop], semantic_temperature)
        structural = temperature_scale(structural_raw[start:stop], structural_temperature)
        if method == "semantic":
            probabilities = semantic
        elif method == "structural":
            probabilities = structural
        elif method == "global":
            probabilities = blend(semantic, structural, global_weight)
        else:
            reliability = gate_model.predict_proba(confidence_features(semantic, structural))[:, 1]
            weights = global_weight + gate_strength * (reliability - global_weight)
            probabilities = blend(semantic, structural, np.clip(weights, 0.0, 1.0))
        probabilities_out[start:stop] = probabilities
    return probabilities_out


def predict_fused_labels_chunked(
    semantic_raw: np.ndarray,
    structural_raw: np.ndarray,
    method: str,
    semantic_temperature: float,
    structural_temperature: float,
    global_weight: float,
    gate_model,
    gate_strength: float,
    chunk_size: int = 20_000,
) -> np.ndarray:
    probabilities = predict_fused_probabilities_chunked(
        semantic_raw,
        structural_raw,
        method,
        semantic_temperature,
        structural_temperature,
        global_weight,
        gate_model,
        gate_strength,
        chunk_size,
    )
    return probabilities.argmax(axis=1)


def metric_key(y_true: np.ndarray, probabilities: np.ndarray, num_classes: int, label_names: list[str]) -> tuple[float, float]:
    metrics = packet_classification_metrics(y_true, probabilities.argmax(axis=1), num_classes, label_names)
    return float(metrics["macro_f1"]), float(metrics["accuracy"])


def select_global_weight(
    y_true: np.ndarray,
    semantic: np.ndarray,
    structural: np.ndarray,
    num_classes: int,
    label_names: list[str],
) -> tuple[float, np.ndarray]:
    best = None
    for weight in np.linspace(0.0, 1.0, 101):
        probabilities = blend(semantic, structural, float(weight))
        key = metric_key(y_true, probabilities, num_classes, label_names)
        if best is None or key > best[0]:
            best = (key, float(weight), probabilities)
    assert best is not None
    return best[1], best[2]


def fit_gate(
    features: np.ndarray,
    y_true: np.ndarray,
    semantic: np.ndarray,
    structural: np.ndarray,
    c_value: float,
    seed: int,
) -> LogisticRegression | None:
    semantic_correct = semantic.argmax(axis=1) == y_true
    structural_correct = structural.argmax(axis=1) == y_true
    decisive = semantic_correct ^ structural_correct
    target = semantic_correct.astype(np.int64)
    if decisive.sum() < 20 or np.unique(target[decisive]).size < 2:
        return None
    model = LogisticRegression(C=c_value, class_weight="balanced", max_iter=1000, random_state=seed)
    model.fit(features[decisive], target[decisive])
    return model


def train_gate(
    features: np.ndarray,
    y_true: np.ndarray,
    semantic: np.ndarray,
    structural: np.ndarray,
    c_value: float,
    seed: int,
) -> LogisticRegression:
    model = fit_gate(features, y_true, semantic, structural, c_value, seed)
    if model is None:
        raise ValueError("reliability gate requires both semantic-only and structural-only successes")
    return model


def candidate_record(
    method: str,
    probabilities: np.ndarray,
    y_true: np.ndarray,
    fold_ids: np.ndarray,
    num_classes: int,
    label_names: list[str],
    gate_c: float | None = None,
    gate_strength: float = 0.0,
) -> dict:
    predictions = probabilities.argmax(axis=1)
    overall = packet_classification_metrics(y_true, predictions, num_classes, label_names)
    fold_f1 = []
    fold_accuracy = []
    for fold_id in sorted(np.unique(fold_ids)):
        mask = fold_ids == fold_id
        metrics = packet_classification_metrics(
            y_true[mask], predictions[mask], num_classes, label_names
        )
        fold_f1.append(float(metrics["macro_f1"]))
        fold_accuracy.append(float(metrics["accuracy"]))
    std = float(np.std(fold_f1, ddof=1)) if len(fold_f1) > 1 else 0.0
    return {
        "method": method,
        "gate_c": gate_c,
        "gate_strength": gate_strength,
        "predictions": predictions,
        "macro_f1": float(overall["macro_f1"]),
        "accuracy": float(overall["accuracy"]),
        "fold_macro_f1": fold_f1,
        "fold_accuracy": fold_accuracy,
        "fold_macro_f1_mean": float(np.mean(fold_f1)),
        "fold_macro_f1_std": std,
        "fold_macro_f1_se": std / np.sqrt(max(1, len(fold_f1))),
    }


def nested_oof_candidates(
    y_true: np.ndarray,
    semantic_raw: np.ndarray,
    structural_raw: np.ndarray,
    num_classes: int,
    label_names: list[str],
    seed: int,
) -> list[dict]:
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    splits = list(splitter.split(semantic_raw, y_true))
    fold_ids = np.empty(len(y_true), dtype=np.int64)
    semantic_oof = np.zeros_like(semantic_raw)
    structural_oof = np.zeros_like(structural_raw)
    global_oof = np.zeros_like(semantic_raw)
    global_weights = np.zeros(len(y_true), dtype=np.float64)
    calibrated = []

    for fold_id, (train_index, valid_index) in enumerate(splits):
        fold_ids[valid_index] = fold_id
        sem_temp, sem_train = select_temperature(semantic_raw[train_index], y_true[train_index])
        str_temp, str_train = select_temperature(structural_raw[train_index], y_true[train_index])
        sem_valid = temperature_scale(semantic_raw[valid_index], sem_temp)
        str_valid = temperature_scale(structural_raw[valid_index], str_temp)
        global_weight, _ = select_global_weight(
            y_true[train_index], sem_train, str_train, num_classes, label_names
        )
        semantic_oof[valid_index] = sem_valid
        structural_oof[valid_index] = str_valid
        global_oof[valid_index] = blend(sem_valid, str_valid, global_weight)
        global_weights[valid_index] = global_weight
        calibrated.append((train_index, valid_index, sem_temp, str_temp))

    candidates = [
        candidate_record("structural", structural_oof, y_true, fold_ids, num_classes, label_names),
        candidate_record("semantic", semantic_oof, y_true, fold_ids, num_classes, label_names, gate_strength=1.0),
        candidate_record("global", global_oof, y_true, fold_ids, num_classes, label_names, gate_strength=1.0),
    ]
    for c_value in (0.01, 0.1, 1.0, 10.0):
        reliability = np.zeros(len(y_true), dtype=np.float64)
        usable = True
        for train_index, valid_index, sem_temp, str_temp in calibrated:
            sem_train = temperature_scale(semantic_raw[train_index], sem_temp)
            str_train = temperature_scale(structural_raw[train_index], str_temp)
            model = fit_gate(
                confidence_features(sem_train, str_train),
                y_true[train_index],
                sem_train,
                str_train,
                c_value,
                seed,
            )
            if model is None:
                usable = False
                break
            sem_valid = semantic_oof[valid_index]
            str_valid = structural_oof[valid_index]
            reliability[valid_index] = model.predict_proba(
                confidence_features(sem_valid, str_valid)
            )[:, 1]
        if not usable:
            continue
        for gate_strength in (0.25, 0.5, 0.75, 1.0):
            weights = global_weights + gate_strength * (reliability - global_weights)
            probabilities = blend(
                semantic_oof, structural_oof, np.clip(weights, 0.0, 1.0)
            )
            candidates.append(
                candidate_record(
                    "reliability_gate",
                    probabilities,
                    y_true,
                    fold_ids,
                    num_classes,
                    label_names,
                    gate_c=c_value,
                    gate_strength=gate_strength,
                )
            )
    return candidates


def select_candidate(candidates: list[dict], selection_rule: str) -> tuple[dict, float | None]:
    best = max(candidates, key=lambda item: (item["macro_f1"], item["accuracy"]))
    if selection_rule == "best":
        return best, None
    threshold = best["fold_macro_f1_mean"] - best["fold_macro_f1_se"]
    eligible = [item for item in candidates if item["fold_macro_f1_mean"] >= threshold]

    def complexity(item: dict) -> tuple[float, float, float]:
        if item["method"] in {"semantic", "structural"}:
            score = 0.0
        elif item["method"] == "global":
            score = 1.0
        else:
            score = 2.0 + float(item["gate_strength"]) + 0.1 * (np.log10(float(item["gate_c"])) + 2.0)
        return score, -item["macro_f1"], -item["accuracy"]

    return min(eligible, key=complexity), float(threshold)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--valid_semantic", required=True)
    ap.add_argument("--valid_structural", required=True)
    ap.add_argument("--test_semantic", default="")
    ap.add_argument("--test_structural", default="")
    ap.add_argument("--label_map", required=True)
    ap.add_argument("--output_json", required=True)
    ap.add_argument(
        "--output_npz",
        default="",
        help="Optional fused test probabilities for cross-fold consensus.",
    )
    ap.add_argument("--gate_out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--selection_rule",
        choices=["one_se", "best"],
        default="one_se",
        help="Use nested OOF one-standard-error selection by default to favor stable, simpler gates.",
    )
    args = ap.parse_args()

    label_names = load_label_names(args.label_map)
    num_classes = len(label_names)
    y_valid, semantic_raw, structural_raw = align_pair(args.valid_semantic, args.valid_structural)
    candidates = nested_oof_candidates(
        y_valid, semantic_raw, structural_raw, num_classes, label_names, args.seed
    )
    selected, one_se_threshold = select_candidate(candidates, args.selection_rule)
    method = selected["method"]
    selected_c = selected["gate_c"]
    gate_strength = selected["gate_strength"]
    valid_predictions = selected["predictions"]

    # Refit calibration and the selected gate on all validation samples only.
    semantic_temperature, semantic = select_temperature(semantic_raw, y_valid)
    structural_temperature, structural = select_temperature(structural_raw, y_valid)
    global_weight, global_probabilities = select_global_weight(
        y_valid, semantic, structural, num_classes, label_names
    )
    features = confidence_features(semantic, structural)
    gate_model = None
    if method == "reliability_gate":
        gate_model = train_gate(features, y_valid, semantic, structural, float(selected_c), args.seed)

    validation_metrics = packet_classification_metrics(
        y_valid, valid_predictions, num_classes, label_names
    )
    result = {
        "task": "packet-level-classification",
        "sample_unit": "one_packet",
        "selection_scope": "validation_only",
        "config": vars(args),
        "calibration": {
            "semantic_temperature": semantic_temperature,
            "structural_temperature": structural_temperature,
            "global_semantic_weight": global_weight,
        },
        "selected": {
            "method": method,
            "gate_c": selected_c,
            "gate_strength": gate_strength,
            "selection_rule": args.selection_rule,
            "one_se_threshold": one_se_threshold,
        },
        "validation_metrics": validation_metrics,
        "validation_candidates": [
            {
                key: value
                for key, value in candidate.items()
                if key != "predictions"
            }
            for candidate in candidates
        ],
    }

    if bool(args.test_semantic) != bool(args.test_structural):
        raise ValueError("provide both --test_semantic and --test_structural, or neither")
    if args.test_semantic:
        y_test, semantic_test_raw, structural_test_raw = align_pair(args.test_semantic, args.test_structural)
        test_probabilities = predict_fused_probabilities_chunked(
            semantic_test_raw,
            structural_test_raw,
            method,
            semantic_temperature,
            structural_temperature,
            global_weight,
            gate_model,
            gate_strength,
        )
        test_predictions = test_probabilities.argmax(axis=1)
        result["test_metrics"] = packet_classification_metrics(
            y_test, test_predictions, num_classes, label_names
        )
        if args.output_npz:
            npz_path = Path(args.output_npz)
            npz_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                npz_path,
                y_true=y_test.astype(np.int64),
                probabilities=test_probabilities.astype(np.float32),
            )
    elif args.output_npz:
        raise ValueError("--output_npz requires test probability inputs")

    output_path = Path(args.output_json)
    gate_path = Path(args.gate_out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gate_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    joblib.dump(
        {
            "method": method,
            "gate_model": gate_model,
            "gate_strength": gate_strength,
            "global_semantic_weight": global_weight,
            "semantic_temperature": semantic_temperature,
            "structural_temperature": structural_temperature,
            "label_names": label_names,
        },
        gate_path,
        compress=3,
    )
    print(
        f"selected={method} valid_accuracy={validation_metrics['accuracy']:.4f} "
        f"valid_macro_f1={validation_metrics['macro_f1']:.4f}"
    )
    if "test_metrics" in result:
        print(
            f"test_accuracy={result['test_metrics']['accuracy']:.4f} "
            f"test_macro_f1={result['test_metrics']['macro_f1']:.4f}"
        )
    print(f"saved {output_path} and {gate_path}")


if __name__ == "__main__":
    main()
