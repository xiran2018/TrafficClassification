#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support

from probability_metrics import calibration_metrics

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


def metric_score(y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> float:
    if len(y_true) == 0:
        return 0.0
    if metric == "accuracy":
        return float(accuracy_score(y_true.tolist(), y_pred.tolist()))
    if metric == "macro_f1":
        return float(f1_score(y_true.tolist(), y_pred.tolist(), average="macro", zero_division=0))
    raise ValueError(metric)


def bootstrap_gain_guard(
    y_valid: np.ndarray,
    selected_prob: np.ndarray,
    base_prob: np.ndarray,
    metric: str,
    samples: int,
    seed: int,
    quantile: float,
) -> Dict[str, float]:
    selected_pred = selected_prob.argmax(axis=1).astype(np.int64)
    base_pred = base_prob.argmax(axis=1).astype(np.int64)
    full_gain = metric_score(y_valid, selected_pred, metric) - metric_score(y_valid, base_pred, metric)
    if samples <= 0:
        return {
            "enabled": False,
            "full_gain": full_gain,
            "samples": 0,
            "win_rate": 1.0 if full_gain > 0 else 0.0,
            "mean_gain": full_gain,
            "gain_quantile": full_gain,
        }
    rng = np.random.default_rng(seed)
    gains = np.zeros(samples, dtype=np.float64)
    n = len(y_valid)
    for i in range(samples):
        idx = rng.integers(0, n, size=n)
        gains[i] = metric_score(y_valid[idx], selected_pred[idx], metric) - metric_score(y_valid[idx], base_pred[idx], metric)
    return {
        "enabled": True,
        "full_gain": full_gain,
        "samples": int(samples),
        "win_rate": float((gains > 0).mean()),
        "mean_gain": float(gains.mean()),
        "gain_quantile": float(np.quantile(gains, quantile)),
        "quantile": float(quantile),
    }


def target_shift_guard(selected_prob: np.ndarray, base_prob: np.ndarray) -> Dict[str, float]:
    selected_pred = selected_prob.argmax(axis=1).astype(np.int64)
    base_pred = base_prob.argmax(axis=1).astype(np.int64)
    num_classes = max(selected_prob.shape[1], base_prob.shape[1])
    selected_hist = np.bincount(selected_pred, minlength=num_classes).astype(np.float64)
    base_hist = np.bincount(base_pred, minlength=num_classes).astype(np.float64)
    selected_hist /= max(float(selected_hist.sum()), 1.0)
    base_hist /= max(float(base_hist.sum()), 1.0)
    midpoint = 0.5 * (selected_hist + base_hist)

    def kl_div(p: np.ndarray, q: np.ndarray) -> float:
        mask = p > 0
        return float((p[mask] * np.log2(p[mask] / np.maximum(q[mask], 1e-12))).sum())

    return {
        "prediction_change_rate": float((selected_pred != base_pred).mean()) if len(selected_pred) else 0.0,
        "prediction_js_divergence": 0.5 * kl_div(selected_hist, midpoint) + 0.5 * kl_div(base_hist, midpoint),
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


def load_payload(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    missing = [key for key in REQUIRED_PROB_FIELDS if key not in data]
    if missing:
        raise ValueError(f"{path} is missing required probability fields: {missing}")
    return data


def load_named_payloads(raw_inputs: List[List[str]]) -> Tuple[List[Tuple[str, Dict[str, Any], str]], List[Dict[str, Any]]]:
    named_payloads: List[Tuple[str, Dict[str, Any], str]] = []
    skipped: List[Dict[str, Any]] = []
    for name, path in raw_inputs:
        try:
            named_payloads.append((name, load_payload(path), path))
        except (FileNotFoundError, json.JSONDecodeError, ValueError) as exc:
            skipped.append({"name": name, "path": path, "reason": str(exc)})
            print("skip_selector_input", json.dumps(skipped[-1], sort_keys=True))
    if not named_payloads:
        raise ValueError("No selector inputs contain the required valid/test probability fields.")
    return named_payloads, skipped


def _id_set(data: Dict[str, Any], split: str) -> set[str]:
    key = "valid_flow_ids" if split == "valid" else "flow_ids"
    return set(map(str, data[key]))


def _compatible_group(
    named_payloads: List[Tuple[str, Dict[str, Any], str]],
    seed_idx: int,
) -> Tuple[List[Tuple[str, Dict[str, Any], str]], set[str], set[str]]:
    seed = named_payloads[seed_idx]
    kept = [seed]
    valid_common = _id_set(seed[1], "valid")
    test_common = _id_set(seed[1], "test")
    for idx, item in enumerate(named_payloads):
        if idx == seed_idx:
            continue
        _, data, _ = item
        next_valid = valid_common & _id_set(data, "valid")
        next_test = test_common & _id_set(data, "test")
        if next_valid and next_test:
            kept.append(item)
            valid_common = next_valid
            test_common = next_test
    return kept, valid_common, test_common


def select_compatible_input_group(
    named_payloads: List[Tuple[str, Dict[str, Any], str]],
    preferred_base: str = "",
) -> Tuple[List[Tuple[str, Dict[str, Any], str]], List[Dict[str, Any]]]:
    if not named_payloads:
        return named_payloads, []
    if preferred_base:
        names = [name for name, _, _ in named_payloads]
        if preferred_base in names:
            kept, _, _ = _compatible_group(named_payloads, names.index(preferred_base))
            chosen_ids = {id(item[1]) for item in kept}
        else:
            raise ValueError(f"--base_input={preferred_base} is not one of loadable selector inputs: {names}")
    else:
        candidates = []
        for idx in range(len(named_payloads)):
            kept, valid_common, test_common = _compatible_group(named_payloads, idx)
            candidates.append((len(kept), len(valid_common), len(test_common), -idx, kept))
        _, _, _, _, kept = max(candidates, key=lambda item: item[:4])
        chosen_ids = {id(item[1]) for item in kept}
    skipped: List[Dict[str, Any]] = []
    chosen_valid = set.intersection(*(_id_set(data, "valid") for _, data, _ in kept))
    chosen_test = set.intersection(*(_id_set(data, "test") for _, data, _ in kept))
    for name, data, path in named_payloads:
        if id(data) in chosen_ids:
            continue
        valid_overlap = len(chosen_valid & _id_set(data, "valid"))
        test_overlap = len(chosen_test & _id_set(data, "test"))
        reason = f"incompatible flow ids with selected selector group: valid_overlap={valid_overlap}, test_overlap={test_overlap}"
        skipped.append({"name": name, "path": path, "reason": reason})
        print("skip_selector_input", json.dumps(skipped[-1], sort_keys=True))
    return kept, skipped


def normalize_prob(prob: np.ndarray) -> np.ndarray:
    prob = np.asarray(prob, dtype=np.float64)
    prob = np.clip(prob, 1e-12, None)
    return prob / prob.sum(axis=1, keepdims=True)


def align_payload(data: Dict[str, Any], split: str, fids: List[str]) -> Tuple[np.ndarray, np.ndarray]:
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
    p = normalize_prob(np.asarray([probs[idx[fid]] for fid in fids], dtype=np.float64))
    return y, p


def confidence_features(prob: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pred = prob.argmax(axis=1).astype(np.int64)
    sorted_prob = np.sort(prob, axis=1)
    conf = sorted_prob[:, -1]
    margin = sorted_prob[:, -1] - sorted_prob[:, -2] if prob.shape[1] > 1 else conf
    return pred, conf, margin


def one_hot_prob(pred: np.ndarray, num_classes: int, smooth: float) -> np.ndarray:
    out = np.full((len(pred), num_classes), smooth / max(num_classes - 1, 1), dtype=np.float64)
    out[np.arange(len(pred)), pred] = 1.0 - smooth
    return out


def selected_prob_from_sources(probs: List[np.ndarray], source_idx: np.ndarray, smooth: float) -> np.ndarray:
    num_classes = probs[0].shape[1]
    out = np.zeros((len(source_idx), num_classes), dtype=np.float64)
    for i, src in enumerate(source_idx):
        out[i] = probs[int(src)][i]
    if smooth > 0:
        pred = out.argmax(axis=1)
        return one_hot_prob(pred, num_classes, smooth)
    return normalize_prob(out)


def class_precision_selector(
    valid_probs: List[np.ndarray],
    y_valid: np.ndarray,
    target_probs: List[np.ndarray],
    alpha: float,
    metric_margin: float,
) -> np.ndarray:
    num_sources = len(valid_probs)
    num_classes = valid_probs[0].shape[1]
    valid_preds = [p.argmax(axis=1).astype(np.int64) for p in valid_probs]
    target_preds = [p.argmax(axis=1).astype(np.int64) for p in target_probs]
    global_acc = np.asarray([(pred == y_valid).mean() for pred in valid_preds], dtype=np.float64)
    scores = np.zeros((num_sources, num_classes), dtype=np.float64)
    for s, pred in enumerate(valid_preds):
        for cls in range(num_classes):
            mask = pred == cls
            correct = float(((pred == y_valid) & mask).sum())
            total = float(mask.sum())
            scores[s, cls] = (correct + alpha * global_acc[s]) / (total + alpha)
    source_idx = np.zeros(len(target_preds[0]), dtype=np.int64)
    target_conf = [confidence_features(p)[1] for p in target_probs]
    for i in range(len(source_idx)):
        best_src = 0
        best_score = scores[0, target_preds[0][i]]
        best_conf = target_conf[0][i]
        for s in range(1, num_sources):
            score = scores[s, target_preds[s][i]]
            conf = target_conf[s][i]
            if score > best_score + metric_margin or (abs(score - best_score) <= metric_margin and conf > best_conf):
                best_src = s
                best_score = score
                best_conf = conf
        source_idx[i] = best_src
    return source_idx


def reliability_tables(valid_probs: List[np.ndarray], y_valid: np.ndarray, alpha: float) -> np.ndarray:
    num_sources = len(valid_probs)
    num_classes = valid_probs[0].shape[1]
    valid_preds = [p.argmax(axis=1).astype(np.int64) for p in valid_probs]
    global_acc = np.asarray([(pred == y_valid).mean() for pred in valid_preds], dtype=np.float64)
    tables = np.zeros((num_sources, num_classes), dtype=np.float64)
    for s, pred in enumerate(valid_preds):
        for cls in range(num_classes):
            mask = pred == cls
            correct = float(((pred == y_valid) & mask).sum())
            total = float(mask.sum())
            tables[s, cls] = (correct + alpha * global_acc[s]) / (total + alpha)
    return tables


def reliability_fusion_prob(
    probs: List[np.ndarray],
    tables: np.ndarray,
    reliability_power: float,
    confidence_power: float,
    min_weight: float,
    temperature: float,
) -> np.ndarray:
    eps = 1e-12
    source_weights = []
    for s, prob in enumerate(probs):
        pred, conf, _ = confidence_features(prob)
        reliability = tables[s, pred]
        score = np.power(np.clip(reliability, eps, 1.0), reliability_power)
        if confidence_power != 0:
            score = score * np.power(np.clip(conf, eps, 1.0), confidence_power)
        source_weights.append(score)
    weights = np.stack(source_weights, axis=1)
    if temperature != 1.0:
        weights = np.power(np.clip(weights, eps, None), 1.0 / max(temperature, eps))
    if min_weight > 0:
        weights = weights + min_weight
    weights = weights / np.maximum(weights.sum(axis=1, keepdims=True), eps)
    fused = np.zeros_like(probs[0], dtype=np.float64)
    for s, prob in enumerate(probs):
        fused += weights[:, s : s + 1] * prob
    return normalize_prob(fused)


def class_bias_from_validation(valid_prob: np.ndarray, y_valid: np.ndarray, alpha: float) -> np.ndarray:
    num_classes = valid_prob.shape[1]
    true_mass = np.bincount(y_valid, minlength=num_classes).astype(np.float64)
    pred_mass = valid_prob.sum(axis=0).astype(np.float64)
    true_prior = (true_mass + alpha) / (true_mass.sum() + alpha * num_classes)
    pred_prior = (pred_mass + alpha) / (pred_mass.sum() + alpha * num_classes)
    return np.log(np.clip(true_prior, 1e-12, None) / np.clip(pred_prior, 1e-12, None))


def class_bias_calibrated_prob(prob: np.ndarray, bias: np.ndarray, strength: float, temperature: float) -> np.ndarray:
    prob = normalize_prob(prob)
    adjusted = np.power(np.clip(prob, 1e-12, None), 1.0 / max(temperature, 1e-12))
    adjusted = adjusted * np.exp(strength * bias)[None, :]
    return normalize_prob(adjusted)


def threshold_switch_selector(
    base_prob: np.ndarray,
    expert_prob: np.ndarray,
    expert_index: int,
    expert_conf_min: float,
    expert_margin_min: float,
    base_conf_max: float,
    delta_conf_min: float,
    delta_margin_min: float,
) -> np.ndarray:
    _, base_conf, base_margin = confidence_features(base_prob)
    _, expert_conf, expert_margin = confidence_features(expert_prob)
    switch = (
        (expert_conf >= expert_conf_min)
        & (expert_margin >= expert_margin_min)
        & (base_conf <= base_conf_max)
        & ((expert_conf - base_conf) >= delta_conf_min)
        & ((expert_margin - base_margin) >= delta_margin_min)
    )
    out = np.zeros(base_prob.shape[0], dtype=np.int64)
    out[switch] = expert_index
    return out


def parse_float_list(text: str) -> List[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def parse_name_list(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def apply_unified_expert_slots(
    inputs: List[Tuple[str, Dict[str, Any], str]],
    slots: List[str],
) -> Tuple[List[Tuple[str, Dict[str, Any], str]], List[Dict[str, Any]]]:
    if not slots:
        return inputs, [{"name": name, "path": path, "status": "provided"} for name, _, path in inputs]
    seen = set()
    duplicates = []
    by_name: Dict[str, Tuple[str, Dict[str, Any], str]] = {}
    for item in inputs:
        name = item[0]
        if name in seen:
            duplicates.append(name)
        seen.add(name)
        by_name[name] = item
    if duplicates:
        raise ValueError(f"Duplicate selector input names are not allowed with --unified_expert_slots: {duplicates}")
    if not inputs:
        raise ValueError("At least one selector input is required.")
    base_item = by_name.get(slots[0], inputs[0])
    unified: List[Tuple[str, Dict[str, Any], str]] = []
    status: List[Dict[str, Any]] = []
    used = set()
    for slot in slots:
        if slot in by_name:
            name, data, path = by_name[slot]
            unified.append((name, data, path))
            status.append({"name": slot, "path": path, "status": "provided"})
            used.add(slot)
        else:
            _, data, path = base_item
            unified.append((slot, data, path))
            status.append({"name": slot, "path": path, "status": "identity_from_base", "source": base_item[0]})
    for name, data, path in inputs:
        if name not in used and name not in slots:
            unified.append((name, data, path))
            status.append({"name": name, "path": path, "status": "extra_provided"})
    return unified, status


def main() -> None:
    ap = argparse.ArgumentParser(description="Validation-gated expert selection for probability JSONs.")
    ap.add_argument("--input", nargs=2, action="append", metavar=("NAME", "JSON"), required=True)
    ap.add_argument("--label_map", default="")
    ap.add_argument("--select_metric", choices=["accuracy", "macro_f1"], default="accuracy")
    ap.add_argument(
        "--rank_select_metric",
        choices=["accuracy", "macro_f1"],
        default="",
        help="Metric used inside bootstrap_* ranking. Defaults to --select_metric.",
    )
    ap.add_argument(
        "--rank_metric",
        choices=[
            "select_metric",
            "accuracy",
            "macro_f1",
            "bootstrap_gain_quantile",
            "bootstrap_mean_gain",
            "bootstrap_win_rate",
        ],
        default="select_metric",
        help="Candidate ordering metric before safety gates. Defaults to the validation select metric.",
    )
    ap.add_argument("--strategies", default="always,class_precision,threshold_switch")
    ap.add_argument("--alpha_grid", default="0.5,1,2,5,10")
    ap.add_argument("--metric_margin_grid", default="0,0.02,0.05,0.1")
    ap.add_argument("--expert_conf_grid", default="0,0.3,0.5,0.7,0.85")
    ap.add_argument("--expert_margin_grid", default="0,0.05,0.15,0.3,0.6")
    ap.add_argument("--base_conf_max_grid", default="1,0.85,0.7,0.55,0.4")
    ap.add_argument("--delta_conf_grid", default="-1,0,0.05,0.1,0.2")
    ap.add_argument("--delta_margin_grid", default="-1,0,0.05,0.1,0.2")
    ap.add_argument("--reliability_power_grid", default="1,2,4")
    ap.add_argument("--confidence_power_grid", default="0,0.5,1,2")
    ap.add_argument("--reliability_min_weight_grid", default="0,0.02,0.05")
    ap.add_argument("--reliability_temperature_grid", default="0.5,1,2")
    ap.add_argument("--calibration_strength_grid", default="0.25,0.5,0.75,1.0")
    ap.add_argument("--calibration_temperature_grid", default="0.75,1,1.25")
    ap.add_argument("--output_smooth", type=float, default=0.0, help="Optional one-hot smoothing for selected predictions.")
    ap.add_argument("--min_valid_gain_over_base", type=float, default=0.0, help="Fallback to the first input unless the selected validation metric improves by at least this amount.")
    ap.add_argument("--bootstrap_samples", type=int, default=0, help="Bootstrap validation resamples for stability-gated fallback. Disabled at 0.")
    ap.add_argument("--rank_bootstrap_samples", type=int, default=0, help="Bootstrap resamples used only for robust candidate ranking. Defaults to --bootstrap_samples.")
    ap.add_argument(
        "--rank_candidate_limit",
        type=int,
        default=256,
        help="When bootstrap ranking is enabled, rescore only the top validation-ranked candidates. Use <=0 to rank all candidates.",
    )
    ap.add_argument("--bootstrap_seed", type=int, default=13)
    ap.add_argument("--bootstrap_quantile", type=float, default=0.05)
    ap.add_argument("--bootstrap_min_win_rate", type=float, default=0.0, help="Fallback unless selected candidate beats base in at least this fraction of bootstrap samples.")
    ap.add_argument("--bootstrap_min_gain_quantile", type=float, default=-1.0, help="Fallback unless the bootstrap gain quantile is at least this value.")
    ap.add_argument("--max_prediction_change_rate", type=float, default=1.0, help="Fallback if selected test predictions differ from the base input by more than this unlabeled target fraction.")
    ap.add_argument("--max_prediction_js_divergence", type=float, default=1.0, help="Fallback if selected test prediction-distribution JS divergence from base is larger than this value.")
    ap.add_argument("--base_input", default="", help="Input name used as the base/fallback expert. Defaults to the first input after slot alignment.")
    ap.add_argument(
        "--calibration_penalty_weight",
        type=float,
        default=0.0,
        help="Subtract weight * calibration metric from candidate ranking scores. Default 0 preserves historical ranking.",
    )
    ap.add_argument(
        "--calibration_penalty_metric",
        choices=["ece", "nll", "brier"],
        default="ece",
        help="Calibration metric used by --calibration_penalty_weight.",
    )
    ap.add_argument(
        "--unified_expert_slots",
        default="",
        help=(
            "Comma-separated expert slots that every dataset should expose to the selector. "
            "Missing slots are filled with the first/base input as identity experts and recorded in feature_config."
        ),
    )
    ap.add_argument("--print_all_candidates", action="store_true", help="Print every validation selector candidate.")
    ap.add_argument("--output_json", required=True)
    args = ap.parse_args()

    named_payloads, skipped_inputs = load_named_payloads(args.input)
    named_payloads, incompatible_inputs = select_compatible_input_group(named_payloads, args.base_input)
    skipped_inputs.extend(incompatible_inputs)
    unified_expert_slots = parse_name_list(args.unified_expert_slots)
    named_payloads, input_slot_status = apply_unified_expert_slots(named_payloads, unified_expert_slots)
    if args.base_input:
        names = [name for name, _, _ in named_payloads]
        if args.base_input not in names:
            raise ValueError(f"--base_input={args.base_input} is not one of selector inputs after slot alignment: {names}")
        base_item = named_payloads[names.index(args.base_input)]
        named_payloads = [base_item] + [item for item in named_payloads if item[0] != args.base_input]
    valid_common = sorted(set.intersection(*(set(map(str, data["valid_flow_ids"])) for _, data, _ in named_payloads)))
    test_common = sorted(set.intersection(*(set(map(str, data["flow_ids"])) for _, data, _ in named_payloads)))
    if not valid_common or not test_common:
        raise ValueError("No common flow ids across selector inputs.")

    valid_probs: List[np.ndarray] = []
    test_probs: List[np.ndarray] = []
    y_valid_ref = None
    y_test_ref = None
    for name, data, _ in named_payloads:
        y_valid, p_valid = align_payload(data, "valid", valid_common)
        y_test, p_test = align_payload(data, "test", test_common)
        if y_valid_ref is None:
            y_valid_ref = y_valid
            y_test_ref = y_test
        elif not np.array_equal(y_valid_ref, y_valid) or not np.array_equal(y_test_ref, y_test):
            raise ValueError(f"Labels do not align for input {name}.")
        valid_probs.append(p_valid)
        test_probs.append(p_test)

    strategies = {x.strip() for x in args.strategies.split(",") if x.strip()}
    reports: List[Dict[str, Any]] = []
    candidates: List[Tuple[Tuple[float, float, float], Dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    best = None
    base_candidate = None

    def add_prob_candidate(
        name: str,
        config: Dict[str, Any],
        valid_prob: np.ndarray,
        test_prob: np.ndarray,
        valid_source: np.ndarray | None = None,
        test_source: np.ndarray | None = None,
    ) -> None:
        nonlocal best, base_candidate
        valid_prob = normalize_prob(valid_prob)
        test_prob = normalize_prob(test_prob)
        if args.output_smooth > 0:
            valid_prob = one_hot_prob(valid_prob.argmax(axis=1).astype(np.int64), valid_prob.shape[1], args.output_smooth)
            test_prob = one_hot_prob(test_prob.argmax(axis=1).astype(np.int64), test_prob.shape[1], args.output_smooth)
        valid_pred = valid_prob.argmax(axis=1).astype(np.int64)
        metrics = compute_metrics(y_valid_ref.tolist(), valid_pred.tolist())
        metrics["calibration"] = calibration_metrics(y_valid_ref, valid_prob)
        row = {"strategy": name, "config": config, "metrics": metrics}
        reports.append(row)
        if args.print_all_candidates:
            print("valid_selector", json.dumps(row, sort_keys=True))
        key = (metrics[args.select_metric], metrics["macro_f1"], metrics["accuracy"])
        if valid_source is None:
            valid_source = np.full(len(y_valid_ref), -1, dtype=np.int64)
        if test_source is None:
            test_source = np.full(len(y_test_ref), -1, dtype=np.int64)
        candidate = (key, row, valid_source.copy(), test_source.copy(), valid_prob, test_prob)
        candidates.append(candidate)
        if best is None or key > best[0]:
            best = candidate
        if name == "always" and config.get("source") == named_payloads[0][0]:
            base_candidate = candidate

    def add_candidate(name: str, config: Dict[str, Any], valid_source: np.ndarray, test_source: np.ndarray) -> None:
        valid_prob = selected_prob_from_sources(valid_probs, valid_source, args.output_smooth)
        test_prob = selected_prob_from_sources(test_probs, test_source, args.output_smooth)
        add_prob_candidate(name, config, valid_prob, test_prob, valid_source, test_source)

    if "always" in strategies:
        for src, (name, _, _) in enumerate(named_payloads):
            valid_src = np.full(len(y_valid_ref), src, dtype=np.int64)
            test_src = np.full(len(y_test_ref), src, dtype=np.int64)
            add_candidate("always", {"source": name}, valid_src, test_src)

    if "class_precision" in strategies:
        for alpha in parse_float_list(args.alpha_grid):
            for margin in parse_float_list(args.metric_margin_grid):
                valid_src = class_precision_selector(valid_probs, y_valid_ref, valid_probs, alpha, margin)
                test_src = class_precision_selector(valid_probs, y_valid_ref, test_probs, alpha, margin)
                add_candidate("class_precision", {"alpha": alpha, "metric_margin": margin}, valid_src, test_src)

    if "reliability_fusion" in strategies:
        for alpha in parse_float_list(args.alpha_grid):
            tables = reliability_tables(valid_probs, y_valid_ref, alpha)
            for reliability_power in parse_float_list(args.reliability_power_grid):
                for confidence_power in parse_float_list(args.confidence_power_grid):
                    for min_weight in parse_float_list(args.reliability_min_weight_grid):
                        for temperature in parse_float_list(args.reliability_temperature_grid):
                            valid_prob = reliability_fusion_prob(
                                valid_probs,
                                tables,
                                reliability_power=reliability_power,
                                confidence_power=confidence_power,
                                min_weight=min_weight,
                                temperature=temperature,
                            )
                            test_prob = reliability_fusion_prob(
                                test_probs,
                                tables,
                                reliability_power=reliability_power,
                                confidence_power=confidence_power,
                                min_weight=min_weight,
                                temperature=temperature,
                            )
                            add_prob_candidate(
                                "reliability_fusion",
                                {
                                    "alpha": alpha,
                                    "reliability_power": reliability_power,
                                    "confidence_power": confidence_power,
                                    "min_weight": min_weight,
                                    "temperature": temperature,
                                },
                                valid_prob,
                                test_prob,
                            )

    if "class_bias_calibration" in strategies:
        for source_idx, (source_name, _, _) in enumerate(named_payloads):
            for alpha in parse_float_list(args.alpha_grid):
                bias = class_bias_from_validation(valid_probs[source_idx], y_valid_ref, alpha)
                for strength in parse_float_list(args.calibration_strength_grid):
                    for temperature in parse_float_list(args.calibration_temperature_grid):
                        valid_prob = class_bias_calibrated_prob(valid_probs[source_idx], bias, strength, temperature)
                        test_prob = class_bias_calibrated_prob(test_probs[source_idx], bias, strength, temperature)
                        add_prob_candidate(
                            "class_bias_calibration",
                            {
                                "source": source_name,
                                "alpha": alpha,
                                "strength": strength,
                                "temperature": temperature,
                            },
                            valid_prob,
                            test_prob,
                        )

    if "threshold_switch" in strategies and len(valid_probs) > 1:
        conf_grid = parse_float_list(args.expert_conf_grid)
        margin_grid = parse_float_list(args.expert_margin_grid)
        base_conf_grid = parse_float_list(args.base_conf_max_grid)
        delta_conf_grid = parse_float_list(args.delta_conf_grid)
        delta_margin_grid = parse_float_list(args.delta_margin_grid)
        for expert_idx in range(1, len(valid_probs)):
            expert_name = named_payloads[expert_idx][0]
            for cmin in conf_grid:
                for mmin in margin_grid:
                    for bmax in base_conf_grid:
                        for dcmin in delta_conf_grid:
                            for dmmin in delta_margin_grid:
                                valid_src = threshold_switch_selector(
                                    valid_probs[0], valid_probs[expert_idx], expert_idx, cmin, mmin, bmax, dcmin, dmmin
                                )
                                if valid_src.max() == 0:
                                    continue
                                test_src = threshold_switch_selector(
                                    test_probs[0], test_probs[expert_idx], expert_idx, cmin, mmin, bmax, dcmin, dmmin
                                )
                                add_candidate(
                                    "threshold_switch",
                                    {
                                        "expert": expert_name,
                                        "expert_conf_min": cmin,
                                        "expert_margin_min": mmin,
                                        "base_conf_max": bmax,
                                        "delta_conf_min": dcmin,
                                        "delta_margin_min": dmmin,
                                    },
                                    valid_src,
                                    test_src,
                                )

    if best is None:
        raise ValueError("No selector candidates were generated.")
    if base_candidate is None:
        valid_src = np.zeros(len(y_valid_ref), dtype=np.int64)
        test_src = np.zeros(len(y_test_ref), dtype=np.int64)
        valid_prob = selected_prob_from_sources(valid_probs, valid_src, args.output_smooth)
        valid_pred = valid_prob.argmax(axis=1).astype(np.int64)
        metrics = compute_metrics(y_valid_ref.tolist(), valid_pred.tolist())
        metrics["calibration"] = calibration_metrics(y_valid_ref, valid_prob)
        row = {"strategy": "always", "config": {"source": named_payloads[0][0]}, "metrics": metrics}
        key = (metrics[args.select_metric], metrics["macro_f1"], metrics["accuracy"])
        test_prob = selected_prob_from_sources(test_probs, test_src, args.output_smooth)
        base_candidate = (key, row, valid_src, test_src, valid_prob, test_prob)
        candidates.append(base_candidate)

    original_best = max(candidates, key=lambda item: item[0])
    rank_select_metric = args.rank_select_metric or args.select_metric
    rank_bootstrap_samples = args.rank_bootstrap_samples if args.rank_bootstrap_samples > 0 else args.bootstrap_samples
    rank_cache: Dict[int, Dict[str, float]] = {}

    def rank_bootstrap_summary(candidate) -> Dict[str, float]:
        cache_key = id(candidate[1])
        if cache_key not in rank_cache:
            rank_cache[cache_key] = bootstrap_gain_guard(
                y_valid_ref,
                candidate[4],
                base_candidate[4],
                rank_select_metric,
                rank_bootstrap_samples,
                args.bootstrap_seed,
                args.bootstrap_quantile,
            )
        return rank_cache[cache_key]

    def candidate_rank_key(candidate) -> Tuple[float, float, float, float, float]:
        primary = candidate_rank_primary(candidate)
        penalty = candidate_calibration_penalty(candidate)
        score = primary - args.calibration_penalty_weight * penalty
        metrics = candidate[1]["metrics"]
        return (float(score), float(primary), metrics[args.select_metric], metrics["macro_f1"], metrics["accuracy"])

    def candidate_rank_primary(candidate) -> float:
        metrics = candidate[1]["metrics"]
        if args.rank_metric == "select_metric":
            return float(metrics[args.select_metric])
        elif args.rank_metric in {"accuracy", "macro_f1"}:
            return float(metrics[args.rank_metric])
        elif args.rank_metric == "bootstrap_gain_quantile":
            return float(rank_bootstrap_summary(candidate)["gain_quantile"])
        elif args.rank_metric == "bootstrap_mean_gain":
            return float(rank_bootstrap_summary(candidate)["mean_gain"])
        elif args.rank_metric == "bootstrap_win_rate":
            return float(rank_bootstrap_summary(candidate)["win_rate"])
        raise ValueError(args.rank_metric)

    def candidate_calibration_penalty(candidate) -> float:
        calibration = candidate[1]["metrics"].get("calibration") or {}
        value = calibration.get(args.calibration_penalty_metric)
        if value is None:
            return 0.0
        return float(value)

    def candidate_rank_details(candidate) -> Dict[str, float | str]:
        primary = candidate_rank_primary(candidate)
        penalty = candidate_calibration_penalty(candidate)
        return {
            "rank_metric": args.rank_metric,
            "rank_select_metric": rank_select_metric,
            "primary": float(primary),
            "calibration_penalty_metric": args.calibration_penalty_metric,
            "calibration_penalty": float(penalty),
            "calibration_penalty_weight": float(args.calibration_penalty_weight),
            "score": float(primary - args.calibration_penalty_weight * penalty),
        }

    selected_rejections = []
    selected_candidate = None
    rank_pool = candidates
    if args.rank_metric.startswith("bootstrap_") and args.rank_candidate_limit > 0:
        rank_pool = sorted(candidates, key=lambda item: item[0], reverse=True)[: args.rank_candidate_limit]
        if not any(item is base_candidate for item in rank_pool):
            rank_pool.append(base_candidate)
    ranked_candidates = sorted(rank_pool, key=candidate_rank_key, reverse=True)
    for candidate in ranked_candidates:
        selected_gain = candidate[1]["metrics"][args.select_metric] - base_candidate[1]["metrics"][args.select_metric]
        bootstrap_summary = bootstrap_gain_guard(
            y_valid_ref,
            candidate[4],
            base_candidate[4],
            args.select_metric,
            args.bootstrap_samples,
            args.bootstrap_seed,
            args.bootstrap_quantile,
        )
        bootstrap_reject = False
        if args.bootstrap_samples > 0:
            bootstrap_reject = (
                bootstrap_summary["win_rate"] < args.bootstrap_min_win_rate
                or bootstrap_summary["gain_quantile"] < args.bootstrap_min_gain_quantile
            )
        shift_summary = target_shift_guard(candidate[5], base_candidate[5])
        shift_reject = (
            shift_summary["prediction_change_rate"] > args.max_prediction_change_rate
            or shift_summary["prediction_js_divergence"] > args.max_prediction_js_divergence
        )
        gain_reject = selected_gain < args.min_valid_gain_over_base
        rank_summary = rank_bootstrap_summary(candidate) if args.rank_metric.startswith("bootstrap_") else None
        if gain_reject or bootstrap_reject or shift_reject:
            selected_rejections.append(
                {
                    "rank_metric": args.rank_metric,
                    "rank_select_metric": rank_select_metric,
                    "rank_details": candidate_rank_details(candidate),
                    "rank_key": list(candidate_rank_key(candidate)),
                    "rank_bootstrap_guard": rank_summary,
                    "selected_gain": selected_gain,
                    "min_valid_gain_over_base": args.min_valid_gain_over_base,
                    "bootstrap_guard": bootstrap_summary,
                    "bootstrap_min_win_rate": args.bootstrap_min_win_rate,
                    "bootstrap_min_gain_quantile": args.bootstrap_min_gain_quantile,
                    "target_shift_guard": shift_summary,
                    "max_prediction_change_rate": args.max_prediction_change_rate,
                    "max_prediction_js_divergence": args.max_prediction_js_divergence,
                    "rejected": candidate[1],
                    "reject_reasons": {
                        "gain": gain_reject,
                        "bootstrap": bootstrap_reject,
                        "target_shift": shift_reject,
                    },
                }
            )
            continue
        accepted_row = dict(candidate[1])
        accepted_row["rank_metric"] = args.rank_metric
        accepted_row["rank_select_metric"] = rank_select_metric
        accepted_row["rank_details"] = candidate_rank_details(candidate)
        accepted_row["rank_key"] = list(candidate_rank_key(candidate))
        if rank_summary is not None:
            accepted_row["rank_bootstrap_guard"] = rank_summary
        accepted_row["bootstrap_guard"] = bootstrap_summary
        accepted_row["target_shift_guard"] = shift_summary
        accepted_row["max_prediction_change_rate"] = args.max_prediction_change_rate
        accepted_row["max_prediction_js_divergence"] = args.max_prediction_js_divergence
        if selected_rejections:
            accepted_row["rejected_before_accept"] = selected_rejections[:5]
        selected_candidate = (candidate[0], accepted_row, candidate[2], candidate[3], candidate[4], candidate[5])
        break
    if selected_candidate is None:
        fallback_row = dict(base_candidate[1])
        fallback_row["fallback_reason"] = {
            "selected_gain": original_best[1]["metrics"][args.select_metric] - base_candidate[1]["metrics"][args.select_metric],
            "min_valid_gain_over_base": args.min_valid_gain_over_base,
            "rejected": original_best[1],
            "top_ranked": ranked_candidates[0][1],
            "top_ranked_rank_details": candidate_rank_details(ranked_candidates[0]),
            "top_ranked_key": list(candidate_rank_key(ranked_candidates[0])),
            "rejected_candidates": selected_rejections[:5],
        }
        selected_candidate = (
            base_candidate[0],
            fallback_row,
            base_candidate[2],
            base_candidate[3],
            base_candidate[4],
            base_candidate[5],
        )
    best = selected_candidate

    _, selected, selected_valid_source, selected_test_source, selected_valid_prob, selected_test_prob = best
    test_pred = selected_test_prob.argmax(axis=1).astype(np.int64)
    test_metrics = compute_metrics(y_test_ref.tolist(), test_pred.tolist())
    test_metrics["calibration"] = calibration_metrics(y_test_ref, selected_test_prob)
    print("selected_selector", json.dumps(selected, sort_keys=True))
    print("test_selector", json.dumps(test_metrics, indent=2, sort_keys=True))

    label_names, label_map = load_label_names(args.label_map)
    if label_names:
        print(classification_report(y_test_ref, test_pred, labels=list(range(len(label_names))), target_names=label_names, zero_division=0))
    else:
        print(classification_report(y_test_ref, test_pred, zero_division=0))

    payload = {
        "metrics": {"flow_level": test_metrics},
        "selected": selected,
        "valid_reports": reports,
        "inputs": [{"name": name, "path": path} for name, _, path in named_payloads],
        "label_map": label_map,
        "flow_ids": test_common,
        "flow_y_true": y_test_ref.tolist(),
        "flow_y_pred": test_pred.tolist(),
        "flow_prob": selected_test_prob.tolist(),
        "flow_source": selected_test_source.astype(int).tolist(),
        "valid_flow_ids": valid_common,
        "valid_y_true": y_valid_ref.tolist(),
        "valid_y_pred": selected_valid_prob.argmax(axis=1).astype(np.int64).tolist(),
        "valid_prob": selected_valid_prob.tolist(),
        "valid_source": selected_valid_source.astype(int).tolist(),
        "feature_config": {
            "select_metric": args.select_metric,
            "rank_select_metric": rank_select_metric,
            "rank_metric": args.rank_metric,
            "strategies": sorted(strategies),
            "unified_expert_slots": unified_expert_slots,
            "input_slot_status": input_slot_status,
            "skipped_inputs": skipped_inputs,
            "output_smooth": args.output_smooth,
            "bootstrap_samples": args.bootstrap_samples,
            "rank_bootstrap_samples": rank_bootstrap_samples,
            "rank_candidate_limit": args.rank_candidate_limit,
            "bootstrap_seed": args.bootstrap_seed,
            "bootstrap_quantile": args.bootstrap_quantile,
            "bootstrap_min_win_rate": args.bootstrap_min_win_rate,
            "bootstrap_min_gain_quantile": args.bootstrap_min_gain_quantile,
            "max_prediction_change_rate": args.max_prediction_change_rate,
            "max_prediction_js_divergence": args.max_prediction_js_divergence,
            "base_input": named_payloads[0][0],
            "calibration_penalty_weight": args.calibration_penalty_weight,
            "calibration_penalty_metric": args.calibration_penalty_metric,
        },
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()
