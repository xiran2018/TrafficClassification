#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support


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


def simplex_project(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, len(u) + 1) > (cssv - 1))[0]
    if len(rho) == 0:
        return np.ones_like(v) / max(len(v), 1)
    rho = int(rho[-1])
    theta = (cssv[rho] - 1) / (rho + 1)
    w = np.maximum(v - theta, 0)
    return w / max(float(w.sum()), 1e-12)


def compute_metrics(y_true, y_pred, sample_weight=None):
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
        sample_weight=sample_weight,
    )
    p_weight, r_weight, f_weight, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
        sample_weight=sample_weight,
    )
    return {
        "accuracy": accuracy_score(y_true, y_pred, sample_weight=sample_weight) if len(y_true) else 0.0,
        "macro_precision": p_macro,
        "macro_recall": r_macro,
        "macro_f1": f_macro,
        "weighted_precision": p_weight,
        "weighted_recall": r_weight,
        "weighted_f1": f_weight,
    }


def uncertainty_gate(prob: np.ndarray, mode: str, threshold: float) -> np.ndarray:
    if mode == "none":
        return np.ones(prob.shape[0], dtype=np.float64)
    prob = prob.astype("float64")
    sorted_prob = np.sort(prob, axis=1)
    confidence = sorted_prob[:, -1]
    margin = sorted_prob[:, -1] - sorted_prob[:, -2] if prob.shape[1] > 1 else confidence
    if mode == "low_margin":
        return np.clip((threshold - margin) / max(threshold, 1e-12), 0.0, 1.0)
    if mode == "low_confidence":
        return np.clip((threshold - confidence) / max(threshold, 1e-12), 0.0, 1.0)
    if mode == "high_entropy":
        entropy = -np.sum(prob * np.log(prob.clip(min=1e-12)), axis=1)
        norm_entropy = entropy / max(np.log(max(prob.shape[1], 2)), 1e-12)
        return np.clip((norm_entropy - threshold) / max(1.0 - threshold, 1e-12), 0.0, 1.0)
    raise ValueError(f"Unknown gate mode: {mode}")


def estimate_prior_hard(y_cal: np.ndarray, pred_cal: np.ndarray, pred_target: np.ndarray, num_classes: int, ridge: float) -> np.ndarray:
    cm = confusion_matrix(y_cal, pred_cal, labels=list(range(num_classes))).astype("float64")
    cm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1.0)
    q = np.bincount(pred_target, minlength=num_classes).astype("float64")
    q = q / max(float(q.sum()), 1.0)
    a = cm.T
    lhs = a.T @ a + ridge * np.eye(num_classes)
    rhs = a.T @ q
    return simplex_project(np.linalg.solve(lhs, rhs))


def estimate_prior_em(target_prob: np.ndarray, source_prior: np.ndarray, max_iter: int = 200, tol: float = 1e-8) -> np.ndarray:
    source_prior = source_prior.astype("float64")
    target_prob = target_prob.astype("float64").clip(min=1e-12)
    prior = source_prior.copy()
    for _ in range(max_iter):
        ratio = prior / np.maximum(source_prior, 1e-12)
        posterior = target_prob * ratio[None, :]
        posterior = posterior / np.maximum(posterior.sum(axis=1, keepdims=True), 1e-12)
        new_prior = posterior.mean(axis=0)
        if np.max(np.abs(new_prior - prior)) < tol:
            prior = new_prior
            break
        prior = new_prior
    return simplex_project(prior)


def kl_div(p: np.ndarray, q: np.ndarray) -> float:
    p = p.astype("float64")
    q = q.astype("float64")
    return float(np.sum(p * (np.log(p + 1e-12) - np.log(q + 1e-12))))


def adjust_prob(prob: np.ndarray, prior_logit: np.ndarray, strength: float, gate_mode: str, gate_threshold: float) -> Tuple[np.ndarray, np.ndarray]:
    gate = uncertainty_gate(prob, gate_mode, gate_threshold)
    logits = np.log(prob.clip(min=1e-12)) + strength * gate[:, None] * prior_logit[None, :]
    logits = logits - logits.max(axis=1, keepdims=True)
    out = np.exp(logits)
    out = out / np.maximum(out.sum(axis=1, keepdims=True), 1e-12)
    return out, gate


def normalize_prob(prob: np.ndarray) -> np.ndarray:
    prob = np.asarray(prob, dtype=np.float64)
    return prob / np.maximum(prob.sum(axis=1, keepdims=True), 1e-12)


def candidate_key(method: str, strength: float, gate_mode: str, gate_threshold: float) -> str:
    if gate_mode == "none":
        return f"{method}:{strength:g}"
    return f"{method}:{strength:g}|{gate_mode}|{gate_threshold:g}"


def make_candidates(
    valid_prob: np.ndarray,
    test_prob: np.ndarray,
    y_valid: np.ndarray,
    y_test: np.ndarray,
    methods: List[str],
    strengths: List[float],
    gate_modes: List[str],
    gate_thresholds: List[float],
    ridge: float,
    select_metric: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    num_classes = test_prob.shape[1]
    pred_valid = valid_prob.argmax(axis=1)
    pred_test = test_prob.argmax(axis=1)
    valid_prior = np.bincount(y_valid, minlength=num_classes).astype("float64")
    valid_prior = valid_prior / max(float(valid_prior.sum()), 1.0)
    hard_prior = estimate_prior_hard(y_valid, pred_valid, pred_test, num_classes, ridge)
    em_prior = estimate_prior_em(test_prob, valid_prior)
    priors = {
        "hard": hard_prior,
        "em": em_prior,
        "blend": simplex_project(0.5 * hard_prior + 0.5 * em_prior),
    }

    candidates = []
    for method in methods:
        estimated_prior = priors[method]
        prior_logit = np.log(estimated_prior + 1e-12) - np.log(valid_prior + 1e-12)
        valid_weight = estimated_prior[y_valid] / np.maximum(valid_prior[y_valid], 1e-12)
        for strength in strengths:
            for gate_mode in gate_modes:
                thresholds = [0.0] if gate_mode == "none" else gate_thresholds
                for gate_threshold in thresholds:
                    valid_adj, valid_gate = adjust_prob(valid_prob, prior_logit, strength, gate_mode, gate_threshold)
                    test_adj, test_gate = adjust_prob(test_prob, prior_logit, strength, gate_mode, gate_threshold)
                    valid_pred = valid_adj.argmax(axis=1).astype(np.int64)
                    test_pred = test_adj.argmax(axis=1).astype(np.int64)
                    pred_prior = np.bincount(test_pred, minlength=num_classes).astype("float64")
                    pred_prior = pred_prior / max(float(pred_prior.sum()), 1.0)
                    soft_prior = test_adj.mean(axis=0)
                    sorted_prob = np.sort(test_adj, axis=1)
                    confidence = float(sorted_prob[:, -1].mean())
                    margin = float((sorted_prob[:, -1] - sorted_prob[:, -2]).mean()) if num_classes > 1 else confidence
                    entropy = float((-test_adj * np.log(test_adj.clip(min=1e-12))).sum(axis=1).mean())
                    valid_metrics = compute_metrics(y_valid.tolist(), valid_pred.tolist())
                    valid_weighted_metrics = compute_metrics(y_valid.tolist(), valid_pred.tolist(), sample_weight=valid_weight)
                    test_metrics = compute_metrics(y_test.tolist(), test_pred.tolist())
                    candidates.append(
                        {
                            "key": candidate_key(method, strength, gate_mode, gate_threshold),
                            "method": method,
                            "strength": strength,
                            "gate_mode": gate_mode,
                            "gate_threshold": gate_threshold,
                            "valid_prob": valid_adj,
                            "test_prob": test_adj,
                            "valid_metrics": valid_metrics,
                            "valid_weighted_metrics": valid_weighted_metrics,
                            "test_metrics": test_metrics,
                            "hard_prior_kl": kl_div(estimated_prior, pred_prior),
                            "soft_prior_kl": kl_div(estimated_prior, soft_prior),
                            "confidence": confidence,
                            "margin": margin,
                            "entropy": entropy,
                            "gate_mean": float(test_gate.mean()),
                        }
                    )
    context = {
        "valid_prior": valid_prior.tolist(),
        "hard_estimated_target_prior": hard_prior.tolist(),
        "em_estimated_target_prior": em_prior.tolist(),
        "blend_estimated_target_prior": priors["blend"].tolist(),
        "select_metric": select_metric,
    }
    return candidates, context


def select_pool(candidates: List[Dict[str, Any]], strategy: str, select_metric: str, top_k: int, hard_cap: float) -> List[Dict[str, Any]]:
    if strategy == "valid_weighted":
        ranked = sorted(
            candidates,
            key=lambda c: (
                c["valid_weighted_metrics"][select_metric],
                c["valid_weighted_metrics"]["macro_f1"],
                c["valid_weighted_metrics"]["accuracy"],
            ),
            reverse=True,
        )
    elif strategy == "valid":
        ranked = sorted(
            candidates,
            key=lambda c: (
                c["valid_metrics"][select_metric],
                c["valid_metrics"]["macro_f1"],
                c["valid_metrics"]["accuracy"],
            ),
            reverse=True,
        )
    elif strategy == "prior_softcap":
        eligible = [c for c in candidates if c["hard_prior_kl"] <= hard_cap]
        if not eligible:
            eligible = candidates
        ranked = sorted(eligible, key=lambda c: (-c["soft_prior_kl"], c["margin"], c["confidence"]), reverse=True)
    elif strategy == "prior_softcap_valid":
        eligible = [c for c in candidates if c["hard_prior_kl"] <= hard_cap]
        if not eligible:
            eligible = candidates
        ranked = sorted(
            eligible,
            key=lambda c: (
                c["valid_weighted_metrics"][select_metric],
                -c["soft_prior_kl"],
                c["margin"],
                c["confidence"],
            ),
            reverse=True,
        )
    elif strategy == "prior_band":
        eligible = [c for c in candidates if c["hard_prior_kl"] <= hard_cap]
        if not eligible:
            eligible = candidates
        ranked = sorted(eligible, key=lambda c: (c["margin"], c["confidence"], -c["soft_prior_kl"]), reverse=True)
    else:
        raise ValueError(f"Unknown pool strategy: {strategy}")
    return ranked[: max(1, top_k)]


def ensemble_prob(pool: List[Dict[str, Any]], split: str, mode: str, temperature: float) -> np.ndarray:
    probs = [c[f"{split}_prob"] for c in pool]
    if mode == "mean":
        out = np.mean(probs, axis=0)
    elif mode == "log_mean":
        logp = np.mean([np.log(p.clip(min=1e-12)) for p in probs], axis=0) / max(temperature, 1e-12)
        logp = logp - logp.max(axis=1, keepdims=True)
        out = np.exp(logp)
    elif mode == "vote":
        num_classes = probs[0].shape[1]
        out = np.zeros_like(probs[0])
        for p in probs:
            pred = p.argmax(axis=1)
            out[np.arange(len(pred)), pred] += 1.0
        out = out + 1e-6
    else:
        raise ValueError(f"Unknown ensemble mode: {mode}")
    return normalize_prob(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_json", required=True)
    ap.add_argument("--label_map", default="")
    ap.add_argument("--methods", default="blend", help="Comma-separated prior estimators: hard,em,blend.")
    ap.add_argument("--strengths", default="0.5,0.6,0.7,0.8,0.85,0.9,0.95,1.0,1.05,1.1,1.2")
    ap.add_argument("--gate_modes", default="none,low_margin,high_entropy,low_confidence")
    ap.add_argument("--gate_thresholds", default="0.45,0.5,0.55,0.6,0.62,0.64,0.66,0.68,0.7,0.72,0.75")
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--select_metric", choices=["accuracy", "macro_f1"], default="accuracy")
    ap.add_argument(
        "--pool_strategy",
        choices=["valid", "valid_weighted", "prior_softcap", "prior_softcap_valid", "prior_band"],
        default="prior_softcap_valid",
    )
    ap.add_argument("--top_k", type=int, default=7)
    ap.add_argument("--hard_prior_kl_cap", type=float, default=0.017)
    ap.add_argument("--ensemble_mode", choices=["mean", "log_mean", "vote"], default="mean")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--oracle_diagnostics", action="store_true", help="Also report test-label oracle single/pool metrics for diagnosis only.")
    ap.add_argument("--output_json", default="")
    args = ap.parse_args()

    with open(args.input_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    valid_prob = normalize_prob(np.asarray(data["valid_prob"], dtype=np.float64))
    test_prob = normalize_prob(np.asarray(data["flow_prob"], dtype=np.float64))
    y_valid = np.asarray(data["valid_y_true"], dtype=np.int64)
    y_test = np.asarray(data["flow_y_true"], dtype=np.int64)
    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    strengths = [float(x) for x in args.strengths.split(",") if x.strip()]
    gate_modes = [x.strip() for x in args.gate_modes.split(",") if x.strip()]
    gate_thresholds = [float(x) for x in args.gate_thresholds.split(",") if x.strip()]

    candidates, context = make_candidates(
        valid_prob,
        test_prob,
        y_valid,
        y_test,
        methods,
        strengths,
        gate_modes,
        gate_thresholds,
        args.ridge,
        args.select_metric,
    )
    pool = select_pool(candidates, args.pool_strategy, args.select_metric, args.top_k, args.hard_prior_kl_cap)
    valid_ens = ensemble_prob(pool, "valid", args.ensemble_mode, args.temperature)
    test_ens = ensemble_prob(pool, "test", args.ensemble_mode, args.temperature)
    y_valid_pred = valid_ens.argmax(axis=1).astype(np.int64)
    y_test_pred = test_ens.argmax(axis=1).astype(np.int64)
    valid_metrics = compute_metrics(y_valid.tolist(), y_valid_pred.tolist())
    test_metrics = compute_metrics(y_test.tolist(), y_test_pred.tolist())
    selected = {
        "pool_strategy": args.pool_strategy,
        "top_k": args.top_k,
        "hard_prior_kl_cap": args.hard_prior_kl_cap,
        "ensemble_mode": args.ensemble_mode,
        "temperature": args.temperature,
        "candidate_keys": [c["key"] for c in pool],
    }
    print("selected_prior_ensemble", json.dumps(selected, sort_keys=True))
    print("valid_prior_ensemble", json.dumps(valid_metrics, indent=2, sort_keys=True))
    print("test_prior_ensemble", json.dumps(test_metrics, indent=2, sort_keys=True))

    oracle = {}
    if args.oracle_diagnostics:
        best_single = max(candidates, key=lambda c: (c["test_metrics"][args.select_metric], c["test_metrics"]["macro_f1"], c["test_metrics"]["accuracy"]))
        oracle["best_single"] = {"key": best_single["key"], "metrics": best_single["test_metrics"]}
        ranked = sorted(candidates, key=lambda c: (c["test_metrics"][args.select_metric], c["test_metrics"]["macro_f1"], c["test_metrics"]["accuracy"]), reverse=True)
        for k in [3, 5, 7, 11, 21, 50]:
            sub = ranked[: min(k, len(ranked))]
            prob = ensemble_prob(sub, "test", args.ensemble_mode, args.temperature)
            pred = prob.argmax(axis=1).astype(np.int64)
            oracle[f"top_{k}_{args.ensemble_mode}"] = compute_metrics(y_test.tolist(), pred.tolist())
        print("oracle_diagnostics", json.dumps(oracle, indent=2, sort_keys=True))

    label_names, label_map = load_label_names(args.label_map)
    if label_names:
        print(classification_report(y_test, y_test_pred, labels=list(range(len(label_names))), target_names=label_names, zero_division=0))
    else:
        print(classification_report(y_test, y_test_pred, zero_division=0))

    if args.output_json:
        payload = {
            "metrics": {"flow_level": test_metrics},
            "valid_metrics": valid_metrics,
            "selected": selected,
            "context": context,
            "candidate_reports": [
                {
                    "key": c["key"],
                    "method": c["method"],
                    "strength": c["strength"],
                    "gate_mode": c["gate_mode"],
                    "gate_threshold": c["gate_threshold"],
                    "valid_metrics": c["valid_metrics"],
                    "valid_weighted_metrics": c["valid_weighted_metrics"],
                    "test_metrics": c["test_metrics"],
                    "hard_prior_kl": c["hard_prior_kl"],
                    "soft_prior_kl": c["soft_prior_kl"],
                    "confidence": c["confidence"],
                    "margin": c["margin"],
                    "entropy": c["entropy"],
                    "gate_mean": c["gate_mean"],
                }
                for c in candidates
            ],
            "oracle_diagnostics": oracle,
            "label_map": label_map,
            "flow_ids": data.get("flow_ids", []),
            "flow_y_true": y_test.tolist(),
            "flow_y_pred": y_test_pred.tolist(),
            "flow_prob": test_ens.tolist(),
            "valid_flow_ids": data.get("valid_flow_ids", []),
            "valid_y_true": y_valid.tolist(),
            "valid_y_pred": y_valid_pred.tolist(),
            "valid_prob": valid_ens.tolist(),
            "input_json": args.input_json,
        }
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()
