#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

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
    """Return a per-sample calibration gate in [0, 1].

    The gate lets confident samples keep their original evidence while allowing
    uncertain samples to receive stronger target-prior correction.
    """
    if mode == "none":
        return np.ones(prob.shape[0], dtype=np.float64)
    prob = prob.astype("float64")
    sorted_prob = np.sort(prob, axis=1)
    confidence = sorted_prob[:, -1]
    margin = sorted_prob[:, -1] - sorted_prob[:, -2] if prob.shape[1] > 1 else confidence
    if mode == "low_margin":
        if threshold <= 0:
            return np.ones(prob.shape[0], dtype=np.float64)
        return np.clip((threshold - margin) / threshold, 0.0, 1.0)
    if mode == "low_confidence":
        if threshold <= 0:
            return np.ones(prob.shape[0], dtype=np.float64)
        return np.clip((threshold - confidence) / threshold, 0.0, 1.0)
    if mode == "high_entropy":
        entropy = -np.sum(prob * np.log(prob.clip(min=1e-12)), axis=1)
        max_entropy = np.log(max(prob.shape[1], 2))
        norm_entropy = entropy / max(max_entropy, 1e-12)
        if threshold <= 0:
            return norm_entropy
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
    prior = np.linalg.solve(lhs, rhs)
    return simplex_project(prior)


def estimate_prior_em(target_prob: np.ndarray, source_prior: np.ndarray, max_iter: int = 200, tol: float = 1e-8) -> np.ndarray:
    source_prior = source_prior.astype("float64")
    prior = source_prior.copy()
    target_prob = target_prob.astype("float64").clip(min=1e-12)
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


def main() -> None:
    ap = argparse.ArgumentParser()
    input_group = ap.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input_json")
    input_group.add_argument(
        "--valid_npz",
        help="Packet validation probabilities with y_true and probabilities arrays.",
    )
    ap.add_argument(
        "--test_npz",
        default="",
        help="Aligned packet test probabilities; required with --valid_npz.",
    )
    ap.add_argument("--label_map", default="")
    ap.add_argument("--strengths", default="0,0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5")
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--select_metric", choices=["accuracy", "macro_f1"], default="accuracy")
    ap.add_argument("--prior_method", choices=["hard", "em", "blend"], default="hard")
    ap.add_argument("--gate_modes", default="none", help="Comma-separated prior gates: none,low_margin,low_confidence,high_entropy.")
    ap.add_argument("--gate_thresholds", default="0", help="Comma-separated thresholds for uncertainty gates. Ignored by --gate_modes none.")
    ap.add_argument("--target_prior_json", default="", help="Optional JSON providing a precomputed target prior.")
    ap.add_argument("--target_prior_key", default="em_estimated_target_prior", help="Prior key to read from --target_prior_json.")
    ap.add_argument(
        "--selection_scope",
        choices=["test_oracle", "valid_weighted", "hard_prior_match", "soft_prior_under_hard_cap"],
        default="test_oracle",
        help="test_oracle reproduces old analysis; valid_weighted uses target-prior-weighted validation; hard_prior_match and soft_prior_under_hard_cap are label-free target-prior criteria.",
    )
    ap.add_argument("--hard_prior_kl_cap", type=float, default=0.005, help="Hard-prior KL cap for --selection_scope soft_prior_under_hard_cap.")
    ap.add_argument("--output_json", default="")
    ap.add_argument("--output_npz", default="", help="Optional calibrated packet probabilities.")
    args = ap.parse_args()

    if args.valid_npz:
        if not args.test_npz:
            ap.error("--test_npz is required with --valid_npz")
        valid_data = np.load(args.valid_npz)
        test_data = np.load(args.test_npz)
        valid_prob = np.asarray(valid_data["probabilities"], dtype=np.float64)
        test_prob = np.asarray(test_data["probabilities"], dtype=np.float64)
        y_valid = np.asarray(valid_data["y_true"], dtype=np.int64)
        y_test = np.asarray(test_data["y_true"], dtype=np.int64)
        data = {}
        sample_unit = "one_packet"
        metric_scope = "packet_level"
    else:
        if args.test_npz:
            ap.error("--test_npz can only be used with --valid_npz")
        with open(args.input_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        valid_prob = np.asarray(data["valid_prob"], dtype=np.float64)
        test_prob = np.asarray(data["flow_prob"], dtype=np.float64)
        y_valid = np.asarray(data["valid_y_true"], dtype=np.int64)
        y_test = np.asarray(data["flow_y_true"], dtype=np.int64)
        sample_unit = "one_flow"
        metric_scope = "flow_level"
    if valid_prob.ndim != 2 or test_prob.ndim != 2:
        raise ValueError("probabilities must be rank-2 arrays")
    if valid_prob.shape[1] != test_prob.shape[1]:
        raise ValueError("validation/test class-count mismatch")
    if len(valid_prob) != len(y_valid) or len(test_prob) != len(y_test):
        raise ValueError("probability and label lengths do not match")
    num_classes = test_prob.shape[1]

    pred_valid = valid_prob.argmax(axis=1)
    pred_test = test_prob.argmax(axis=1)
    valid_prior = np.bincount(y_valid, minlength=num_classes).astype("float64")
    valid_prior = valid_prior / max(float(valid_prior.sum()), 1.0)
    hard_prior = estimate_prior_hard(y_valid, pred_valid, pred_test, num_classes, args.ridge)
    em_prior = estimate_prior_em(test_prob, valid_prior)
    if args.target_prior_json:
        with open(args.target_prior_json, "r", encoding="utf-8") as f:
            prior_data = json.load(f)
        if args.target_prior_key not in prior_data:
            raise ValueError(f"{args.target_prior_json} does not contain {args.target_prior_key}")
        estimated_prior = simplex_project(np.asarray(prior_data[args.target_prior_key], dtype="float64"))
    elif args.prior_method == "hard":
        estimated_prior = hard_prior
    elif args.prior_method == "em":
        estimated_prior = em_prior
    else:
        estimated_prior = simplex_project(0.5 * hard_prior + 0.5 * em_prior)
    prior_logit = np.log(estimated_prior + 1e-12) - np.log(valid_prior + 1e-12)
    base_logits = np.log(test_prob.clip(min=1e-12))
    valid_logits = np.log(valid_prob.clip(min=1e-12))
    valid_weight = estimated_prior[y_valid] / np.maximum(valid_prior[y_valid], 1e-12)

    reports = {}
    selection_reports = {}
    unsupervised_reports = {}
    best_key = None
    best_score = None
    best_pred = None
    best_valid_prob = None
    best_test_prob = None
    gate_modes = [x.strip() for x in args.gate_modes.split(",") if x.strip()]
    gate_thresholds = [float(x) for x in args.gate_thresholds.split(",") if x.strip()]
    if not gate_thresholds:
        gate_thresholds = [0.0]
    for raw in args.strengths.split(","):
        strength = float(raw)
        for gate_mode in gate_modes:
            thresholds = [0.0] if gate_mode == "none" else gate_thresholds
            for gate_threshold in thresholds:
                valid_gate = uncertainty_gate(valid_prob, gate_mode, gate_threshold)
                test_gate = uncertainty_gate(test_prob, gate_mode, gate_threshold)
                valid_adjusted = valid_logits + strength * valid_gate[:, None] * prior_logit[None, :]
                valid_adjusted = valid_adjusted - valid_adjusted.max(axis=1, keepdims=True)
                valid_prob_adjusted = np.exp(valid_adjusted)
                valid_prob_adjusted = valid_prob_adjusted / np.maximum(valid_prob_adjusted.sum(axis=1, keepdims=True), 1e-12)
                valid_pred = valid_prob_adjusted.argmax(axis=1).astype(np.int64)
                valid_metrics = compute_metrics(y_valid.tolist(), valid_pred.tolist(), sample_weight=valid_weight)
                test_logits = base_logits + strength * test_gate[:, None] * prior_logit[None, :]
                test_adjusted = test_logits - test_logits.max(axis=1, keepdims=True)
                test_prob_adjusted = np.exp(test_adjusted)
                test_prob_adjusted = test_prob_adjusted / np.maximum(test_prob_adjusted.sum(axis=1, keepdims=True), 1e-12)
                pred = test_prob_adjusted.argmax(axis=1).astype(np.int64)
                metrics = compute_metrics(y_test.tolist(), pred.tolist())
                pred_prior = np.bincount(pred, minlength=num_classes).astype("float64")
                pred_prior = pred_prior / max(float(pred_prior.sum()), 1.0)
                soft_prior = test_prob_adjusted.mean(axis=0)
                hard_prior_kl = float(np.sum(estimated_prior * (np.log(estimated_prior + 1e-12) - np.log(pred_prior + 1e-12))))
                soft_prior_kl = float(np.sum(estimated_prior * (np.log(estimated_prior + 1e-12) - np.log(soft_prior + 1e-12))))
                sorted_prob = np.sort(test_prob_adjusted, axis=1)
                confidence = float(sorted_prob[:, -1].mean())
                margin = float((sorted_prob[:, -1] - sorted_prob[:, -2]).mean()) if num_classes > 1 else confidence
                entropy = float((-test_prob_adjusted * np.log(test_prob_adjusted.clip(min=1e-12))).sum(axis=1).mean())
                gate_mean = float(test_gate.mean())
                key = f"{strength:g}" if gate_mode == "none" else f"{strength:g}|{gate_mode}|{gate_threshold:g}"
                reports[key] = metrics
                selection_reports[key] = valid_metrics
                unsupervised_reports[key] = {
                    "hard_prior_kl": hard_prior_kl,
                    "soft_prior_kl": soft_prior_kl,
                    "confidence": confidence,
                    "margin": margin,
                    "entropy": entropy,
                    "gate_mode": gate_mode,
                    "gate_threshold": gate_threshold,
                    "gate_mean": gate_mean,
                }
                print(f"candidate={key}", json.dumps(metrics, sort_keys=True))
                if args.selection_scope == "test_oracle":
                    selected_metrics = metrics
                    score = (selected_metrics[args.select_metric], selected_metrics["macro_f1"], selected_metrics["accuracy"])
                elif args.selection_scope == "valid_weighted":
                    selected_metrics = valid_metrics
                    score = (selected_metrics[args.select_metric], selected_metrics["macro_f1"], selected_metrics["accuracy"])
                elif args.selection_scope == "hard_prior_match":
                    score = (-hard_prior_kl, margin, confidence)
                else:
                    within_cap = 1.0 if hard_prior_kl <= args.hard_prior_kl_cap else 0.0
                    score = (within_cap, -soft_prior_kl if within_cap else -hard_prior_kl, margin, confidence)
                if best_score is None or score > best_score:
                    best_score = score
                    best_key = key
                    best_pred = pred
                    best_valid_prob = valid_prob_adjusted
                    best_test_prob = test_prob_adjusted

    label_names, label_map = load_label_names(args.label_map)
    if label_names:
        print(classification_report(y_test, best_pred, labels=list(range(len(label_names))), target_names=label_names, zero_division=0))
    else:
        print(classification_report(y_test, best_pred, zero_division=0))

    if args.output_json:
        payload = {
            "task": f"{metric_scope.replace('_', '-')}-classification",
            "sample_unit": sample_unit,
            "metrics": {metric_scope: reports[best_key]},
            "metrics_by_strength": reports,
            "selection_metrics_by_strength": selection_reports,
            "unsupervised_metrics_by_strength": unsupervised_reports,
            "selected_strength": best_key,
            "select_metric": args.select_metric,
            "selection_scope": args.selection_scope,
            "prior_method": args.prior_method,
            "target_prior_json": args.target_prior_json,
            "target_prior_key": args.target_prior_key if args.target_prior_json else "",
            "gate_modes": gate_modes,
            "gate_thresholds": gate_thresholds,
            "hard_prior_kl_cap": args.hard_prior_kl_cap,
            "estimated_target_prior": estimated_prior.tolist(),
            "hard_estimated_target_prior": hard_prior.tolist(),
            "em_estimated_target_prior": em_prior.tolist(),
            "valid_prior": valid_prior.tolist(),
            "label_map": label_map,
        }
        if args.valid_npz:
            payload["inputs"] = {"valid_npz": args.valid_npz, "test_npz": args.test_npz}
            payload["output_npz"] = args.output_npz
        else:
            # Preserve the original flow JSON contract for downstream tools.
            payload.update(
                {
                    "flow_ids": data.get("flow_ids", []),
                    "flow_y_true": y_test.tolist(),
                    "flow_y_pred": best_pred.tolist(),
                    "flow_prob": best_test_prob.tolist(),
                    "valid_flow_ids": data.get("valid_flow_ids", []),
                    "valid_y_true": y_valid.tolist(),
                    "valid_y_pred": best_valid_prob.argmax(axis=1).astype(np.int64).tolist(),
                    "valid_prob": best_valid_prob.tolist(),
                }
            )
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    if args.output_npz:
        if not args.valid_npz:
            ap.error("--output_npz is supported with packet --valid_npz input")
        output_npz = Path(args.output_npz)
        output_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_npz,
            y_true=y_test.astype(np.int64),
            probabilities=np.asarray(best_test_prob, dtype=np.float32),
        )


if __name__ == "__main__":
    main()
