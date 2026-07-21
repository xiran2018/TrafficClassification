#!/usr/bin/env python3
"""Fuse same-dataset predictions from multiple ready-made folds.

The SWEET split folders expose a shared test set with several train/valid
partitions. This script treats fold-specific models as a stability ensemble:
all inputs must predict the same shared test flow ids, and the output is a
test-label-free consensus over those fold models. Some optional modes estimate
reliability from each fold's held-out validation labels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support


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


def normalize_prob(prob: np.ndarray) -> np.ndarray:
    prob = np.asarray(prob, dtype=np.float64)
    prob = np.clip(prob, 1e-12, None)
    return prob / np.maximum(prob.sum(axis=1, keepdims=True), 1e-12)


def parse_input(raw: List[str]) -> Tuple[str, str]:
    if len(raw) != 2:
        raise ValueError("--input expects NAME JSON")
    return raw[0], raw[1]


def load_payload(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    required = ["flow_ids", "flow_y_true", "flow_prob"]
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"{path} is missing required fields: {missing}")
    return data


def align_test(data: Dict[str, Any], fids: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    idx = {str(fid): i for i, fid in enumerate(data["flow_ids"])}
    y = np.asarray([data["flow_y_true"][idx[fid]] for fid in fids], dtype=np.int64)
    p = normalize_prob(np.asarray([data["flow_prob"][idx[fid]] for fid in fids], dtype=np.float64))
    return y, p


def validation_class_reliability(
    data: Dict[str, Any],
    num_classes: int,
    smoothing: float = 4.0,
) -> np.ndarray:
    y_true = np.asarray(data.get("valid_y_true", []), dtype=np.int64)
    valid_prob = np.asarray(data.get("valid_prob", []), dtype=np.float64)
    if y_true.size == 0 or valid_prob.ndim != 2 or valid_prob.shape[0] != y_true.size:
        raise ValueError("class_reliability modes require valid_y_true and valid_prob in every input")
    y_pred = valid_prob.argmax(axis=1)
    global_accuracy = float((y_true == y_pred).mean())
    reliability = np.zeros(num_classes, dtype=np.float64)
    smoothing = max(float(smoothing), 0.0)
    for cls in range(num_classes):
        tp = float(np.sum((y_true == cls) & (y_pred == cls)))
        predicted = float(np.sum(y_pred == cls))
        support = float(np.sum(y_true == cls))
        precision = (tp + smoothing * global_accuracy) / max(predicted + smoothing, 1e-12)
        recall = (tp + smoothing * global_accuracy) / max(support + smoothing, 1e-12)
        reliability[cls] = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return np.clip(reliability, 1e-3, 1.0)


def validation_confusion_likelihood(
    data: Dict[str, Any],
    num_classes: int,
    smoothing: float = 1.0,
) -> np.ndarray:
    y_true = np.asarray(data.get("valid_y_true", []), dtype=np.int64)
    valid_prob = np.asarray(data.get("valid_prob", []), dtype=np.float64)
    if y_true.size == 0 or valid_prob.ndim != 2 or valid_prob.shape != (y_true.size, num_classes):
        raise ValueError("confusion_em requires valid_y_true and valid_prob in every input")
    counts = np.full((num_classes, num_classes), max(float(smoothing), 0.0), dtype=np.float64)
    for label, prob in zip(y_true, normalize_prob(valid_prob)):
        if 0 <= label < num_classes:
            counts[label] += prob
    return counts / np.maximum(counts.sum(axis=1, keepdims=True), 1e-12)


def confusion_em_fusion(
    probs: List[np.ndarray],
    confusion_likelihoods: List[np.ndarray],
    prior_anchor_weight: float = 0.05,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray, int]:
    num_classes = probs[0].shape[1]
    uniform = np.full(num_classes, 1.0 / num_classes, dtype=np.float64)
    prior = normalize_prob(np.mean(probs, axis=0)).mean(axis=0)
    prior = prior / prior.sum()
    posterior = np.tile(prior, (probs[0].shape[0], 1))
    iterations = 0
    for iteration in range(max(1, int(max_iter))):
        log_posterior = np.log(np.clip(prior, 1e-12, 1.0))[None, :].repeat(len(posterior), axis=0)
        for prob, confusion in zip(probs, confusion_likelihoods):
            likelihood = normalize_prob(prob) @ confusion.T
            log_posterior += np.log(np.clip(likelihood, 1e-12, 1.0))
        log_posterior -= log_posterior.max(axis=1, keepdims=True)
        posterior = normalize_prob(np.exp(log_posterior))
        updated = posterior.mean(axis=0)
        anchor = max(float(prior_anchor_weight), 0.0)
        updated = (updated + anchor * uniform) / (1.0 + anchor)
        updated /= updated.sum()
        iterations = iteration + 1
        if np.max(np.abs(updated - prior)) < tol:
            prior = updated
            break
        prior = updated
    return posterior, prior, iterations


def fuse_probs(
    probs: List[np.ndarray],
    mode: str,
    class_reliability: List[np.ndarray] | None = None,
) -> np.ndarray:
    if mode == "mean":
        return normalize_prob(np.mean(probs, axis=0))
    if mode == "log_mean":
        logp = np.mean([np.log(np.clip(p, 1e-12, 1.0)) for p in probs], axis=0)
        logp = logp - logp.max(axis=1, keepdims=True)
        return normalize_prob(np.exp(logp))
    if mode in {"vote", "vote_priority"}:
        out = np.zeros_like(probs[0], dtype=np.float64)
        for p in probs:
            pred = p.argmax(axis=1)
            out[np.arange(len(pred)), pred] += 1.0
        if mode == "vote_priority":
            priority_pred = probs[0].argmax(axis=1)
            max_votes = out.max(axis=1)
            tied = out[np.arange(out.shape[0]), priority_pred] == max_votes
            out[tied] += 0.0
            out[np.where(tied)[0], priority_pred[tied]] += 1e-3
        out += 1e-6
        return normalize_prob(out)
    if mode in {"class_reliability_mean", "class_reliability_log_mean", "class_reliability_vote"}:
        if not class_reliability or len(class_reliability) != len(probs):
            raise ValueError(f"{mode} requires one validation reliability vector per input")
        weights = np.stack(class_reliability, axis=0)
        weights = weights / np.maximum(weights.sum(axis=0, keepdims=True), 1e-12)
        if mode == "class_reliability_mean":
            stacked = np.stack(probs, axis=0)
            return normalize_prob((stacked * weights[:, None, :]).sum(axis=0))
        if mode == "class_reliability_log_mean":
            stacked_log = np.stack([np.log(np.clip(p, 1e-12, 1.0)) for p in probs], axis=0)
            logp = (stacked_log * weights[:, None, :]).sum(axis=0)
            logp -= logp.max(axis=1, keepdims=True)
            return normalize_prob(np.exp(logp))
        out = np.zeros_like(probs[0], dtype=np.float64)
        for model_idx, p in enumerate(probs):
            pred = p.argmax(axis=1)
            out[np.arange(len(pred)), pred] += class_reliability[model_idx][pred]
        out += 1e-3 * np.mean(probs, axis=0)
        return normalize_prob(out)
    raise ValueError(mode)


def selected_mode(probs: List[np.ndarray], requested: str, confidence_threshold: float) -> str:
    if requested != "auto_confidence":
        return requested
    mean_conf = float(np.mean([p.max(axis=1).mean() for p in probs]))
    return "vote_priority" if mean_conf >= confidence_threshold else "log_mean"


def validation_selective_threshold(
    data: Dict[str, Any],
    min_precision: float = 0.9,
    min_coverage: float = 0.1,
) -> Tuple[float, Dict[str, float]]:
    """Choose the widest validation subset meeting a precision guarantee.

    The threshold is learned from the anchor fold only. Test labels and the
    other experts' test performance never participate in this decision.
    """
    y_true = np.asarray(data.get("valid_y_true", []), dtype=np.int64)
    valid_prob = np.asarray(data.get("valid_prob", []), dtype=np.float64)
    if y_true.size == 0 or valid_prob.ndim != 2 or valid_prob.shape[0] != y_true.size:
        raise ValueError(
            "selective_anchor_vote requires valid_y_true and valid_prob in the anchor input"
        )
    valid_prob = normalize_prob(valid_prob)
    confidence = valid_prob.max(axis=1)
    correct = valid_prob.argmax(axis=1) == y_true
    order = np.argsort(-confidence, kind="stable")
    sorted_confidence = confidence[order]
    cumulative_precision = np.cumsum(correct[order]) / np.arange(1, y_true.size + 1)
    coverage = np.arange(1, y_true.size + 1) / float(y_true.size)
    eligible = np.flatnonzero(
        (cumulative_precision >= float(min_precision))
        & (coverage >= float(min_coverage))
    )
    if eligible.size == 0:
        return float("inf"), {
            "validation_precision": 0.0,
            "validation_coverage": 0.0,
            "selected_count": 0,
        }
    # The last eligible prefix maximizes anchor coverage under the guarantee.
    selected = int(eligible[-1])
    return float(sorted_confidence[selected]), {
        "validation_precision": float(cumulative_precision[selected]),
        "validation_coverage": float(coverage[selected]),
        "selected_count": selected + 1,
    }


def selective_anchor_vote(
    probs: List[np.ndarray],
    threshold: float,
) -> Tuple[np.ndarray, int]:
    if not probs:
        raise ValueError("selective_anchor_vote requires at least one input")
    fused = fuse_probs(probs, "vote_priority")
    anchor = normalize_prob(probs[0])
    selected = anchor.max(axis=1) >= float(threshold)
    fused[selected] = anchor[selected]
    return normalize_prob(fused), int(selected.sum())


def write_payload(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", nargs=2, action="append", metavar=("NAME", "JSON"), required=True)
    ap.add_argument(
        "--mode",
        choices=[
            "mean", "log_mean", "vote", "vote_priority", "auto_confidence",
            "class_reliability_mean", "class_reliability_log_mean", "class_reliability_vote",
            "class_reliability_tie_break",
            "selective_anchor_vote",
            "confusion_em", "confusion_em_tie_break",
        ],
        default="auto_confidence",
    )
    ap.add_argument("--confidence_threshold", type=float, default=0.9)
    ap.add_argument("--reliability_smoothing", type=float, default=4.0)
    ap.add_argument("--anchor_min_precision", type=float, default=0.9)
    ap.add_argument("--anchor_min_coverage", type=float, default=0.1)
    ap.add_argument("--confusion_smoothing", type=float, default=1.0)
    ap.add_argument("--confusion_prior_anchor_weight", type=float, default=0.05)
    ap.add_argument("--confusion_em_max_iter", type=int, default=100)
    ap.add_argument("--label_map", default="")
    ap.add_argument("--output_json", required=True)
    ap.add_argument(
        "--fold_alias",
        action="append",
        default=[],
        help="Deprecated compatibility flag. Consensus outputs are never copied as single-fold results.",
    )
    ap.add_argument("--no_report", action="store_true")
    args = ap.parse_args()

    named = [parse_input(raw) for raw in args.input]
    payloads = [(name, load_payload(path), path) for name, path in named]
    common = sorted(set.intersection(*(set(map(str, data["flow_ids"])) for _, data, _ in payloads)))
    if not common:
        raise ValueError("No common flow ids across inputs.")

    probs = []
    class_reliability = []
    confusion_likelihoods = []
    y_ref = None
    input_reports = []
    for name, data, path in payloads:
        y, p = align_test(data, common)
        if y_ref is None:
            y_ref = y
        elif not np.array_equal(y_ref, y):
            raise ValueError(f"Labels do not align for input {name}.")
        pred = p.argmax(axis=1).astype(np.int64)
        probs.append(p)
        reliability = (
            validation_class_reliability(data, p.shape[1], args.reliability_smoothing)
            if args.mode.startswith("class_reliability_")
            else np.ones(p.shape[1], dtype=np.float64)
        )
        class_reliability.append(reliability)
        confusion = (
            validation_confusion_likelihood(data, p.shape[1], args.confusion_smoothing)
            if args.mode in {"confusion_em", "confusion_em_tie_break"}
            else np.eye(p.shape[1], dtype=np.float64)
        )
        confusion_likelihoods.append(confusion)
        input_reports.append({
            "name": name,
            "path": path,
            "metrics": compute_metrics(y.tolist(), pred.tolist()),
            "mean_confidence": float(p.max(axis=1).mean()),
            "validation_class_reliability": reliability.tolist(),
            "validation_confusion_likelihood": confusion.tolist()
            if args.mode in {"confusion_em", "confusion_em_tie_break"} else None,
        })

    assert y_ref is not None
    mode = selected_mode(probs, args.mode, args.confidence_threshold)
    estimated_target_prior = None
    em_iterations = 0
    em_tie_count = 0
    selective_threshold = None
    selective_report = None
    selective_test_count = 0
    if mode == "selective_anchor_vote":
        selective_threshold, selective_report = validation_selective_threshold(
            payloads[0][1],
            args.anchor_min_precision,
            args.anchor_min_coverage,
        )
        fused, selective_test_count = selective_anchor_vote(probs, selective_threshold)
    elif mode == "class_reliability_tie_break":
        fused = fuse_probs(probs, "vote_priority")
        hard_predictions = np.stack([prob.argmax(axis=1) for prob in probs], axis=1)
        tie_mask = np.asarray([len(set(row.tolist())) == len(probs) for row in hard_predictions])
        for row_idx in np.flatnonzero(tie_mask):
            candidates = hard_predictions[row_idx]
            scores = np.asarray([
                class_reliability[model_idx][class_id]
                for model_idx, class_id in enumerate(candidates)
            ])
            selected_model = int(scores.argmax())
            selected_class = int(candidates[selected_model])
            fused[row_idx] = 1e-6 * np.mean([prob[row_idx] for prob in probs], axis=0)
            fused[row_idx, selected_class] += 1.0
        fused = normalize_prob(fused)
        em_tie_count = int(tie_mask.sum())
    elif mode in {"confusion_em", "confusion_em_tie_break"}:
        em_prob, estimated_target_prior, em_iterations = confusion_em_fusion(
            probs,
            confusion_likelihoods,
            args.confusion_prior_anchor_weight,
            args.confusion_em_max_iter,
        )
        if mode == "confusion_em":
            fused = em_prob
        else:
            fused = fuse_probs(probs, "vote_priority")
            hard_predictions = np.stack([prob.argmax(axis=1) for prob in probs], axis=1)
            tie_mask = np.asarray([len(set(row.tolist())) == len(probs) for row in hard_predictions])
            fused[tie_mask] = em_prob[tie_mask]
            em_tie_count = int(tie_mask.sum())
    else:
        fused = fuse_probs(probs, mode, class_reliability)
    pred = fused.argmax(axis=1).astype(np.int64)
    metrics = compute_metrics(y_ref.tolist(), pred.tolist())
    label_names, label_map = load_label_names(args.label_map)
    config = {
        "result_scope": "cross_fold_consensus",
        "requested_mode": args.mode,
        "selected_mode": mode,
        "confidence_threshold": args.confidence_threshold,
        "reliability_smoothing": args.reliability_smoothing,
        "anchor_min_precision": args.anchor_min_precision,
        "anchor_min_coverage": args.anchor_min_coverage,
        "selective_anchor_threshold": selective_threshold,
        "selective_anchor_validation": selective_report,
        "selective_anchor_test_count": selective_test_count,
        "confusion_smoothing": args.confusion_smoothing,
        "confusion_prior_anchor_weight": args.confusion_prior_anchor_weight,
        "confusion_em_iterations": em_iterations,
        "confusion_em_tie_count": em_tie_count,
        "tie_break_count": em_tie_count,
        "estimated_target_prior": estimated_target_prior.tolist() if estimated_target_prior is not None else None,
        "mean_input_confidence": float(np.mean([r["mean_confidence"] for r in input_reports])),
        "num_inputs": len(input_reports),
        "num_flows": len(common),
    }
    print("cross_fold_consensus", json.dumps({"metrics": metrics, "config": config}, indent=2, sort_keys=True))
    if not args.no_report:
        if label_names:
            print(classification_report(y_ref, pred, labels=list(range(len(label_names))), target_names=label_names, zero_division=0))
        else:
            print(classification_report(y_ref, pred, zero_division=0))

    payload = {
        "result_scope": "cross_fold_consensus",
        "metrics": {"flow_level": metrics},
        "label_map": label_map,
        "flow_ids": common,
        "flow_y_true": y_ref.tolist(),
        "flow_y_pred": pred.tolist(),
        "flow_prob": fused.tolist(),
        "inputs": input_reports,
        "config": config,
    }
    output = Path(args.output_json)
    write_payload(output, payload)
    if args.fold_alias:
        print(
            "warning: --fold_alias is deprecated and ignored; a cross-fold "
            "ensemble must not be reported as an independent fold result."
        )


if __name__ == "__main__":
    main()
