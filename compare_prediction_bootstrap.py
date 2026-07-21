#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, f1_score


def load_predictions(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    required = ("flow_ids", "flow_y_true", "flow_prob")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"{path} is missing fields: {missing}")
    return payload


def align_predictions(
    baseline: Dict[str, Any], candidate: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Sequence[str]]:
    base = {
        str(fid): (int(label), np.asarray(prob, dtype=np.float64))
        for fid, label, prob in zip(
            baseline["flow_ids"], baseline["flow_y_true"], baseline["flow_prob"]
        )
    }
    other = {
        str(fid): (int(label), np.asarray(prob, dtype=np.float64))
        for fid, label, prob in zip(
            candidate["flow_ids"], candidate["flow_y_true"], candidate["flow_prob"]
        )
    }
    if len(base) != len(baseline["flow_ids"]) or len(other) != len(candidate["flow_ids"]):
        raise ValueError("duplicate flow_ids in prediction payload")
    if set(base) != set(other):
        raise ValueError("baseline and candidate flow_ids differ")
    flow_ids = sorted(base)
    labels = np.asarray([base[fid][0] for fid in flow_ids], dtype=np.int64)
    candidate_labels = np.asarray([other[fid][0] for fid in flow_ids], dtype=np.int64)
    if not np.array_equal(labels, candidate_labels):
        raise ValueError("baseline and candidate labels differ")
    base_prob = np.stack([base[fid][1] for fid in flow_ids])
    candidate_prob = np.stack([other[fid][1] for fid in flow_ids])
    if base_prob.shape != candidate_prob.shape:
        raise ValueError("baseline and candidate probability shapes differ")
    return labels, base_prob, candidate_prob, flow_ids


def metric_values(y: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
    }


def paired_bootstrap(
    y: np.ndarray,
    base_prob: np.ndarray,
    candidate_prob: np.ndarray,
    samples: int = 2000,
    seed: int = 42,
) -> Dict[str, Any]:
    base_pred = base_prob.argmax(axis=1)
    candidate_pred = candidate_prob.argmax(axis=1)
    base_metrics = metric_values(y, base_pred)
    candidate_metrics = metric_values(y, candidate_pred)
    rng = np.random.default_rng(seed)
    deltas = {"accuracy": [], "macro_f1": []}
    for _ in range(max(0, int(samples))):
        index = rng.integers(0, len(y), size=len(y))
        base_sample = metric_values(y[index], base_pred[index])
        candidate_sample = metric_values(y[index], candidate_pred[index])
        for metric in deltas:
            deltas[metric].append(candidate_sample[metric] - base_sample[metric])
    delta_report = {}
    for metric, values in deltas.items():
        full = candidate_metrics[metric] - base_metrics[metric]
        if values:
            array = np.asarray(values, dtype=np.float64)
            interval = np.percentile(array, [2.5, 97.5]).tolist()
            win_rate = float((array > 0).mean())
            mean = float(array.mean())
        else:
            interval = [full, full]
            win_rate = float(full > 0)
            mean = full
        delta_report[metric] = {
            "delta": float(full),
            "bootstrap_mean": mean,
            "bootstrap_95_ci": [float(value) for value in interval],
            "win_rate": win_rate,
        }
    base_correct = base_pred == y
    candidate_correct = candidate_pred == y
    candidate_only = int((candidate_correct & ~base_correct).sum())
    baseline_only = int((base_correct & ~candidate_correct).sum())
    discordant = candidate_only + baseline_only
    mcnemar_p = (
        float(binomtest(min(candidate_only, baseline_only), discordant, 0.5).pvalue)
        if discordant > 0
        else 1.0
    )
    return {
        "num_flows": int(len(y)),
        "samples": int(samples),
        "seed": int(seed),
        "baseline": base_metrics,
        "candidate": candidate_metrics,
        "delta": delta_report,
        "mcnemar": {
            "candidate_only_correct": candidate_only,
            "baseline_only_correct": baseline_only,
            "discordant": discordant,
            "exact_p_value": mcnemar_p,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    baseline = load_predictions(args.baseline)
    candidate = load_predictions(args.candidate)
    y, base_prob, candidate_prob, _ = align_predictions(baseline, candidate)
    report = paired_bootstrap(y, base_prob, candidate_prob, args.samples, args.seed)
    report["baseline_path"] = args.baseline
    report["candidate_path"] = args.candidate
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
