#!/usr/bin/env python3
"""Describe how hierarchy-aware class weighting changes held-out class F1."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from scipy.stats import spearmanr


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_history(path: Path) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            step = int(row["step"])
            if step in rows:
                raise ValueError(f"duplicate validation step {step} in {path}")
            rows[step] = row
    return rows


def class_counts(
    path: Path,
) -> tuple[dict[int, int], dict[int, int], dict[int, str]]:
    packet_counts: dict[int, int] = defaultdict(int)
    flow_ids: dict[int, set[str]] = defaultdict(set)
    label_names: dict[int, str] = {}
    flow_labels: dict[str, int] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            label = int(row["label_id"])
            flow_id = str(row.get("flow_id") or "")
            if not flow_id:
                raise ValueError("every training packet must have a flow_id")
            previous = flow_labels.setdefault(flow_id, label)
            if previous != label:
                raise ValueError(f"flow_id has conflicting labels: {flow_id}")
            name = str(row["label"])
            if label in label_names and label_names[label] != name:
                raise ValueError(f"label_id has conflicting names: {label}")
            label_names[label] = name
            packet_counts[label] += 1
            flow_ids[label].add(flow_id)
    if not label_names:
        raise ValueError("training packet file is empty")
    expected = set(range(len(label_names)))
    if set(label_names) != expected:
        raise ValueError("label IDs must be contiguous from zero")
    return (
        dict(packet_counts),
        {label: len(flows) for label, flows in flow_ids.items()},
        label_names,
    )


def effective_weights(
    counts: dict[int, int], *, beta: float, strength: float
) -> dict[int, float]:
    raw = {
        label: (1.0 - beta) / max(1.0 - beta ** int(count), 1e-12)
        for label, count in counts.items()
    }
    raw_mean = statistics.fmean(raw.values())
    powered = {label: (value / raw_mean) ** strength for label, value in raw.items()}
    powered_mean = statistics.fmean(powered.values())
    return {label: value / powered_mean for label, value in powered.items()}


def rank_correlation(left: list[float], right: list[float]) -> dict[str, Any]:
    if len(set(left)) < 2 or len(set(right)) < 2:
        return {"spearman_rho": None, "pvalue": None, "reason": "constant_input"}
    rho, pvalue = spearmanr(left, right)
    if not math.isfinite(float(rho)) or not math.isfinite(float(pvalue)):
        return {"spearman_rho": None, "pvalue": None, "reason": "non_finite"}
    return {"spearman_rho": float(rho), "pvalue": float(pvalue)}


def analyze(
    train_jsonl: Path,
    baseline_history_path: Path,
    candidate_history_path: Path,
    *,
    step: int,
    baseline_basis: str,
    candidate_basis: str,
    baseline_strength: float,
    candidate_strength: float,
    beta: float,
) -> dict[str, Any]:
    if baseline_basis not in {"packet", "flow"} or candidate_basis not in {
        "packet",
        "flow",
    }:
        raise ValueError("class-weight basis must be packet or flow")
    if not 0.0 <= baseline_strength <= 1.0 or not 0.0 <= candidate_strength <= 1.0:
        raise ValueError("class-weight strengths must be in [0,1]")
    if not 0.0 < beta < 1.0:
        raise ValueError("effective-number beta must be in (0,1)")

    packet_counts, flow_counts, label_names = class_counts(train_jsonl)
    histories = {
        "baseline": load_history(baseline_history_path),
        "candidate": load_history(candidate_history_path),
    }
    if any(step not in history for history in histories.values()):
        raise ValueError(f"step {step} must exist in both validation histories")
    metrics = {
        name: history[step]["metrics"] for name, history in histories.items()
    }
    expected_names = set(label_names.values())
    for name, values in metrics.items():
        if set((values.get("per_class") or {}).keys()) != expected_names:
            raise ValueError(f"{name} validation classes do not match training labels")

    count_sources = {"packet": packet_counts, "flow": flow_counts}
    baseline_weights = effective_weights(
        count_sources[baseline_basis], beta=beta, strength=baseline_strength
    )
    candidate_weights = effective_weights(
        count_sources[candidate_basis], beta=beta, strength=candidate_strength
    )
    rows = []
    for label_id in sorted(label_names):
        label = label_names[label_id]
        baseline_f1 = float(metrics["baseline"]["per_class"][label]["f1"])
        candidate_f1 = float(metrics["candidate"]["per_class"][label]["f1"])
        rows.append(
            {
                "label_id": label_id,
                "label": label,
                "train_packets": packet_counts[label_id],
                "train_flows": flow_counts[label_id],
                "train_packets_per_flow": packet_counts[label_id]
                / flow_counts[label_id],
                "baseline_weight": baseline_weights[label_id],
                "candidate_weight": candidate_weights[label_id],
                "candidate_to_baseline_weight_ratio": candidate_weights[label_id]
                / baseline_weights[label_id],
                "baseline_f1": baseline_f1,
                "candidate_f1": candidate_f1,
                "f1_delta": candidate_f1 - baseline_f1,
            }
        )
    deltas = [row["f1_delta"] for row in rows]
    weight_ratios = [row["candidate_to_baseline_weight_ratio"] for row in rows]
    return {
        "schema": "class_weight_mechanism_analysis_v1",
        "scope": "heldout_validation_descriptive_only",
        "selection_role": "reporting_only_not_model_selection",
        "test_labels_used": False,
        "step": step,
        "inputs": {
            "train_jsonl": {
                "path": str(train_jsonl.resolve()),
                "sha256": file_sha256(train_jsonl),
            },
            "baseline_history": {
                "path": str(baseline_history_path.resolve()),
                "sha256": file_sha256(baseline_history_path),
            },
            "candidate_history": {
                "path": str(candidate_history_path.resolve()),
                "sha256": file_sha256(candidate_history_path),
            },
        },
        "weighting": {
            "effective_number_beta": beta,
            "baseline": {"basis": baseline_basis, "strength": baseline_strength},
            "candidate": {"basis": candidate_basis, "strength": candidate_strength},
        },
        "metrics": {
            "baseline": {
                "accuracy": float(metrics["baseline"]["accuracy"]),
                "macro_f1": float(metrics["baseline"]["macro_f1"]),
            },
            "candidate": {
                "accuracy": float(metrics["candidate"]["accuracy"]),
                "macro_f1": float(metrics["candidate"]["macro_f1"]),
            },
            "delta": {
                "accuracy": float(
                    metrics["candidate"]["accuracy"] - metrics["baseline"]["accuracy"]
                ),
                "macro_f1": float(
                    metrics["candidate"]["macro_f1"] - metrics["baseline"]["macro_f1"]
                ),
            },
        },
        "correlations": {
            "train_flow_count_vs_f1_delta": rank_correlation(
                [math.log1p(row["train_flows"]) for row in rows], deltas
            ),
            "packets_per_flow_vs_f1_delta": rank_correlation(
                [math.log1p(row["train_packets_per_flow"]) for row in rows], deltas
            ),
            "weight_ratio_vs_f1_delta": rank_correlation(
                [math.log(value) for value in weight_ratios], deltas
            ),
        },
        "num_classes_improved": sum(delta > 0 for delta in deltas),
        "num_classes_degraded": sum(delta < 0 for delta in deltas),
        "largest_f1_gains": sorted(
            rows, key=lambda row: row["f1_delta"], reverse=True
        )[:10],
        "largest_f1_losses": sorted(rows, key=lambda row: row["f1_delta"])[:10],
        "per_class": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_jsonl", type=Path, required=True)
    parser.add_argument("--baseline_history", type=Path, required=True)
    parser.add_argument("--candidate_history", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--baseline_basis", choices=["packet", "flow"], required=True)
    parser.add_argument("--candidate_basis", choices=["packet", "flow"], required=True)
    parser.add_argument("--baseline_strength", type=float, required=True)
    parser.add_argument("--candidate_strength", type=float, required=True)
    parser.add_argument("--beta", type=float, default=0.9999)
    parser.add_argument("--output_json", type=Path, required=True)
    args = parser.parse_args()
    payload = analyze(
        args.train_jsonl,
        args.baseline_history,
        args.candidate_history,
        step=args.step,
        baseline_basis=args.baseline_basis,
        candidate_basis=args.candidate_basis,
        baseline_strength=args.baseline_strength,
        candidate_strength=args.candidate_strength,
        beta=args.beta,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"metrics": payload["metrics"], "correlations": payload["correlations"]}, indent=2))


if __name__ == "__main__":
    main()
