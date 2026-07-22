#!/usr/bin/env python3
"""Measure whether a learned view router tracks counterfactual view quality."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


EPS = 1e-12


def _file_evidence(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(resolved),
        "sha256": digest.hexdigest(),
        "size_bytes": int(resolved.stat().st_size),
    }


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or np.std(left) <= EPS or np.std(right) <= EPS:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _cross_entropy(probabilities: np.ndarray, labels: np.ndarray) -> np.ndarray:
    if probabilities.ndim != 2 or len(probabilities) != len(labels):
        raise ValueError("probability/label shape mismatch")
    return -np.log(np.clip(probabilities[np.arange(len(labels)), labels], EPS, 1.0))


def _load_packet(path: str) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        required = {"y_true", "probabilities", "packet_uids", "flow_ids"}
        missing = sorted(required - set(data.files))
        if missing:
            raise ValueError(f"{path}: missing packet arrays {missing}")
        payload = {name: np.asarray(data[name]) for name in required}
        gate_key = "effective_intervention_view_gate"
        if gate_key in data.files:
            payload["gate"] = np.asarray(data[gate_key], dtype=np.float64)
    payload["ids"] = payload.pop("packet_uids").astype(str)
    payload["groups"] = payload.pop("flow_ids").astype(str)
    return payload


def _aggregate_window_gate(payload: dict[str, Any]) -> np.ndarray:
    window_gates = payload.get("window_effective_gate_values", {}).get(
        "intervention_view_gate"
    )
    window_flow_ids = payload.get("window_flow_ids")
    flow_ids = payload.get("flow_ids")
    if window_gates is None or window_flow_ids is None or flow_ids is None:
        raise ValueError("flow result lacks aligned intervention gate values")
    values = np.asarray(window_gates, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("flow intervention gate must have shape [windows, 2]")
    if len(values) != len(window_flow_ids):
        raise ValueError("window gate/flow-id length mismatch")
    grouped: dict[str, list[np.ndarray]] = {}
    for flow_id, value in zip(window_flow_ids, values):
        grouped.setdefault(str(flow_id), []).append(value)
    missing = [str(flow_id) for flow_id in flow_ids if str(flow_id) not in grouped]
    if missing:
        raise ValueError(f"missing window gates for {len(missing)} evaluated flows")
    return np.stack(
        [np.mean(grouped[str(flow_id)], axis=0) for flow_id in flow_ids], axis=0
    )


def _load_flow(path: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {"flow_y_true", "flow_prob", "flow_ids"}
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"{path}: missing flow fields {missing}")
    result = {
        "y_true": np.asarray(payload["flow_y_true"], dtype=np.int64),
        "probabilities": np.asarray(payload["flow_prob"], dtype=np.float64),
        "ids": np.asarray(payload["flow_ids"], dtype=str),
        "groups": np.asarray(payload["flow_ids"], dtype=str),
    }
    if payload.get("window_effective_gate_values"):
        result["gate"] = _aggregate_window_gate(payload)
    return result


def _validate_alignment(reference: dict[str, Any], candidate: dict[str, Any], name: str) -> None:
    if not np.array_equal(reference["ids"], candidate["ids"]):
        raise ValueError(f"{name}: sample identities are not aligned")
    if not np.array_equal(reference["y_true"], candidate["y_true"]):
        raise ValueError(f"{name}: labels are not aligned")
    if not np.array_equal(reference["groups"], candidate["groups"]):
        raise ValueError(f"{name}: resampling groups are not aligned")


def _bootstrap_pearson(
    gate_preference: np.ndarray,
    advantage: np.ndarray,
    groups: np.ndarray,
    samples: int,
    seed: int,
) -> dict[str, float]:
    if samples <= 0:
        return {}
    rng = np.random.default_rng(seed)
    unique_groups, inverse = np.unique(groups.astype(str), return_inverse=True)
    group_members = [np.flatnonzero(inverse == index) for index in range(len(unique_groups))]
    correlations = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        sampled_groups = rng.integers(
            0, len(unique_groups), size=len(unique_groups)
        )
        sample = np.concatenate([group_members[group] for group in sampled_groups])
        correlations[index] = _correlation(
            gate_preference[sample], advantage[sample]
        )
    return {
        "samples": int(samples),
        "seed": int(seed),
        "resampling_unit": "flow_cluster",
        "num_clusters": int(len(unique_groups)),
        "mean": float(correlations.mean()),
        "ci95_low": float(np.quantile(correlations, 0.025)),
        "ci95_high": float(np.quantile(correlations, 0.975)),
        "positive_fraction": float(np.mean(correlations > 0.0)),
    }


def analyze(
    full: dict[str, Any],
    factual: dict[str, Any],
    intervened: dict[str, Any],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    _validate_alignment(full, factual, "factual-only")
    _validate_alignment(full, intervened, "intervened-only")
    gate = np.asarray(full.get("gate"), dtype=np.float64)
    if gate.ndim != 2 or gate.shape != (len(full["y_true"]), 2):
        raise ValueError("full prediction lacks an aligned [samples, 2] effective gate")
    if not np.all(np.isfinite(gate)):
        raise ValueError("effective gate contains non-finite values")
    factual_loss = _cross_entropy(factual["probabilities"], full["y_true"])
    intervened_loss = _cross_entropy(intervened["probabilities"], full["y_true"])
    advantage = intervened_loss - factual_loss
    gate_preference = gate[:, 0] - gate[:, 1]
    pearson = _correlation(gate_preference, advantage)
    spearman = _correlation(_rank(gate_preference), _rank(advantage))
    informative = (np.abs(advantage) > 1e-8) & (np.abs(gate_preference) > 1e-8)
    direction_accuracy = (
        float(np.mean(np.sign(gate_preference[informative]) == np.sign(advantage[informative])))
        if np.any(informative)
        else None
    )
    order = np.argsort(advantage)
    quintile = max(1, len(order) // 5)
    bottom = gate[:, 0][order[:quintile]]
    top = gate[:, 0][order[-quintile:]]
    bootstrap = _bootstrap_pearson(
        gate_preference,
        advantage,
        np.asarray(full["groups"]),
        bootstrap_samples,
        seed,
    )
    robust_positive = bool(
        pearson > 0.0
        and spearman > 0.0
        and (not bootstrap or bootstrap["ci95_low"] > 0.0)
    )
    return {
        "status": "positive_association" if robust_positive else "not_demonstrated",
        "claim_scope": (
            "diagnostic association between the learned effective router and "
            "counterfactual single-view predictive loss; not a causal guarantee"
        ),
        "num_samples": int(len(advantage)),
        "num_flow_clusters": int(len(np.unique(np.asarray(full["groups"]).astype(str)))),
        "gate": {
            "view_names": ["factual", "intervened"],
            "effective_mean": gate.mean(axis=0).tolist(),
            "effective_std": gate.std(axis=0).tolist(),
            "preference_std": float(gate_preference.std()),
        },
        "counterfactual_loss": {
            "factual_mean": float(factual_loss.mean()),
            "intervened_mean": float(intervened_loss.mean()),
            "factual_advantage_mean": float(advantage.mean()),
            "factual_advantage_std": float(advantage.std()),
        },
        "association": {
            "pearson": pearson,
            "spearman": spearman,
            "direction_accuracy": direction_accuracy,
            "direction_sample_count": int(informative.sum()),
            "factual_weight_bottom_advantage_quintile": float(bottom.mean()),
            "factual_weight_top_advantage_quintile": float(top.mean()),
            "top_minus_bottom": float(top.mean() - bottom.mean()),
            "bootstrap_pearson": bootstrap,
            "robust_positive": robust_positive,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["packet", "flow"])
    ap.add_argument("--full_prediction", required=True)
    ap.add_argument("--factual_prediction", required=True)
    ap.add_argument("--intervened_prediction", required=True)
    ap.add_argument("--output_json", required=True)
    ap.add_argument("--bootstrap_samples", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260722)
    args = ap.parse_args()
    if args.bootstrap_samples < 0:
        ap.error("--bootstrap_samples must be non-negative")
    loader = _load_packet if args.task == "packet" else _load_flow
    report = analyze(
        loader(args.full_prediction),
        loader(args.factual_prediction),
        loader(args.intervened_prediction),
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    report["task"] = args.task
    report["inputs"] = {
        "full": _file_evidence(args.full_prediction),
        "factual_only": _file_evidence(args.factual_prediction),
        "intervened_only": _file_evidence(args.intervened_prediction),
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"saved {output}")


if __name__ == "__main__":
    main()
