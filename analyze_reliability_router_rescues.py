#!/usr/bin/env python3
"""Measure whether a flow reliability router uses complementary evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np


def load_payload(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    required = {"flow_ids", "flow_y_true", "flow_y_pred"}
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"{path} is missing fields: {missing}")
    return payload


def unique_index(flow_ids: Iterable[Any], source: str) -> Dict[str, int]:
    index: Dict[str, int] = {}
    for row, flow_id in enumerate(flow_ids):
        key = str(flow_id)
        if key in index:
            raise ValueError(f"{source} contains duplicate flow_id {key}")
        index[key] = row
    return index


def align(payload: Dict[str, Any], flow_ids: List[str], key: str, source: str) -> np.ndarray:
    index = unique_index(payload["flow_ids"], source)
    missing = [flow_id for flow_id in flow_ids if flow_id not in index]
    if missing:
        raise ValueError(f"{source} is missing {len(missing)} router flow ids")
    values = payload[key]
    return np.asarray([values[index[flow_id]] for flow_id in flow_ids])


def conditional_gate(gate: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
    return {
        "count": int(mask.sum()),
        "gate_mean": float(gate[mask].mean()) if mask.any() else None,
    }


def analyze(
    semantic: Dict[str, Any],
    structural: Dict[str, Any],
    router: Dict[str, Any],
) -> Dict[str, Any]:
    flow_ids = [str(flow_id) for flow_id in router["flow_ids"]]
    if len(set(flow_ids)) != len(flow_ids):
        raise ValueError("router contains duplicate flow ids")

    y_true = np.asarray(router["flow_y_true"], dtype=np.int64)
    router_pred = np.asarray(router["flow_y_pred"], dtype=np.int64)
    semantic_y = align(semantic, flow_ids, "flow_y_true", "semantic")
    structural_y = align(structural, flow_ids, "flow_y_true", "structural")
    if not np.array_equal(y_true, semantic_y) or not np.array_equal(y_true, structural_y):
        raise ValueError("expert and router labels do not align")

    semantic_pred = align(semantic, flow_ids, "flow_y_pred", "semantic").astype(np.int64)
    structural_pred = align(structural, flow_ids, "flow_y_pred", "structural").astype(np.int64)
    gates = np.asarray(router.get("flow_structural_gate"), dtype=np.float64)
    if gates.ndim == 1:
        gates = gates[:, None]
    if gates.ndim != 2 or gates.shape[0] != y_true.size:
        raise ValueError("flow_structural_gate must have shape [num_flows, num_environments]")
    gate = gates.mean(axis=1)

    masks = {
        "semantic_correct": semantic_pred == y_true,
        "semantic_wrong": semantic_pred != y_true,
        "structural_only_correct": (structural_pred == y_true) & (semantic_pred != y_true),
        "semantic_only_correct": (semantic_pred == y_true) & (structural_pred != y_true),
        "router_rescue": (semantic_pred != y_true) & (router_pred == y_true),
        "router_harm": (semantic_pred == y_true) & (router_pred != y_true),
        "both_experts_wrong": (semantic_pred != y_true) & (structural_pred != y_true),
    }
    conditions = {name: conditional_gate(gate, mask) for name, mask in masks.items()}
    return {
        "schema": "reliability_router_rescue_analysis_v1",
        "num_flows": int(y_true.size),
        "num_environments": int(gates.shape[1]),
        "accuracy": {
            "semantic": float((semantic_pred == y_true).mean()),
            "structural": float((structural_pred == y_true).mean()),
            "router": float((router_pred == y_true).mean()),
        },
        "gate": {
            "mean": float(gate.mean()),
            "std_across_flows": float(gate.std()),
            "per_environment_mean": gates.mean(axis=0).tolist(),
        },
        "conditions": conditions,
        "net_rescues": conditions["router_rescue"]["count"] - conditions["router_harm"]["count"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--semantic", required=True)
    parser.add_argument("--structural", required=True)
    parser.add_argument("--router", required=True)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    report = analyze(
        load_payload(args.semantic),
        load_payload(args.structural),
        load_payload(args.router),
    )
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
