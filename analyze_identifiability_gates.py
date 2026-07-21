#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F

from test_tower2 import load_model
from train_tower2 import GraphDataset, SeqDataset, collate_seq, window_identifiability


@torch.no_grad()
def collect(args) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    model, ckpt, _ = load_model(args.checkpoint, args.device)
    model_type = ckpt["model_type"]
    dataset = GraphDataset(args.dataset) if model_type == "graph" else SeqDataset(args.dataset)
    gates = []
    adapter_norms = []
    reliability = []
    limit = len(dataset) if args.max_samples <= 0 else min(len(dataset), args.max_samples)
    for index in range(limit):
        item = dataset[index]
        if model_type == "graph":
            out = model(
                item["x"].to(args.device),
                item["edge_index"].to(args.device),
                item["edge_attr"].to(args.device),
            )
        else:
            batch = collate_seq([item])
            out = model(batch["x"].to(args.device), batch["mask"].to(args.device))
        gate = out.get("identifiability_gate")
        adapter_norm = out.get("identifiability_adapter_norm")
        if gate is None and adapter_norm is None:
            raise ValueError("checkpoint does not use an identifiability routing module")
        if gate is not None:
            gates.append(gate.detach().float().cpu().reshape(-1, 3).mean(dim=0).numpy())
        if adapter_norm is not None:
            adapter_norms.append(float(adapter_norm.detach().float().mean().cpu().item()))
        reliability.append(window_identifiability(item))
    pool = model.pool
    learned = {
        "identifiable_prior_scale": float(F.softplus(pool.reliability_prior_raw_scale).item())
        if hasattr(pool, "reliability_prior_raw_scale")
        else None,
        "context_prior_scale": float(F.softplus(pool.unidentifiable_prior_raw_scale).item())
        if hasattr(pool, "unidentifiable_prior_raw_scale")
        else None,
        "adapter_max_delta": float(getattr(pool, "reliability_adapter_max_delta", 0.0)),
    }
    return (
        np.asarray(gates),
        np.asarray(adapter_norms, dtype=np.float64),
        np.asarray(reliability, dtype=np.float64),
        learned,
    )


def summarize(gates: np.ndarray, adapter_norms: np.ndarray, reliability: np.ndarray, learned: dict) -> dict:
    names = ["base", "identifiable", "low_identifiability"]
    correlations = []
    if gates.size:
        for branch in range(gates.shape[1]):
            if np.std(gates[:, branch]) <= 1e-12 or np.std(reliability) <= 1e-12:
                correlations.append(None)
            else:
                correlations.append(float(np.corrcoef(gates[:, branch], reliability)[0, 1]))
    report = {
        "num_windows": int(len(reliability)),
        "window_reliability_mean": float(reliability.mean()),
        "window_reliability_std": float(reliability.std()),
        "learned": learned,
    }
    if gates.size:
        report.update(
            {
                "branches": names,
                "gate_mean": gates.mean(axis=0).astype(float).tolist(),
                "gate_std": gates.std(axis=0).astype(float).tolist(),
                "gate_quantiles": {
                    name: np.quantile(gates[:, index], [0.05, 0.5, 0.95]).astype(float).tolist()
                    for index, name in enumerate(names)
                },
                "gate_reliability_correlation": correlations,
            }
        )
    if adapter_norms.size:
        correlation = (
            float(np.corrcoef(adapter_norms, reliability)[0, 1])
            if np.std(adapter_norms) > 1e-12 and np.std(reliability) > 1e-12
            else None
        )
        report["adapter_correction_norm"] = {
            "mean": float(adapter_norms.mean()),
            "std": float(adapter_norms.std()),
            "quantiles": np.quantile(adapter_norms, [0.05, 0.5, 0.95]).astype(float).tolist(),
            "reliability_correlation": correlation,
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output_json", default="")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    gates, adapter_norms, reliability, learned = collect(args)
    report = summarize(gates, adapter_norms, reliability, learned)
    text = json.dumps(report, indent=2)
    print(text)
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")


if __name__ == "__main__":
    main()
