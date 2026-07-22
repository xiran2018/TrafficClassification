#!/usr/bin/env python3
"""Replay patience-based Tower1 stopping decisions from validation histories."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_history(path: Path, metric: str = "macro_f1") -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"empty validation history: {path}")
    previous_step = -1
    for index, row in enumerate(rows):
        step = int(row["step"])
        if step <= previous_step:
            raise ValueError(f"history steps must increase strictly: row={index}")
        previous_step = step
        value = float(row["metrics"][metric])
        if not math.isfinite(value):
            raise ValueError(f"non-finite {metric}: row={index}")
    return rows


def replay_patience(
    values: Sequence[float], patience: int, min_delta: float = 0.0
) -> dict[str, Any]:
    if not values:
        raise ValueError("values must not be empty")
    if patience <= 0:
        raise ValueError("patience must be positive")
    if min_delta < 0:
        raise ValueError("min_delta must be non-negative")
    best = float("-inf")
    best_index = -1
    stale = 0
    stop_index = len(values) - 1
    for index, raw_value in enumerate(values):
        value = float(raw_value)
        if not math.isfinite(value):
            raise ValueError(f"non-finite metric value at index {index}")
        if value > best + min_delta:
            best = value
            best_index = index
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            stop_index = index
            break
    oracle_index = max(range(len(values)), key=lambda index: float(values[index]))
    oracle = float(values[oracle_index])
    return {
        "patience": int(patience),
        "min_delta": float(min_delta),
        "stop_epoch": stop_index + 1,
        "selected_epoch": best_index + 1,
        "selected_metric": best,
        "oracle_epoch": oracle_index + 1,
        "oracle_metric": oracle,
        "metric_regret": oracle - best,
        "epochs_saved": len(values) - (stop_index + 1),
    }


def analyze_history(
    name: str,
    path: Path,
    metric: str,
    patience_values: Sequence[int],
    min_delta: float,
) -> dict[str, Any]:
    rows = load_history(path, metric=metric)
    values = [float(row["metrics"][metric]) for row in rows]
    return {
        "name": name,
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "num_epochs": len(rows),
        "steps": [int(row["step"]) for row in rows],
        "trajectory": values,
        "simulations": [
            replay_patience(values, patience, min_delta)
            for patience in patience_values
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        nargs=2,
        action="append",
        metavar=("NAME", "HISTORY_JSONL"),
        required=True,
    )
    parser.add_argument("--metric", default="macro_f1")
    parser.add_argument("--patience", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--min_delta", type=float, default=0.0)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()
    if len(set(args.patience)) != len(args.patience):
        parser.error("--patience values must be unique")
    if any(value <= 0 for value in args.patience):
        parser.error("--patience values must be positive")
    if args.min_delta < 0:
        parser.error("--min_delta must be non-negative")
    names = [name for name, _ in args.input]
    if len(set(names)) != len(names):
        parser.error("--input names must be unique")

    report = {
        "schema": "tower1_early_stopping_replay_v1",
        "metric": args.metric,
        "test_labels_used": False,
        "selection_role": "validation_only_compute_budget_audit",
        "histories": [
            analyze_history(
                name,
                Path(path),
                args.metric,
                args.patience,
                args.min_delta,
            )
            for name, path in args.input
        ],
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
