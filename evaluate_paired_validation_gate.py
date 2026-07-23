#!/usr/bin/env python3
"""Evaluate one paired-view candidate using held-out validation evidence only."""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any


GATE_COMPARISON_EPSILON = 1e-12


def load_metrics(path: str | Path, task: str) -> dict[str, float]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    metrics = payload["metrics"]
    if task == "flow":
        metrics = metrics["flow_level"]
    return {
        "accuracy": float(metrics["accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
    }


def unique_glob(pattern: str) -> Path:
    matches = [Path(path) for path in glob.glob(pattern)]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one result for {pattern!r}, found {len(matches)}: {matches}"
        )
    return matches[0]


def compare_metrics(
    baseline: dict[str, float],
    candidate: dict[str, float],
    *,
    min_macro_f1_delta: float,
    max_accuracy_drop: float,
) -> dict[str, Any]:
    delta = {
        name: candidate[name] - baseline[name]
        for name in ("accuracy", "macro_f1")
    }
    macro_f1_pass = (
        delta["macro_f1"] + GATE_COMPARISON_EPSILON >= min_macro_f1_delta
    )
    accuracy_pass = (
        delta["accuracy"] + GATE_COMPARISON_EPSILON >= -max_accuracy_drop
    )
    return {
        "base": baseline,
        "candidate": candidate,
        "delta": delta,
        "macro_f1_improvement_pass": macro_f1_pass,
        "accuracy_guard_pass": accuracy_pass,
        "pass": macro_f1_pass and accuracy_pass,
    }


def metric_delta(
    baseline: dict[str, float], candidate: dict[str, float]
) -> dict[str, Any]:
    return {
        "base": baseline,
        "candidate": candidate,
        "delta": {
            name: candidate[name] - baseline[name]
            for name in ("accuracy", "macro_f1")
        },
    }


def sensitivity_metrics(
    root: str | Path, dataset: str, task: str, view: str
) -> dict[str, float]:
    root = Path(root)
    candidates = (
        root / f"{dataset}_fold0_valid_{task}_{view}.json",
        root / f"{task}_valid_{view}.json",
    )
    matches = [path for path in candidates if path.is_file()]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one {task}/{view} sensitivity result under {root}, "
            f"found {matches}"
        )
    return load_metrics(matches[0], task)


def mechanism_diagnostics(
    baseline_root: str | Path,
    candidate_root: str | Path,
    dataset: str,
) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {}
    for task in ("packet", "flow"):
        task_rows: dict[str, Any] = {}
        for view in ("factual_only", "intervened_only"):
            task_rows[view] = metric_delta(
                sensitivity_metrics(baseline_root, dataset, task, view),
                sensitivity_metrics(candidate_root, dataset, task, view),
            )
        task_rows["intervention_gap"] = {
            metric: {
                "base": (
                    task_rows["factual_only"]["base"][metric]
                    - task_rows["intervened_only"]["base"][metric]
                ),
                "candidate": (
                    task_rows["factual_only"]["candidate"][metric]
                    - task_rows["intervened_only"]["candidate"][metric]
                ),
            }
            for metric in ("accuracy", "macro_f1")
        }
        for metric in ("accuracy", "macro_f1"):
            row = task_rows["intervention_gap"][metric]
            row["reduction"] = row["base"] - row["candidate"]
        diagnostics[task] = task_rows
    return diagnostics


def build_gate_payload(
    *,
    dataset: str,
    baseline_packet: str | Path,
    candidate_packet: str | Path,
    baseline_flow: str | Path,
    candidate_flow: str | Path,
    min_macro_f1_delta: float = 0.005,
    max_accuracy_drop: float = 0.005,
    baseline_sensitivity_dir: str | Path | None = None,
    candidate_sensitivity_dir: str | Path | None = None,
) -> dict[str, Any]:
    sensitivity_pair = (
        baseline_sensitivity_dir is not None,
        candidate_sensitivity_dir is not None,
    )
    if sensitivity_pair[0] != sensitivity_pair[1]:
        raise ValueError(
            "candidate and baseline sensitivity directories must be supplied together"
        )
    paths = {
        "packet": {
            "baseline": str(baseline_packet),
            "candidate": str(candidate_packet),
        },
        "flow": {
            "baseline": str(baseline_flow),
            "candidate": str(candidate_flow),
        },
    }
    tasks = {
        task: compare_metrics(
            load_metrics(task_paths["baseline"], task),
            load_metrics(task_paths["candidate"], task),
            min_macro_f1_delta=min_macro_f1_delta,
            max_accuracy_drop=max_accuracy_drop,
        )
        for task, task_paths in paths.items()
    }
    strict_pass = all(row["pass"] for row in tasks.values())
    payload: dict[str, Any] = {
        "protocol": "paired_robust_validation_gate_v1",
        "scope": "heldout_validation_only",
        "dataset": dataset,
        "thresholds": {
            "min_macro_f1_delta": min_macro_f1_delta,
            "max_accuracy_drop": max_accuracy_drop,
        },
        "taskwise_same_candidate_required": True,
        "test_metrics_used": False,
        "inputs": paths,
        "tasks": tasks,
        "strict_pass": strict_pass,
        "decision": (
            "advance_to_next_dataset_validation"
            if strict_pass
            else "reject_or_revise_candidate"
        ),
    }
    if baseline_sensitivity_dir is not None:
        payload["mechanism_diagnostics"] = mechanism_diagnostics(
            baseline_sensitivity_dir,
            candidate_sensitivity_dir,
            dataset,
        )
        payload["mechanism_diagnostics_role"] = (
            "validation_only_explanatory_evidence_not_an_additional_selection_gate"
        )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["vpn-app", "tls-120"], required=True)
    parser.add_argument("--candidate_root", required=True)
    parser.add_argument("--candidate_worktree", required=True)
    parser.add_argument("--baseline_root", required=True)
    parser.add_argument("--baseline_worktree", required=True)
    parser.add_argument("--candidate_tag", required=True)
    parser.add_argument("--baseline_tag", default="base_milestone_dev_fold0")
    parser.add_argument("--min_macro_f1_delta", type=float, default=0.005)
    parser.add_argument("--max_accuracy_drop", type=float, default=0.005)
    parser.add_argument("--candidate_sensitivity_dir", default="")
    parser.add_argument("--baseline_sensitivity_dir", default="")
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    candidate_root = Path(args.candidate_root)
    baseline_root = Path(args.baseline_root)
    packet_tail = Path("packet_artifacts") / args.dataset / "fold0"
    candidate_packet = candidate_root / packet_tail / "valid_unified_packet_single_head.json"
    baseline_packet = baseline_root / packet_tail / "valid_unified_packet_single_head.json"
    candidate_flow = unique_glob(
        str(
            Path(args.candidate_worktree)
            / "reasoningDataset"
            / args.dataset
            / f"valid_seq_metrics_flow_*{args.candidate_tag}*_probs.json"
        )
    )
    baseline_flow = unique_glob(
        str(
            Path(args.baseline_worktree)
            / "reasoningDataset"
            / args.dataset
            / f"valid_seq_metrics_flow_*{args.baseline_tag}*_probs.json"
        )
    )
    paired_sensitivity = bool(args.candidate_sensitivity_dir) and bool(
        args.baseline_sensitivity_dir
    )
    if bool(args.candidate_sensitivity_dir) != bool(args.baseline_sensitivity_dir):
        parser.error(
            "candidate and baseline sensitivity directories must be supplied together"
        )
    payload = build_gate_payload(
        dataset=args.dataset,
        baseline_packet=baseline_packet,
        candidate_packet=candidate_packet,
        baseline_flow=baseline_flow,
        candidate_flow=candidate_flow,
        min_macro_f1_delta=args.min_macro_f1_delta,
        max_accuracy_drop=args.max_accuracy_drop,
        baseline_sensitivity_dir=(
            args.baseline_sensitivity_dir if paired_sensitivity else None
        ),
        candidate_sensitivity_dir=(
            args.candidate_sensitivity_dir if paired_sensitivity else None
        ),
    )
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
