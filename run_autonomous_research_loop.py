#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import shlex
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Tuple

from recommend_next_experiment import cuda_summary
from summarize_experiment_results import DEFAULT_TARGETS, collect_dataset, parse_target


def split_csv(raw: str) -> List[str]:
    out: List[str] = []
    for item in raw.split(","):
        item = item.strip()
        if item and item not in out:
            out.append(item)
    return out


def run_cmd(cmd: List[str], execute: bool = True) -> Dict[str, Any]:
    print("$ " + " ".join(shlex.quote(x) for x in cmd), flush=True)
    if not execute:
        return {"cmd": cmd, "returncode": 0, "skipped": True}
    proc = subprocess.run(cmd, check=False)
    return {"cmd": cmd, "returncode": int(proc.returncode), "skipped": False}


def target_map(overrides: List[str]) -> Dict[str, Tuple[float, float]]:
    targets = DEFAULT_TARGETS.copy()
    for raw in overrides:
        dataset, acc, macro_f1 = parse_target(raw)
        targets[dataset] = (acc, macro_f1)
    return targets


def dataset_status(datasets: List[str], targets: Dict[str, Tuple[float, float]]) -> List[Dict[str, Any]]:
    rows = []
    for dataset in datasets:
        results = collect_dataset(dataset, ["test*.json"])
        best = results[0] if results else None
        target = targets.get(dataset)
        achieved = None
        if target and best:
            achieved = bool(best["accuracy"] >= target[0] and (best.get("macro_f1") or 0.0) >= target[1])
        rows.append(
            {
                "dataset": dataset,
                "target": None if target is None else {"accuracy": target[0], "macro_f1": target[1]},
                "target_met": achieved,
                "num_results": len(results),
                "best": best,
            }
        )
    return rows


def all_goals_met(status: List[Dict[str, Any]], goal_datasets: List[str]) -> bool:
    by_dataset = {row["dataset"]: row for row in status}
    for dataset in goal_datasets:
        row = by_dataset.get(dataset)
        if not row or row.get("target_met") is not True:
            return False
    return True


def write_ledger(args, payload: Dict[str, Any]) -> None:
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out}", flush=True)


def report_commands(args, datasets: List[str]) -> List[List[str]]:
    rec_cmd = [
        "python",
        "recommend_next_experiment.py",
        "--output_json",
        "reasoningDataset/next_experiment_recommendation.json",
        "--output_md",
        "reasoningDataset/next_experiment_recommendation.md",
    ]
    for dataset in datasets:
        rec_cmd += ["--dataset", dataset]
    return [
        rec_cmd,
        [
            "python",
            "make_paper_framework_report.py",
            "--output_json",
            "reasoningDataset/paper_framework_report.json",
            "--output_md",
            "reasoningDataset/paper_framework_report.md",
        ],
        [
            "python",
            "make_paper_ablation_report.py",
            "--output_json",
            "reasoningDataset/paper_ablation_report.json",
            "--output_md",
            "reasoningDataset/paper_ablation_report.md",
        ],
    ]


def suite_cmd(args, datasets: List[str], iteration: int) -> List[str]:
    cmd = [
        "python",
        "run_recommended_suite.py",
        "--datasets",
        ",".join(datasets),
        "--run_tag",
        args.run_tag,
        "--model_types",
        args.model_types,
        "--tower2_epochs",
        str(args.tower2_epochs),
        "--tower2_early_stop_patience",
        str(args.tower2_early_stop_patience),
        "--output_json",
        str(Path(args.suite_output_dir) / f"recommended_suite_plan_iter{iteration:02d}.json"),
    ]
    if args.execute:
        cmd.append("--execute")
    if args.allow_no_cuda:
        cmd.append("--allow_no_cuda")
    if args.continue_on_error:
        cmd.append("--continue_on_error")
    return cmd


def load_framework_consistency() -> Dict[str, Any] | None:
    path = Path("reasoningDataset/paper_framework_report.json")
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"error": str(exc)}
    return data.get("framework_consistency")


def main() -> None:
    ap = argparse.ArgumentParser(description="Autonomous research loop for the unified traffic framework.")
    ap.add_argument("--datasets", default="vpn-app,tls-120,ustc-app")
    ap.add_argument("--goal_datasets", default="vpn-app,tls-120")
    ap.add_argument("--target", action="append", default=[], help="Optional DATASET:ACC:MACRO_F1 target override.")
    ap.add_argument("--max_iters", type=int, default=1)
    ap.add_argument("--execute", action="store_true", help="Run recommended experiments when goals are not met.")
    ap.add_argument("--continue_after_targets", action="store_true", help="Keep running recommended suite even if target gates already pass.")
    ap.add_argument("--allow_no_cuda", action="store_true")
    ap.add_argument("--continue_on_error", action="store_true")
    ap.add_argument("--run_tag", default="paired_ipport")
    ap.add_argument("--model_types", default="graph,seq")
    ap.add_argument("--tower2_epochs", type=int, default=30)
    ap.add_argument("--tower2_early_stop_patience", type=int, default=8)
    ap.add_argument("--suite_output_dir", default="reasoningDataset/autonomous_loop")
    ap.add_argument("--output_json", default="reasoningDataset/autonomous_loop/research_loop_ledger.json")
    args = ap.parse_args()

    datasets = split_csv(args.datasets)
    goal_datasets = split_csv(args.goal_datasets)
    targets = target_map(args.target)
    if not datasets:
        raise SystemExit("--datasets must contain at least one dataset")
    if not goal_datasets:
        raise SystemExit("--goal_datasets must contain at least one dataset")
    if args.max_iters < 1:
        raise SystemExit("--max_iters must be >= 1")

    ledger: Dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "datasets": datasets,
        "goal_datasets": goal_datasets,
        "execute": bool(args.execute),
        "continue_after_targets": bool(args.continue_after_targets),
        "cuda": cuda_summary(),
        "iterations": [],
        "stop_reason": "",
    }

    for iteration in range(1, args.max_iters + 1):
        before = dataset_status(datasets, targets)
        commands = []
        for cmd in report_commands(args, datasets):
            result = run_cmd(cmd, execute=True)
            commands.append(result)
            if result["returncode"] and not args.continue_on_error:
                ledger["iterations"].append({"iteration": iteration, "status_before": before, "commands": commands})
                ledger["stop_reason"] = f"command_failed:{cmd[1]}"
                write_ledger(args, ledger)
                raise SystemExit(result["returncode"])

        framework = load_framework_consistency()
        goals_met = all_goals_met(before, goal_datasets)
        record: Dict[str, Any] = {
            "iteration": iteration,
            "status_before": before,
            "framework_consistency": framework,
            "goals_met_before": goals_met,
            "commands": commands,
        }
        if goals_met and not args.continue_after_targets:
            record["action"] = "stop_targets_met"
            ledger["iterations"].append(record)
            ledger["stop_reason"] = "targets_met"
            write_ledger(args, ledger)
            return

        suite_result = run_cmd(suite_cmd(args, datasets, iteration), execute=True)
        record["commands"].append(suite_result)
        record["status_after"] = dataset_status(datasets, targets)
        record["goals_met_after"] = all_goals_met(record["status_after"], goal_datasets)
        record["action"] = "ran_recommended_suite" if args.execute else "dry_run_recommended_suite"
        ledger["iterations"].append(record)
        write_ledger(args, ledger)
        if suite_result["returncode"] and not args.continue_on_error:
            ledger["stop_reason"] = "suite_failed"
            write_ledger(args, ledger)
            raise SystemExit(suite_result["returncode"])
        if record["goals_met_after"] and not args.continue_after_targets:
            ledger["stop_reason"] = "targets_met_after_iteration"
            write_ledger(args, ledger)
            return

    ledger["stop_reason"] = "max_iters_reached"
    write_ledger(args, ledger)


if __name__ == "__main__":
    main()
