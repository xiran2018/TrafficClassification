#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import shlex
import subprocess
from pathlib import Path
from typing import Any, Dict, List

from recommend_next_experiment import cuda_summary
from run_recommended_experiment import DATASET_PRESETS
from summarize_experiment_results import DEFAULT_TARGETS, collect_dataset


def split_csv(raw: str) -> List[str]:
    out: List[str] = []
    for item in raw.split(","):
        item = item.strip()
        if item and item not in out:
            out.append(item)
    return out


def run(cmd: List[str], execute: bool) -> int:
    print("$ " + " ".join(shlex.quote(x) for x in cmd), flush=True)
    if not execute:
        return 0
    proc = subprocess.run(cmd, check=False)
    return int(proc.returncode)


def dataset_num_classes(dataset: str) -> int:
    preset = DATASET_PRESETS.get(dataset, {})
    if "num_classes" not in preset:
        raise ValueError(f"No num_classes preset for {dataset}; pass fewer datasets or add a preset.")
    return int(preset["num_classes"])


def dataset_cmd(args, dataset: str) -> List[str]:
    cmd = [
        "python",
        "run_recommended_experiment.py",
        "--dataset",
        dataset,
        "--num_classes",
        str(dataset_num_classes(dataset)),
        "--run_tag",
        args.run_tag,
        "--model_types",
        args.model_types,
        "--tower2_epochs",
        str(args.tower2_epochs),
        "--tower2_early_stop_patience",
        str(args.tower2_early_stop_patience),
        "--plan_json",
        str(Path("reasoningDataset") / dataset / f"recommended_experiment_plan_{args.run_tag}.json"),
    ]
    if args.execute:
        cmd.append("--execute")
    if args.allow_no_cuda:
        cmd.append("--allow_no_cuda")
    if args.no_skip_existing:
        cmd.append("--no-skip_existing")
    if args.require_cuda_for_tower2:
        cmd.append("--require_cuda_for_tower2")
    return cmd


def dataset_status(dataset: str) -> Dict[str, Any]:
    rows = collect_dataset(dataset, ["test*.json"])
    target = DEFAULT_TARGETS.get(dataset)
    best = rows[0] if rows else None
    achieved = None
    if target and best:
        achieved = bool(best["accuracy"] >= target[0] and (best.get("macro_f1") or 0.0) >= target[1])
    return {
        "dataset": dataset,
        "target": None if target is None else {"accuracy": target[0], "macro_f1": target[1]},
        "target_met": achieved,
        "num_results": len(rows),
        "best": best,
    }


def write_suite_plan(
    args,
    datasets: List[str],
    commands: List[List[str]],
    cuda: Dict[str, Any],
    command_results: List[Dict[str, Any]] | None = None,
) -> None:
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "datasets": datasets,
        "execute": bool(args.execute),
        "run_tag": args.run_tag,
        "model_types": args.model_types,
        "cuda": cuda,
        "dataset_status": [dataset_status(dataset) for dataset in datasets],
        "commands": [
            {"dataset": dataset, "cmd": cmd}
            for dataset, cmd in zip(datasets, commands)
        ],
        "command_results": command_results or [],
    }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run or dry-run the recommended autopilot across multiple datasets.")
    ap.add_argument("--datasets", default="vpn-app,tls-120,ustc-app")
    ap.add_argument("--run_tag", default="paired_ipport")
    ap.add_argument("--model_types", default="graph,seq")
    ap.add_argument("--tower2_epochs", type=int, default=30)
    ap.add_argument("--tower2_early_stop_patience", type=int, default=8)
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--allow_no_cuda", action="store_true")
    ap.add_argument("--require_cuda_for_tower2", action="store_true")
    ap.add_argument("--no-skip_existing", action="store_true")
    ap.add_argument("--continue_on_error", action="store_true")
    ap.add_argument("--output_json", default="reasoningDataset/recommended_suite_plan.json")
    args = ap.parse_args()

    datasets = split_csv(args.datasets)
    if not datasets:
        raise SystemExit("--datasets must contain at least one dataset")
    unknown = [dataset for dataset in datasets if dataset not in DATASET_PRESETS]
    if unknown:
        raise SystemExit(f"No presets for datasets: {', '.join(unknown)}")

    cuda = cuda_summary()
    commands = [dataset_cmd(args, dataset) for dataset in datasets]
    write_suite_plan(args, datasets, commands, cuda)
    command_results: List[Dict[str, Any]] = []
    for cmd in commands:
        dataset = cmd[cmd.index("--dataset") + 1] if "--dataset" in cmd else ""
        returncode = run(cmd, execute=args.execute)
        command_results.append({"dataset": dataset, "returncode": returncode, "cmd": cmd})
        if returncode and not args.continue_on_error:
            write_suite_plan(args, datasets, commands, cuda, command_results)
            raise SystemExit(returncode)
    write_suite_plan(args, datasets, commands, cuda, command_results)


if __name__ == "__main__":
    main()
