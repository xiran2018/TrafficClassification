#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path
from typing import Any, Dict, List

from recommend_next_experiment import cuda_summary
from run_recommended_experiment import DATASET_PRESETS


def split_csv(raw: str) -> List[str]:
    out: List[str] = []
    for item in raw.split(","):
        item = item.strip()
        if item and item not in out:
            out.append(item)
    return out


def run(cmd: List[str], execute: bool, continue_on_error: bool) -> int:
    print("$ " + " ".join(shlex.quote(x) for x in cmd), flush=True)
    if not execute:
        return 0
    proc = subprocess.run(cmd, check=False)
    if proc.returncode and not continue_on_error:
        raise SystemExit(proc.returncode)
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


def write_suite_plan(args, datasets: List[str], commands: List[List[str]], cuda: Dict[str, Any]) -> None:
    payload = {
        "datasets": datasets,
        "execute": bool(args.execute),
        "run_tag": args.run_tag,
        "model_types": args.model_types,
        "cuda": cuda,
        "commands": commands,
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
    for cmd in commands:
        run(cmd, execute=args.execute, continue_on_error=args.continue_on_error)


if __name__ == "__main__":
    main()
