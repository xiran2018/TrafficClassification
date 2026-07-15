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
        "--paired_view_weight",
        str(args.paired_view_weight),
        "--paired_consistency_weight",
        str(args.paired_consistency_weight),
        "--consistency_weight",
        str(args.consistency_weight),
        "--meta_dropout_prob",
        str(args.meta_dropout_prob),
        "--embedding_dropout_prob",
        str(args.embedding_dropout_prob),
        "--window_dropout_prob",
        str(args.window_dropout_prob),
        "--edge_attr_dropout_prob",
        str(args.edge_attr_dropout_prob),
        "--seed",
        str(args.seed),
        "--flow_pooling",
        args.flow_pooling,
        "--multi_view_gate_entropy_weight",
        str(args.multi_view_gate_entropy_weight),
        "--final_selector_unified_expert_slots",
        args.final_selector_unified_expert_slots,
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


def cmd_value(cmd: List[str], flag: str) -> str:
    if flag not in cmd:
        return ""
    idx = cmd.index(flag) + 1
    return cmd[idx] if idx < len(cmd) else ""


def child_plan_summary(dataset: str, cmd: List[str]) -> Dict[str, Any]:
    plan_json = cmd_value(cmd, "--plan_json")
    summary: Dict[str, Any] = {
        "dataset": dataset,
        "plan_json": plan_json,
        "exists": bool(plan_json and Path(plan_json).exists()),
    }
    if not summary["exists"]:
        return summary
    try:
        data = json.loads(Path(plan_json).read_text(encoding="utf-8"))
    except Exception as exc:
        summary["error"] = str(exc)
        return summary
    for key in [
        "run_tag",
        "embedding_suffix",
        "paired_embedding_suffix",
        "base_selector_input",
        "paired_prior_output",
        "final_selector_output",
        "experiment_config",
    ]:
        if key in data:
            summary[key] = data[key]
    stages = data.get("stages", [])
    summary["num_stages"] = len(stages)
    summary["skipped_stages"] = [stage.get("name") for stage in stages if stage.get("skip_if")]
    summary["required_cuda_stages"] = [stage.get("name") for stage in stages if stage.get("requires_cuda")]
    return summary


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
        "materialize_child_plans": bool(args.materialize_child_plans),
        "run_tag": args.run_tag,
        "model_types": args.model_types,
        "experiment_config": {
            "paired_view_weight": args.paired_view_weight,
            "paired_consistency_weight": args.paired_consistency_weight,
            "consistency_weight": args.consistency_weight,
            "meta_dropout_prob": args.meta_dropout_prob,
            "embedding_dropout_prob": args.embedding_dropout_prob,
            "window_dropout_prob": args.window_dropout_prob,
            "edge_attr_dropout_prob": args.edge_attr_dropout_prob,
            "tower2_epochs": args.tower2_epochs,
            "tower2_early_stop_patience": args.tower2_early_stop_patience,
            "model_types": args.model_types,
            "seed": args.seed,
            "flow_pooling": args.flow_pooling,
            "multi_view_gate_entropy_weight": args.multi_view_gate_entropy_weight,
            "final_selector_unified_expert_slots": args.final_selector_unified_expert_slots,
        },
        "cuda": cuda,
        "dataset_status": [dataset_status(dataset) for dataset in datasets],
        "commands": [
            {
                "dataset": dataset,
                "run_tag": args.run_tag,
                "plan_json": cmd_value(cmd, "--plan_json"),
                "cmd": cmd,
            }
            for dataset, cmd in zip(datasets, commands)
        ],
        "child_plans": [child_plan_summary(dataset, cmd) for dataset, cmd in zip(datasets, commands)],
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
    ap.add_argument("--paired_view_weight", type=float, default=0.2)
    ap.add_argument("--paired_consistency_weight", type=float, default=0.1)
    ap.add_argument("--consistency_weight", type=float, default=0.05)
    ap.add_argument("--meta_dropout_prob", type=float, default=0.1)
    ap.add_argument("--embedding_dropout_prob", type=float, default=0.05)
    ap.add_argument("--window_dropout_prob", type=float, default=0.1)
    ap.add_argument("--edge_attr_dropout_prob", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--flow_pooling", choices=["mean", "attention", "late_fusion", "transformer", "multi_view"], default="mean")
    ap.add_argument("--multi_view_gate_entropy_weight", type=float, default=0.0)
    ap.add_argument(
        "--final_selector_unified_expert_slots",
        default="base,graph,seq,prior_base,emb_lr,emb_et,paired",
        help="Comma-separated final-selector expert slots shared by every dataset; missing slots become identity experts.",
    )
    ap.add_argument("--execute", action="store_true")
    ap.add_argument(
        "--materialize_child_plans",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="In suite dry-run mode, run each dataset autopilot without --execute so child JSON plans are written.",
    )
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
        plan_json = cmd_value(cmd, "--plan_json")
        returncode = run(cmd, execute=bool(args.execute or args.materialize_child_plans))
        command_results.append({"dataset": dataset, "plan_json": plan_json, "returncode": returncode, "cmd": cmd})
        if returncode and not args.continue_on_error:
            write_suite_plan(args, datasets, commands, cuda, command_results)
            raise SystemExit(returncode)
    write_suite_plan(args, datasets, commands, cuda, command_results)


if __name__ == "__main__":
    main()
