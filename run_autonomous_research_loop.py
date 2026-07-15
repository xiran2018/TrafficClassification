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
from summarize_experiment_results import DEFAULT_TARGETS, RANK_METRICS, collect_dataset, parse_target


DEFAULT_UNIFIED_EXPERT_SLOTS = "base,graph,seq,prior_base,emb_lr,emb_et,proto_emb,paired,slot_stacker"


VARIANT_SCHEDULES: Dict[str, List[Dict[str, Any]]] = {
    "none": [
        {
            "name": "default",
            "paired_view_weight": 0.2,
            "paired_consistency_weight": 0.1,
            "consistency_weight": 0.05,
            "meta_dropout_prob": 0.1,
            "embedding_dropout_prob": 0.05,
            "window_dropout_prob": 0.1,
            "edge_attr_dropout_prob": 0.1,
            "seed": 42,
            "flow_pooling": "mean",
            "multi_view_gate_entropy_weight": 0.0,
        }
    ],
    "stage8_balanced": [
        {
            "name": "default_balanced",
            "paired_view_weight": 0.2,
            "paired_consistency_weight": 0.1,
            "consistency_weight": 0.05,
            "meta_dropout_prob": 0.1,
            "embedding_dropout_prob": 0.05,
            "window_dropout_prob": 0.1,
            "edge_attr_dropout_prob": 0.1,
            "seed": 42,
            "flow_pooling": "mean",
            "multi_view_gate_entropy_weight": 0.0,
        },
        {
            "name": "stronger_invariance",
            "paired_view_weight": 0.1,
            "paired_consistency_weight": 0.15,
            "consistency_weight": 0.1,
            "meta_dropout_prob": 0.2,
            "embedding_dropout_prob": 0.1,
            "window_dropout_prob": 0.15,
            "edge_attr_dropout_prob": 0.15,
            "seed": 43,
            "flow_pooling": "mean",
            "multi_view_gate_entropy_weight": 0.0,
        },
        {
            "name": "higher_paired_view_multiview_gate",
            "paired_view_weight": 0.3,
            "paired_consistency_weight": 0.1,
            "consistency_weight": 0.05,
            "meta_dropout_prob": 0.1,
            "embedding_dropout_prob": 0.05,
            "window_dropout_prob": 0.1,
            "edge_attr_dropout_prob": 0.1,
            "seed": 44,
            "flow_pooling": "multi_view",
            "multi_view_gate_entropy_weight": 0.01,
        },
        {
            "name": "dropout_regularized",
            "paired_view_weight": 0.15,
            "paired_consistency_weight": 0.1,
            "consistency_weight": 0.05,
            "meta_dropout_prob": 0.25,
            "embedding_dropout_prob": 0.1,
            "window_dropout_prob": 0.2,
            "edge_attr_dropout_prob": 0.2,
            "seed": 45,
            "flow_pooling": "mean",
            "multi_view_gate_entropy_weight": 0.0,
        },
    ],
}


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


def cmd_value(cmd: List[str], flag: str) -> str:
    if flag not in cmd:
        return ""
    idx = cmd.index(flag) + 1
    return cmd[idx] if idx < len(cmd) else ""


def target_map(overrides: List[str]) -> Dict[str, Tuple[float, float]]:
    targets = DEFAULT_TARGETS.copy()
    for raw in overrides:
        dataset, acc, macro_f1 = parse_target(raw)
        targets[dataset] = (acc, macro_f1)
    return targets


def dataset_status(datasets: List[str], targets: Dict[str, Tuple[float, float]], rank_metric: str = "accuracy") -> List[Dict[str, Any]]:
    rows = []
    for dataset in datasets:
        target = targets.get(dataset)
        results = collect_dataset(dataset, ["test*.json"], rank_metric, target)
        best = results[0] if results else None
        achieved = None
        if target and best:
            achieved = bool(best["accuracy"] >= target[0] and (best.get("macro_f1") or 0.0) >= target[1])
        rows.append(
            {
                "dataset": dataset,
                "target": None if target is None else {"accuracy": target[0], "macro_f1": target[1]},
                "rank_metric": rank_metric,
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


def best_delta_summary(before: List[Dict[str, Any]], after: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    before_map = {row["dataset"]: row for row in before}
    rows: List[Dict[str, Any]] = []
    for after_row in after:
        dataset = after_row["dataset"]
        before_best = (before_map.get(dataset) or {}).get("best")
        after_best = after_row.get("best")
        item: Dict[str, Any] = {
            "dataset": dataset,
            "before_best": before_best,
            "after_best": after_best,
            "best_changed": bool(
                before_best
                and after_best
                and before_best.get("path") != after_best.get("path")
            ),
            "new_best_found": False,
        }
        if before_best and after_best:
            item["delta_accuracy"] = float(after_best["accuracy"] - before_best["accuracy"])
            item["delta_macro_f1"] = float((after_best.get("macro_f1") or 0.0) - (before_best.get("macro_f1") or 0.0))
            item["new_best_found"] = bool(
                item["delta_accuracy"] > 0
                or (item["delta_accuracy"] == 0 and item["delta_macro_f1"] > 0 and item["best_changed"])
            )
        elif after_best and not before_best:
            item["new_best_found"] = True
        rows.append(item)
    return rows


def framework_ready(framework: Dict[str, Any] | None, required: bool) -> bool:
    if not required:
        return True
    return bool(isinstance(framework, dict) and framework.get("consistent") is True)


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
            "--required_expert_slots",
            args.final_selector_unified_expert_slots,
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
        [
            "python",
            "make_paper_evidence_pack.py",
            "--output_json",
            "reasoningDataset/paper_evidence_pack.json",
            "--output_md",
            "reasoningDataset/paper_evidence_pack.md",
        ],
    ]


def iteration_run_tag(args, iteration: int) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    try:
        return args.run_tag_template.format(
            run_tag=args.run_tag,
            iteration=iteration,
            iter=iteration,
            timestamp=timestamp,
        )
    except Exception as exc:
        raise ValueError(f"Invalid --run_tag_template={args.run_tag_template!r}: {exc}") from exc


def iteration_variant(args, iteration: int) -> Dict[str, Any]:
    variants = VARIANT_SCHEDULES[args.variant_schedule]
    return dict(variants[(iteration - 1) % len(variants)])


def suite_cmd(args, datasets: List[str], iteration: int, run_tag: str, variant: Dict[str, Any]) -> List[str]:
    cmd = [
        "python",
        "run_recommended_suite.py",
        "--datasets",
        ",".join(datasets),
        "--run_tag",
        run_tag,
        "--model_types",
        args.model_types,
        "--tower2_epochs",
        str(args.tower2_epochs),
        "--tower2_early_stop_patience",
        str(args.tower2_early_stop_patience),
        "--paired_view_weight",
        str(variant["paired_view_weight"]),
        "--paired_consistency_weight",
        str(variant["paired_consistency_weight"]),
        "--consistency_weight",
        str(variant["consistency_weight"]),
        "--meta_dropout_prob",
        str(variant["meta_dropout_prob"]),
        "--embedding_dropout_prob",
        str(variant["embedding_dropout_prob"]),
        "--window_dropout_prob",
        str(variant["window_dropout_prob"]),
        "--edge_attr_dropout_prob",
        str(variant["edge_attr_dropout_prob"]),
        "--seed",
        str(variant["seed"]),
        "--flow_pooling",
        str(variant["flow_pooling"]),
        "--multi_view_gate_entropy_weight",
        str(variant["multi_view_gate_entropy_weight"]),
        "--final_selector_unified_expert_slots",
        args.final_selector_unified_expert_slots,
        "--output_json",
        str(Path(args.suite_output_dir) / f"recommended_suite_plan_iter{iteration:02d}.json"),
    ]
    if not args.enable_slot_stacker:
        cmd.append("--no-enable_slot_stacker")
    if args.execute:
        cmd.append("--execute")
    if args.allow_no_cuda:
        cmd.append("--allow_no_cuda")
    if args.continue_on_error:
        cmd.append("--continue_on_error")
    return cmd


def load_suite_summary(cmd: List[str]) -> Dict[str, Any]:
    path = cmd_value(cmd, "--output_json")
    summary: Dict[str, Any] = {"output_json": path, "exists": bool(path and Path(path).exists())}
    if not summary["exists"]:
        return summary
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        summary["error"] = str(exc)
        return summary
    for key in [
        "run_tag",
        "execute",
        "materialize_child_plans",
        "dataset_status",
        "experiment_config",
        "commands",
        "child_plans",
        "command_results",
    ]:
        if key in data:
            summary[key] = data[key]
    return summary


def load_framework_consistency() -> Dict[str, Any] | None:
    path = Path("reasoningDataset/paper_framework_report.json")
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"error": str(exc)}
    return data.get("framework_consistency")


def load_evidence_pack() -> Dict[str, Any] | None:
    path = Path("reasoningDataset/paper_evidence_pack.json")
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"error": str(exc)}


def ci_targets_ready(evidence: Dict[str, Any] | None, goal_datasets: List[str], required: bool) -> bool:
    if not required:
        return True
    if not isinstance(evidence, dict):
        return False
    by_dataset = {row.get("dataset"): row for row in evidence.get("claims", [])}
    for dataset in goal_datasets:
        row = by_dataset.get(dataset)
        if not row or row.get("ci_target_met") is not True:
            return False
    return True


def framework_point_targets_ready(evidence: Dict[str, Any] | None, goal_datasets: List[str], required: bool) -> bool:
    if not required:
        return True
    if not isinstance(evidence, dict):
        return False
    by_dataset = {row.get("dataset"): row for row in evidence.get("claims", [])}
    for dataset in goal_datasets:
        row = by_dataset.get(dataset)
        if not row or row.get("point_target_met") is not True:
            return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="Autonomous research loop for the unified traffic framework.")
    ap.add_argument("--datasets", default="vpn-app,tls-120,ustc-app")
    ap.add_argument("--goal_datasets", default="vpn-app,tls-120")
    ap.add_argument("--target", action="append", default=[], help="Optional DATASET:ACC:MACRO_F1 target override.")
    ap.add_argument("--max_iters", type=int, default=1)
    ap.add_argument("--execute", action="store_true", help="Run recommended experiments when goals are not met.")
    ap.add_argument("--continue_after_targets", action="store_true", help="Keep running recommended suite even if target gates already pass.")
    ap.add_argument("--require_framework_consistency", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--require_ci_targets", action="store_true", help="Stop only when goal datasets pass bootstrap CI target gates in the evidence pack.")
    ap.add_argument("--allow_no_cuda", action="store_true")
    ap.add_argument("--continue_on_error", action="store_true")
    ap.add_argument("--run_tag", default="paired_ipport")
    ap.add_argument(
        "--run_tag_template",
        default="{run_tag}",
        help="Per-iteration run tag template. Use e.g. '{run_tag}_iter{iteration:02d}' to avoid reusing existing outputs.",
    )
    ap.add_argument("--model_types", default="graph,seq")
    ap.add_argument("--tower2_epochs", type=int, default=30)
    ap.add_argument("--tower2_early_stop_patience", type=int, default=8)
    ap.add_argument(
        "--variant_schedule",
        choices=sorted(VARIANT_SCHEDULES),
        default="stage8_balanced",
        help="Stage-8 training hyperparameter schedule used across autonomous-loop iterations.",
    )
    ap.add_argument(
        "--status_rank_metric",
        choices=RANK_METRICS,
        default="accuracy",
        help="Rank test JSONs for best before/after status by accuracy, macro_f1, balanced, or target_margin.",
    )
    ap.add_argument(
        "--final_selector_unified_expert_slots",
        default=DEFAULT_UNIFIED_EXPERT_SLOTS,
        help="Comma-separated expert slots required by the paper framework audit and passed to every suite child plan.",
    )
    ap.add_argument(
        "--enable_slot_stacker",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep the trainable unified-slot probability stacker in every recommended-suite child plan.",
    )
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
        "require_framework_consistency": bool(args.require_framework_consistency),
        "require_ci_targets": bool(args.require_ci_targets),
        "run_tag": args.run_tag,
        "run_tag_template": args.run_tag_template,
        "variant_schedule": args.variant_schedule,
        "status_rank_metric": args.status_rank_metric,
        "final_selector_unified_expert_slots": args.final_selector_unified_expert_slots,
        "enable_slot_stacker": bool(args.enable_slot_stacker),
        "cuda": cuda_summary(),
        "iterations": [],
        "stop_reason": "",
    }

    for iteration in range(1, args.max_iters + 1):
        run_tag = iteration_run_tag(args, iteration)
        variant = iteration_variant(args, iteration)
        before = dataset_status(datasets, targets, args.status_rank_metric)
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
        evidence = load_evidence_pack()
        goals_met = all_goals_met(before, goal_datasets)
        framework_met = framework_ready(framework, args.require_framework_consistency)
        framework_point_targets_met = framework_point_targets_ready(evidence, goal_datasets, args.require_framework_consistency)
        ci_targets_met = ci_targets_ready(evidence, goal_datasets, args.require_ci_targets)
        ready_to_stop = goals_met and framework_met and framework_point_targets_met and ci_targets_met
        record: Dict[str, Any] = {
            "iteration": iteration,
            "status_before": before,
            "framework_consistency": framework,
            "evidence_pack": evidence,
            "goals_met_before": goals_met,
            "framework_met_before": framework_met,
            "framework_point_targets_met_before": framework_point_targets_met,
            "ci_targets_met_before": ci_targets_met,
            "ready_to_stop_before": ready_to_stop,
            "run_tag": run_tag,
            "variant": variant,
            "commands": commands,
        }
        if ready_to_stop and not args.continue_after_targets:
            record["action"] = "stop_targets_and_framework_met"
            ledger["iterations"].append(record)
            ledger["stop_reason"] = "targets_and_framework_met"
            write_ledger(args, ledger)
            return

        suite_command = suite_cmd(args, datasets, iteration, run_tag, variant)
        suite_result = run_cmd(suite_command, execute=True)
        record["commands"].append(suite_result)
        record["suite_summary"] = load_suite_summary(suite_command)
        if suite_result["returncode"] and not args.continue_on_error:
            ledger["iterations"].append(record)
            ledger["stop_reason"] = "suite_failed"
            write_ledger(args, ledger)
            raise SystemExit(suite_result["returncode"])

        post_report_commands = []
        for cmd in report_commands(args, datasets):
            result = run_cmd(cmd, execute=True)
            post_report_commands.append(result)
            record["commands"].append(result)
            if result["returncode"] and not args.continue_on_error:
                record["post_report_commands"] = post_report_commands
                ledger["iterations"].append(record)
                ledger["stop_reason"] = f"post_suite_command_failed:{cmd[1]}"
                write_ledger(args, ledger)
                raise SystemExit(result["returncode"])

        framework_after = load_framework_consistency()
        evidence_after = load_evidence_pack()
        record["status_after"] = dataset_status(datasets, targets, args.status_rank_metric)
        record["best_delta"] = best_delta_summary(before, record["status_after"])
        record["goals_met_after"] = all_goals_met(record["status_after"], goal_datasets)
        record["framework_consistency_after"] = framework_after
        record["evidence_pack_after"] = evidence_after
        record["framework_met_after"] = framework_ready(framework_after, args.require_framework_consistency)
        record["framework_point_targets_met_after"] = framework_point_targets_ready(
            evidence_after, goal_datasets, args.require_framework_consistency
        )
        record["ci_targets_met_after"] = ci_targets_ready(evidence_after, goal_datasets, args.require_ci_targets)
        record["ready_to_stop_after"] = (
            record["goals_met_after"]
            and record["framework_met_after"]
            and record["framework_point_targets_met_after"]
            and record["ci_targets_met_after"]
        )
        record["action"] = "ran_recommended_suite" if args.execute else "dry_run_recommended_suite"
        ledger["iterations"].append(record)
        write_ledger(args, ledger)
        if record["ready_to_stop_after"] and not args.continue_after_targets:
            ledger["stop_reason"] = "targets_and_framework_met_after_iteration"
            write_ledger(args, ledger)
            return

    ledger["stop_reason"] = "max_iters_reached"
    write_ledger(args, ledger)


if __name__ == "__main__":
    main()
