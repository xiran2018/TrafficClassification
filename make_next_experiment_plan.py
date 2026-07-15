#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path
from typing import Any, Dict, List

from recommend_next_experiment import cuda_summary


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def fmt(value: Any, digits: int = 4, signed: bool = False) -> str:
    if value is None:
        return "-"
    prefix = "+" if signed and float(value) > 0 else ""
    return f"{prefix}{float(value):.{digits}f}"


def ci_lower(ci: Any) -> float | None:
    if isinstance(ci, list) and ci:
        return float(ci[0])
    return None


def format_ci(ci: Any) -> str:
    if not isinstance(ci, list) or len(ci) != 2:
        return "-"
    return f"[{fmt(ci[0])}, {fmt(ci[1])}]"


def positive_gap(target: Any, value: Any) -> float | None:
    if target is None or value is None:
        return None
    return max(0.0, float(target) - float(value))


def recommendation_by_dataset(pack: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {row.get("dataset"): row for row in pack.get("recommendations", [])}


def classify_row(claim: Dict[str, Any], rec: Dict[str, Any]) -> Dict[str, Any]:
    target_acc = claim.get("target_accuracy")
    target_f1 = claim.get("target_macro_f1")
    acc = claim.get("accuracy")
    macro_f1 = claim.get("macro_f1")
    acc_lo = ci_lower(claim.get("accuracy_ci95"))
    f1_lo = ci_lower(claim.get("macro_f1_ci95"))
    point_acc_gap = positive_gap(target_acc, acc)
    point_f1_gap = positive_gap(target_f1, macro_f1)
    ci_acc_gap = positive_gap(target_acc, acc_lo)
    ci_f1_gap = positive_gap(target_f1, f1_lo)
    raw_gap_acc = rec.get("raw_minus_paper_safe_accuracy")
    raw_gap_f1 = rec.get("raw_minus_paper_safe_macro_f1")
    raw_acceptance_gap = max(
        0.0,
        float(raw_gap_acc or 0.0),
        float(raw_gap_f1 or 0.0),
    )

    if target_acc is None:
        kind = "evidence_only"
        priority = 20.0
        next_action = "Increase split size/repeats before making a strong paper claim; keep the same selector and expert-slot interface."
    elif point_acc_gap or point_f1_gap:
        kind = "point_metric_gap"
        priority = 100.0 + max(point_acc_gap or 0.0, point_f1_gap or 0.0)
        next_action = "Run representation-learning experiments first; probability-level fusion is secondary until point targets pass."
    elif ci_acc_gap or ci_f1_gap:
        kind = "ci_lower_bound_gap"
        priority = 80.0 + max(ci_acc_gap or 0.0, ci_f1_gap or 0.0)
        next_action = "Run repeated Stage-8 flow-aware A800 experiments and accept only validation-gated candidates that improve CI lower bounds."
    elif raw_acceptance_gap > 0.003:
        kind = "raw_best_acceptance_gap"
        priority = 50.0 + raw_acceptance_gap
        next_action = "Search selector thresholds/guards that safely admit the raw-best expert without increasing target-shift risk."
    else:
        kind = "stable_or_optional_search"
        priority = 10.0
        next_action = "Keep as current paper-safe baseline; spend compute only on ablations or cross-dataset validation."

    return {
        "dataset": claim.get("dataset"),
        "claim_strength": claim.get("claim_strength"),
        "point_target_met": claim.get("point_target_met"),
        "ci_target_met": claim.get("ci_target_met"),
        "accuracy": acc,
        "macro_f1": macro_f1,
        "target_accuracy": target_acc,
        "target_macro_f1": target_f1,
        "accuracy_ci95": claim.get("accuracy_ci95"),
        "macro_f1_ci95": claim.get("macro_f1_ci95"),
        "point_accuracy_gap": point_acc_gap,
        "point_macro_f1_gap": point_f1_gap,
        "ci_accuracy_gap": ci_acc_gap,
        "ci_macro_f1_gap": ci_f1_gap,
        "raw_minus_paper_safe_accuracy": raw_gap_acc,
        "raw_minus_paper_safe_macro_f1": raw_gap_f1,
        "raw_best_path": rec.get("raw_best_path"),
        "paper_safe_path": rec.get("paper_safe_path"),
        "gap_kind": kind,
        "priority_score": priority,
        "next_action": next_action,
    }


def build_commands(args: argparse.Namespace) -> List[Dict[str, Any]]:
    base_cmd = [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        "llm-factory",
        "python",
        "run_autonomous_research_loop.py",
        "--datasets",
        args.datasets,
        "--goal_datasets",
        args.goal_datasets,
        "--max_iters",
        str(args.max_iters),
        "--execute",
        "--continue_after_targets",
        "--require_ci_targets",
        "--status_rank_metric",
        "target_margin",
        "--run_tag",
        args.run_tag,
        "--run_tag_template",
        args.run_tag_template,
        "--output_json",
        args.loop_output_json,
    ]
    return [
        {
            "name": "a800_strict_ci_stage8_loop",
            "purpose": "Continue unified-framework Stage-8 search in the real A800 llm-factory shell until point targets and CI gates pass.",
            "cmd": base_cmd,
        },
        {
            "name": "dry_run_plan_refresh",
            "purpose": "Refresh reports without launching CUDA stages.",
            "cmd": [
                "conda",
                "run",
                "--no-capture-output",
                "-n",
                "llm-factory",
                "python",
                "run_autonomous_research_loop.py",
                "--max_iters",
                "1",
                "--output_json",
                "reasoningDataset/autonomous_loop/research_loop_plan_refresh.json",
            ],
        },
    ]


def build_plan(args: argparse.Namespace) -> Dict[str, Any]:
    pack = load_json(args.evidence_pack)
    recs = recommendation_by_dataset(pack)
    rows = [classify_row(claim, recs.get(claim.get("dataset")) or {}) for claim in pack.get("claims", [])]
    rows.sort(key=lambda row: (-float(row["priority_score"]), str(row["dataset"])))
    return {
        "cuda": cuda_summary(),
        "source": {
            "evidence_pack": args.evidence_pack,
            "framework_consistency": pack.get("framework_consistency"),
        },
        "priority": rows,
        "commands": build_commands(args),
    }


def render_cmd(cmd: List[str]) -> str:
    return " ".join(shlex.quote(item) for item in cmd)


def render_markdown(plan: Dict[str, Any]) -> str:
    lines = [
        "# Next Experiment Plan",
        "",
        f"CUDA available in this session: `{plan['cuda'].get('available')}`; devices: `{plan['cuda'].get('device_count', 0)}`",
        "",
        "## Priority Gaps",
        "",
        "| Rank | Dataset | Gap Kind | Acc | F1 | Target | Acc CI | F1 CI | CI Gap | Raw-Paper Gap | Action |",
        "|---:|---|---|---:|---:|---|---|---|---|---|---|",
    ]
    for idx, row in enumerate(plan["priority"], 1):
        target = "-"
        if row["target_accuracy"] is not None:
            target = f"{fmt(row['target_accuracy'])}/{fmt(row['target_macro_f1'])}"
        ci_gap = f"{fmt(row['ci_accuracy_gap'])}/{fmt(row['ci_macro_f1_gap'])}"
        raw_gap = f"{fmt(row['raw_minus_paper_safe_accuracy'], signed=True)}/{fmt(row['raw_minus_paper_safe_macro_f1'], signed=True)}"
        lines.append(
            "| {idx} | {dataset} | {kind} | {acc} | {f1} | {target} | {acc_ci} | {f1_ci} | {ci_gap} | {raw_gap} | {action} |".format(
                idx=idx,
                dataset=row["dataset"],
                kind=row["gap_kind"],
                acc=fmt(row["accuracy"]),
                f1=fmt(row["macro_f1"]),
                target=target,
                acc_ci=format_ci(row.get("accuracy_ci95")),
                f1_ci=format_ci(row.get("macro_f1_ci95")),
                ci_gap=ci_gap,
                raw_gap=raw_gap,
                action=row["next_action"],
            )
        )
    lines += ["", "## Commands", ""]
    for item in plan["commands"]:
        lines += [
            f"### {item['name']}",
            "",
            item["purpose"],
            "",
            "```bash",
            render_cmd(item["cmd"]),
            "```",
            "",
        ]
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a CI-gap-aware next experiment plan from the paper evidence pack.")
    ap.add_argument("--evidence_pack", default="reasoningDataset/paper_evidence_pack.json")
    ap.add_argument("--datasets", default="vpn-app,tls-120,ustc-app")
    ap.add_argument("--goal_datasets", default="vpn-app,tls-120")
    ap.add_argument("--max_iters", type=int, default=4)
    ap.add_argument("--run_tag", default="stage8_flowaware")
    ap.add_argument("--run_tag_template", default="{run_tag}_ci_iter{iteration:02d}")
    ap.add_argument("--loop_output_json", default="reasoningDataset/autonomous_loop/research_loop_ci_stage8.json")
    ap.add_argument("--output_json", default="reasoningDataset/next_experiment_plan.json")
    ap.add_argument("--output_md", default="reasoningDataset/next_experiment_plan.md")
    args = ap.parse_args()

    plan = build_plan(args)
    md = render_markdown(plan)
    print(md)
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.output_md:
        Path(args.output_md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_md).write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()
