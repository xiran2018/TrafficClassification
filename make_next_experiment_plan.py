#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path
from typing import Any, Dict, List

from recommend_next_experiment import cuda_summary


NATIVE_STRUCTURAL_PILOT_RESULT = Path(
    "reasoningDataset/vpn-app/"
    "test_selector_base_prior_stacker_graph_seq_rawproj_flowaware_change_weight_split2_retrain_"
    "native_struct_pilot_stage8_flowaware_native_structural_pilot_accuracy.json"
)
NEGATIVE_NATIVE_STATUSES = {"negative_target_gap", "metrics_missing", "invalid_json"}
DISTILL_STUDENT_SELECTOR_RESULTS = {
    "vpn-app": Path(
        "reasoningDataset/vpn-app/"
        "test_selector_best_plus_rawproj_flowaware_change_weight_split2_retrain_"
        "stage8_flowaware_consensus_distill_student_valid_macro.json"
    ),
    "tls-120": Path(
        "reasoningDataset/tls-120/"
        "test_selector_best_plus_rawproj_flowaware_change_weight_fold2_"
        "stage8_flowaware_consensus_distill_student_valid_macro.json"
    ),
}
DISTILL_COVERAGE_AUDIT_RESULTS = {
    "vpn-app": Path("reasoningDataset/vpn-app/distill_teacher_coverage_flowaware_split2_current.json"),
    "tls-120": Path("reasoningDataset/tls-120/distill_teacher_coverage_flowaware_fold2_current.json"),
}


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


def nested_flow_metrics(data: Dict[str, Any]) -> Dict[str, Any]:
    metrics = data.get("metrics")
    if isinstance(metrics, dict):
        flow = metrics.get("flow_level")
        if isinstance(flow, dict):
            return flow
        return metrics
    return {}


def native_structural_pilot_status(path: Path = NATIVE_STRUCTURAL_PILOT_RESULT) -> Dict[str, Any]:
    if not path.exists():
        return {"status": "not_run", "path": str(path)}
    try:
        data = load_json(str(path))
    except Exception as exc:
        return {"status": "invalid_json", "path": str(path), "error": str(exc)}
    metrics = nested_flow_metrics(data)
    acc = metrics.get("accuracy")
    macro_f1 = metrics.get("macro_f1")
    if acc is None or macro_f1 is None:
        return {"status": "metrics_missing", "path": str(path)}
    status = "negative_target_gap" if float(acc) < 0.74 or float(macro_f1) < 0.65 else "candidate"
    return {"status": status, "path": str(path), "accuracy": acc, "macro_f1": macro_f1}


def distill_student_suite_status(paths: Dict[str, Path] = DISTILL_STUDENT_SELECTOR_RESULTS) -> Dict[str, Any]:
    rows = []
    for dataset, path in paths.items():
        row: Dict[str, Any] = {"dataset": dataset, "path": str(path), "exists": path.exists()}
        if path.exists():
            try:
                metrics = nested_flow_metrics(load_json(str(path)))
                row.update({"accuracy": metrics.get("accuracy"), "macro_f1": metrics.get("macro_f1")})
                row["metric_status"] = "available" if row["accuracy"] is not None and row["macro_f1"] is not None else "metrics_missing"
            except Exception as exc:
                row.update({"metric_status": "invalid_json", "error": str(exc)})
        else:
            row["metric_status"] = "missing"
        rows.append(row)
    complete = bool(rows) and all(row.get("metric_status") == "available" for row in rows)
    return {"status": "complete" if complete else "incomplete", "results": rows}


def distill_coverage_audit_status(paths: Dict[str, Path] = DISTILL_COVERAGE_AUDIT_RESULTS) -> Dict[str, Any]:
    rows = []
    for dataset, path in paths.items():
        row: Dict[str, Any] = {"dataset": dataset, "path": str(path), "exists": path.exists()}
        if path.exists():
            try:
                data = load_json(str(path))
                recommendation = data.get("recommendation")
                datasets = data.get("datasets") or []
                row.update(
                    {
                        "recommendation": recommendation,
                        "min_coverage": data.get("min_coverage"),
                        "coverage": min(
                            (float(item.get("coverage", 0.0)) for item in datasets),
                            default=None,
                        ),
                    }
                )
                row["audit_status"] = "passed" if recommendation == "flow_id_distillation_safe" else "not_safe"
            except Exception as exc:
                row.update({"audit_status": "invalid_json", "error": str(exc)})
        else:
            row["audit_status"] = "missing"
        rows.append(row)
    complete = bool(rows) and all(row.get("audit_status") == "passed" for row in rows)
    return {"status": "complete" if complete else "incomplete", "results": rows}


def insert_before(commands: List[Dict[str, Any]], before_name: str, item: Dict[str, Any]) -> None:
    for idx, command in enumerate(commands):
        if command.get("name") == before_name:
            commands.insert(idx, item)
            return
    commands.append(item)


def classify_row(claim: Dict[str, Any], rec: Dict[str, Any]) -> Dict[str, Any]:
    target_acc = claim.get("target_accuracy")
    target_f1 = claim.get("target_macro_f1")
    acc = claim.get("accuracy")
    macro_f1 = claim.get("macro_f1")
    acc_lo = ci_lower(claim.get("accuracy_ci95"))
    f1_lo = ci_lower(claim.get("macro_f1_ci95"))
    group_acc_lo = ci_lower(claim.get("content_group_accuracy_ci95"))
    group_f1_lo = ci_lower(claim.get("content_group_macro_f1_ci95"))
    point_acc_gap = positive_gap(target_acc, acc)
    point_f1_gap = positive_gap(target_f1, macro_f1)
    ci_acc_gap = positive_gap(target_acc, acc_lo)
    ci_f1_gap = positive_gap(target_f1, f1_lo)
    group_ci_acc_gap = positive_gap(target_acc, group_acc_lo)
    group_ci_f1_gap = positive_gap(target_f1, group_f1_lo)
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
    elif group_ci_acc_gap or group_ci_f1_gap:
        kind = "content_grouped_ci_lower_bound_gap"
        priority = 90.0 + max(group_ci_acc_gap or 0.0, group_ci_f1_gap or 0.0)
        next_action = (
            "Content-grouped robustness evidence is available and still below the target lower bound. "
            "Prioritize unified coverage-audited consensus distillation into trainable graph/seq students, repeated seeds, "
            "and validation-gated promotion using the same expert-slot interface."
        )
    elif ci_acc_gap or ci_f1_gap:
        kind = "ci_lower_bound_gap"
        priority = 80.0 + max(ci_acc_gap or 0.0, ci_f1_gap or 0.0)
        recommendation = str(rec.get("recommendation") or "")
        if "Fresh flow-aware paired-view probes are also negative" in recommendation:
            next_action = (
                "Do not rerun stronger Tower-2 paired IP/port intervention. If content-grouped evidence is missing, build it first; "
                "otherwise prioritize coverage-audited consensus distillation into the same graph/seq student interface. "
                "Keep native structural pretraining as a negative ablation unless its objective or gate is redesigned."
            )
        else:
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
        "content_group_accuracy_ci95": claim.get("content_group_accuracy_ci95"),
        "content_group_macro_f1_ci95": claim.get("content_group_macro_f1_ci95"),
        "content_group_ci_accuracy_gap": group_ci_acc_gap,
        "content_group_ci_macro_f1_gap": group_ci_f1_gap,
        "raw_minus_paper_safe_accuracy": raw_gap_acc,
        "raw_minus_paper_safe_macro_f1": raw_gap_f1,
        "raw_best_path": rec.get("raw_best_path"),
        "paper_safe_path": rec.get("paper_safe_path"),
        "gap_kind": kind,
        "priority_score": priority,
        "next_action": next_action,
    }


def build_commands(
    args: argparse.Namespace,
    native_status: Dict[str, Any] | None = None,
    distill_status: Dict[str, Any] | None = None,
    coverage_status: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    native_status = native_status or native_structural_pilot_status()
    distill_status = distill_status or distill_student_suite_status()
    coverage_status = coverage_status or distill_coverage_audit_status()
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
    commands = [
        {
            "name": "content_unique_robustness_refresh",
            "purpose": "Refresh exact-PCAP content-unique and content-grouped bootstrap robustness metrics for the current VPN/TLS paper-safe predictions.",
            "cmd": [
                "conda",
                "run",
                "--no-capture-output",
                "-n",
                "llm-factory",
                "python",
                "make_paper_evidence_pack.py",
                "--output_json",
                "reasoningDataset/paper_evidence_pack.json",
                "--output_md",
                "reasoningDataset/paper_evidence_pack.md",
            ],
        },
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
    if coverage_status.get("status") != "complete":
        commands[1:1] = [
            {
                "name": "vpn_distillation_coverage_audit",
                "purpose": "Verify whether the VPN consensus teacher is in the same flow-id namespace as the flow-aware student train split before enabling flow-id KL.",
                "cmd": [
                    "conda",
                    "run",
                    "--no-capture-output",
                    "-n",
                    "llm-factory",
                    "python",
                    "audit_distillation_teacher_coverage.py",
                    "--teacher_json",
                    "reasoningDataset/vpn-app/train_namespace_oof_teacher_from_valid_consensus_flowaware_split2_retrain.json",
                    "--dataset",
                    "train_graph",
                    "reasoningDataset/vpn-app/train_tower2_rawproj_flowaware_change_weight_split2_retrain/graph_dataset.pt",
                    "--dataset",
                    "train_seq",
                    "reasoningDataset/vpn-app/train_tower2_rawproj_flowaware_change_weight_split2_retrain/seq_dataset.pt",
                    "--gate_dataset",
                    "train_graph",
                    "--gate_dataset",
                    "train_seq",
                    "--min_coverage",
                    "0.50",
                    "--low_coverage_action",
                    "fail",
                    "--output_json",
                    "reasoningDataset/vpn-app/distill_teacher_coverage_flowaware_split2_current.json",
                ],
            },
            {
                "name": "tls120_distillation_coverage_audit",
                "purpose": "Verify whether the TLS-120 consensus teacher is in the same flow-id namespace as the flow-aware student train split before enabling flow-id KL.",
                "cmd": [
                    "conda",
                    "run",
                    "--no-capture-output",
                    "-n",
                    "llm-factory",
                    "python",
                    "audit_distillation_teacher_coverage.py",
                    "--teacher_json",
                    "reasoningDataset/tls-120/train_namespace_oof_teacher_from_valid_consensus_flowaware_fold2.json",
                    "--dataset",
                    "train_graph",
                    "reasoningDataset/tls-120/train_tower2_rawproj_flowaware_change_weight_fold2/graph_dataset.pt",
                    "--dataset",
                    "train_seq",
                    "reasoningDataset/tls-120/train_tower2_rawproj_flowaware_change_weight_fold2/seq_dataset.pt",
                    "--gate_dataset",
                    "train_graph",
                    "--gate_dataset",
                    "train_seq",
                    "--min_coverage",
                    "0.50",
                    "--low_coverage_action",
                    "fail",
                    "--output_json",
                    "reasoningDataset/tls-120/distill_teacher_coverage_flowaware_fold2_current.json",
                ],
            },
        ]
    if native_status.get("status") not in NEGATIVE_NATIVE_STATUSES:
        insert_before(
            commands,
            "a800_strict_ci_stage8_loop",
            {
                "name": "native_structural_pretraining_pilot",
                "purpose": "Run the next paper-grade direction end to end: label-free native protocol-structural pretraining, structural embedding extraction, Tower-2 preprocessing, and validation-gated flow evaluation.",
                "cmd": [
                    "conda",
                    "run",
                    "--no-capture-output",
                    "-n",
                    "llm-factory",
                    "python",
                    "run_stage8_flowaware_pipeline.py",
                    "--dataset",
                    "vpn-app",
                    "--num_classes",
                    "16",
                    "--stage",
                    "paper_unified",
                    "--framework_profile",
                    "paper_unified",
                    "--embedding_suffix",
                    "rawproj_flowaware_change_weight_split2_retrain",
                    "--output_suffix",
                    "flowaware_change_weight_split2_retrain",
                    "--tower1_data_suffix",
                    "flowaware_change_weight_split2_retrain",
                    "--tower1_output_dir",
                    "checkpoints/tower1_qwen_multitask_vpn_app_flowaware_change_weight_split2_retrain",
                    "--native_structural_suffix",
                    "struct_pilot",
                    "--native_epochs",
                    "20",
                    "--native_batch_size",
                    "8",
                    "--native_extract_batch_size",
                    "16",
                    "--run_tag",
                    "native_structural_pilot",
                ],
            },
        )
    if distill_status.get("status") != "complete":
        insert_before(
            commands,
            "a800_strict_ci_stage8_loop",
            {
                "name": "coverage_audited_calibrated_distillation_ablation",
                "purpose": "Reproduce the calibrated paper_unified distillation ablation only after coverage audits pass; do not promote it unless a new teacher construction closes the test gap.",
                "cmd": [
                    "conda",
                    "run",
                    "--no-capture-output",
                    "-n",
                    "llm-factory",
                    "python",
                    "run_recommended_suite.py",
                    "--datasets",
                    args.datasets,
                    "--run_tag",
                    "consensus_distill_student",
                    "--model_types",
                    "graph,seq",
                    "--framework_profile",
                    "paper_unified",
                    "--tower2_epochs",
                    "30",
                    "--paired_view_weight",
                    "0.0",
                    "--paired_consistency_weight",
                    "0.0",
                    "--paired_alignment_weight",
                    "0.0",
                    "--paired_crossview_contrastive_weight",
                    "0.0",
                    "--paired_variance_weight",
                    "0.0",
                    "--view_domain_adversarial_weight",
                    "0.0",
                    "--distill_target",
                    "vpn-app=reasoningDataset/vpn-app/train_namespace_oof_teacher_from_valid_consensus_flowaware_split2_retrain.json",
                    "--distill_target",
                    "tls-120=reasoningDataset/tls-120/train_namespace_oof_teacher_from_valid_consensus_flowaware_fold2.json",
                    "--distill_weight",
                    "0.02",
                    "--distill_class_prior_weight",
                    "0.01",
                    "--distill_temperature",
                    "6.0",
                    "--distill_min_confidence",
                    "0.45",
                    "--distill_max_confidence",
                    "0.85",
                    "--distill_confidence_power",
                    "0.0",
                    "--distill_min_coverage",
                    "0.50",
                    "--distill_low_coverage_action",
                    "disable_flow",
                    "--execute",
                ],
            },
        )
    return commands


def build_plan(args: argparse.Namespace) -> Dict[str, Any]:
    pack = load_json(args.evidence_pack)
    recs = recommendation_by_dataset(pack)
    rows = [classify_row(claim, recs.get(claim.get("dataset")) or {}) for claim in pack.get("claims", [])]
    rows.sort(key=lambda row: (-float(row["priority_score"]), str(row["dataset"])))
    native_status = native_structural_pilot_status()
    distill_status = distill_student_suite_status()
    coverage_status = distill_coverage_audit_status()
    return {
        "cuda": cuda_summary(),
        "source": {
            "evidence_pack": args.evidence_pack,
            "framework_consistency": pack.get("framework_consistency"),
            "native_structural_pilot": native_status,
            "distill_coverage_audit": coverage_status,
            "distill_student_suite": distill_status,
        },
        "priority": rows,
        "commands": build_commands(args, native_status, distill_status, coverage_status),
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
        "| Rank | Dataset | Gap Kind | Acc | F1 | Target | Acc CI | F1 CI | CI Gap (Std/Group) | Raw-Paper Gap | Action |",
        "|---:|---|---|---:|---:|---|---|---|---|---|---|",
    ]
    for idx, row in enumerate(plan["priority"], 1):
        target = "-"
        if row["target_accuracy"] is not None:
            target = f"{fmt(row['target_accuracy'])}/{fmt(row['target_macro_f1'])}"
        ci_gap = f"{fmt(row['ci_accuracy_gap'])}/{fmt(row['ci_macro_f1_gap'])}"
        group_ci_gap = f"{fmt(row.get('content_group_ci_accuracy_gap'))}/{fmt(row.get('content_group_ci_macro_f1_gap'))}"
        raw_gap = f"{fmt(row['raw_minus_paper_safe_accuracy'], signed=True)}/{fmt(row['raw_minus_paper_safe_macro_f1'], signed=True)}"
        lines.append(
            "| {idx} | {dataset} | {kind} | {acc} | {f1} | {target} | {acc_ci} | {f1_ci} | {ci_gap} (group {group_ci_gap}) | {raw_gap} | {action} |".format(
                idx=idx,
                dataset=row["dataset"],
                kind=row["gap_kind"],
                acc=fmt(row["accuracy"]),
                f1=fmt(row["macro_f1"]),
                target=target,
                acc_ci=format_ci(row.get("accuracy_ci95")),
                f1_ci=format_ci(row.get("macro_f1_ci95")),
                ci_gap=ci_gap,
                group_ci_gap=group_ci_gap,
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
    ap.add_argument("--datasets", default="vpn-app,tls-120")
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
