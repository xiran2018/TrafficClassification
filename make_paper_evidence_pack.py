#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


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


def ci_upper(ci: Any) -> float | None:
    if isinstance(ci, list) and len(ci) > 1:
        return float(ci[1])
    return None


def format_ci(ci: Any, signed: bool = False) -> str:
    if not isinstance(ci, list) or len(ci) != 2:
        return "-"
    return f"[{fmt(ci[0], signed=signed)}, {fmt(ci[1], signed=signed)}]"


def format_module_usage(module_usage: Any) -> str:
    if not isinstance(module_usage, dict):
        return "-"
    parts = []
    preferred = [
        "packet_embedding_backbone",
        "flow_base_expert",
        "validation_gated_selector",
        "bootstrap_guard",
        "target_shift_guard",
        "expert_switch_or_fusion",
        "class_bias_calibration_candidate",
        "base",
        "selector",
        "expert",
        "calib",
        "guards",
        "trainable_multiview_gate",
    ]
    keys = preferred + [key for key in module_usage if key not in preferred]
    for key in keys:
        value = module_usage.get(key)
        if value is None:
            continue
        if isinstance(value, dict):
            value = ",".join(f"{sub_key}:{sub_value}" for sub_key, sub_value in value.items())
        parts.append(f"{key}={value}")
    return "; ".join(parts) if parts else "-"


def format_selector(selector: Any) -> str:
    if isinstance(selector, str):
        return selector
    if not isinstance(selector, dict):
        return "-"
    strategy = selector.get("strategy") or selector.get("pool_strategy") or selector.get("kind")
    config = selector.get("config") or {}
    if isinstance(config, dict) and config:
        items = ", ".join(f"{key}={value}" for key, value in config.items())
        return f"{strategy}: {items}"
    return str(strategy) if strategy else "-"


def claim_rows(framework: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for row in framework.get("results", []):
        target_acc = row.get("target_accuracy")
        target_f1 = row.get("target_macro_f1")
        unc = row.get("uncertainty") or {}
        acc_ci = unc.get("accuracy_ci95")
        f1_ci = unc.get("macro_f1_ci95")
        ci_target_met = None
        if target_acc is not None and target_f1 is not None:
            ci_target_met = bool(
                ci_lower(acc_ci) is not None
                and ci_lower(f1_ci) is not None
                and ci_lower(acc_ci) >= float(target_acc)
                and ci_lower(f1_ci) >= float(target_f1)
            )
        if row.get("achieved") is True and ci_target_met is True:
            strength = "strong"
        elif row.get("achieved") is True:
            strength = "point_pass_ci_mixed"
        elif target_acc is None:
            strength = "evidence_only"
        else:
            strength = "not_met"
        rows.append(
            {
                "dataset": row["dataset"],
                "accuracy": row.get("accuracy"),
                "macro_f1": row.get("macro_f1"),
                "target_accuracy": target_acc,
                "target_macro_f1": target_f1,
                "point_target_met": row.get("achieved"),
                "ci_target_met": ci_target_met,
                "claim_strength": strength,
                "accuracy_ci95": acc_ci,
                "macro_f1_ci95": f1_ci,
                "module_usage": row.get("module_usage"),
                "selector_slot_summary": row.get("selector_slot_summary"),
                "multi_view_gate": row.get("multi_view_gate"),
                "selector": row.get("selector"),
                "num_flows": row.get("num_flows"),
            }
        )
    return rows


def classify_delta(row: Dict[str, Any]) -> str:
    unc = row.get("paired_delta_uncertainty") or {}
    dacc_ci = unc.get("delta_accuracy_ci95")
    df1_ci = unc.get("delta_macro_f1_ci95")
    if not dacc_ci or not df1_ci:
        return "no_paired_ci"
    acc_lo, acc_hi = ci_lower(dacc_ci), ci_upper(dacc_ci)
    f1_lo, f1_hi = ci_lower(df1_ci), ci_upper(df1_ci)
    if acc_hi is not None and f1_hi is not None and acc_hi < 0 and f1_hi < 0:
        return "harmful"
    if acc_lo is not None and f1_lo is not None and acc_lo > 0 and f1_lo > 0:
        return "helpful"
    if (acc_lo is not None and acc_hi is not None and acc_lo <= 0 <= acc_hi) or (
        f1_lo is not None and f1_hi is not None and f1_lo <= 0 <= f1_hi
    ):
        return "uncertain_or_neutral"
    return "mixed"


def ablation_rows(ablation: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for row in ablation.get("ablations", []):
        unc = row.get("paired_delta_uncertainty") or {}
        rows.append(
            {
                "dataset": row["dataset"],
                "stage": row["stage"],
                "delta_accuracy": row.get("delta_accuracy"),
                "delta_macro_f1": row.get("delta_macro_f1"),
                "delta_accuracy_ci95": unc.get("delta_accuracy_ci95"),
                "delta_macro_f1_ci95": unc.get("delta_macro_f1_ci95"),
                "effect": classify_delta(row),
                "note": row.get("note"),
                "selector": row.get("selector"),
            }
        )
    return rows


def recommendation_rows(recommendation: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for row in recommendation.get("datasets", []):
        best = row.get("best") or {}
        rows.append(
            {
                "dataset": row["dataset"],
                "target_met": row.get("target_met"),
                "best_accuracy": best.get("accuracy"),
                "best_macro_f1": best.get("macro_f1"),
                "recommendation": row.get("recommendation"),
            }
        )
    return rows


def paper_positioning(claims: List[Dict[str, Any]], ablations: List[Dict[str, Any]], framework: Dict[str, Any]) -> Dict[str, Any]:
    strong = [row["dataset"] for row in claims if row["claim_strength"] == "strong"]
    mixed = [row["dataset"] for row in claims if row["claim_strength"] == "point_pass_ci_mixed"]
    evidence_only = [row["dataset"] for row in claims if row["claim_strength"] == "evidence_only"]
    harmful = [row for row in ablations if row["effect"] == "harmful"]
    neutral = [row for row in ablations if row["effect"] == "uncertain_or_neutral"]
    risks = []
    if mixed:
        risks.append(
            "Some datasets pass point targets but not bootstrap lower-bound targets; avoid overclaiming statistical dominance."
        )
    if evidence_only:
        risks.append(
            "Some datasets are evidence-only because the test split is too small or lacks a paper target; present them as framework-transfer evidence."
        )
    if not framework.get("consistent"):
        risks.append("Framework consistency audit failed; do not claim a single unified framework until this is fixed.")
    recommended_claims = [
        "The method should be framed as a unified candidate-expert traffic classification framework with validation-gated safety controls.",
        "The strongest performance claim is supported on datasets whose point estimates and bootstrap lower bounds both pass target gates.",
        "Dataset-specific behavior should be described as automatic expert activation, gating, or identity fallback inside the same module family.",
    ]
    risk_controls = [
        "Use bootstrap and target-shift guards to prevent validation-favorable but target-unstable experts from overriding the base model.",
        "Report harmful expert candidates as negative ablations instead of hiding them; they motivate the gated-selector design.",
        "Separate strong performance claims from cross-dataset framework evidence when confidence intervals are wide.",
    ]
    next_experiments = [
        "Run fresh A800 Stage-8 paired-view experiments with flow-aware embeddings instead of old-embedding paired probes.",
        "Add larger or repeated USTC splits before making strong USTC performance claims.",
        "For VPN, prioritize representation learning improvements because probability-level expert switching is already gated off by target-shift evidence.",
    ]
    return {
        "strong_claim_datasets": strong,
        "point_pass_ci_mixed_datasets": mixed,
        "evidence_only_datasets": evidence_only,
        "harmful_ablation_count": len(harmful),
        "neutral_or_uncertain_ablation_count": len(neutral),
        "recommended_claims": recommended_claims,
        "risk_controls": risk_controls,
        "reviewer_risks": risks,
        "next_experiments": next_experiments,
    }


def render_markdown(pack: Dict[str, Any]) -> str:
    lines = [
        "# Paper Evidence Pack",
        "",
        f"Framework consistency: `{pack['framework_consistency'].get('consistent')}`",
        "",
        "## Claims",
        "",
        "| Dataset | Acc | Macro-F1 | Target | Point Gate | CI Gate | Claim | Acc 95% CI | Macro-F1 95% CI |",
        "|---|---:|---:|---|---|---|---|---:|---:|",
    ]
    for row in pack["claims"]:
        target = "-"
        if row["target_accuracy"] is not None:
            target = f"{fmt(row['target_accuracy'])}/{fmt(row['target_macro_f1'])}"
        lines.append(
            "| {dataset} | {acc} | {f1} | {target} | {point} | {ci_gate} | {claim} | {acc_ci} | {f1_ci} |".format(
                dataset=row["dataset"],
                acc=fmt(row["accuracy"]),
                f1=fmt(row["macro_f1"]),
                target=target,
                point=row["point_target_met"],
                ci_gate=row["ci_target_met"],
                claim=row["claim_strength"],
                acc_ci=format_ci(row["accuracy_ci95"]),
                f1_ci=format_ci(row["macro_f1_ci95"]),
            )
        )
    lines += [
        "",
        "## Unified Module Usage",
        "",
        "| Dataset | Module family evidence | Selector decision |",
        "|---|---|---|",
    ]
    for row in pack["claims"]:
        lines.append(
            "| {dataset} | {modules} | {selector} |".format(
                dataset=row["dataset"],
                modules=format_module_usage(row.get("module_usage")),
                selector=format_selector(row.get("selector")),
            )
        )
    slot_claims = [row for row in pack["claims"] if row.get("selector_slot_summary")]
    if slot_claims:
        lines += [
            "",
            "## Unified Expert Slots",
            "",
            "| Dataset | Mode | Slots | Provided | Identity-from-base | Extra |",
            "|---|---|---|---|---|---|",
        ]
        for row in slot_claims:
            summary = row["selector_slot_summary"]
            lines.append(
                "| {dataset} | {mode} | {slots} | {provided} | {identity} | {extra} |".format(
                    dataset=row["dataset"],
                    mode=summary.get("mode", "-"),
                    slots=", ".join(summary.get("slots") or []),
                    provided=", ".join(summary.get("provided") or []),
                    identity=", ".join(summary.get("identity_from_base") or []),
                    extra=", ".join(summary.get("extra_provided") or []),
                )
            )
    gated_claims = [row for row in pack["claims"] if row.get("multi_view_gate")]
    if gated_claims:
        lines += [
            "",
            "## Trainable Multi-View Gates",
            "",
            "| Dataset | Dominant branch | Effective branches | Norm. entropy | Mean weights: mean/max/std/attention |",
            "|---|---|---:|---:|---|",
        ]
        for row in gated_claims:
            gate = row["multi_view_gate"]
            branches = gate.get("branches") or []
            means = gate.get("mean") or []
            weight_by_branch = {branch: value for branch, value in zip(branches, means)}
            ordered = [weight_by_branch.get(branch) for branch in ["mean", "max", "std", "attention"]]
            weights = "/".join(fmt(value) for value in ordered)
            lines.append(
                "| {dataset} | {branch} ({weight}) | {eff} | {entropy} | {weights} |".format(
                    dataset=row["dataset"],
                    branch=gate.get("dominant_branch", "-"),
                    weight=fmt(gate.get("dominant_weight")),
                    eff=fmt(gate.get("effective_branches_mean")),
                    entropy=fmt(gate.get("normalized_entropy_mean")),
                    weights=weights,
                )
            )
    lines += [
        "",
        "## Ablation Effects",
        "",
        "| Dataset | Stage | Delta Acc | Delta F1 | Delta Acc 95% CI | Delta F1 95% CI | Effect |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in pack["ablations"]:
        lines.append(
            "| {dataset} | {stage} | {dacc} | {df1} | {dacc_ci} | {df1_ci} | {effect} |".format(
                dataset=row["dataset"],
                stage=row["stage"],
                dacc=fmt(row["delta_accuracy"], signed=True),
                df1=fmt(row["delta_macro_f1"], signed=True),
                dacc_ci=format_ci(row["delta_accuracy_ci95"], signed=True),
                df1_ci=format_ci(row["delta_macro_f1_ci95"], signed=True),
                effect=row["effect"],
            )
        )
    lines += ["", "## Next-Step Recommendations", ""]
    for row in pack["recommendations"]:
        lines.append(f"- {row['dataset']}: {row['recommendation']}")
    positioning = pack.get("paper_positioning") or {}
    lines += ["", "## Paper Positioning", "", "Recommended claims:"]
    for item in positioning.get("recommended_claims", []):
        lines.append(f"- {item}")
    lines += ["", "Risk controls:"]
    for item in positioning.get("risk_controls", []):
        lines.append(f"- {item}")
    lines += ["", "Reviewer risks:"]
    for item in positioning.get("reviewer_risks", []):
        lines.append(f"- {item}")
    lines += ["", "Next experiments:"]
    for item in positioning.get("next_experiments", []):
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a compact paper evidence pack from framework, ablation, and recommendation reports.")
    ap.add_argument("--framework_json", default="reasoningDataset/paper_framework_report.json")
    ap.add_argument("--ablation_json", default="reasoningDataset/paper_ablation_report.json")
    ap.add_argument("--recommendation_json", default="reasoningDataset/next_experiment_recommendation.json")
    ap.add_argument("--output_json", default="reasoningDataset/paper_evidence_pack.json")
    ap.add_argument("--output_md", default="reasoningDataset/paper_evidence_pack.md")
    args = ap.parse_args()

    framework = load_json(args.framework_json)
    ablation = load_json(args.ablation_json)
    recommendation = load_json(args.recommendation_json)
    claims = claim_rows(framework)
    ablations = ablation_rows(ablation)
    pack = {
        "framework_consistency": framework.get("framework_consistency") or {},
        "claims": claims,
        "ablations": ablations,
        "recommendations": recommendation_rows(recommendation),
        "paper_positioning": paper_positioning(claims, ablations, framework.get("framework_consistency") or {}),
    }
    md = render_markdown(pack)
    print(md)
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(pack, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.output_md:
        Path(args.output_md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_md).write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()
