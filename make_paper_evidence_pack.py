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
                "calibration": row.get("calibration"),
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
        paper_safe = row.get("paper_safe") or {}
        delta = row.get("raw_vs_paper_safe_delta") or {}
        rows.append(
            {
                "dataset": row["dataset"],
                "target_met": row.get("target_met"),
                "raw_best_accuracy": best.get("accuracy"),
                "raw_best_macro_f1": best.get("macro_f1"),
                "raw_best_path": best.get("path"),
                "paper_safe_accuracy": paper_safe.get("accuracy"),
                "paper_safe_macro_f1": paper_safe.get("macro_f1"),
                "paper_safe_path": paper_safe.get("path"),
                "raw_minus_paper_safe_accuracy": delta.get("delta_accuracy"),
                "raw_minus_paper_safe_macro_f1": delta.get("delta_macro_f1"),
                "raw_and_paper_safe_same_path": delta.get("same_path"),
                "recommendation": row.get("recommendation"),
            }
        )
    return rows


def content_unique_default_path(dataset: str) -> str:
    return f"reasoningDataset/{dataset}/test_crossfold_consensus_auto_confidence_content_unique.json"


def content_unique_rows(claims: List[Dict[str, Any]], overrides: Dict[str, str] | None = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    overrides = overrides or {}
    for claim in claims:
        dataset = str(claim["dataset"])
        path = overrides.get(dataset, content_unique_default_path(dataset))
        if not Path(path).exists():
            continue
        payload = load_json(path)
        metrics = payload.get("metrics") or {}
        original = metrics.get("original_flow_level") or {}
        unique = metrics.get("content_unique_flow_level") or {}
        audit = payload.get("audit") or {}
        bootstrap = payload.get("content_unique_bootstrap") or {}
        group_bootstrap = payload.get("content_group_bootstrap") or {}
        rows.append(
            {
                "dataset": dataset,
                "path": path,
                "original_accuracy": original.get("accuracy"),
                "original_macro_f1": original.get("macro_f1"),
                "content_unique_accuracy": unique.get("accuracy"),
                "content_unique_macro_f1": unique.get("macro_f1"),
                "delta_accuracy": None
                if unique.get("accuracy") is None or original.get("accuracy") is None
                else float(unique["accuracy"]) - float(original["accuracy"]),
                "delta_macro_f1": None
                if unique.get("macro_f1") is None or original.get("macro_f1") is None
                else float(unique["macro_f1"]) - float(original["macro_f1"]),
                "input_flows": audit.get("input_flows"),
                "unique_content_flows": audit.get("unique_content_flows"),
                "duplicate_content_groups": audit.get("duplicate_content_groups"),
                "duplicate_rows_removed": audit.get("duplicate_rows_removed"),
                "accuracy_ci95": bootstrap.get("accuracy_95_ci"),
                "macro_f1_ci95": bootstrap.get("macro_f1_95_ci"),
                "bootstrap_samples": bootstrap.get("samples"),
                "content_group_accuracy_ci95": group_bootstrap.get("accuracy_95_ci"),
                "content_group_macro_f1_ci95": group_bootstrap.get("macro_f1_95_ci"),
                "content_group_bootstrap_samples": group_bootstrap.get("samples"),
                "content_group_count": group_bootstrap.get("num_groups"),
                "content_group_rows": group_bootstrap.get("num_rows"),
            }
        )
    return rows


def attach_content_robustness_to_claims(
    claims: List[Dict[str, Any]],
    content_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    by_dataset = {row["dataset"]: row for row in content_rows}
    enriched: List[Dict[str, Any]] = []
    for claim in claims:
        row = dict(claim)
        content = by_dataset.get(str(claim.get("dataset")))
        if content:
            target_acc = row.get("target_accuracy")
            target_f1 = row.get("target_macro_f1")
            group_acc_ci = content.get("content_group_accuracy_ci95")
            group_f1_ci = content.get("content_group_macro_f1_ci95")
            row.update(
                {
                    "content_unique_accuracy": content.get("content_unique_accuracy"),
                    "content_unique_macro_f1": content.get("content_unique_macro_f1"),
                    "content_group_accuracy_ci95": group_acc_ci,
                    "content_group_macro_f1_ci95": group_f1_ci,
                    "content_group_count": content.get("content_group_count"),
                    "content_group_rows": content.get("content_group_rows"),
                    "content_group_ci_target_met": None
                    if target_acc is None or target_f1 is None
                    else bool(
                        ci_lower(group_acc_ci) is not None
                        and ci_lower(group_f1_ci) is not None
                        and ci_lower(group_acc_ci) >= float(target_acc)
                        and ci_lower(group_f1_ci) >= float(target_f1)
                    ),
                }
            )
        enriched.append(row)
    return enriched


def paper_positioning(claims: List[Dict[str, Any]], ablations: List[Dict[str, Any]], framework: Dict[str, Any]) -> Dict[str, Any]:
    strong = [row["dataset"] for row in claims if row["claim_strength"] == "strong"]
    mixed = [row["dataset"] for row in claims if row["claim_strength"] == "point_pass_ci_mixed"]
    evidence_only = [row["dataset"] for row in claims if row["claim_strength"] == "evidence_only"]
    grouped_mixed = [
        row["dataset"]
        for row in claims
        if row.get("point_target_met") is True and row.get("content_group_ci_target_met") is False
    ]
    harmful = [row for row in ablations if row["effect"] == "harmful"]
    neutral = [row for row in ablations if row["effect"] == "uncertain_or_neutral"]
    risks = []
    if mixed:
        risks.append(
            "Some datasets pass point targets but not bootstrap lower-bound targets; avoid overclaiming statistical dominance."
        )
    if grouped_mixed:
        risks.append(
            "Some datasets pass point targets but not content-grouped bootstrap lower-bound targets; present them as point-estimate gains until group-level stability improves."
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
        "Separate strong performance claims from exploratory evidence when confidence intervals are wide.",
    ]
    next_experiments = [
        "Do not increase Tower-2 paired IP/port-randomization weight; fresh flow-aware paired-view probes are negative.",
        "Use content-grouped bootstrap lower bounds as the VPN promotion gate, and prioritize coverage-audited consensus distillation for group-level stability.",
        "Distill the cross-fold consensus back into trainable graph/seq models; keep native structural pretraining as a negative ablation until its objective or gate is redesigned.",
        "Keep per-packet-split datasets outside the flow-level main table unless a leakage-free per-flow split is released.",
    ]
    return {
        "strong_claim_datasets": strong,
        "point_pass_ci_mixed_datasets": mixed,
        "content_group_ci_mixed_datasets": grouped_mixed,
        "evidence_only_datasets": evidence_only,
        "harmful_ablation_count": len(harmful),
        "neutral_or_uncertain_ablation_count": len(neutral),
        "recommended_claims": recommended_claims,
        "risk_controls": risk_controls,
        "reviewer_risks": risks,
        "next_experiments": next_experiments,
    }


def paper_audit_gates(unified_audit: Dict[str, Any], defaults_audit: Dict[str, Any]) -> Dict[str, Any]:
    framework = unified_audit.get("framework") or {}
    defaults_framework = defaults_audit.get("unified_framework") or {}
    flow_rows = defaults_audit.get("flow_datasets") or defaults_audit.get("datasets") or []
    packet_rows = defaults_audit.get("packet_datasets") or []

    def authoritative(key: str) -> Any:
        # Zero and an empty scope are valid strict-audit results. Fall back only
        # for legacy audits that do not expose the field at all.
        return framework[key] if key in framework else defaults_framework.get(key)

    flow_manifest_passes = authoritative("paper_unified_flow_manifest_passes")
    packet_manifest_passes = authoritative("paper_unified_packet_manifest_passes")
    gates = {
        "unified_framework_status": unified_audit.get("status"),
        "paper_defaults_ok": defaults_audit.get("ok"),
        "shared_core_match": defaults_framework.get("shared_core_matches_defaults"),
        "paper_unified_flow_manifest_passes": flow_manifest_passes,
        "paper_unified_packet_manifest_passes": packet_manifest_passes,
        "flow_scope": authoritative("flow_scope"),
        "packet_scope": authoritative("packet_scope"),
        "flow_default_count": len(flow_rows),
        "packet_default_count": len(packet_rows),
        "flow_defaults_pass": bool(flow_rows) and all(row.get("ok") is True for row in flow_rows),
        "packet_defaults_pass": bool(packet_rows)
        and all(row.get("publication_status") == "pass" for row in packet_rows),
    }
    gates["strict_reproduction_complete"] = bool(
        gates["unified_framework_status"] == "pass"
        and gates["paper_defaults_ok"] is True
        and int(flow_manifest_passes or 0) > 0
        and int(packet_manifest_passes or 0) > 0
        and gates["flow_defaults_pass"]
        and gates["packet_defaults_pass"]
    )
    return gates


def render_markdown(pack: Dict[str, Any]) -> str:
    lines = [
        "# Paper Evidence Pack",
        "",
        f"Framework consistency: `{pack['framework_consistency'].get('consistent')}`",
        "",
        "## Claims",
        "",
        "| Dataset | Acc | Macro-F1 | Target | Point Gate | CI Gate | Group CI Gate | Claim | Acc 95% CI | Macro-F1 95% CI | Grouped Acc/F1 95% CI |",
        "|---|---:|---:|---|---|---|---|---|---:|---:|---:|",
    ]
    for row in pack["claims"]:
        target = "-"
        if row["target_accuracy"] is not None:
            target = f"{fmt(row['target_accuracy'])}/{fmt(row['target_macro_f1'])}"
        lines.append(
            "| {dataset} | {acc} | {f1} | {target} | {point} | {ci_gate} | {group_ci_gate} | {claim} | {acc_ci} | {f1_ci} | {group_ci} |".format(
                dataset=row["dataset"],
                acc=fmt(row["accuracy"]),
                f1=fmt(row["macro_f1"]),
                target=target,
                point=row["point_target_met"],
                ci_gate=row["ci_target_met"],
                group_ci_gate=row.get("content_group_ci_target_met", "-"),
                claim=row["claim_strength"],
                acc_ci=format_ci(row["accuracy_ci95"]),
                f1_ci=format_ci(row["macro_f1_ci95"]),
                group_ci="{}/{}".format(
                    format_ci(row.get("content_group_accuracy_ci95")),
                    format_ci(row.get("content_group_macro_f1_ci95")),
                ),
            )
        )
    gates = pack.get("paper_audit_gates") or {}
    if gates:
        lines += [
            "",
            "## Paper Audit Gates",
            "",
            "| Unified Audit | Defaults Audit | Shared Core | Flow Manifests | Packet Manifests | Flow Defaults | Packet Defaults | Strict Reproduction |",
            "|---|---|---|---:|---:|---:|---:|---|",
            "| {unified} | {defaults} | {shared} | {flow_manifest} | {packet_manifest} | {flow_pass}/{flow_count} | {packet_pass}/{packet_count} | {strict} |".format(
                unified=gates.get("unified_framework_status"),
                defaults=gates.get("paper_defaults_ok"),
                shared=gates.get("shared_core_match"),
                flow_manifest=gates.get("paper_unified_flow_manifest_passes", "-"),
                packet_manifest=gates.get("paper_unified_packet_manifest_passes", "-"),
                flow_pass=gates.get("flow_defaults_pass"),
                flow_count=gates.get("flow_default_count", "-"),
                packet_pass=gates.get("packet_defaults_pass"),
                packet_count=gates.get("packet_default_count", "-"),
                strict=gates.get("strict_reproduction_complete"),
            ),
            "",
            "Flow scope: `{}`".format(", ".join(gates.get("flow_scope") or [])),
            "",
            "Packet scope: `{}`".format(", ".join(gates.get("packet_scope") or [])),
        ]
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
    calibration_claims = [row for row in pack["claims"] if row.get("calibration")]
    if calibration_claims:
        lines += [
            "",
            "## Flow Calibration",
            "",
            "| Dataset | ECE | NLL | Brier | Avg Confidence | Samples |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for row in pack["claims"]:
            cal = row.get("calibration") or {}
            lines.append(
                "| {dataset} | {ece} | {nll} | {brier} | {conf} | {samples} |".format(
                    dataset=row["dataset"],
                    ece=fmt(cal.get("ece")),
                    nll=fmt(cal.get("nll")),
                    brier=fmt(cal.get("brier")),
                    conf=fmt(cal.get("avg_confidence")),
                    samples=cal.get("num_samples", "-"),
                )
            )
    content_rows = pack.get("content_unique") or []
    if content_rows:
        lines += [
            "",
            "## Content-Unique Robustness",
            "",
            "| Dataset | Original Acc/F1 | Content-Unique Acc/F1 | Delta Acc/F1 | Unique/Original Flows | Duplicate Groups | Unique Acc/F1 95% CI | Grouped Acc/F1 95% CI |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in content_rows:
            lines.append(
                "| {dataset} | {orig_acc}/{orig_f1} | {uniq_acc}/{uniq_f1} | {dacc}/{df1} | {unique}/{total} | {dups} | {acc_ci}/{f1_ci} | {group_acc_ci}/{group_f1_ci} |".format(
                    dataset=row["dataset"],
                    orig_acc=fmt(row.get("original_accuracy")),
                    orig_f1=fmt(row.get("original_macro_f1")),
                    uniq_acc=fmt(row.get("content_unique_accuracy")),
                    uniq_f1=fmt(row.get("content_unique_macro_f1")),
                    dacc=fmt(row.get("delta_accuracy"), signed=True),
                    df1=fmt(row.get("delta_macro_f1"), signed=True),
                    unique=row.get("unique_content_flows", "-"),
                    total=row.get("input_flows", "-"),
                    dups=row.get("duplicate_content_groups", "-"),
                    acc_ci=format_ci(row.get("accuracy_ci95")),
                    f1_ci=format_ci(row.get("macro_f1_ci95")),
                    group_acc_ci=format_ci(row.get("content_group_accuracy_ci95")),
                    group_f1_ci=format_ci(row.get("content_group_macro_f1_ci95")),
                )
            )
    lines += [
        "",
        "## Raw Best vs Paper-Safe Result",
        "",
        "| Dataset | Raw Best Acc | Raw Best F1 | Paper-Safe Acc | Paper-Safe F1 | Raw-Paper Acc | Raw-Paper F1 | Same Path |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in pack["recommendations"]:
        lines.append(
            "| {dataset} | {raw_acc} | {raw_f1} | {safe_acc} | {safe_f1} | {dacc} | {df1} | {same} |".format(
                dataset=row["dataset"],
                raw_acc=fmt(row.get("raw_best_accuracy")),
                raw_f1=fmt(row.get("raw_best_macro_f1")),
                safe_acc=fmt(row.get("paper_safe_accuracy")),
                safe_f1=fmt(row.get("paper_safe_macro_f1")),
                dacc=fmt(row.get("raw_minus_paper_safe_accuracy"), signed=True),
                df1=fmt(row.get("raw_minus_paper_safe_macro_f1"), signed=True),
                same=row.get("raw_and_paper_safe_same_path"),
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
    ap.add_argument("--unified_audit_json", default="reasoningDataset/unified_framework_audit.json")
    ap.add_argument("--defaults_audit_json", default="reasoningDataset/paper_framework_defaults_audit.json")
    ap.add_argument(
        "--content_unique_result",
        nargs=2,
        action="append",
        default=[],
        metavar=("DATASET", "JSON"),
        help="Optional DATASET JSON override for content-unique evaluation results.",
    )
    ap.add_argument("--output_json", default="reasoningDataset/paper_evidence_pack.json")
    ap.add_argument("--output_md", default="reasoningDataset/paper_evidence_pack.md")
    args = ap.parse_args()

    framework = load_json(args.framework_json)
    ablation = load_json(args.ablation_json)
    recommendation = load_json(args.recommendation_json)
    unified_audit = load_json(args.unified_audit_json)
    defaults_audit = load_json(args.defaults_audit_json)
    claims = claim_rows(framework)
    ablations = ablation_rows(ablation)
    content_rows = content_unique_rows(claims, dict(args.content_unique_result))
    claims = attach_content_robustness_to_claims(claims, content_rows)
    pack = {
        "framework_consistency": framework.get("framework_consistency") or {},
        "paper_audit_gates": paper_audit_gates(unified_audit, defaults_audit),
        "claims": claims,
        "ablations": ablations,
        "content_unique": content_rows,
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
