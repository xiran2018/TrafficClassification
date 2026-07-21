#!/usr/bin/env python3
"""Build a paper-facing method card for the unified traffic framework.

The evidence pack proves metrics; this card turns the same machine-readable
evidence into a reviewer-oriented method narrative. It intentionally separates
paper-ready claims from risks so the project does not drift back into
dataset-specific best-result storytelling.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from unified_framework_spec import (
    FLOW_LEVEL_RESULTS,
    MODEL_SHARED_CORE_MODULES,
    PACKET_LEVEL_RESULTS,
    SHARED_PROTOCOL_GUARDS,
)


PROBLEM_TO_MODULES = [
    {
        "problem": "Shortcut learning from endpoints, ports, and split artifacts",
        "modules": ["per_flow_split_guard", "field_aware_header_intervention"],
        "paper_angle": "Treat header intervention and split provenance as part of the model protocol, not cleanup.",
    },
    {
        "problem": "Packet-level evidence does not automatically transfer to flow-level decisions",
        "modules": [
            "label_free_protocol_content_pretraining",
            "semantic_tower1_channel",
            "current_packet_structural_encoder",
            "packet_to_window_flow_aggregator",
        ],
        "paper_angle": "Use a shared packet representation contract and a task-specific flow aggregator.",
    },
    {
        "problem": "Semantic, packet-content, and structural evidence have split-dependent reliability",
        "modules": [
            "shared_intervention_view_fusion",
            "bounded_tri_channel_router",
        ],
        "paper_angle": "Route every sample through the same three channels and learn bounded, data-dependent reliability weights.",
    },
    {
        "problem": "Validation-selected recipes and duplicate content can inflate evidence",
        "modules": ["content_group_empirical_risk", "validation_only_selection", "fixed_cross_fold_consensus"],
        "paper_angle": "Freeze one cross-dataset recipe on validation and report a fixed equal-weight consensus.",
    },
]


CORE_CONTRIBUTIONS = [
    {
        "name": "Field-aware shortcut intervention",
        "claim": (
            "A paired factual/intervened view masks endpoint and port fields under the same policy for both tasks, "
            "so shortcut resistance is learned inside the shared representation rather than added at prediction time."
        ),
    },
    {
        "name": "Packet-to-flow dual representation contract",
        "claim": (
            "Packet-level and flow-level classification reuse the same label-free protocol-content, Tower1 semantic, "
            "13-dimensional packet-local structural encoders and bounded router under a strict current-packet input "
            "policy; supervised parameters are re-trained from each task's own training split, while sequence position "
            "and window context enter only through the Flow aggregator. The shared core may retain a direction cue "
            "inferred from the current packet alone, but never previous-packet IAT or whole-flow server-role inference."
        ),
    },
    {
        "name": "Counterfactual semantic-structural routing",
        "claim": (
            "Factual and header-intervened semantic views are fused before one semantic-anchored bounded router combines "
            "semantic, protocol-content, and packet-local structural evidence for every dataset and both tasks."
        ),
    },
    {
        "name": "Validation-safe publication protocol",
        "claim": (
            "One validation-frozen cross-dataset recipe, content-group empirical risk, fixed equal-weight log-mean, "
            "flow-cluster bootstrap, executed-policy/checkpoint audits, and reporting-only train-signature novelty "
            "strata distinguish publishable evidence from exploratory probes."
        ),
    },
]

DEFAULT_CANDIDATE_RANKINGS = [
    "reasoningDataset/vpn-app/content_group_candidate_ranking.json",
    "reasoningDataset/tls-120/content_group_candidate_ranking_crossfold.json",
]

STRICT_DATASETS = ("vpn-app", "tls-120")


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return str(value)
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def compact_bool(value: Any) -> bool:
    return bool(value is True or value == "pass")


def flow_claim_rows(evidence: Dict[str, Any]) -> List[Dict[str, Any]]:
    return list(evidence.get("claims") or [])


def packet_claim_rows(defaults_audit: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        row for row in defaults_audit.get("packet_datasets") or []
        if row.get("dataset") in STRICT_DATASETS
    ]


def supplementary_packet_claim_rows(
    defaults_audit: Dict[str, Any],
) -> List[Dict[str, Any]]:
    return [
        row for row in defaults_audit.get("packet_datasets") or []
        if row.get("dataset") not in STRICT_DATASETS
    ]


def ablation_summary(evidence: Dict[str, Any]) -> Dict[str, Any]:
    rows = list(evidence.get("ablations") or [])
    helpful = [row for row in rows if row.get("effect") == "helpful"]
    harmful = [row for row in rows if row.get("effect") == "harmful"]
    neutral = [row for row in rows if row.get("effect") not in {"helpful", "harmful"}]
    return {
        "num_ablations": len(rows),
        "helpful_count": len(helpful),
        "harmful_count": len(harmful),
        "neutral_or_uncertain_count": len(neutral),
        "helpful": helpful[:5],
        "harmful": harmful[:5],
    }


def candidate_ranking_summary(paths: List[str]) -> List[Dict[str, Any]]:
    rows = []
    for path in paths:
        p = Path(path)
        if not p.exists():
            continue
        try:
            data = load_json(str(p))
        except Exception:
            continue
        ranked = list(data.get("rows") or [])
        best = next((row for row in ranked if row.get("status") == "ok"), None)
        if not best:
            continue
        acc_ci = best.get("content_group_accuracy_ci95")
        f1_ci = best.get("content_group_macro_f1_ci95")
        rows.append(
            {
                "dataset": data.get("dataset"),
                "path": str(p),
                "num_candidates": data.get("num_candidates"),
                "num_ok": data.get("num_ok"),
                "best_path": best.get("path"),
                "best_accuracy": best.get("accuracy"),
                "best_macro_f1": best.get("macro_f1"),
                "best_content_group_accuracy_lower": acc_ci[0] if isinstance(acc_ci, list) and acc_ci else None,
                "best_content_group_macro_f1_lower": f1_ci[0] if isinstance(f1_ci, list) and f1_ci else None,
                "best_content_group_target_met": best.get("content_group_target_met"),
            }
        )
    return rows


def strict_shared_core_publication_status() -> Dict[str, Any]:
    """Audit the four canonical VPN/TLS task results for exact-v2 provenance."""
    rows: List[Dict[str, Any]] = []
    fingerprints = set()
    for task, specs in (
        ("packet-level", PACKET_LEVEL_RESULTS),
        ("flow-level", FLOW_LEVEL_RESULTS),
    ):
        for dataset in STRICT_DATASETS:
            path = Path(specs[dataset].path)
            provenance: Dict[str, Any] = {}
            error = None
            if path.is_file():
                try:
                    provenance = load_json(str(path)).get("publication_provenance") or {}
                except Exception as exc:
                    error = str(exc)
            fingerprint = provenance.get("shared_core_config_sha256")
            novelty_path = Path(str(provenance.get("session_novelty") or ""))
            novelty_hash = provenance.get("session_novelty_sha256")
            novelty_verified = bool(
                novelty_path.is_file()
                and novelty_hash
                and sha256_file(novelty_path) == novelty_hash
            )
            if fingerprint:
                fingerprints.add(str(fingerprint))
            passed = bool(
                error is None
                and provenance.get("status") == "strict_shared_core_v2"
                and provenance.get("fixed_consensus") == "equal_log_mean_three_folds"
                and provenance.get("runtime_mechanism_evidence_required") is True
                and provenance.get("flow_native_extraction_evidence_required") is True
                and fingerprint
                and len(provenance.get("audit_paths") or []) == 3
                and novelty_verified
            )
            rows.append(
                {
                    "dataset": dataset,
                    "task": task,
                    "path": str(path),
                    "exists": path.is_file(),
                    "status": provenance.get("status"),
                    "shared_core_config_sha256": fingerprint,
                    "session_novelty": provenance.get("session_novelty"),
                    "session_novelty_sha256": provenance.get(
                        "session_novelty_sha256"
                    ),
                    "session_novelty_verified": novelty_verified,
                    "passed": passed,
                    "error": error,
                }
            )
    all_pass = len(rows) == 4 and all(row["passed"] for row in rows)
    session_novelty_ready = len(rows) == 4 and all(
        row["session_novelty_verified"] for row in rows
    )
    same_fingerprint = all_pass and len(fingerprints) == 1
    return {
        "required_schema": "strict_shared_core_v2",
        "required_datasets": list(STRICT_DATASETS),
        "required_tasks": ["packet-level", "flow-level"],
        "canonical_results": rows,
        "all_results_have_strict_provenance": all_pass,
        "all_session_novelty_evidence_verified": session_novelty_ready,
        "single_shared_core_config": same_fingerprint,
        "shared_core_config_sha256": next(iter(fingerprints)) if same_fingerprint else None,
        "ready": bool(all_pass and same_fingerprint),
    }


def readiness(
    evidence: Dict[str, Any],
    unified_audit: Dict[str, Any],
    defaults_audit: Dict[str, Any],
    strict_status: Dict[str, Any],
    sweet_comparison: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    claims = flow_claim_rows(evidence)
    gates = evidence.get("paper_audit_gates") or {}
    flow_point_ready = bool(claims) and all(row.get("point_target_met") is True for row in claims)
    flow_ci_ready = bool(claims) and all(row.get("ci_target_met") is True for row in claims)
    flow_group_ci_ready = bool(claims) and all(row.get("content_group_ci_target_met") is True for row in claims)
    strict_packet_rows = packet_claim_rows(defaults_audit)
    packet_ready = bool(strict_packet_rows) and all(
        row.get("publication_status") == "pass" for row in defaults_audit.get("packet_datasets") or []
        if row.get("dataset") in STRICT_DATASETS
    )
    unified_ready = unified_audit.get("status") == "pass"
    exact_common_reference_ready = (
        (unified_audit.get("exact_shared_core_v2") or {}).get("status") == "pass"
    )
    unified_method_v2_ready = (
        (unified_audit.get("unified_method_v2") or {}).get("status") == "pass"
    )
    defaults_ready = defaults_audit.get("ok") is True
    legacy_protocol_ready = unified_ready and defaults_ready
    strict_ready = strict_status.get("ready") is True
    ccf_a_risks = []
    if not strict_ready:
        ccf_a_risks.append(
            "VPN/TLS packet/flow canonical results do not yet share complete strict_shared_core_v2 publication provenance."
        )
    sweet_summary = (sweet_comparison or {}).get("summary") or {}
    sweet_end_to_end_ready = sweet_summary.get(
        "all_results_exceed_end_to_end_accuracy_and_macro_f1"
    )
    if sweet_end_to_end_ready is False:
        ccf_a_risks.append(
            "At least one current VPN/TLS task result does not exceed the protocol-matched SWEET end-to-end accuracy and macro-F1 pair."
        )
    if not flow_group_ci_ready:
        ccf_a_risks.append("At least one flow dataset misses the content-grouped bootstrap lower-bound gate.")
    if not flow_ci_ready:
        ccf_a_risks.append("At least one flow dataset passes point targets but not the ordinary bootstrap lower-bound gate.")
    if not legacy_protocol_ready:
        ccf_a_risks.append("The legacy unified/defaults protocol audit is not fully passing.")
    if not packet_ready:
        ccf_a_risks.append("At least one packet-level default result is not publication-ready.")
    if not ccf_a_risks and strict_ready:
        recommended_claim = (
            "Unified shortcut-resistant packet-to-flow framework with point, CI, "
            "group-CI, and provenance gates satisfied."
        )
    elif packet_ready and strict_ready:
        recommended_claim = (
            "Unified shortcut-resistant packet-to-flow framework with publication-ready "
            "packet-level evidence and flow-level point-target gains; CI/group-CI claims "
            "remain limited to the datasets whose gates pass."
        )
    else:
        recommended_claim = (
            "Candidate unified shortcut-resistant packet-to-flow framework; historical "
            "packet-level scores and flow-level point gains remain non-headline evidence "
            "until strict_shared_core_v2 cross-task provenance, fixed-consensus, and CI gates pass."
        )
    return {
        "legacy_protocol_audit_ready": legacy_protocol_ready,
        "exact_common_reference_v2_ready": exact_common_reference_ready,
        "unified_method_v2_ready": unified_method_v2_ready,
        "strict_shared_core_v2_ready": strict_ready,
        "session_novelty_evidence_ready": strict_status.get(
            "all_session_novelty_evidence_verified"
        )
        is True,
        "sweet_end_to_end_all_tasks_exceeded": sweet_end_to_end_ready,
        "paper_unified_profile_ready": legacy_protocol_ready and strict_ready,
        "flow_point_targets_ready": flow_point_ready,
        "flow_ci_targets_ready": flow_ci_ready,
        "flow_content_group_ci_targets_ready": flow_group_ci_ready,
        "packet_publication_defaults_ready": packet_ready,
        "audit_gates": gates,
        "ccf_a_risk_level": "moderate" if not ccf_a_risks else "high",
        "ccf_a_risks": ccf_a_risks,
        "recommended_main_claim": recommended_claim,
    }


def build_card(
    evidence: Dict[str, Any],
    unified_audit: Dict[str, Any],
    defaults_audit: Dict[str, Any],
    *,
    candidate_rankings: List[Dict[str, Any]] | None = None,
    strict_status: Dict[str, Any] | None = None,
    sweet_comparison: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    framework = unified_audit.get("framework") or {}
    strict_status = strict_status or {
        "ready": False,
        "canonical_results": [],
        "single_shared_core_config": False,
        "shared_core_config_sha256": None,
    }
    return {
        "method_name": "Unified Shortcut-Resistant Packet-to-Flow Framework",
        "framework_profile": (defaults_audit.get("defaults") or {}).get("framework_profile", "paper_unified"),
        "scope": {
            "flow": list(STRICT_DATASETS),
            "packet": list(STRICT_DATASETS),
        },
        "supplementary_packet_only_scope": [
            dataset for dataset in (framework.get("packet_scope") or [])
            if dataset not in STRICT_DATASETS
        ],
        "architecture_rule": (
            "Every dataset and both tasks execute all shared representation modules; "
            "only learned parameters and gates differ. Supervised Packet-module weights "
            "are trained independently from each task's own split and are never transferred "
            "across tasks. Flow adds packet_to_flow_proj and aggregation after the shared "
            "packet representation."
        ),
        "shared_core_modules": framework.get("shared_core_modules") or [],
        "model_shared_core_modules": framework.get("model_shared_core_modules")
        or list(MODEL_SHARED_CORE_MODULES),
        "shared_protocol_guards": framework.get("shared_protocol_guards")
        or list(SHARED_PROTOCOL_GUARDS),
        "flow_modules": framework.get("flow_modules") or [],
        "packet_modules": framework.get("packet_modules") or [],
        "problems": PROBLEM_TO_MODULES,
        "contributions": CORE_CONTRIBUTIONS,
        "flow_claims": flow_claim_rows(evidence),
        "packet_claims": packet_claim_rows(defaults_audit),
        "supplementary_packet_claims": supplementary_packet_claim_rows(defaults_audit),
        "candidate_rankings": candidate_rankings or [],
        "ablation_summary": ablation_summary(evidence),
        "strict_shared_core_v2": strict_status,
        "sweet_protocol_comparison": sweet_comparison or {},
        "readiness": readiness(
            evidence,
            unified_audit,
            defaults_audit,
            strict_status,
            sweet_comparison,
        ),
    }


def md_list(items: Iterable[str]) -> str:
    return ", ".join(f"`{item}`" for item in items) if items else "-"


def render_markdown(card: Dict[str, Any]) -> str:
    ready = card.get("readiness") or {}
    lines = [
        "# Paper Method Card",
        "",
        f"Method: **{card['method_name']}**",
        "",
        f"Framework profile: `{card['framework_profile']}`",
        "",
        f"Recommended main claim: {ready.get('recommended_main_claim', '-')}",
        "",
        f"Architecture rule: {card.get('architecture_rule', '-')}",
        "",
        "## Scope",
        "",
        f"- Flow-level datasets: {md_list(card.get('scope', {}).get('flow') or [])}",
        f"- Packet-level datasets: {md_list(card.get('scope', {}).get('packet') or [])}",
        f"- Supplementary packet-only datasets (not cross-task evidence): "
        f"{md_list(card.get('supplementary_packet_only_scope') or [])}",
        "",
        "## Shared Contract",
        "",
        f"Shared representation modules: {md_list(card.get('model_shared_core_modules') or [])}",
        "",
        f"Shared training/evaluation protocol: {md_list(card.get('shared_protocol_guards') or [])}",
        "",
        f"Flow-only modules: {md_list([m for m in card.get('flow_modules') or [] if m not in set(card.get('shared_core_modules') or [])])}",
        "",
        f"Packet-only modules: {md_list([m for m in card.get('packet_modules') or [] if m not in set(card.get('shared_core_modules') or [])])}",
        "",
        "## Problems To Modules",
        "",
        "| Reviewer-facing problem | Unified modules | Paper angle |",
        "|---|---|---|",
    ]
    for row in card.get("problems") or []:
        lines.append(
            "| {problem} | {modules} | {angle} |".format(
                problem=row["problem"],
                modules=md_list(row.get("modules") or []),
                angle=row["paper_angle"],
            )
        )
    lines += [
        "",
        "## Contributions",
        "",
    ]
    for idx, row in enumerate(card.get("contributions") or [], 1):
        lines.append(f"{idx}. **{row['name']}**: {row['claim']}")
    lines += [
        "",
        "## Flow Evidence",
        "",
        "| Dataset | Acc | Macro-F1 | Target | Point | CI | Group CI | Claim |",
        "|---|---:|---:|---|---|---|---|---|",
    ]
    for row in card.get("flow_claims") or []:
        target = f"{fmt(row.get('target_accuracy'))}/{fmt(row.get('target_macro_f1'))}"
        lines.append(
            "| {dataset} | {acc} | {f1} | {target} | {point} | {ci} | {group} | {claim} |".format(
                dataset=row.get("dataset"),
                acc=fmt(row.get("accuracy")),
                f1=fmt(row.get("macro_f1")),
                target=target,
                point=row.get("point_target_met"),
                ci=row.get("ci_target_met"),
                group=row.get("content_group_ci_target_met"),
                claim=row.get("claim_strength", "-"),
            )
        )
    lines += [
        "",
        "## Packet Evidence",
        "",
        "| Dataset | Acc | Macro-F1 | Publication | Path |",
        "|---|---:|---:|---|---|",
    ]
    for row in card.get("packet_claims") or []:
        lines.append(
            "| {dataset} | {acc} | {f1} | {status} | `{path}` |".format(
                dataset=row.get("dataset"),
                acc=fmt(row.get("accuracy")),
                f1=fmt(row.get("macro_f1")),
                status=row.get("publication_status"),
                path=row.get("path"),
            )
        )
    supplementary = card.get("supplementary_packet_claims") or []
    if supplementary:
        lines += [
            "",
            "## Supplementary Packet-Only Evidence",
            "",
            "These datasets do not establish the unified Packet-to-Flow claim because a protocol-matched Flow task is not in scope.",
            "",
            "| Dataset | Acc | Macro-F1 | Publication | Path |",
            "|---|---:|---:|---|---|",
        ]
        for row in supplementary:
            lines.append(
                "| {dataset} | {acc} | {f1} | {status} | `{path}` |".format(
                    dataset=row.get("dataset"),
                    acc=fmt(row.get("accuracy")),
                    f1=fmt(row.get("macro_f1")),
                    status=row.get("publication_status", "-"),
                    path=row.get("path", "-"),
                )
            )
    ablation = card.get("ablation_summary") or {}
    lines += [
        "",
        "## Ablation Positioning",
        "",
        "- Total ablations: `{}`".format(ablation.get("num_ablations", 0)),
        "- Helpful: `{}`".format(ablation.get("helpful_count", 0)),
        "- Harmful: `{}`".format(ablation.get("harmful_count", 0)),
        "- Neutral or uncertain: `{}`".format(ablation.get("neutral_or_uncertain_count", 0)),
        "",
        "Candidate modules remain controlled ablations unless they pass the same pre-registered paper_unified promotion rule on every in-scope dataset; harmful candidates motivate the shared guards rather than becoming dataset-specific branches.",
        "",
    ]
    rankings = card.get("candidate_rankings") or []
    if rankings:
        lines += [
            "## Content-Group Candidate Scan",
            "",
            "| Dataset | Candidates | Best Acc | Best Macro-F1 | Best Group Acc/F1 Lower | Target | Best Path |",
            "|---|---:|---:|---:|---:|---|---|",
        ]
        for row in rankings:
            lines.append(
                "| {dataset} | {num_ok}/{num_candidates} | {acc} | {f1} | {gacc}/{gf1} | {target} | `{path}` |".format(
                    dataset=row.get("dataset"),
                    num_ok=row.get("num_ok"),
                    num_candidates=row.get("num_candidates"),
                    acc=fmt(row.get("best_accuracy")),
                    f1=fmt(row.get("best_macro_f1")),
                    gacc=fmt(row.get("best_content_group_accuracy_lower")),
                    gf1=fmt(row.get("best_content_group_macro_f1_lower")),
                    target=row.get("best_content_group_target_met"),
                    path=row.get("best_path"),
                )
            )
        lines.append("")
    lines += [
        "## Readiness",
        "",
        "| Gate | Status |",
        "|---|---|",
        f"| legacy protocol audit | {ready.get('legacy_protocol_audit_ready')} |",
        f"| exact common-reference v2 | {ready.get('exact_common_reference_v2_ready')} |",
        f"| unified method v2 | {ready.get('unified_method_v2_ready')} |",
        f"| strict shared-core v2 provenance | {ready.get('strict_shared_core_v2_ready')} |",
        f"| durable session-novelty evidence | {ready.get('session_novelty_evidence_ready')} |",
        f"| SWEET end-to-end exceeded on all four tasks | {ready.get('sweet_end_to_end_all_tasks_exceeded')} |",
        f"| paper_unified profile | {ready.get('paper_unified_profile_ready')} |",
        f"| flow point targets | {ready.get('flow_point_targets_ready')} |",
        f"| flow bootstrap CI targets | {ready.get('flow_ci_targets_ready')} |",
        f"| flow content-grouped CI targets | {ready.get('flow_content_group_ci_targets_ready')} |",
        f"| packet publication defaults | {ready.get('packet_publication_defaults_ready')} |",
        f"| CCF-A risk level | {ready.get('ccf_a_risk_level')} |",
        "",
        "## Reviewer Risks",
        "",
    ]
    risks = ready.get("ccf_a_risks") or []
    if not risks:
        lines.append("- No current audit-level risks.")
    else:
        for risk in risks:
            lines.append(f"- {risk}")
    lines += [
        "",
        "## Next Paper-Grade Action",
        "",
        (
            "Complete the frozen exact-v2 VPN/TLS packet/flow cross-fold matrix, checkpoint-schema audits, "
            "content-group bootstrap, and matched ablations. Do not add another expert unless the same module is "
            "pre-registered for every dataset and both tasks."
        ),
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a reviewer-facing paper method card from current evidence.")
    ap.add_argument("--evidence_json", default="reasoningDataset/paper_evidence_pack.json")
    ap.add_argument("--unified_audit_json", default="reasoningDataset/unified_framework_audit.json")
    ap.add_argument("--defaults_audit_json", default="reasoningDataset/paper_framework_defaults_audit.json")
    ap.add_argument(
        "--sweet_comparison_json",
        default="reasoningDataset/sweet_protocol_comparison.json",
    )
    ap.add_argument(
        "--candidate_ranking_json",
        action="append",
        default=[],
        help="Optional content-group candidate ranking JSON. Defaults to known VPN/TLS ranking files when present.",
    )
    ap.add_argument("--output_json", default="reasoningDataset/paper_method_card.json")
    ap.add_argument("--output_md", default="reasoningDataset/paper_method_card.md")
    args = ap.parse_args()

    ranking_paths = args.candidate_ranking_json or [path for path in DEFAULT_CANDIDATE_RANKINGS if Path(path).exists()]
    card = build_card(
        load_json(args.evidence_json),
        load_json(args.unified_audit_json),
        load_json(args.defaults_audit_json),
        candidate_rankings=candidate_ranking_summary(ranking_paths),
        strict_status=strict_shared_core_publication_status(),
        sweet_comparison=(
            load_json(args.sweet_comparison_json)
            if Path(args.sweet_comparison_json).is_file()
            else None
        ),
    )
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(card, indent=2, ensure_ascii=False), encoding="utf-8")
    Path(args.output_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_md).write_text(render_markdown(card), encoding="utf-8")
    print(json.dumps({"output_json": args.output_json, "output_md": args.output_md, "risk": card["readiness"]["ccf_a_risk_level"]}, indent=2))


if __name__ == "__main__":
    main()
