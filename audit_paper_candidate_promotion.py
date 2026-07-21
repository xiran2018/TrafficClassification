#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from evaluate_content_unique_predictions import (
    content_group_bootstrap_ci,
    flow_content_map,
    group_indices_by_content,
)
from paper_framework_defaults import DEFAULT_UNIFIED_EXPERT_SLOTS
from summarize_experiment_results import metric_from_payload


def load_json(path: str) -> Dict[str, Any] | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def fmt(value: Any, digits: int = 4, signed: bool = False) -> str:
    if value is None:
        return "-"
    prefix = "+" if signed and float(value) > 0 else ""
    return f"{prefix}{float(value):.{digits}f}"


def slot_summary(data: Dict[str, Any] | None) -> Dict[str, Any]:
    if not isinstance(data, dict):
        return {"mode": "missing", "matches_required": False, "slots": []}
    feature_config = data.get("feature_config") or {}
    slots = feature_config.get("unified_expert_slots") or []
    if slots:
        return {
            "mode": "recorded",
            "matches_required": list(slots) == list(DEFAULT_UNIFIED_EXPERT_SLOTS),
            "slots": slots,
        }
    if data.get("inputs"):
        return {
            "mode": "legacy_inferred",
            "matches_required": False,
            "slots": [],
        }
    return {"mode": "missing", "matches_required": False, "slots": []}


def selector_safety(data: Dict[str, Any] | None) -> Dict[str, Any]:
    selected = (data or {}).get("selected") or {}
    has_bootstrap = isinstance(selected, dict) and isinstance(selected.get("bootstrap_guard"), dict)
    has_shift = isinstance(selected, dict) and isinstance(selected.get("target_shift_guard"), dict)
    strategy = selected.get("strategy") if isinstance(selected, dict) else None
    rejected = []
    fallback = selected.get("fallback_reason") if isinstance(selected, dict) else None
    if isinstance(fallback, dict):
        for row in fallback.get("rejected_candidates") or []:
            candidate = row.get("rejected") or {}
            rejected.append(
                {
                    "strategy": candidate.get("strategy"),
                    "reject_reasons": row.get("reject_reasons"),
                    "target_shift_guard": row.get("target_shift_guard"),
                    "bootstrap_guard": row.get("bootstrap_guard"),
                }
            )
    return {
        "strategy": strategy,
        "has_bootstrap_guard": has_bootstrap,
        "has_target_shift_guard": has_shift,
        "has_required_guards": has_bootstrap and has_shift,
        "rejected_candidates": rejected[:5],
    }


def claims_by_dataset(pack: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {
        str(row.get("dataset")): row
        for row in pack.get("claims", [])
        if row.get("dataset") is not None
    }


def parse_dataset_paths(items: List[str]) -> Dict[str, str]:
    parsed = {}
    for raw in items:
        if "=" not in raw:
            raise SystemExit("--raw_content_group_index expects DATASET=PATH")
        dataset, path = raw.split("=", 1)
        dataset = dataset.strip()
        path = path.strip()
        if not dataset or not path:
            raise SystemExit("--raw_content_group_index expects non-empty DATASET=PATH")
        parsed[dataset] = path
    return parsed


def raw_candidate_content_group_ci(
    data: Dict[str, Any] | None,
    index_path: str,
    samples: int,
    seed: int,
) -> Dict[str, Any]:
    if not isinstance(data, dict):
        return {"status": "missing_raw_candidate_payload", "target_met": False}
    if samples <= 0:
        return {"status": "disabled", "target_met": False}
    missing = [key for key in ("flow_ids", "flow_y_true", "flow_prob") if key not in data]
    if missing:
        return {"status": "missing_probability_fields", "target_met": False, "missing": missing}
    path = Path(index_path)
    if not path.exists():
        return {"status": "missing_flow_embedding_index", "target_met": False, "index_path": index_path}
    flow_ids = [str(value) for value in data["flow_ids"]]
    y_true = np.asarray(data["flow_y_true"], dtype=np.int64)
    prob = np.asarray(data["flow_prob"], dtype=np.float64)
    try:
        flow_to_hash, _ = flow_content_map(index_path)
        _, grouped_indices = group_indices_by_content(flow_ids, flow_to_hash)
        ci = content_group_bootstrap_ci(y_true, prob, grouped_indices, samples, seed)
    except Exception as exc:
        return {"status": "compute_failed", "target_met": False, "error": str(exc), "index_path": index_path}
    ci.update({"status": "computed", "target_met": None, "index_path": index_path})
    return ci


def content_group_ci_status(
    row: Dict[str, Any],
    claim: Dict[str, Any] | None,
    *,
    raw_data: Dict[str, Any] | None = None,
    raw_content_group_index: str = "",
    bootstrap_samples: int = 0,
    bootstrap_seed: int = 42,
) -> Dict[str, Any]:
    raw_path = row.get("raw_best_path")
    safe_path = row.get("paper_safe_path")
    if not isinstance(claim, dict):
        return {
            "status": "missing_claim",
            "target_met": False,
            "reason": "evidence_pack_has_no_dataset_claim",
        }
    if raw_path != safe_path:
        computed: Dict[str, Any] | None = None
        if raw_content_group_index:
            computed = raw_candidate_content_group_ci(
                raw_data,
                raw_content_group_index,
                bootstrap_samples,
                bootstrap_seed,
            )
            if computed.get("status") == "computed":
                target_acc = claim.get("target_accuracy")
                target_f1 = claim.get("target_macro_f1")
                acc_ci = computed.get("accuracy_95_ci")
                f1_ci = computed.get("macro_f1_95_ci")
                target_met = bool(
                    isinstance(acc_ci, list)
                    and isinstance(f1_ci, list)
                    and len(acc_ci) == 2
                    and len(f1_ci) == 2
                    and (target_acc is None or float(acc_ci[0]) >= float(target_acc))
                    and (target_f1 is None or float(f1_ci[0]) >= float(target_f1))
                )
                computed.update(
                    {
                        "target_met": target_met,
                        "accuracy_ci95": acc_ci,
                        "macro_f1_ci95": f1_ci,
                    }
                )
                return computed
        return {
            "status": "missing_raw_candidate_group_ci",
            "target_met": False,
            "reason": "raw_best_path_differs_from_paper_safe_path",
            "computed": computed,
            "paper_safe_target_met": claim.get("content_group_ci_target_met"),
            "paper_safe_accuracy_ci95": claim.get("content_group_accuracy_ci95"),
            "paper_safe_macro_f1_ci95": claim.get("content_group_macro_f1_ci95"),
        }
    return {
        "status": "available",
        "target_met": claim.get("content_group_ci_target_met") is True,
        "accuracy_ci95": claim.get("content_group_accuracy_ci95"),
        "macro_f1_ci95": claim.get("content_group_macro_f1_ci95"),
        "group_count": claim.get("content_group_count"),
        "row_count": claim.get("content_group_rows"),
    }


def metric_row(data: Dict[str, Any] | None) -> Dict[str, Any]:
    if not isinstance(data, dict):
        return {"accuracy": None, "macro_f1": None, "num_flows": 0}
    acc, f1 = metric_from_payload(data)
    return {
        "accuracy": acc,
        "macro_f1": f1,
        "num_flows": len(data.get("flow_y_true", [])),
    }


def audit_candidate(
    row: Dict[str, Any],
    min_raw_gain: float,
    *,
    claim: Dict[str, Any] | None = None,
    require_content_group_ci: bool = True,
    raw_content_group_index: str = "",
    content_group_bootstrap_samples: int = 0,
    content_group_bootstrap_seed: int = 42,
) -> Dict[str, Any]:
    dataset = row.get("dataset")
    raw_path = row.get("raw_best_path")
    safe_path = row.get("paper_safe_path")
    raw_data = load_json(raw_path)
    safe_data = load_json(safe_path)
    raw_metric = metric_row(raw_data)
    safe_metric = metric_row(safe_data)
    raw_gain_acc = row.get("raw_minus_paper_safe_accuracy")
    raw_gain_f1 = row.get("raw_minus_paper_safe_macro_f1")
    raw_has_gain = max(float(raw_gain_acc or 0.0), float(raw_gain_f1 or 0.0)) >= min_raw_gain
    raw_slots = slot_summary(raw_data)
    safe_slots = slot_summary(safe_data)
    raw_safety = selector_safety(raw_data)
    safe_safety = selector_safety(safe_data)
    group_ci = content_group_ci_status(
        row,
        claim,
        raw_data=raw_data,
        raw_content_group_index=raw_content_group_index,
        bootstrap_samples=content_group_bootstrap_samples,
        bootstrap_seed=content_group_bootstrap_seed,
    )
    direct_promotable = bool(
        raw_has_gain
        and raw_slots["matches_required"]
        and raw_safety["has_required_guards"]
        and (not require_content_group_ci or group_ci["target_met"])
    )
    blockers: List[str] = []
    if not raw_has_gain:
        blockers.append("raw_gain_below_threshold")
    if not raw_slots["matches_required"]:
        blockers.append("raw_candidate_missing_required_unified_slots")
    if not raw_safety["has_required_guards"]:
        blockers.append("raw_candidate_missing_selector_safety_guards")
    if require_content_group_ci and not group_ci["target_met"]:
        blockers.append(f"raw_candidate_content_group_ci_not_ready:{group_ci['status']}")
    decision = "promotable" if direct_promotable else "keep_as_probe_or_wrap_with_selector"
    if safe_safety["has_required_guards"] and not direct_promotable:
        decision = "paper_safe_selector_already_required"
    return {
        "dataset": dataset,
        "raw_best_path": raw_path,
        "paper_safe_path": safe_path,
        "raw_metrics": raw_metric,
        "paper_safe_metrics": safe_metric,
        "raw_minus_paper_safe_accuracy": raw_gain_acc,
        "raw_minus_paper_safe_macro_f1": raw_gain_f1,
        "raw_slots": raw_slots,
        "paper_safe_slots": safe_slots,
        "raw_safety": raw_safety,
        "paper_safe_safety": safe_safety,
        "content_group_ci": group_ci,
        "require_content_group_ci": require_content_group_ci,
        "direct_promotable": direct_promotable,
        "blockers": blockers,
        "decision": decision,
    }


def build_audit(args: argparse.Namespace) -> Dict[str, Any]:
    pack = load_json(args.evidence_pack)
    if not isinstance(pack, dict):
        raise SystemExit(f"Cannot load evidence pack: {args.evidence_pack}")
    claims = claims_by_dataset(pack)
    raw_group_indices = parse_dataset_paths(args.raw_content_group_index or [])
    rows = []
    for rec in pack.get("recommendations", []):
        raw_path = rec.get("raw_best_path")
        safe_path = rec.get("paper_safe_path")
        same_path = rec.get("raw_and_paper_safe_same_path")
        raw_gain = max(
            float(rec.get("raw_minus_paper_safe_accuracy") or 0.0),
            float(rec.get("raw_minus_paper_safe_macro_f1") or 0.0),
        )
        if args.only_changed and (same_path or raw_gain < args.min_raw_gain):
            continue
        if raw_path and safe_path:
            rows.append(
                audit_candidate(
                    rec,
                    args.min_raw_gain,
                    claim=claims.get(str(rec.get("dataset"))),
                    require_content_group_ci=args.require_content_group_ci,
                    raw_content_group_index=raw_group_indices.get(str(rec.get("dataset")), ""),
                    content_group_bootstrap_samples=args.content_group_bootstrap_samples,
                    content_group_bootstrap_seed=args.content_group_bootstrap_seed,
                )
            )
    return {
        "source": {"evidence_pack": args.evidence_pack},
        "min_raw_gain": args.min_raw_gain,
        "require_content_group_ci": args.require_content_group_ci,
        "raw_content_group_indices": raw_group_indices,
        "content_group_bootstrap_samples": args.content_group_bootstrap_samples,
        "content_group_bootstrap_seed": args.content_group_bootstrap_seed,
        "required_unified_expert_slots": DEFAULT_UNIFIED_EXPERT_SLOTS,
        "candidates": rows,
        "num_direct_promotable": sum(1 for row in rows if row["direct_promotable"]),
    }


def render_markdown(audit: Dict[str, Any]) -> str:
    lines = [
        "# Paper Candidate Promotion Audit",
        "",
        f"Minimum raw gain threshold: `{audit['min_raw_gain']}`",
        "",
        f"Require raw candidate content-group CI: `{audit.get('require_content_group_ci')}`",
        "",
        "| Dataset | Raw Acc/F1 | Paper-Safe Acc/F1 | Raw-Paper Gap | Raw Slots | Raw Guards | Group CI | Decision | Blockers |",
        "|---|---:|---:|---:|---|---|---|---|---|",
    ]
    for row in audit["candidates"]:
        raw = row["raw_metrics"]
        safe = row["paper_safe_metrics"]
        raw_slots = row["raw_slots"]
        raw_safety = row["raw_safety"]
        guards = "boot={}; shift={}".format(
            raw_safety.get("has_bootstrap_guard"),
            raw_safety.get("has_target_shift_guard"),
        )
        group_ci = row.get("content_group_ci") or {}
        group = "{}; target={}".format(group_ci.get("status"), group_ci.get("target_met"))
        lines.append(
            "| {dataset} | {raw_acc}/{raw_f1} | {safe_acc}/{safe_f1} | {dacc}/{df1} | {slot_mode}, match={slot_match} | {guards} | {group} | {decision} | {blockers} |".format(
                dataset=row["dataset"],
                raw_acc=fmt(raw["accuracy"]),
                raw_f1=fmt(raw["macro_f1"]),
                safe_acc=fmt(safe["accuracy"]),
                safe_f1=fmt(safe["macro_f1"]),
                dacc=fmt(row.get("raw_minus_paper_safe_accuracy"), signed=True),
                df1=fmt(row.get("raw_minus_paper_safe_macro_f1"), signed=True),
                slot_mode=raw_slots.get("mode"),
                slot_match=raw_slots.get("matches_required"),
                guards=guards,
                group=group,
                decision=row["decision"],
                blockers=", ".join(row["blockers"]) or "-",
            )
        )
    if not audit["candidates"]:
        lines.append("| - | - | - | - | - | - | no_changed_candidates | - |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit whether raw-best candidates can be promoted to paper-safe defaults.")
    ap.add_argument("--evidence_pack", default="reasoningDataset/paper_evidence_pack.json")
    ap.add_argument("--min_raw_gain", type=float, default=0.003)
    ap.add_argument("--only_changed", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--require_content_group_ci", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument(
        "--raw_content_group_index",
        action="append",
        default=[],
        help="Optional DATASET=flow_embedding_index.jsonl used to compute raw-best content-group CI.",
    )
    ap.add_argument("--content_group_bootstrap_samples", type=int, default=300)
    ap.add_argument("--content_group_bootstrap_seed", type=int, default=42)
    ap.add_argument("--output_json", default="reasoningDataset/paper_candidate_promotion_audit.json")
    ap.add_argument("--output_md", default="reasoningDataset/paper_candidate_promotion_audit.md")
    args = ap.parse_args()

    audit = build_audit(args)
    md = render_markdown(audit)
    print(md)
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.output_md:
        Path(args.output_md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_md).write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()
