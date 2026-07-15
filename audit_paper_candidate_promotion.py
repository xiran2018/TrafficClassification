#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

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


def metric_row(data: Dict[str, Any] | None) -> Dict[str, Any]:
    if not isinstance(data, dict):
        return {"accuracy": None, "macro_f1": None, "num_flows": 0}
    acc, f1 = metric_from_payload(data)
    return {
        "accuracy": acc,
        "macro_f1": f1,
        "num_flows": len(data.get("flow_y_true", [])),
    }


def audit_candidate(row: Dict[str, Any], min_raw_gain: float) -> Dict[str, Any]:
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
    direct_promotable = bool(
        raw_has_gain
        and raw_slots["matches_required"]
        and raw_safety["has_required_guards"]
    )
    blockers: List[str] = []
    if not raw_has_gain:
        blockers.append("raw_gain_below_threshold")
    if not raw_slots["matches_required"]:
        blockers.append("raw_candidate_missing_required_unified_slots")
    if not raw_safety["has_required_guards"]:
        blockers.append("raw_candidate_missing_selector_safety_guards")
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
        "direct_promotable": direct_promotable,
        "blockers": blockers,
        "decision": decision,
    }


def build_audit(args: argparse.Namespace) -> Dict[str, Any]:
    pack = load_json(args.evidence_pack)
    if not isinstance(pack, dict):
        raise SystemExit(f"Cannot load evidence pack: {args.evidence_pack}")
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
            rows.append(audit_candidate(rec, args.min_raw_gain))
    return {
        "source": {"evidence_pack": args.evidence_pack},
        "min_raw_gain": args.min_raw_gain,
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
        "| Dataset | Raw Acc/F1 | Paper-Safe Acc/F1 | Raw-Paper Gap | Raw Slots | Raw Guards | Decision | Blockers |",
        "|---|---:|---:|---:|---|---|---|---|",
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
        lines.append(
            "| {dataset} | {raw_acc}/{raw_f1} | {safe_acc}/{safe_f1} | {dacc}/{df1} | {slot_mode}, match={slot_match} | {guards} | {decision} | {blockers} |".format(
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
