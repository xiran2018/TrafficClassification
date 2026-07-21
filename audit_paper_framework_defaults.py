#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from paper_framework_defaults import (
    DEFAULT_FRAMEWORK_PROFILE,
    DEFAULT_FRAMEWORK_PROFILE_DESCRIPTION,
    DEFAULT_PAPER_SAFE_RESULTS,
    DEFAULT_SHARED_CORE_MODULES,
    DEFAULT_TARGETS,
    DEFAULT_UNIFIED_EXPERT_SLOTS,
    DEFAULT_UNIFIED_EXPERT_SLOTS_CSV,
    default_framework_results,
)
from summarize_experiment_results import metric_from_payload
from audit_unified_framework import build_report as build_unified_framework_report


def load_json(path: Path) -> Dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def slot_status(data: Dict[str, Any]) -> Dict[str, Any]:
    feature_config = data.get("feature_config") or {}
    slots = feature_config.get("unified_expert_slots") or []
    status = feature_config.get("input_slot_status") or []
    if not slots:
        # Legacy selector outputs are compatible if their inputs can be mapped
        # onto the required slots and missing slots are identity-from-base.
        inputs = data.get("inputs") or []
        slots = list(DEFAULT_UNIFIED_EXPERT_SLOTS) if inputs else []
        if inputs and (data.get("config") or {}).get("selected_mode"):
            mode = "crossfold_consensus_identity_compatible"
        else:
            mode = "inferred_identity_compatible" if inputs else "missing"
    else:
        mode = "recorded"
    provided = []
    identity = []
    if isinstance(status, list):
        provided = [row.get("name") for row in status if row.get("status") in {"provided", "extra_provided"}]
        identity = [row.get("name") for row in status if row.get("status") == "identity_from_base"]
    return {
        "mode": mode,
        "slots": slots,
        "matches_required": list(slots) == list(DEFAULT_UNIFIED_EXPERT_SLOTS),
        "provided": provided,
        "identity_from_base": identity,
    }


def audit_dataset(dataset: str, path: str, target: tuple[float, float] | None) -> Dict[str, Any]:
    p = Path(path)
    row: Dict[str, Any] = {
        "dataset": dataset,
        "path": path,
        "exists": p.exists(),
        "target": None if target is None else {"accuracy": target[0], "macro_f1": target[1]},
        "ok": False,
        "errors": [],
    }
    if not p.exists():
        row["errors"].append("missing_result_json")
        return row
    data = load_json(p)
    if data is None:
        row["errors"].append("invalid_json")
        return row
    acc, macro_f1 = metric_from_payload(data)
    row["accuracy"] = acc
    row["macro_f1"] = macro_f1
    row["num_flows"] = len(data.get("flow_y_true", []))
    if acc is None:
        row["errors"].append("missing_flow_metrics")
    if target is not None:
        target_met = bool(acc is not None and macro_f1 is not None and acc >= target[0] and macro_f1 >= target[1])
        row["target_met"] = target_met
        if not target_met:
            row["errors"].append("target_not_met")
    else:
        row["target_met"] = None
    row["slot_status"] = slot_status(data)
    if not row["slot_status"]["matches_required"]:
        row["errors"].append("selector_slots_do_not_match_required")
    row["ok"] = not row["errors"]
    return row


def build_audit() -> Dict[str, Any]:
    configured = {
        dataset: {
            "path": path,
            "target": None if target_acc is None else (float(target_acc), float(target_f1)),
        }
        for dataset, path, target_acc, target_f1 in default_framework_results()
    }
    rows = [
        audit_dataset(dataset, item["path"], item["target"])
        for dataset, item in configured.items()
    ]
    unified_report = build_unified_framework_report()
    packet_rows = list(unified_report.get("packet_level") or [])
    unified_framework = unified_report.get("framework") or {}
    shared_core_matches = list(DEFAULT_SHARED_CORE_MODULES) == list(unified_framework.get("shared_core_modules") or [])
    unified_profile_ok = (
        unified_report.get("status") == "pass"
        and int(unified_framework.get("paper_unified_flow_manifest_passes") or 0) > 0
        and int(unified_framework.get("paper_unified_packet_manifest_passes") or 0) > 0
    )
    packet_profile_ok = bool(packet_rows) and all(
        row.get("publication_status") == "pass" for row in packet_rows
    )
    return {
        "defaults": {
            "framework_profile": DEFAULT_FRAMEWORK_PROFILE,
            "framework_profile_description": DEFAULT_FRAMEWORK_PROFILE_DESCRIPTION,
            "shared_core_modules": list(DEFAULT_SHARED_CORE_MODULES),
            "targets": {key: {"accuracy": value[0], "macro_f1": value[1]} for key, value in DEFAULT_TARGETS.items()},
            "paper_safe_results": DEFAULT_PAPER_SAFE_RESULTS,
            "unified_expert_slots": DEFAULT_UNIFIED_EXPERT_SLOTS,
            "unified_expert_slots_csv": DEFAULT_UNIFIED_EXPERT_SLOTS_CSV,
        },
        "unified_framework": {
            "status": unified_report.get("status"),
            "shared_core_matches_defaults": shared_core_matches,
            "paper_unified_flow_manifest_passes": unified_framework.get("paper_unified_flow_manifest_passes"),
            "paper_unified_packet_manifest_passes": unified_framework.get("paper_unified_packet_manifest_passes"),
            "flow_scope": unified_framework.get("flow_scope"),
            "packet_scope": unified_framework.get("packet_scope"),
        },
        "flow_datasets": rows,
        "packet_datasets": packet_rows,
        "datasets": rows,
        "ok": all(row["ok"] for row in rows) and shared_core_matches and unified_profile_ok and packet_profile_ok,
    }


def render_markdown(audit: Dict[str, Any]) -> str:
    lines = [
        "# Paper Framework Defaults Audit",
        "",
        f"Overall status: `{audit['ok']}`",
        "",
        f"Default framework profile: `{audit['defaults']['framework_profile']}`",
        "",
        "Shared core modules:",
        "",
    ]
    for module in audit["defaults"]["shared_core_modules"]:
        lines.append(f"- `{module}`")
    unified = audit.get("unified_framework") or {}
    lines += [
        "",
        "Unified framework gate:",
        "",
        "| Status | Shared Core Match | paper_unified Flow Manifests | paper_unified Packet Manifests |",
        "|---|---|---:|---:|",
        "| {status} | {shared} | {flow} | {packet} |".format(
            status=unified.get("status"),
            shared=unified.get("shared_core_matches_defaults"),
            flow=unified.get("paper_unified_flow_manifest_passes"),
            packet=unified.get("paper_unified_packet_manifest_passes"),
        ),
        "",
        "## Flow-Level Defaults",
        "",
        "| Dataset | Exists | Acc | Macro-F1 | Target | Target Met | Slot Mode | Slots Match | Errors |",
        "|---|---|---:|---:|---|---|---|---|---|",
    ]
    for row in audit["flow_datasets"]:
        target = row.get("target")
        target_text = "-" if target is None else f"{target['accuracy']:.4f}/{target['macro_f1']:.4f}"
        slot = row.get("slot_status") or {}
        errors = ", ".join(row.get("errors") or [])
        lines.append(
            "| {dataset} | {exists} | {acc} | {f1} | {target} | {target_met} | {slot_mode} | {slots_match} | {errors} |".format(
                dataset=row["dataset"],
                exists=row["exists"],
                acc="-" if row.get("accuracy") is None else f"{row['accuracy']:.4f}",
                f1="-" if row.get("macro_f1") is None else f"{row['macro_f1']:.4f}",
                target=target_text,
                target_met=row.get("target_met"),
                slot_mode=slot.get("mode", "-"),
                slots_match=slot.get("matches_required", "-"),
                errors=errors or "-",
            )
        )
    lines += [
        "",
        "## Packet-Level Defaults",
        "",
        "| Dataset | Exists | Acc | Macro-F1 | Publication Status | Provenance | Path |",
        "|---|---|---:|---:|---|---:|---|",
    ]
    for row in audit.get("packet_datasets") or []:
        prov = row.get("framework_provenance") or {}
        lines.append(
            "| {dataset} | {exists} | {acc} | {f1} | {pub} | {prov_count} | `{path}` |".format(
                dataset=row["dataset"],
                exists=row["exists"],
                acc="-" if row.get("accuracy") is None else f"{row['accuracy']:.4f}",
                f1="-" if row.get("macro_f1") is None else f"{row['macro_f1']:.4f}",
                pub=row.get("publication_status"),
                prov_count=prov.get("matching_manifest_count", 0),
                path=row["path"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit paper-facing default result paths, targets, and unified expert slots.")
    ap.add_argument("--output_json", default="")
    ap.add_argument("--output_md", default="")
    args = ap.parse_args()

    audit = build_audit()
    md = render_markdown(audit)
    print(md)
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.output_md:
        Path(args.output_md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_md).write_text(md, encoding="utf-8")
    if not audit["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
