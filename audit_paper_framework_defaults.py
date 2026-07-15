#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from paper_framework_defaults import (
    DEFAULT_PAPER_SAFE_RESULTS,
    DEFAULT_TARGETS,
    DEFAULT_UNIFIED_EXPERT_SLOTS,
    DEFAULT_UNIFIED_EXPERT_SLOTS_CSV,
    default_framework_results,
)
from summarize_experiment_results import metric_from_payload


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
    return {
        "defaults": {
            "targets": {key: {"accuracy": value[0], "macro_f1": value[1]} for key, value in DEFAULT_TARGETS.items()},
            "paper_safe_results": DEFAULT_PAPER_SAFE_RESULTS,
            "unified_expert_slots": DEFAULT_UNIFIED_EXPERT_SLOTS,
            "unified_expert_slots_csv": DEFAULT_UNIFIED_EXPERT_SLOTS_CSV,
        },
        "datasets": rows,
        "ok": all(row["ok"] for row in rows),
    }


def render_markdown(audit: Dict[str, Any]) -> str:
    lines = [
        "# Paper Framework Defaults Audit",
        "",
        f"Overall status: `{audit['ok']}`",
        "",
        "| Dataset | Exists | Acc | Macro-F1 | Target | Target Met | Slot Mode | Slots Match | Errors |",
        "|---|---|---:|---:|---|---|---|---|---|",
    ]
    for row in audit["datasets"]:
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
