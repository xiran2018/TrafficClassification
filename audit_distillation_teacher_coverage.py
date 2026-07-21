#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


def load_pt(path: str) -> Any:
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def item_flow_id(item: Any, fallback: int) -> str:
    if isinstance(item, dict):
        return str(item.get("flow_id", fallback))
    return str(fallback)


def dataset_flow_ids(path: str) -> list[str]:
    data = load_pt(path)
    return sorted({item_flow_id(item, idx) for idx, item in enumerate(data)})


def load_teacher(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    flow_ids = [str(fid) for fid in data.get("flow_ids", [])]
    if not flow_ids:
        raise ValueError(f"{path} does not contain non-empty flow_ids")
    return data


def parse_dataset(raw: list[str]) -> tuple[str, str]:
    if len(raw) != 2:
        raise ValueError("--dataset expects NAME PT_PATH")
    return raw[0], raw[1]


def coverage_row(name: str, dataset_path: str, teacher_ids: set[str], min_coverage: float) -> dict[str, Any]:
    ids = dataset_flow_ids(dataset_path)
    matched = sorted(set(ids) & teacher_ids)
    coverage = len(matched) / max(1, len(ids))
    return {
        "name": name,
        "dataset_path": dataset_path,
        "unique_flow_count": len(ids),
        "matched_flow_count": len(matched),
        "coverage": coverage,
        "min_coverage": float(min_coverage),
        "passes_min_coverage": coverage >= float(min_coverage),
    }


def recommendation(rows: Iterable[dict[str, Any]], action: str, gate_names: set[str] | None = None) -> str:
    rows = [row for row in rows if not gate_names or row.get("name") in gate_names]
    if not rows:
        return "no_datasets"
    if all(row["passes_min_coverage"] for row in rows):
        return "flow_id_distillation_safe"
    if action == "fail":
        return "fail_before_training"
    if action == "disable_flow":
        return "disable_flow_id_kl_keep_class_prior"
    return "warn_only"


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit flow_id coverage before consensus distillation.")
    ap.add_argument("--teacher_json", required=True)
    ap.add_argument("--dataset", nargs=2, action="append", required=True, metavar=("NAME", "PT"))
    ap.add_argument("--gate_dataset", action="append", default=[], help="Dataset NAME used for the final coverage recommendation. Defaults to all.")
    ap.add_argument("--min_coverage", type=float, default=0.5)
    ap.add_argument("--low_coverage_action", choices=["warn", "disable_flow", "fail"], default="disable_flow")
    ap.add_argument("--output_json", default="")
    args = ap.parse_args()

    teacher = load_teacher(args.teacher_json)
    teacher_ids = {str(fid) for fid in teacher.get("flow_ids", [])}
    rows = [
        coverage_row(name, path, teacher_ids, args.min_coverage)
        for name, path in (parse_dataset(raw) for raw in args.dataset)
    ]
    gate_names = {str(name) for name in args.gate_dataset}
    payload = {
        "teacher_json": args.teacher_json,
        "teacher_flow_count": len(teacher_ids),
        "teacher_metrics": (teacher.get("metrics") or {}).get("flow_level", {}),
        "teacher_config": teacher.get("config", {}),
        "min_coverage": float(args.min_coverage),
        "low_coverage_action": args.low_coverage_action,
        "gate_datasets": sorted(gate_names) if gate_names else "all",
        "datasets": rows,
        "recommendation": recommendation(rows, args.low_coverage_action, gate_names),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
