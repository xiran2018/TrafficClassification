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


def teacher_contract(
    teacher: dict[str, Any], min_teachers_per_flow: int, require_oof_exclusion_proof: bool
) -> dict[str, Any]:
    flow_ids = [str(fid) for fid in teacher.get("flow_ids", [])]
    multiplicity = teacher.get("teacher_multiplicity")
    counts: list[int] = []
    oof_counts: list[int] = []
    aligned = False
    oof_aligned = False
    if isinstance(multiplicity, dict):
        count_ids = [str(fid) for fid in multiplicity.get("flow_ids", [])]
        raw_counts = multiplicity.get("teacher_counts", [])
        if len(count_ids) == len(raw_counts) and len(set(count_ids)) == len(count_ids):
            count_by_id = {fid: int(count) for fid, count in zip(count_ids, raw_counts)}
            if set(count_by_id) == set(flow_ids):
                counts = [count_by_id[fid] for fid in flow_ids]
                aligned = True
                raw_oof_counts = multiplicity.get("oof_teacher_counts", [])
                if len(raw_oof_counts) == len(count_ids):
                    oof_by_id = {
                        fid: int(count)
                        for fid, count in zip(count_ids, raw_oof_counts)
                    }
                    oof_counts = [oof_by_id[fid] for fid in flow_ids]
                    oof_aligned = True
    count_pass = bool(aligned and counts and min(counts) >= int(min_teachers_per_flow))
    oof_proven = bool(
        isinstance(multiplicity, dict) and multiplicity.get("oof_exclusion_proven") is True
    )
    all_contributors_oof = bool(
        aligned
        and oof_aligned
        and counts
        and all(oof == count for oof, count in zip(oof_counts, counts))
    )
    multi_teacher_oof = bool(
        isinstance(multiplicity, dict)
        and multiplicity.get("oof_multi_teacher_consensus_proven") is True
    )
    strict_oof_pass = bool(
        oof_proven
        and all_contributors_oof
        and (min_teachers_per_flow <= 1 or multi_teacher_oof)
    )
    return {
        "multiplicity_available_and_aligned": aligned,
        "minimum_teacher_count": min(counts) if counts else None,
        "maximum_teacher_count": max(counts) if counts else None,
        "mean_teacher_count": (sum(counts) / len(counts)) if counts else None,
        "required_min_teachers_per_flow": int(min_teachers_per_flow),
        "passes_teacher_count": count_pass,
        "oof_counts_available_and_aligned": oof_aligned,
        "minimum_oof_teacher_count": min(oof_counts) if oof_counts else None,
        "all_contributing_teachers_oof": all_contributors_oof,
        "oof_exclusion_proven": oof_proven,
        "oof_multi_teacher_consensus_proven": multi_teacher_oof,
        "require_oof_exclusion_proof": bool(require_oof_exclusion_proof),
        "passes_oof_exclusion": strict_oof_pass or not require_oof_exclusion_proof,
        "passes": count_pass and (strict_oof_pass or not require_oof_exclusion_proof),
    }


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


def recommendation(
    rows: Iterable[dict[str, Any]],
    action: str,
    gate_names: set[str] | None = None,
    teacher_gate_passes: bool = True,
) -> str:
    rows = [row for row in rows if not gate_names or row.get("name") in gate_names]
    if not rows:
        return "no_datasets"
    if teacher_gate_passes and all(row["passes_min_coverage"] for row in rows):
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
    ap.add_argument("--min_teachers_per_flow", type=int, default=1)
    ap.add_argument(
        "--require_oof_exclusion_proof",
        action="store_true",
        help="Require machine-readable proof that every teacher excluded its target flow. Legacy targets fail this gate.",
    )
    ap.add_argument("--low_coverage_action", choices=["warn", "disable_flow", "fail"], default="disable_flow")
    ap.add_argument("--output_json", default="")
    args = ap.parse_args()

    teacher = load_teacher(args.teacher_json)
    if args.min_teachers_per_flow <= 0:
        raise ValueError("--min_teachers_per_flow must be positive")
    contract = teacher_contract(
        teacher, args.min_teachers_per_flow, args.require_oof_exclusion_proof
    )
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
        "teacher_contract": contract,
        "min_coverage": float(args.min_coverage),
        "low_coverage_action": args.low_coverage_action,
        "gate_datasets": sorted(gate_names) if gate_names else "all",
        "datasets": rows,
        "recommendation": recommendation(
            rows,
            args.low_coverage_action,
            gate_names,
            teacher_gate_passes=contract["passes"],
        ),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
