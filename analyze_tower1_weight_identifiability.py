#!/usr/bin/env python3
"""Relate Tower-1 weighting outcomes to packet identifiability on validation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scipy.stats import spearmanr


def rank_association(values: list[float], outcomes: list[float]) -> dict[str, float]:
    result = spearmanr(values, outcomes)
    return {"rho": float(result.statistic), "p_value": float(result.pvalue)}


def build_report(
    matched_report: dict[str, Any],
    identifiability_report: dict[str, Any],
    sampling_report: dict[str, Any],
    label_map: dict[str, int],
    signature_level: str = "session",
    weight_strength: str = "0.5",
) -> dict[str, Any]:
    comparison = matched_report.get("matched_comparison") or {}
    latest = comparison.get("latest_matched") or {}
    class_deltas = latest.get("per_class") or []
    if not class_deltas:
        raise ValueError("matched report does not contain latest per-class deltas")

    levels = identifiability_report.get("levels") or {}
    if signature_level not in levels:
        raise ValueError(f"identifiability report has no level={signature_level!r}")
    level = levels[signature_level]
    train_conflicts = (level.get("train") or {}).get("per_class") or {}
    valid_conflicts = (level.get("test") or {}).get("per_class") or {}
    weights = (sampling_report.get("flow_count_weights") or {}).get(weight_strength)
    if not weights:
        raise ValueError(f"sampling report has no flow-count weight strength={weight_strength!r}")

    inverse_labels = {int(label_id): str(name) for name, label_id in label_map.items()}
    if len(inverse_labels) != len(label_map):
        raise ValueError("label map contains duplicate label IDs")
    deltas_by_name = {str(row["label"]): row for row in class_deltas}
    if set(deltas_by_name) != set(label_map):
        raise ValueError("matched per-class labels do not align with label map")

    rows = []
    for label_id, class_name in sorted(inverse_labels.items()):
        key = str(label_id)
        if key not in train_conflicts or key not in valid_conflicts or key not in weights:
            raise ValueError(f"missing identifiability/weight evidence for label_id={label_id}")
        delta = deltas_by_name[class_name]
        rows.append(
            {
                "class": class_name,
                "label_id": label_id,
                "flow_count_weight": float(weights[key]),
                "train_conflicting_sample_rate": float(
                    train_conflicts[key]["conflicting_sample_rate"]
                ),
                "validation_conflicting_sample_rate": float(
                    valid_conflicts[key]["conflicting_sample_rate"]
                ),
                "validation_support": int(delta["support"]),
                "f1_delta": float(delta["f1_delta"]),
                "recall_delta": float(delta["recall_delta"]),
            }
        )

    f1_delta = [row["f1_delta"] for row in rows]
    return {
        "scope": "heldout_validation_weight_identifiability_association",
        "selection_role": "descriptive_only",
        "causal_claim": False,
        "matched_step": int(latest["step"]),
        "signature_level": signature_level,
        "flow_count_weight_strength": weight_strength,
        "num_classes": len(rows),
        "association_with_f1_delta": {
            "flow_count_weight": rank_association(
                [row["flow_count_weight"] for row in rows], f1_delta
            ),
            "train_conflicting_sample_rate": rank_association(
                [row["train_conflicting_sample_rate"] for row in rows], f1_delta
            ),
            "validation_conflicting_sample_rate": rank_association(
                [row["validation_conflicting_sample_rate"] for row in rows], f1_delta
            ),
        },
        "classes": rows,
        "interpretation_guard": (
            "Associations may motivate a pre-registered identifiability-aware risk "
            "ablation, but cannot establish causality or promote a model."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matched_report", required=True)
    parser.add_argument("--identifiability_report", required=True)
    parser.add_argument("--sampling_report", required=True)
    parser.add_argument("--report_index", type=int, default=0)
    parser.add_argument("--label_map", required=True)
    parser.add_argument("--signature_level", default="session")
    parser.add_argument("--weight_strength", default="0.5")
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    matched = json.loads(Path(args.matched_report).read_text(encoding="utf-8"))
    identifiability = json.loads(
        Path(args.identifiability_report).read_text(encoding="utf-8")
    )
    sampling_payload = json.loads(Path(args.sampling_report).read_text(encoding="utf-8"))
    reports = sampling_payload["reports"]
    if not 0 <= args.report_index < len(reports):
        raise IndexError("--report_index is outside the sampling report")
    label_map = json.loads(Path(args.label_map).read_text(encoding="utf-8"))
    result = build_report(
        matched,
        identifiability,
        reports[args.report_index],
        label_map,
        signature_level=args.signature_level,
        weight_strength=args.weight_strength,
    )
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["association_with_f1_delta"], sort_keys=True))


if __name__ == "__main__":
    main()
