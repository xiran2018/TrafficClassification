#!/usr/bin/env python3
"""Audit candidate experts across ready-made train/valid/test folds.

Each fold has its own validation split and a shared test split. This script
compares same-named candidate probability JSONs against a fold-specific base and
reports whether the candidate is consistently helpful across folds instead of
only overfitting one validation set.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from validation_gated_selector import (
    align_payload,
    compute_metrics,
    load_label_names,
    load_payload,
    target_shift_guard,
)


def parse_fold_arg(values: List[str]) -> Tuple[str, str, Dict[str, str]]:
    if len(values) < 3:
        raise ValueError("--fold expects: FOLD_NAME BASE_JSON CANDIDATE=JSON ...")
    fold_name = values[0]
    base_json = values[1]
    candidates: Dict[str, str] = {}
    for raw in values[2:]:
        if "=" not in raw:
            raise ValueError(f"Candidate spec must be NAME=JSON, got: {raw}")
        name, path = raw.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"Empty candidate name in spec: {raw}")
        if name in candidates:
            raise ValueError(f"Duplicate candidate {name} in fold {fold_name}")
        candidates[name] = path
    return fold_name, base_json, candidates


def metric_value(metrics: Dict[str, float], metric: str) -> float:
    if metric == "accuracy":
        return float(metrics["accuracy"])
    if metric == "macro_f1":
        return float(metrics["macro_f1"])
    if metric == "target_margin":
        return min(float(metrics["accuracy"]), float(metrics["macro_f1"]))
    raise ValueError(metric)


def evaluate_candidate(
    base: Dict[str, Any],
    candidate: Dict[str, Any],
    metric: str,
) -> Dict[str, Any]:
    valid_common = sorted(set(map(str, base["valid_flow_ids"])) & set(map(str, candidate["valid_flow_ids"])))
    test_common = sorted(set(map(str, base["flow_ids"])) & set(map(str, candidate["flow_ids"])))
    if not valid_common or not test_common:
        raise ValueError("No common flow ids between base and candidate.")

    y_valid_base, base_valid_prob = align_payload(base, "valid", valid_common)
    y_valid_cand, cand_valid_prob = align_payload(candidate, "valid", valid_common)
    y_test_base, base_test_prob = align_payload(base, "test", test_common)
    y_test_cand, cand_test_prob = align_payload(candidate, "test", test_common)
    if not np.array_equal(y_valid_base, y_valid_cand) or not np.array_equal(y_test_base, y_test_cand):
        raise ValueError("Base and candidate labels do not align.")

    base_valid_pred = base_valid_prob.argmax(axis=1).astype(np.int64)
    cand_valid_pred = cand_valid_prob.argmax(axis=1).astype(np.int64)
    base_test_pred = base_test_prob.argmax(axis=1).astype(np.int64)
    cand_test_pred = cand_test_prob.argmax(axis=1).astype(np.int64)

    base_valid_metrics = compute_metrics(y_valid_base.tolist(), base_valid_pred.tolist())
    cand_valid_metrics = compute_metrics(y_valid_cand.tolist(), cand_valid_pred.tolist())
    base_test_metrics = compute_metrics(y_test_base.tolist(), base_test_pred.tolist())
    cand_test_metrics = compute_metrics(y_test_cand.tolist(), cand_test_pred.tolist())

    valid_gain = metric_value(cand_valid_metrics, metric) - metric_value(base_valid_metrics, metric)
    test_gain = metric_value(cand_test_metrics, metric) - metric_value(base_test_metrics, metric)
    return {
        "valid": {
            "base": base_valid_metrics,
            "candidate": cand_valid_metrics,
            "gain": valid_gain,
        },
        "test": {
            "base": base_test_metrics,
            "candidate": cand_test_metrics,
            "gain": test_gain,
        },
        "target_shift": target_shift_guard(cand_test_prob, base_test_prob),
        "counts": {"valid": len(valid_common), "test": len(test_common)},
    }


def summarize_candidate(rows: List[Dict[str, Any]], metric: str) -> Dict[str, Any]:
    valid_gains = [float(row["valid"]["gain"]) for row in rows]
    test_gains = [float(row["test"]["gain"]) for row in rows]
    test_scores = [metric_value(row["test"]["candidate"], metric) for row in rows]
    shifts = [float(row["target_shift"]["prediction_change_rate"]) for row in rows]
    return {
        "fold_count": len(rows),
        "valid_gain_mean": float(np.mean(valid_gains)),
        "valid_gain_min": float(np.min(valid_gains)),
        "valid_gain_win_rate": float(np.mean(np.asarray(valid_gains) > 0)),
        "test_gain_mean": float(np.mean(test_gains)),
        "test_gain_min": float(np.min(test_gains)),
        "test_gain_win_rate": float(np.mean(np.asarray(test_gains) > 0)),
        "test_metric_mean": float(np.mean(test_scores)),
        "test_metric_min": float(np.min(test_scores)),
        "target_change_mean": float(np.mean(shifts)),
        "target_change_max": float(np.max(shifts)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--fold",
        nargs="+",
        action="append",
        required=True,
        metavar="FOLD_SPEC",
        help="Fold spec. Repeat for each fold.",
    )
    ap.add_argument("--label_map", default="")
    ap.add_argument("--select_metric", choices=["accuracy", "macro_f1", "target_margin"], default="target_margin")
    ap.add_argument("--min_valid_gain", type=float, default=0.0)
    ap.add_argument("--min_valid_win_rate", type=float, default=1.0)
    ap.add_argument("--max_target_change", type=float, default=1.0)
    ap.add_argument("--rank_metric", choices=["valid_gain_min", "valid_gain_mean", "test_metric_min"], default="valid_gain_min")
    ap.add_argument("--output_json", required=True)
    args = ap.parse_args()

    fold_specs = [parse_fold_arg(values) for values in args.fold]
    common_names = set(fold_specs[0][2])
    for _, _, candidates in fold_specs[1:]:
        common_names &= set(candidates)
    if not common_names:
        raise ValueError("No candidate names are common to every fold.")

    fold_reports: Dict[str, Any] = {}
    candidate_rows: Dict[str, List[Dict[str, Any]]] = {name: [] for name in sorted(common_names)}
    for fold_name, base_json, candidates in fold_specs:
        base_payload = load_payload(base_json)
        fold_reports[fold_name] = {"base_json": base_json, "candidates": {}}
        for name in sorted(common_names):
            candidate_json = candidates[name]
            candidate_payload = load_payload(candidate_json)
            row = evaluate_candidate(base_payload, candidate_payload, args.select_metric)
            row["candidate_json"] = candidate_json
            fold_reports[fold_name]["candidates"][name] = row
            candidate_rows[name].append(row)

    summaries = {}
    accepted = []
    for name, rows in candidate_rows.items():
        summary = summarize_candidate(rows, args.select_metric)
        summary["accepted"] = (
            summary["valid_gain_min"] >= args.min_valid_gain
            and summary["valid_gain_win_rate"] >= args.min_valid_win_rate
            and summary["target_change_max"] <= args.max_target_change
        )
        summaries[name] = summary
        if summary["accepted"]:
            accepted.append(name)

    def rank_key(name: str) -> Tuple[float, float, float]:
        s = summaries[name]
        primary = float(s[args.rank_metric])
        return (primary, float(s["test_metric_min"]), float(s["valid_gain_mean"]))

    ranked = sorted(summaries, key=rank_key, reverse=True)
    selected = ranked[0] if ranked else None
    selected_accepted = selected if selected in accepted else None

    _, label_map = load_label_names(args.label_map)
    payload = {
        "selected": selected,
        "selected_accepted": selected_accepted,
        "ranked_candidates": ranked,
        "candidate_summaries": summaries,
        "fold_reports": fold_reports,
        "label_map": label_map,
        "config": {
            "select_metric": args.select_metric,
            "min_valid_gain": args.min_valid_gain,
            "min_valid_win_rate": args.min_valid_win_rate,
            "max_target_change": args.max_target_change,
            "rank_metric": args.rank_metric,
        },
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print("cross_fold_stability", json.dumps({
        "selected": selected,
        "selected_accepted": selected_accepted,
        "ranked_candidates": ranked,
        "candidate_summaries": summaries,
    }, indent=2, sort_keys=True))
    print(f"wrote {args.output_json}")


if __name__ == "__main__":
    main()
