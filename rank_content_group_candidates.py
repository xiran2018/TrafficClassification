#!/usr/bin/env python3
"""Rank prediction candidates by exact-content group bootstrap robustness."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np

from evaluate_content_unique_predictions import (
    flow_content_map,
    group_indices_by_content,
    metrics,
)


REQUIRED_FIELDS = ("flow_ids", "flow_y_true", "flow_prob")


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def percentile_lower(ci: Any) -> float:
    if isinstance(ci, list) and len(ci) == 2 and ci[0] is not None:
        return float(ci[0])
    return float("-inf")


def fast_metrics_from_pred(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> tuple[float, float]:
    if len(y_true) == 0:
        return 0.0, 0.0
    accuracy = float((y_true == y_pred).mean())
    encoded = y_true.astype(np.int64) * int(num_classes) + y_pred.astype(np.int64)
    confusion = np.bincount(encoded, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
    tp = np.diag(confusion).astype(np.float64)
    true_support = confusion.sum(axis=1).astype(np.float64)
    pred_support = confusion.sum(axis=0).astype(np.float64)
    denom = true_support + pred_support
    observed = denom > 0
    f1 = np.zeros(num_classes, dtype=np.float64)
    f1[observed] = (2.0 * tp[observed]) / np.maximum(denom[observed], 1e-12)
    macro_f1 = float(f1[observed].mean()) if observed.any() else 0.0
    return accuracy, macro_f1


def fast_content_group_bootstrap_ci(
    y_true: np.ndarray,
    prob: np.ndarray,
    groups: List[List[int]],
    samples: int,
    seed: int,
) -> Dict[str, Any]:
    if samples <= 0 or not groups:
        return {}
    rng = np.random.default_rng(seed)
    group_arrays = [np.asarray(indices, dtype=np.int64) for indices in groups]
    y_pred = prob.argmax(axis=1).astype(np.int64)
    num_classes = int(prob.shape[1])
    accuracy = np.zeros(samples, dtype=np.float64)
    macro_f1 = np.zeros(samples, dtype=np.float64)
    for sample_idx in range(samples):
        sampled_groups = rng.integers(0, len(group_arrays), size=len(group_arrays))
        sampled_indices = np.concatenate([group_arrays[index] for index in sampled_groups])
        accuracy[sample_idx], macro_f1[sample_idx] = fast_metrics_from_pred(
            y_true[sampled_indices],
            y_pred[sampled_indices],
            num_classes,
        )
    return {
        "method": "cluster_bootstrap_by_exact_pcap_sha256_fast_numpy",
        "samples": int(samples),
        "seed": int(seed),
        "num_groups": len(group_arrays),
        "num_rows": int(len(y_true)),
        "accuracy_95_ci": np.percentile(accuracy, [2.5, 97.5]).astype(float).tolist(),
        "macro_f1_95_ci": np.percentile(macro_f1, [2.5, 97.5]).astype(float).tolist(),
    }


def candidate_paths(root: Path, patterns: Iterable[str], max_files: int) -> List[Path]:
    seen = set()
    paths: List[Path] = []
    for pattern in patterns:
        for path in sorted(root.glob(pattern)):
            if path in seen or not path.is_file():
                continue
            seen.add(path)
            paths.append(path)
            if max_files >= 0 and len(paths) >= max_files:
                return paths
    return paths


def evaluate_candidate(
    path: Path,
    *,
    flow_to_hash: Dict[str, str],
    samples: int,
    seed: int,
    target_accuracy: float | None,
    target_macro_f1: float | None,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {"path": str(path), "status": "pending"}
    try:
        data = load_json(path)
    except Exception as exc:
        row.update({"status": "invalid_json", "error": str(exc)})
        return row
    missing = [field for field in REQUIRED_FIELDS if field not in data]
    if missing:
        row.update({"status": "missing_fields", "missing": missing})
        return row
    try:
        flow_ids = [str(value) for value in data["flow_ids"]]
        y_true = np.asarray(data["flow_y_true"], dtype=np.int64)
        prob = np.asarray(data["flow_prob"], dtype=np.float64)
        _, groups = group_indices_by_content(flow_ids, flow_to_hash)
        base = metrics(y_true, prob.argmax(axis=1))
        group_ci = fast_content_group_bootstrap_ci(y_true, prob, groups, samples, seed)
    except Exception as exc:
        row.update({"status": "compute_failed", "error": str(exc)})
        return row
    acc_ci = group_ci.get("accuracy_95_ci")
    f1_ci = group_ci.get("macro_f1_95_ci")
    target_met = bool(
        (target_accuracy is None or percentile_lower(acc_ci) >= float(target_accuracy))
        and (target_macro_f1 is None or percentile_lower(f1_ci) >= float(target_macro_f1))
    )
    row.update(
        {
            "status": "ok",
            "accuracy": base["accuracy"],
            "macro_f1": base["macro_f1"],
            "content_group_accuracy_ci95": acc_ci,
            "content_group_macro_f1_ci95": f1_ci,
            "content_group_count": group_ci.get("num_groups"),
            "content_group_rows": group_ci.get("num_rows"),
            "target_accuracy": target_accuracy,
            "target_macro_f1": target_macro_f1,
            "content_group_target_met": target_met,
        }
    )
    return row


def rank_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            row.get("status") == "ok",
            percentile_lower(row.get("content_group_accuracy_ci95")),
            percentile_lower(row.get("content_group_macro_f1_ci95")),
            float(row.get("accuracy") or float("-inf")),
            float(row.get("macro_f1") or float("-inf")),
        ),
        reverse=True,
    )


def render_markdown(payload: Dict[str, Any]) -> str:
    lines = [
        "# Content-Group Candidate Ranking",
        "",
        f"Dataset: `{payload['dataset']}`",
        "",
        f"Candidates scanned: `{payload['num_candidates']}`",
        f"OK candidates: `{payload['num_ok']}`",
        "",
        "| Rank | Status | Acc | Macro-F1 | Group Acc 95% CI | Group F1 95% CI | Target | Path |",
        "|---:|---|---:|---:|---|---|---|---|",
    ]
    for idx, row in enumerate(payload.get("rows", [])[: payload.get("top_k", 20)], 1):
        acc_ci = row.get("content_group_accuracy_ci95")
        f1_ci = row.get("content_group_macro_f1_ci95")
        lines.append(
            "| {idx} | {status} | {acc} | {f1} | {acc_ci} | {f1_ci} | {target} | `{path}` |".format(
                idx=idx,
                status=row.get("status"),
                acc=fmt(row.get("accuracy")),
                f1=fmt(row.get("macro_f1")),
                acc_ci=(
                    "-"
                    if not isinstance(acc_ci, list)
                    else f"[{fmt(acc_ci[0])}, {fmt(acc_ci[1])}]"
                ),
                f1_ci=(
                    "-"
                    if not isinstance(f1_ci, list)
                    else f"[{fmt(f1_ci[0])}, {fmt(f1_ci[1])}]"
                ),
                target=row.get("content_group_target_met", "-"),
                path=row.get("path"),
            )
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Rank test prediction JSONs by content-group bootstrap robustness.")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--root", default="")
    ap.add_argument("--prediction_glob", action="append", default=["test*.json"])
    ap.add_argument("--flow_embedding_index", required=True)
    ap.add_argument("--target_accuracy", type=float, default=None)
    ap.add_argument("--target_macro_f1", type=float, default=None)
    ap.add_argument("--bootstrap_samples", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--max_files", type=int, default=-1)
    ap.add_argument("--output_json", required=True)
    ap.add_argument("--output_md", required=True)
    args = ap.parse_args()

    root = Path(args.root or Path("reasoningDataset") / args.dataset)
    flow_to_hash, _ = flow_content_map(args.flow_embedding_index)
    paths = candidate_paths(root, args.prediction_glob, args.max_files)
    rows = [
        evaluate_candidate(
            path,
            flow_to_hash=flow_to_hash,
            samples=args.bootstrap_samples,
            seed=args.seed,
            target_accuracy=args.target_accuracy,
            target_macro_f1=args.target_macro_f1,
        )
        for path in paths
    ]
    rows = rank_rows(rows)
    payload = {
        "dataset": args.dataset,
        "root": str(root),
        "prediction_glob": args.prediction_glob,
        "flow_embedding_index": args.flow_embedding_index,
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "target_accuracy": args.target_accuracy,
        "target_macro_f1": args.target_macro_f1,
        "top_k": args.top_k,
        "num_candidates": len(rows),
        "num_ok": sum(1 for row in rows if row.get("status") == "ok"),
        "rows": rows,
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    Path(args.output_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_md).write_text(render_markdown(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "dataset": args.dataset,
                "num_candidates": payload["num_candidates"],
                "num_ok": payload["num_ok"],
                "best": rows[0] if rows else None,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
