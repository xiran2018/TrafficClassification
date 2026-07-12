#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from sklearn.metrics import accuracy_score, f1_score


DEFAULT_TARGETS = {
    "vpn-app": (0.74, 0.65),
    "tls-120": (0.78, 0.70),
    "tls": (0.78, 0.70),
}


def parse_target(raw: str) -> Tuple[str, float, float]:
    parts = raw.split(":")
    if len(parts) != 3:
        raise ValueError("--target expects DATASET:ACC:MACRO_F1")
    return parts[0], float(parts[1]), float(parts[2])


def metric_from_payload(data: Dict[str, Any]) -> Tuple[float | None, float | None]:
    nested = data.get("metrics")
    if isinstance(nested, dict):
        flow_level = nested.get("flow_level")
        if isinstance(flow_level, dict):
            acc = flow_level.get("accuracy") or flow_level.get("flow_acc")
            f1 = flow_level.get("macro_f1") or flow_level.get("flow_macro_f1")
            if acc is not None:
                return float(acc), float(f1) if f1 is not None else None
    for acc_key in ("flow_acc", "accuracy", "acc"):
        if acc_key in data:
            f1 = data.get("flow_macro_f1", data.get("macro_f1"))
            return float(data[acc_key]), float(f1) if f1 is not None else None
    y_true = data.get("flow_y_true")
    y_pred = data.get("flow_y_pred")
    if y_pred is None and "flow_prob" in data:
        y_pred = [max(range(len(row)), key=lambda i: row[i]) for row in data["flow_prob"]]
    if y_true is None or y_pred is None:
        return None, None
    if len(y_true) != len(y_pred) or not y_true:
        return None, None
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return float(acc), float(f1)


def iter_result_files(root: Path, patterns: Iterable[str]) -> Iterable[Path]:
    seen = set()
    for pattern in patterns:
        for path in root.glob(pattern):
            if path in seen or not path.is_file():
                continue
            seen.add(path)
            yield path


def collect_dataset(dataset: str, patterns: List[str]) -> List[Dict[str, Any]]:
    root = Path("reasoningDataset") / dataset
    rows = []
    if not root.exists():
        return rows
    for path in iter_result_files(root, patterns):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        acc, macro_f1 = metric_from_payload(data)
        if acc is None:
            continue
        rows.append(
            {
                "dataset": dataset,
                "path": str(path),
                "accuracy": acc,
                "macro_f1": macro_f1,
                "num_flows": len(data.get("flow_y_true", [])),
            }
        )
    rows.sort(key=lambda row: (row["accuracy"], row["macro_f1"] or -1.0, row["path"]), reverse=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", action="append", default=[], help="Dataset under reasoningDataset/. Can be repeated.")
    ap.add_argument("--target", action="append", default=[], help="Optional DATASET:ACC:MACRO_F1 target. Can be repeated.")
    ap.add_argument("--pattern", action="append", default=["test*.json"], help="Glob under each dataset directory. Can be repeated.")
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--output_json", default="")
    args = ap.parse_args()

    datasets = args.dataset or ["vpn-app", "tls-120", "ustc-app", "ustc-binary"]
    targets = DEFAULT_TARGETS.copy()
    for raw in args.target:
        dataset, acc, macro_f1 = parse_target(raw)
        targets[dataset] = (acc, macro_f1)

    summary = []
    for dataset in datasets:
        rows = collect_dataset(dataset, args.pattern)
        best = rows[0] if rows else None
        target = targets.get(dataset)
        target_acc, target_f1 = target if target else (None, None)
        achieved = False
        if best and target:
            achieved = best["accuracy"] >= target_acc and (best["macro_f1"] or 0.0) >= target_f1
        item = {
            "dataset": dataset,
            "target_accuracy": target_acc,
            "target_macro_f1": target_f1,
            "achieved": achieved if target else None,
            "best": best,
            "top_results": rows[: args.top_k],
            "num_results": len(rows),
        }
        summary.append(item)

        print(f"\n[{dataset}] results={len(rows)}")
        if target:
            status = "PASS" if achieved else "MISS"
            print(f"target acc>={target_acc:.4f} macro_f1>={target_f1:.4f}: {status}")
        if not rows:
            print("no metric JSON found")
            continue
        for rank, row in enumerate(rows[: args.top_k], start=1):
            f1 = row["macro_f1"]
            f1_text = "nan" if f1 is None else f"{f1:.6f}"
            print(f"{rank:02d} acc={row['accuracy']:.6f} macro_f1={f1_text} flows={row['num_flows']} {row['path']}")

    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump({"datasets": summary}, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
