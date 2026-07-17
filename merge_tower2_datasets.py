#!/usr/bin/env python3
"""Merge Tower-2 `.pt` datasets for final train+valid student training."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import torch


def load_pt(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def item_key(item: Dict[str, Any], fallback: str, mode: str) -> str:
    flow_id = str(item.get("flow_id", fallback))
    if mode == "flow":
        return flow_id
    if mode == "flow_label":
        return f"{flow_id}\t{int(item.get('label', -999999))}"
    if mode == "none":
        return fallback
    raise ValueError(mode)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", action="append", required=True, help="Input Tower-2 .pt dataset. Repeat for train/valid.")
    ap.add_argument("--output", required=True)
    ap.add_argument(
        "--dedupe",
        choices=["none", "flow", "flow_label"],
        default="none",
        help="Default keeps all windows/items. Use flow/flow_label only for already flow-level datasets.",
    )
    ap.add_argument("--manifest_json", default="")
    args = ap.parse_args()

    merged: List[dict] = []
    seen = set()
    reports = []
    duplicate_count = 0
    for path in args.input:
        data = load_pt(path)
        kept = 0
        duplicates = 0
        labels = Counter()
        flow_ids = set()
        for idx, item in enumerate(data):
            key = item_key(item, f"{path}:{idx}", args.dedupe)
            if args.dedupe != "none" and key in seen:
                duplicates += 1
                duplicate_count += 1
                continue
            seen.add(key)
            merged.append(item)
            kept += 1
            label = int(item.get("label", -1))
            if label >= 0:
                labels[label] += 1
            flow_ids.add(str(item.get("flow_id", f"{path}:{idx}")))
        reports.append({
            "path": path,
            "input_items": len(data),
            "kept_items": kept,
            "duplicates_skipped": duplicates,
            "unique_flows_seen": len(flow_ids),
            "label_counts": dict(sorted(labels.items())),
        })

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, output)
    manifest = {
        "inputs": reports,
        "output": str(output),
        "output_items": len(merged),
        "dedupe": args.dedupe,
        "duplicates_skipped_total": duplicate_count,
    }
    manifest_path = Path(args.manifest_json) if args.manifest_json else output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print("merged_tower2_dataset " + json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
