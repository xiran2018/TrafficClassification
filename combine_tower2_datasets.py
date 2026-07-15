#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def load_pt(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="Input Tower2 .pt dataset files.")
    ap.add_argument("--output", required=True, help="Output merged .pt dataset file.")
    ap.add_argument(
        "--exclude_flow_ids_from",
        nargs="*",
        default=[],
        help="Optional Tower2 .pt datasets whose flow_id values are excluded from the merged output.",
    )
    args = ap.parse_args()

    excluded_flow_ids = set()
    for path in args.exclude_flow_ids_from:
        data = load_pt(path)
        for item in data:
            flow_id = str(item.get("flow_id", ""))
            if flow_id:
                excluded_flow_ids.add(flow_id)

    merged = []
    seen = set()
    for path in args.inputs:
        data = load_pt(path)
        for item in data:
            flow_id = str(item.get("flow_id", ""))
            if flow_id in excluded_flow_ids:
                continue
            window = tuple(item.get("window", ()))
            key = (flow_id, window, int(item.get("label", -1)))
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, out)
    flow_count = len({str(item.get("flow_id", "")) for item in merged if int(item.get("label", -1)) >= 0})
    print(
        f"saved {len(merged)} windows from {flow_count} flows to {out}; "
        f"excluded_flows={len(excluded_flow_ids)}"
    )


if __name__ == "__main__":
    main()
