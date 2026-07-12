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
    args = ap.parse_args()

    merged = []
    seen = set()
    for path in args.inputs:
        data = load_pt(path)
        for item in data:
            flow_id = str(item.get("flow_id", ""))
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
    print(f"saved {len(merged)} windows from {flow_count} flows to {out}")


if __name__ == "__main__":
    main()
