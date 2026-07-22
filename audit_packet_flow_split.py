#!/usr/bin/env python3
"""Audit reconstructed flow IDs and packet distributions across packet splits."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def load_summary(path: str) -> dict:
    flow_ids = set()
    packets_by_class = Counter()
    flows_by_class: dict[str, set[str]] = {}
    packets = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            flow_id = str(row["flow_id"])
            label = str(row["label"])
            flow_ids.add(flow_id)
            flows_by_class.setdefault(label, set()).add(flow_id)
            packets_by_class[label] += 1
            packets += 1
    return {
        "path": path,
        "packets": packets,
        "flows": len(flow_ids),
        "flow_ids": flow_ids,
        "packets_by_class": dict(sorted(packets_by_class.items())),
        "flows_by_class": {label: len(ids) for label, ids in sorted(flows_by_class.items())},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--valid", required=True)
    ap.add_argument("--test", default="")
    ap.add_argument("--output_json", default="")
    ap.add_argument("--fail_on_overlap", action="store_true")
    args = ap.parse_args()

    split_paths = [("train", args.train), ("valid", args.valid)]
    if args.test:
        split_paths.append(("test", args.test))
    summaries = {name: load_summary(path) for name, path in split_paths}
    overlaps = {}
    pairs = [("train", "valid")]
    if "test" in summaries:
        pairs.extend([("train", "test"), ("valid", "test")])
    for left, right in pairs:
        shared = summaries[left]["flow_ids"] & summaries[right]["flow_ids"]
        overlaps[f"{left}_{right}"] = {"count": len(shared), "examples": sorted(shared)[:20]}
    output = {
        "protocol": "per-flow-split_packet-level-classification",
        "splits": {
            name: {key: value for key, value in summary.items() if key != "flow_ids"}
            for name, summary in summaries.items()
        },
        "flow_overlap": overlaps,
        "valid": all(item["count"] == 0 for item in overlaps.values()),
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
    if args.fail_on_overlap and not output["valid"]:
        raise SystemExit("reconstructed flow overlap detected")


if __name__ == "__main__":
    main()
