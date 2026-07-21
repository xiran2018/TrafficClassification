#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize(path: str | Path) -> dict:
    path = Path(path)
    rows = 0
    identity_rows = Counter()
    flow_identities: dict[str, set[str]] = defaultdict(set)
    flow_labels: dict[str, set[int]] = defaultdict(set)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            flow_id = str(row.get("flow_id") or "")
            packet_uid = str(row.get("packet_uid") or "")
            if not flow_id or not packet_uid:
                raise ValueError(f"{path}:{line_number}: flow_id and packet_uid are required")
            rows += 1
            identity_rows[packet_uid] += 1
            flow_identities[flow_id].add(packet_uid)
            flow_labels[flow_id].add(int(row["label_id"]))

    conflicting = sorted(flow for flow, labels in flow_labels.items() if len(labels) != 1)
    if conflicting:
        raise ValueError(f"flows have conflicting labels: {conflicting[:5]}")
    identity_counts = [len(values) for values in flow_identities.values()]
    eligible_flows = sum(count >= 2 for count in identity_counts)
    eligible_identities = sum(count for count in identity_counts if count >= 2)
    unique_identities = len(identity_rows)
    flows = len(flow_identities)
    return {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "rows": rows,
        "unique_packet_identities": unique_identities,
        "duplicate_identity_rows": rows - unique_identities,
        "flows": flows,
        "singleton_flows": flows - eligible_flows,
        "context_eligible_flows": eligible_flows,
        "context_eligible_flow_rate": eligible_flows / max(1, flows),
        "context_eligible_packet_identities": eligible_identities,
        "context_eligible_packet_identity_rate": eligible_identities
        / max(1, unique_identities),
        "equal_flow_sampler_context_rate": eligible_flows / max(1, flows),
        "min_packets_per_flow": min(identity_counts, default=0),
        "median_packets_per_flow": sorted(identity_counts)[len(identity_counts) // 2]
        if identity_counts
        else 0,
        "max_packets_per_flow": max(identity_counts, default=0),
    }


def parse_named_input(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--input must be NAME=PATH")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("--input must be NAME=PATH")
    return name, path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", type=parse_named_input, required=True)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()
    names = [name for name, _ in args.input]
    if len(names) != len(set(names)):
        parser.error("--input names must be unique")
    payload = {
        "schema": "cross_scale_context_availability_v1",
        "identity_policy": "exact_packet_uid",
        "context_policy": "distinct_same_flow_packet_leave_one_out",
        "inputs": {name: summarize(path) for name, path in args.input},
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
