#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable


def load_jsonl(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def file_sha256(path: str, cache: Dict[str, str]) -> str:
    if path not in cache:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        cache[path] = digest.hexdigest()
    return cache[path]


def parse_environment_spec(spec: str) -> tuple[str, str]:
    if "=" not in spec:
        raise ValueError(f"Environment spec must be NAME=INDEX_PATH, got: {spec}")
    name, path = spec.split("=", 1)
    if not name.strip() or not path.strip():
        raise ValueError(f"Invalid environment spec: {spec}")
    return name.strip(), path.strip()


def main() -> None:
    ap = argparse.ArgumentParser(description="Map training flow IDs to source environments by exact PCAP content hash.")
    ap.add_argument("--train_index", required=True)
    ap.add_argument("--environment_index", action="append", required=True, help="Repeat NAME=flow_embedding_index.jsonl.")
    ap.add_argument("--output_json", required=True)
    ap.add_argument("--require_full_coverage", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    cache: Dict[str, str] = {}
    environment_names = []
    hash_to_environment: Dict[str, int] = {}
    reference_counts = Counter()
    for environment_id, spec in enumerate(args.environment_index):
        name, index_path = parse_environment_spec(spec)
        environment_names.append(name)
        for row in load_jsonl(index_path):
            digest = file_sha256(str(row["pcap_path"]), cache)
            previous = hash_to_environment.get(digest)
            if previous is not None and previous != environment_id:
                raise ValueError(
                    f"PCAP content occurs in both environments {environment_names[previous]} and {name}; "
                    "environment labels would be ambiguous."
                )
            hash_to_environment[digest] = environment_id
            reference_counts[environment_id] += 1

    flow_to_environment: Dict[str, int] = {}
    unmatched = []
    train_hashes = Counter()
    for row in load_jsonl(args.train_index):
        flow_id = str(row["flow_id"])
        digest = file_sha256(str(row["pcap_path"]), cache)
        train_hashes[digest] += 1
        if digest in hash_to_environment:
            flow_to_environment[flow_id] = hash_to_environment[digest]
        else:
            unmatched.append(flow_id)
    if args.require_full_coverage and unmatched:
        raise ValueError(f"{len(unmatched)} training flows have no environment match; first={unmatched[0]}")

    mapped_counts = Counter(flow_to_environment.values())
    payload = {
        "method": "exact_pcap_sha256",
        "train_index": args.train_index,
        "environment_specs": args.environment_index,
        "environment_names": environment_names,
        "flow_to_environment": flow_to_environment,
        "audit": {
            "training_rows": sum(train_hashes.values()),
            "unique_training_content": len(train_hashes),
            "duplicate_training_rows": sum(train_hashes.values()) - len(train_hashes),
            "mapped_flows": len(flow_to_environment),
            "unmatched_flows": len(unmatched),
            "mapped_per_environment": {
                environment_names[idx]: mapped_counts[idx] for idx in range(len(environment_names))
            },
            "reference_rows_per_environment": {
                environment_names[idx]: reference_counts[idx] for idx in range(len(environment_names))
            },
        },
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps(payload["audit"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
