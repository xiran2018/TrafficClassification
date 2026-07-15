#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def load_pt(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def sha256_file(path: str, cache: dict[str, str]) -> str:
    if path in cache:
        return cache[path]
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    digest = h.hexdigest()
    cache[path] = digest
    return digest


def flow_content_hashes(index_paths: list[str], cache: dict[str, str]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in index_paths:
        for row in load_jsonl(path):
            flow_id = str(row.get("flow_id", ""))
            pcap_path = str(row.get("pcap_path", ""))
            if flow_id and pcap_path and Path(pcap_path).exists():
                hashes[flow_id] = sha256_file(pcap_path, cache)
    return hashes


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
    ap.add_argument(
        "--flow_embedding_indices",
        nargs="*",
        default=[],
        help="Optional flow_embedding_index.jsonl files used to map flow_id to pcap content hashes.",
    )
    ap.add_argument(
        "--exclude_content_from_indices",
        nargs="*",
        default=[],
        help="Optional flow_embedding_index.jsonl files whose pcap SHA256 hashes are excluded from the merged output.",
    )
    args = ap.parse_args()

    excluded_flow_ids = set()
    for path in args.exclude_flow_ids_from:
        data = load_pt(path)
        for item in data:
            flow_id = str(item.get("flow_id", ""))
            if flow_id:
                excluded_flow_ids.add(flow_id)

    hash_cache: dict[str, str] = {}
    flow_content = flow_content_hashes(args.flow_embedding_indices, hash_cache)
    excluded_content_hashes = set(flow_content_hashes(args.exclude_content_from_indices, hash_cache).values())

    merged = []
    seen = set()
    content_excluded = 0
    missing_content_hash = 0
    for path in args.inputs:
        data = load_pt(path)
        for item in data:
            flow_id = str(item.get("flow_id", ""))
            if flow_id in excluded_flow_ids:
                continue
            if excluded_content_hashes:
                content_hash = flow_content.get(flow_id, "")
                if not content_hash:
                    missing_content_hash += 1
                elif content_hash in excluded_content_hashes:
                    content_excluded += 1
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
        f"excluded_flows={len(excluded_flow_ids)}; "
        f"excluded_content_hashes={len(excluded_content_hashes)}; "
        f"content_excluded_windows={content_excluded}; "
        f"missing_content_hash_windows={missing_content_hash}"
    )


if __name__ == "__main__":
    main()
