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


def merge_items(
    input_paths: list[str],
    excluded_flow_ids: set[str],
    flow_content: dict[str, str],
    excluded_content_hashes: set[str],
    dedupe_content: bool,
) -> tuple[list[dict], dict[str, int]]:
    merged = []
    seen_windows = set()
    content_owner: dict[str, str] = {}
    content_label: dict[str, int] = {}
    stats = {
        "content_excluded_windows": 0,
        "content_duplicate_flows": 0,
        "content_duplicate_windows": 0,
        "missing_content_hash_windows": 0,
        "label_conflicts": 0,
    }
    duplicate_flows = set()
    for path in input_paths:
        for item in load_pt(path):
            flow_id = str(item.get("flow_id", ""))
            if flow_id in excluded_flow_ids:
                continue
            content_hash = flow_content.get(flow_id, "")
            if excluded_content_hashes:
                if not content_hash:
                    stats["missing_content_hash_windows"] += 1
                elif content_hash in excluded_content_hashes:
                    stats["content_excluded_windows"] += 1
                    continue
            if dedupe_content:
                if not content_hash:
                    stats["missing_content_hash_windows"] += 1
                    continue
                label = int(item.get("label", -1))
                owner = content_owner.get(content_hash)
                if owner is None:
                    content_owner[content_hash] = flow_id
                    content_label[content_hash] = label
                elif content_label[content_hash] != label:
                    stats["label_conflicts"] += 1
                    raise ValueError(
                        f"Content hash {content_hash} has conflicting labels "
                        f"{content_label[content_hash]} and {label}."
                    )
                elif owner != flow_id:
                    duplicate_flows.add(flow_id)
                    stats["content_duplicate_windows"] += 1
                    continue
            window = tuple(item.get("window", ()))
            key = (flow_id, window, int(item.get("label", -1)))
            if key in seen_windows:
                continue
            seen_windows.add(key)
            merged.append(item)
    stats["content_duplicate_flows"] = len(duplicate_flows)
    return merged, stats


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
    ap.add_argument(
        "--dedupe_content",
        action="store_true",
        help="Keep all windows for the first flow with each PCAP SHA256 and drop later content-identical flow IDs.",
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

    if args.dedupe_content and not args.flow_embedding_indices:
        raise ValueError("--dedupe_content requires --flow_embedding_indices")
    merged, stats = merge_items(
        args.inputs,
        excluded_flow_ids,
        flow_content,
        excluded_content_hashes,
        args.dedupe_content,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, out)
    flow_count = len({str(item.get("flow_id", "")) for item in merged if int(item.get("label", -1)) >= 0})
    print(
        f"saved {len(merged)} windows from {flow_count} flows to {out}; "
        f"excluded_flows={len(excluded_flow_ids)}; "
        f"excluded_content_hashes={len(excluded_content_hashes)}; "
        f"content_excluded_windows={stats['content_excluded_windows']}; "
        f"content_duplicate_flows={stats['content_duplicate_flows']}; "
        f"content_duplicate_windows={stats['content_duplicate_windows']}; "
        f"missing_content_hash_windows={stats['missing_content_hash_windows']}; "
        f"label_conflicts={stats['label_conflicts']}"
    )


if __name__ == "__main__":
    main()
