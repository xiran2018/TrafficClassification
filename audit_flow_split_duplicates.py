#!/usr/bin/env python3
"""Audit exact and endpoint-invariant duplicate flows across dataset splits.

The goal is not to prove semantic equivalence perfectly. It gives a fast,
repeatable warning signal when train/valid/test or train_val_split_* folders
share identical files or endpoint-invariant packet patterns.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from tqdm import tqdm

from traffic_utils import extract_flow_packets, iter_labeled_pcaps, stable_id


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def digest_obj(obj: Any) -> str:
    raw = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8", errors="ignore")).hexdigest()


def split_dirs(root: Path, include_test: bool) -> List[Tuple[str, Path]]:
    out: List[Tuple[str, Path]] = []
    for split_root in sorted(root.glob("train_val_split_*")):
        if not split_root.is_dir():
            continue
        for subset in ("train", "val"):
            p = split_root / subset
            if p.exists():
                out.append((f"{split_root.name}/{subset}", p))
    if include_test and (root / "test").exists():
        out.append(("test", root / "test"))
    return out


def iat_bin(value: float) -> str:
    if value <= 0:
        return "0"
    if value < 0.001:
        return "<1ms"
    if value < 0.01:
        return "<10ms"
    if value < 0.1:
        return "<100ms"
    if value < 1.0:
        return "<1s"
    return ">=1s"


def prompt_payload_prefix(prompt: str) -> str:
    for line in prompt.splitlines():
        if line.startswith("PayloadPrefix:"):
            return line.split(":", 1)[1].strip()
    return ""


def flow_signatures(
    path: Path,
    max_packets: int,
    payload_prefix_len: int,
    l3_prefix_len: int,
) -> Dict[str, str | int]:
    metas, _, prompts = extract_flow_packets(
        path,
        max_packets=max_packets,
        payload_prefix_len=payload_prefix_len,
        l3_prefix_len=l3_prefix_len,
        embedding_header_policy="mask_ip_port",
    )
    behavior_seq = []
    endpoint_invariant_seq = []
    payload_seq = []
    for meta, prompt in zip(metas, prompts):
        payload = prompt_payload_prefix(prompt)
        payload_seq.append(payload)
        behavior_seq.append(
            {
                "dir": meta.direction,
                "l4": meta.l4,
                "pkt_len": meta.packet_len,
                "l3_len": meta.l3_captured_len,
                "payload_len": meta.payload_len,
                "payload_prefix_len": meta.payload_prefix_len,
                "payload_truncated": meta.payload_truncated,
                "entropy_bin": round(float(meta.payload_entropy), 1),
                "flags": meta.tcp_flags,
                "iat_bin": iat_bin(float(meta.iat)),
            }
        )
        endpoint_invariant_seq.append(
            {
                "dir": meta.direction,
                "l4": meta.l4,
                "ip_ttl": meta.ip_ttl,
                "ip_total_len": meta.ip_total_len,
                "ip_header_len": meta.ip_header_len,
                "pkt_len": meta.packet_len,
                "payload_len": meta.payload_len,
                "udp_len": meta.udp_len,
                "tcp_flags": meta.tcp_flags,
                "tcp_window": meta.tcp_window,
                "tcp_data_offset": meta.tcp_data_offset,
                "payload": payload,
            }
        )
    return {
        "packet_count": len(metas),
        "file_sha256": sha256_file(path),
        "behavior_hash": digest_obj(behavior_seq),
        "endpoint_invariant_hash": digest_obj(endpoint_invariant_seq),
        "payload_prefix_hash": digest_obj(payload_seq),
    }


def flow_records(
    root: Path,
    include_test: bool,
    max_packets: int,
    payload_prefix_len: int,
    l3_prefix_len: int,
    show_progress: bool,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    dirs = split_dirs(root, include_test)
    all_items: List[Tuple[str, str, Path]] = []
    for partition, directory in dirs:
        for label, path in iter_labeled_pcaps(directory):
            all_items.append((partition, label, path))
    iterator: Iterable[Tuple[str, str, Path]] = all_items
    if show_progress:
        iterator = tqdm(all_items, desc=f"audit {root.name}", unit="flow")
    for partition, label, path in iterator:
        rec: Dict[str, Any] = {
            "dataset": root.name,
            "partition": partition,
            "label": label,
            "path": str(path),
            "flow_id": stable_id(str(path.resolve())),
        }
        try:
            rec.update(flow_signatures(path, max_packets, payload_prefix_len, l3_prefix_len))
        except Exception as exc:  # keep the audit robust on malformed pcaps
            rec["error"] = str(exc)
        records.append(rec)
    return records


def summarize_key(records: List[Dict[str, Any]], key: str, sample_groups: int, max_members: int) -> Dict[str, Any]:
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rec in records:
        value = rec.get(key)
        if value:
            buckets[str(value)].append(rec)
    dup_groups = [items for items in buckets.values() if len(items) > 1]

    def cross(items: List[Dict[str, Any]], field: str) -> bool:
        return len({str(item.get(field, "")) for item in items}) > 1

    cross_partition = [items for items in dup_groups if cross(items, "partition")]
    cross_label = [items for items in dup_groups if cross(items, "label")]
    cross_split_root = [
        items
        for items in dup_groups
        if len({str(item.get("partition", "")).split("/", 1)[0] for item in items}) > 1
    ]
    sample = []
    for items in cross_partition[:sample_groups]:
        sample.append(
            [
                {
                    "partition": item.get("partition"),
                    "label": item.get("label"),
                    "flow_id": item.get("flow_id"),
                    "packet_count": item.get("packet_count"),
                    "path": item.get("path"),
                }
                for item in items[:max_members]
            ]
        )
    return {
        "key": key,
        "duplicate_groups": len(dup_groups),
        "duplicate_flows": sum(len(items) for items in dup_groups),
        "cross_partition_groups": len(cross_partition),
        "cross_split_root_groups": len(cross_split_root),
        "cross_label_groups": len(cross_label),
        "sample_cross_partition_groups": sample,
    }


def partition_counts(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    errors = 0
    for rec in records:
        counts[str(rec.get("partition", ""))][str(rec.get("label", ""))] += 1
        if rec.get("error"):
            errors += 1
    return {
        "total_flows": len(records),
        "parse_errors": errors,
        "by_partition": {k: dict(v) for k, v in sorted(counts.items())},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Dataset root with train_val_split_* and optional test.")
    ap.add_argument("--output_json", required=True)
    ap.add_argument("--include_test", action="store_true", help="Also audit the shared test folder.")
    ap.add_argument("--max_packets", type=int, default=64)
    ap.add_argument("--payload_prefix_len", type=int, default=64)
    ap.add_argument("--l3_prefix_len", type=int, default=256)
    ap.add_argument("--sample_groups", type=int, default=10)
    ap.add_argument("--max_members_per_group", type=int, default=8)
    ap.add_argument("--no_progress", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    records = flow_records(
        root,
        include_test=args.include_test,
        max_packets=args.max_packets,
        payload_prefix_len=args.payload_prefix_len,
        l3_prefix_len=args.l3_prefix_len,
        show_progress=not args.no_progress,
    )
    keys = ["file_sha256", "endpoint_invariant_hash", "payload_prefix_hash", "behavior_hash"]
    payload = {
        "config": {
            "root": str(root),
            "include_test": args.include_test,
            "max_packets": args.max_packets,
            "payload_prefix_len": args.payload_prefix_len,
            "l3_prefix_len": args.l3_prefix_len,
        },
        "counts": partition_counts(records),
        "duplicate_summary": {
            key: summarize_key(records, key, args.sample_groups, args.max_members_per_group) for key in keys
        },
        "records_with_errors": [rec for rec in records if rec.get("error")][: args.sample_groups],
    }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    compact = {
        key: {
            name: value
            for name, value in row.items()
            if name != "sample_cross_partition_groups"
        }
        for key, row in payload["duplicate_summary"].items()
    }
    print(json.dumps({"output_json": str(out), "counts": payload["counts"], "duplicate_summary": compact}, indent=2))


if __name__ == "__main__":
    main()
