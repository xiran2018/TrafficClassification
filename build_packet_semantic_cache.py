#!/usr/bin/env python3
"""Align per-flow Tower-1 embeddings to strict packet-index row order."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_embedding_index(path: str | Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            flow_id = str(row["flow_id"])
            if flow_id in rows:
                raise ValueError(f"duplicate semantic embedding flow_id={flow_id}")
            rows[flow_id] = row
    if not rows:
        raise ValueError("semantic embedding index is empty")
    return rows


def packet_keys(path: str | Path) -> list[tuple[str, int]]:
    keys: list[tuple[str, int]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            keys.append((str(row["flow_id"]), int(row["packet_id"])))
    if not keys:
        raise ValueError("packet index is empty")
    if len(set(keys)) != len(keys):
        raise ValueError("packet index contains duplicate (flow_id, packet_id) keys")
    return keys


class FlowEmbeddingCache:
    def __init__(self, index: dict[str, dict[str, Any]], capacity: int = 32) -> None:
        self.index = index
        self.capacity = int(max(1, capacity))
        self.cache: OrderedDict[str, tuple[np.ndarray, dict[int, int]]] = OrderedDict()

    def get(self, flow_id: str) -> tuple[np.ndarray, dict[int, int]]:
        cached = self.cache.pop(flow_id, None)
        if cached is not None:
            self.cache[flow_id] = cached
            return cached
        row = self.index.get(flow_id)
        if row is None:
            raise ValueError(f"semantic embedding index is missing flow_id={flow_id}")
        values = np.load(row["embedding_path"], mmap_mode="r")
        metas = row.get("packet_metas") or []
        if len(metas) != len(values):
            raise ValueError(
                f"semantic packet count mismatch for flow_id={flow_id}: "
                f"embedding={len(values)} metadata={len(metas)}"
            )
        ids = [int(meta.get("packet_id", index)) for index, meta in enumerate(metas)]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate semantic packet_id for flow_id={flow_id}")
        cached = (values, {packet_id: index for index, packet_id in enumerate(ids)})
        self.cache[flow_id] = cached
        while len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
        return cached


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet_index", required=True)
    parser.add_argument("--flow_embedding_index", required=True)
    parser.add_argument("--output_npy", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--cache_flows", type=int, default=32)
    parser.add_argument("--no_progress", action="store_true")
    args = parser.parse_args()

    keys = packet_keys(args.packet_index)
    embedding_rows = load_embedding_index(args.flow_embedding_index)
    first_row = next(iter(embedding_rows.values()))
    first_values = np.load(first_row["embedding_path"], mmap_mode="r")
    if first_values.ndim != 2 or first_values.shape[1] <= 0:
        raise ValueError("semantic embeddings must have shape [packets, dimension]")
    dimension = int(first_values.shape[1])
    output_path = Path(args.output_npy)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(keys), dimension),
    )
    cache = FlowEmbeddingCache(embedding_rows, args.cache_flows)
    used_flows: set[str] = set()
    for row_index, (flow_id, packet_id) in enumerate(
        tqdm(keys, desc="align packet semantic embeddings", disable=args.no_progress)
    ):
        values, positions = cache.get(flow_id)
        if int(values.shape[1]) != dimension:
            raise ValueError(
                f"semantic dimension mismatch for flow_id={flow_id}: "
                f"observed={values.shape[1]} expected={dimension}"
            )
        position = positions.get(packet_id)
        if position is None:
            raise ValueError(
                f"semantic embedding is missing packet_id={packet_id} for flow_id={flow_id}"
            )
        output[row_index] = values[position]
        used_flows.add(flow_id)
    output.flush()

    embedding_config_path = Path(args.flow_embedding_index).parent / "embedding_config.json"
    embedding_config = {}
    if embedding_config_path.exists():
        with open(embedding_config_path, "r", encoding="utf-8") as handle:
            embedding_config = json.load(handle)
    manifest = {
        "version": 1,
        "scope": "strict_current_packet_row_aligned_semantic_embedding",
        "packet_index": args.packet_index,
        "packet_index_sha256": sha256_file(args.packet_index),
        "flow_embedding_index": args.flow_embedding_index,
        "flow_embedding_index_sha256": sha256_file(args.flow_embedding_index),
        "output_npy": str(output_path),
        "num_packets": len(keys),
        "num_flows": len(used_flows),
        "embedding_dim": dimension,
        "embedding_mode": first_row.get("embedding_mode"),
        "embedding_scheduler": embedding_config.get("scheduler", "unknown"),
        "embedding_batch_size": embedding_config.get("batch_size"),
        "embedding_flow_batch_packets": embedding_config.get("flow_batch_packets"),
        "embedding_num_shards": embedding_config.get("num_shards", 1),
        "embedding_merge_scheduler": embedding_config.get("merge_scheduler", "none"),
        "header_policy": embedding_config.get("packet_index_header_policy", "unknown"),
        "packet_context_policy": embedding_config.get(
            "packet_index_context_policy", "unknown"
        ),
    }
    manifest_path = Path(args.output_json)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
