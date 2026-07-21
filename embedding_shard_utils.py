#!/usr/bin/env python3
"""Deterministic flow-level sharding helpers for semantic extraction."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping


CONFIG_CONTRACT_KEYS = (
    "base_model",
    "lora_path",
    "tower1_heads",
    "embedding_mode",
    "batch_size",
    "flow_batch_packets",
    "scheduler",
    "max_length",
    "packet_index",
    "packet_index_header_policy",
    "packet_index_context_policy",
    "model_provenance",
)


def flow_shard_index(flow_id: str, num_shards: int) -> int:
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    digest = hashlib.sha1(flow_id.encode("utf-8", errors="ignore")).digest()
    return int.from_bytes(digest[:8], "big") % num_shards


def expected_embedding_shard_counts(packet_index: Path, num_shards: int) -> list[int]:
    counts = [0 for _ in range(num_shards)]
    seen_flows: set[str] = set()
    with packet_index.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            flow_id = str(json.loads(line)["flow_id"])
            if flow_id in seen_flows:
                continue
            seen_flows.add(flow_id)
            counts[flow_shard_index(flow_id, num_shards)] += 1
    return counts


def jsonl_row_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def merge_embedding_shards(
    *,
    packet_index: Path,
    output_dir: Path,
    num_shards: int,
    shard_devices: Iterable[str] = (),
) -> dict:
    """Merge complete deterministic shards and preserve their extraction contract."""
    shard_root = output_dir / "_shards"
    expected_counts = expected_embedding_shard_counts(packet_index, num_shards)
    output_dir.mkdir(parents=True, exist_ok=True)

    canonical_config: dict | None = None
    rows: list[str] = []
    seen_flows: set[str] = set()
    shard_summaries = []
    for shard_index, expected_count in enumerate(expected_counts):
        shard_dir = shard_root / f"shard_{shard_index}"
        index_path = shard_dir / "flow_embedding_index.jsonl"
        config_path = shard_dir / "embedding_config.json"
        if not index_path.is_file() or not config_path.is_file():
            raise FileNotFoundError(
                f"Missing embedding shard artifacts: shard={shard_index} dir={shard_dir}"
            )
        config = _load_json(config_path)
        if int(config.get("num_shards", -1)) != num_shards:
            raise ValueError(f"Shard {shard_index} num_shards does not match {num_shards}")
        if int(config.get("shard_index", -1)) != shard_index:
            raise ValueError(f"Shard directory {shard_index} contains a different shard_index")
        if Path(str(config.get("packet_index", ""))).resolve() != packet_index.resolve():
            raise ValueError(f"Shard {shard_index} packet_index does not match the merge input")

        contract = {key: config.get(key) for key in CONFIG_CONTRACT_KEYS}
        if canonical_config is None:
            canonical_config = contract
        elif contract != canonical_config:
            drift = [
                key for key in CONFIG_CONTRACT_KEYS
                if contract.get(key) != canonical_config.get(key)
            ]
            raise ValueError(f"Embedding shard configuration drift: shard={shard_index} keys={drift}")

        actual_count = 0
        with index_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                flow_id = str(row["flow_id"])
                assigned = flow_shard_index(flow_id, num_shards)
                if assigned != shard_index:
                    raise ValueError(
                        f"Flow {flow_id} belongs to shard {assigned}, found in {shard_index}"
                    )
                if flow_id in seen_flows:
                    raise ValueError(f"Duplicate flow across embedding shards: {flow_id}")
                embedding_path = Path(str(row.get("embedding_path", "")))
                if not embedding_path.is_file():
                    raise FileNotFoundError(
                        f"Missing embedding file for flow {flow_id}: {embedding_path}"
                    )
                seen_flows.add(flow_id)
                rows.append(json.dumps(row, ensure_ascii=False) + "\n")
                actual_count += 1
        if actual_count != expected_count:
            raise ValueError(
                f"Incomplete embedding shard {shard_index}: {actual_count}/{expected_count} flows"
            )
        shard_summaries.append(
            {
                "shard_index": shard_index,
                "expected_flows": expected_count,
                "actual_flows": actual_count,
                "config_path": str(config_path),
                "index_path": str(index_path),
            }
        )

    merged_index = output_dir / "flow_embedding_index.jsonl"
    temporary_index = output_dir / "flow_embedding_index.jsonl.tmp"
    with temporary_index.open("w", encoding="utf-8") as handle:
        handle.writelines(rows)
    temporary_index.replace(merged_index)

    merged_config = dict(canonical_config or {})
    merged_config.update(
        {
            "device": "multi_shard",
            "num_shards": num_shards,
            "shard_index": None,
            "shard_root": str(shard_root),
            "shard_devices": list(shard_devices),
            "merged_flows": len(rows),
            "merge_scheduler": "deterministic_flow_sha1_v1",
            "shards": shard_summaries,
        }
    )
    config_path = output_dir / "embedding_config.json"
    temporary_config = output_dir / "embedding_config.json.tmp"
    with temporary_config.open("w", encoding="utf-8") as handle:
        json.dump(merged_config, handle, indent=2, ensure_ascii=False)
    temporary_config.replace(config_path)
    return {
        "merged_index": str(merged_index),
        "config_path": str(config_path),
        "merged_flows": len(rows),
        "expected_shard_counts": expected_counts,
    }
