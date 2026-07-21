import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from embedding_shard_utils import flow_shard_index, merge_embedding_shards
from run_packet_level_pipeline import (
    semantic_embedding_audit_command,
    semantic_embedding_command,
)


def write_fixture(tmp_path: Path, *, num_shards: int = 2):
    packet_index = tmp_path / "packet_index.jsonl"
    flow_ids = ["flow-a", "flow-b", "flow-c", "flow-d"]
    packet_index.write_text(
        "".join(
            json.dumps({"flow_id": flow_id, "packet_id": 0}) + "\n"
            for flow_id in flow_ids
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "embeddings"
    common = {
        "base_model": "qwen",
        "lora_path": "adapter",
        "tower1_heads": "heads.pt",
        "embedding_mode": "concat",
        "batch_size": 8,
        "flow_batch_packets": 128,
        "scheduler": "cross_flow_length_bucketed_v1",
        "max_length": 640,
        "packet_index": str(packet_index),
        "packet_index_header_policy": "full",
        "packet_index_context_policy": "single_packet",
        "num_shards": num_shards,
    }
    for shard_index in range(num_shards):
        shard_dir = output_dir / "_shards" / f"shard_{shard_index}"
        embedding_dir = shard_dir / "packet_embeddings"
        embedding_dir.mkdir(parents=True)
        rows = []
        for flow_id in flow_ids:
            if flow_shard_index(flow_id, num_shards) != shard_index:
                continue
            path = embedding_dir / f"{flow_id}.npy"
            np.save(path, np.ones((1, 3), dtype=np.float32))
            rows.append(
                {
                    "flow_id": flow_id,
                    "embedding_path": str(path),
                    "embedding_mode": "concat",
                }
            )
        (shard_dir / "flow_embedding_index.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        (shard_dir / "embedding_config.json").write_text(
            json.dumps({**common, "shard_index": shard_index}), encoding="utf-8"
        )
    return packet_index, output_dir


def test_merge_embedding_shards_preserves_execution_contract(tmp_path):
    packet_index, output_dir = write_fixture(tmp_path)

    result = merge_embedding_shards(
        packet_index=packet_index,
        output_dir=output_dir,
        num_shards=2,
        shard_devices=["0", "1"],
    )

    config = json.loads((output_dir / "embedding_config.json").read_text())
    rows = [
        json.loads(line)
        for line in (output_dir / "flow_embedding_index.jsonl").read_text().splitlines()
    ]
    assert result["merged_flows"] == 4
    assert {row["flow_id"] for row in rows} == {"flow-a", "flow-b", "flow-c", "flow-d"}
    assert config["scheduler"] == "cross_flow_length_bucketed_v1"
    assert config["batch_size"] == 8
    assert config["flow_batch_packets"] == 128
    assert config["merge_scheduler"] == "deterministic_flow_sha1_v1"
    assert config["shard_devices"] == ["0", "1"]


def test_merge_embedding_shards_rejects_config_drift(tmp_path):
    packet_index, output_dir = write_fixture(tmp_path)
    config_path = output_dir / "_shards" / "shard_1" / "embedding_config.json"
    config = json.loads(config_path.read_text())
    config["scheduler"] = "legacy_per_flow_v1"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="configuration drift"):
        merge_embedding_shards(
            packet_index=packet_index, output_dir=output_dir, num_shards=2
        )


def test_merge_embedding_shards_rejects_duplicate_flow(tmp_path):
    packet_index, output_dir = write_fixture(tmp_path)
    shard_path = output_dir / "_shards" / "shard_0" / "flow_embedding_index.jsonl"
    lines = shard_path.read_text().splitlines()
    if not lines:
        shard_path = output_dir / "_shards" / "shard_1" / "flow_embedding_index.jsonl"
        lines = shard_path.read_text().splitlines()
    shard_path.write_text("\n".join([*lines, lines[0]]) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate flow"):
        merge_embedding_shards(
            packet_index=packet_index, output_dir=output_dir, num_shards=2
        )


def test_packet_semantic_shard_command_uses_shared_extractor_contract(tmp_path):
    args = SimpleNamespace(
        base_model="qwen",
        semantic_embedding_mode="concat",
        semantic_embedding_batch_size=8,
        semantic_embedding_flow_batch_packets=128,
        max_packet_length=640,
        semantic_embedding_device="auto",
        semantic_embedding_num_shards=4,
        local_files_only=True,
    )
    command = semantic_embedding_command(
        args,
        packet_index=tmp_path / "packet_index.jsonl",
        output_dir=tmp_path / "shard_2",
        checkpoint=tmp_path / "checkpoint",
        shard_index=2,
        device="cuda:3",
    )

    assert command[command.index("--num_shards") + 1] == "4"
    assert command[command.index("--shard_index") + 1] == "2"
    assert command[command.index("--device") + 1] == "cuda:3"
    assert command[command.index("--flow_batch_packets") + 1] == "128"
    assert "--resume_existing" in command
    assert "--local_files_only" in command


def test_paper_semantic_embedding_audit_requires_model_provenance(tmp_path):
    command = semantic_embedding_audit_command(
        tmp_path / "packet_index.jsonl",
        tmp_path / "embeddings",
        require_model_provenance=True,
    )

    assert "--require_model_provenance" in command
