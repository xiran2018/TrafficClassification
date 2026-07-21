import json
from pathlib import Path

import numpy as np

from audit_flow_embeddings import (
    artifact_sha256,
    audit_embeddings,
    audit_report_evidence,
    sha256_file,
)
from extract_packet_embeddings_qwen import artifact_evidence


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_extractor_and_auditor_share_model_artifact_hash_contract(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")

    extracted = artifact_evidence(adapter)
    audited_sha, audited_count = artifact_sha256(adapter)

    assert extracted["sha256"] == audited_sha
    assert extracted["file_count"] == audited_count == 2


def test_embedding_audit_passes_aligned_artifacts(tmp_path: Path) -> None:
    packet_index = tmp_path / "packets.jsonl"
    embedding_index = tmp_path / "flow_embedding_index.jsonl"
    values = np.ones((2, 4), dtype=np.float16)
    embedding_path = tmp_path / "flow.npy"
    np.save(embedding_path, values)
    write_jsonl(
        packet_index,
        [
            {"flow_id": "f", "packet_id": 0, "label": "a", "label_id": 0},
            {"flow_id": "f", "packet_id": 1, "label": "a", "label_id": 0},
        ],
    )
    write_jsonl(
        embedding_index,
        [
            {
                "flow_id": "f",
                "label": "a",
                "label_id": 0,
                "embedding_path": str(embedding_path),
                "embedding_dim": 4,
                "packet_metas": [{"packet_id": 0}, {"packet_id": 1}],
            }
        ],
    )

    report = audit_embeddings(packet_index, embedding_index)

    assert report["status"] == "pass"
    assert report["counts"]["output_packets"] == 2
    assert report["embedding_dimensions"] == {4: 1}
    assert report["embedding_dtypes"] == {"float16": 1}


def test_embedding_audit_rejects_order_and_nonfinite_values(tmp_path: Path) -> None:
    packet_index = tmp_path / "packets.jsonl"
    embedding_index = tmp_path / "flow_embedding_index.jsonl"
    embedding_path = tmp_path / "flow.npy"
    np.save(embedding_path, np.array([[np.nan], [1.0]], dtype=np.float32))
    write_jsonl(
        packet_index,
        [
            {"flow_id": "f", "packet_id": 0, "label": "a", "label_id": 0},
            {"flow_id": "f", "packet_id": 1, "label": "a", "label_id": 0},
        ],
    )
    write_jsonl(
        embedding_index,
        [
            {
                "flow_id": "f",
                "label": "a",
                "label_id": 0,
                "embedding_path": str(embedding_path),
                "embedding_dim": 1,
                "packet_metas": [{"packet_id": 1}, {"packet_id": 0}],
            }
        ],
    )

    report = audit_embeddings(packet_index, embedding_index)

    assert report["status"] == "fail"
    assert report["integrity"]["packet_order_mismatches"] == 1
    assert report["integrity"]["nonfinite_embeddings"] == 1


def test_embedding_audit_binds_model_artifacts_when_required(tmp_path: Path) -> None:
    packet_index = tmp_path / "packets.jsonl"
    embedding_index = tmp_path / "flow_embedding_index.jsonl"
    embedding_path = tmp_path / "flow.npy"
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
    heads = tmp_path / "tower1_heads.pt"
    heads.write_bytes(b"heads")
    np.save(embedding_path, np.ones((1, 2), dtype=np.float32))
    write_jsonl(
        packet_index,
        [{"flow_id": "f", "packet_id": 0, "label": "a", "label_id": 0}],
    )
    write_jsonl(
        embedding_index,
        [
            {
                "flow_id": "f",
                "label": "a",
                "label_id": 0,
                "embedding_path": str(embedding_path),
                "embedding_dim": 2,
                "packet_metas": [{"packet_id": 0}],
            }
        ],
    )
    adapter_sha, adapter_files = artifact_sha256(adapter)
    heads_sha, heads_files = artifact_sha256(heads)
    (tmp_path / "embedding_config.json").write_text(
        json.dumps(
            {
                "model_provenance": {
                    "lora_adapter": {
                        "path": str(adapter),
                        "sha256": adapter_sha,
                        "file_count": adapter_files,
                    },
                    "tower1_heads": {
                        "path": str(heads),
                        "sha256": heads_sha,
                        "file_count": heads_files,
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    accepted = audit_embeddings(
        packet_index, embedding_index, require_model_provenance=True
    )
    assert accepted["status"] == "pass"
    assert accepted["model_provenance"]["verified"] is True
    audit_path = tmp_path / "embedding_audit.json"
    audit_path.write_text(json.dumps(accepted), encoding="utf-8")
    assert audit_report_evidence(audit_path)["audit_verified"] is True

    heads.write_bytes(b"changed")
    rejected = audit_embeddings(
        packet_index, embedding_index, require_model_provenance=True
    )
    assert rejected["status"] == "fail"
    assert rejected["model_provenance"]["verified"] is False
    evidence = audit_report_evidence(audit_path)
    assert evidence["audit_verified"] is False
    assert evidence["model_provenance_verified"] is False


def test_audit_report_evidence_requires_hashes_counts_and_zero_integrity(
    tmp_path: Path,
) -> None:
    path = tmp_path / "audit.json"
    packet_index = tmp_path / "packet_index.jsonl"
    embedding_index = tmp_path / "flow_embedding_index.jsonl"
    embedding_config = tmp_path / "embedding_config.json"
    packet_index.write_text("packet\n", encoding="utf-8")
    embedding_index.write_text("embedding\n", encoding="utf-8")
    embedding_config.write_text("{}\n", encoding="utf-8")
    path.write_text(
        json.dumps(
            {
                "schema": "flow_embedding_audit_v1",
                "status": "pass",
                "require_finite": True,
                "inputs": {
                    "packet_index": str(packet_index),
                    "packet_index_sha256": sha256_file(packet_index),
                    "embedding_index": str(embedding_index),
                    "embedding_index_sha256": sha256_file(embedding_index),
                    "embedding_config": str(embedding_config),
                    "embedding_config_sha256": sha256_file(embedding_config),
                },
                "counts": {
                    "input_flows": 1,
                    "input_packets": 2,
                    "output_rows": 1,
                    "unique_output_flows": 1,
                    "output_packets": 2,
                },
                "integrity": {"missing_flows": 0, "bad_embeddings": 0},
            }
        ),
        encoding="utf-8",
    )

    assert audit_report_evidence(path)["audit_verified"] is True

    embedding_config.write_text('{"changed":true}\n', encoding="utf-8")
    evidence = audit_report_evidence(path)
    assert evidence["audit_verified"] is False
    assert evidence["hash_bindings_verified"] is False
    embedding_config.write_text("{}\n", encoding="utf-8")

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["integrity"]["missing_flows"] = 1
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert audit_report_evidence(path)["audit_verified"] is False

    payload["integrity"]["missing_flows"] = 0
    payload["require_model_provenance"] = True
    payload["model_provenance"] = {"verified": False}
    path.write_text(json.dumps(payload), encoding="utf-8")
    evidence = audit_report_evidence(path)
    assert evidence["audit_verified"] is False
    assert evidence["model_provenance_required"] is True
