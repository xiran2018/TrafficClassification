#!/usr/bin/env python3
"""Audit packet-to-flow embedding artifacts before downstream training."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def load_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON at {path}:{line_number}") from exc


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_sha256(path: Path) -> tuple[str, int]:
    if path.is_file():
        files = [path]
        root = path.parent
    elif path.is_dir():
        files = sorted(item for item in path.rglob("*") if item.is_file())
        root = path
    else:
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    for item in files:
        digest.update(item.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest(), len(files)


def verify_model_provenance(config: dict[str, Any]) -> dict[str, Any]:
    declared = config.get("model_provenance") or {}
    artifacts = {}
    for name in ("lora_adapter", "tower1_heads"):
        row = declared.get(name)
        if not isinstance(row, dict) or not row.get("path") or not row.get("sha256"):
            artifacts[name] = {"verified": False, "reason": "missing_declaration"}
            continue
        path = Path(str(row["path"]))
        try:
            actual_sha256, actual_file_count = artifact_sha256(path)
        except OSError:
            actual_sha256, actual_file_count = None, 0
        artifacts[name] = {
            "path": str(path),
            "declared_sha256": row.get("sha256"),
            "actual_sha256": actual_sha256,
            "declared_file_count": row.get("file_count"),
            "actual_file_count": actual_file_count,
            "verified": bool(
                actual_sha256
                and actual_sha256 == row.get("sha256")
                and int(row.get("file_count", -1)) == actual_file_count
            ),
        }
    return {
        "verified": bool(artifacts) and all(row["verified"] for row in artifacts.values()),
        "artifacts": artifacts,
    }


def audit_report_evidence(path: Path) -> dict[str, Any]:
    """Return a compact, independently checked manifest binding for one audit."""
    report: dict[str, Any] = {}
    if path.is_file():
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            report = {}
    counts = report.get("counts") or {}
    integrity = report.get("integrity") or {}
    inputs = report.get("inputs") or {}
    hashes = {
        key: inputs.get(key)
        for key in (
            "packet_index_sha256",
            "embedding_index_sha256",
            "embedding_config_sha256",
        )
    }
    valid_hashes = all(
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
        for value in hashes.values()
    )
    path_fields = {
        "packet_index_sha256": "packet_index",
        "embedding_index_sha256": "embedding_index",
        "embedding_config_sha256": "embedding_config",
    }
    actual_hashes: dict[str, str | None] = {}
    hash_bindings: dict[str, bool] = {}
    for hash_field, path_field in path_fields.items():
        raw_path = inputs.get(path_field)
        artifact_path = Path(str(raw_path)) if raw_path else None
        if (
            artifact_path is not None
            and not artifact_path.is_absolute()
            and not artifact_path.is_file()
        ):
            report_relative = path.parent / artifact_path
            if report_relative.is_file():
                artifact_path = report_relative
        actual_hash = None
        if artifact_path is not None and artifact_path.is_file():
            try:
                actual_hash = sha256_file(artifact_path)
            except OSError:
                actual_hash = None
        actual_hashes[hash_field] = actual_hash
        hash_bindings[hash_field] = bool(
            actual_hash is not None and actual_hash == hashes.get(hash_field)
        )
    hash_bindings_verified = bool(hash_bindings) and all(hash_bindings.values())
    integrity_zero = bool(integrity) and all(int(value) == 0 for value in integrity.values())
    counts_match = bool(counts) and (
        int(counts.get("input_flows", -1))
        == int(counts.get("unique_output_flows", -2))
        == int(counts.get("output_rows", -3))
        and int(counts.get("input_packets", -1))
        == int(counts.get("output_packets", -2))
        and int(counts.get("input_flows", 0)) > 0
        and int(counts.get("input_packets", 0)) > 0
    )
    model_provenance_required = report.get("require_model_provenance") is True
    reported_model_provenance_verified = bool(
        (report.get("model_provenance") or {}).get("verified") is True
    )
    current_model_provenance: dict[str, Any] = {"verified": False, "artifacts": {}}
    raw_config_path = inputs.get("embedding_config")
    config_path = Path(str(raw_config_path)) if raw_config_path else None
    if config_path is not None and not config_path.is_absolute() and not config_path.is_file():
        candidate = path.parent / config_path
        if candidate.is_file():
            config_path = candidate
    if config_path is not None and config_path.is_file():
        try:
            current_config = json.loads(config_path.read_text(encoding="utf-8"))
            current_model_provenance = verify_model_provenance(current_config)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            current_model_provenance = {"verified": False, "artifacts": {}}
    model_provenance_verified = bool(
        reported_model_provenance_verified
        and current_model_provenance.get("verified") is True
    )
    verified = bool(
        report.get("schema") == "flow_embedding_audit_v1"
        and report.get("status") == "pass"
        and report.get("require_finite") is True
        and integrity_zero
        and counts_match
        and valid_hashes
        and hash_bindings_verified
        and (not model_provenance_required or model_provenance_verified)
    )
    return {
        "audit_path": str(path),
        "audit_verified": verified,
        "audit_status": report.get("status", "missing"),
        "counts": counts,
        "integrity": integrity,
        "input_hashes": hashes,
        "actual_input_hashes": actual_hashes,
        "hash_bindings": hash_bindings,
        "hash_bindings_verified": hash_bindings_verified,
        "model_provenance_required": model_provenance_required,
        "model_provenance_verified": model_provenance_verified,
        "current_model_provenance": current_model_provenance,
    }


def audit_embeddings(
    packet_index: Path,
    embedding_index: Path,
    *,
    require_finite: bool = True,
    require_model_provenance: bool = False,
) -> dict[str, Any]:
    packet_ids: dict[str, list[int]] = defaultdict(list)
    labels: dict[str, tuple[str, int]] = {}
    duplicate_packet_keys = 0
    seen_packet_keys: set[tuple[str, int]] = set()
    input_packets = 0
    for row in load_jsonl(packet_index):
        flow_id = str(row["flow_id"])
        packet_id = int(row["packet_id"])
        key = (flow_id, packet_id)
        duplicate_packet_keys += key in seen_packet_keys
        seen_packet_keys.add(key)
        packet_ids[flow_id].append(packet_id)
        label = (str(row["label"]), int(row["label_id"]))
        if flow_id in labels and labels[flow_id] != label:
            raise ValueError(f"inconsistent input label for flow_id={flow_id}")
        labels[flow_id] = label
        input_packets += 1

    seen_flows: set[str] = set()
    duplicate_flows = 0
    extra_flows = 0
    output_packets = 0
    packet_order_mismatches = 0
    label_mismatches = 0
    bad_embeddings = 0
    nonfinite_embeddings = 0
    dimensions: Counter[int] = Counter()
    dtypes: Counter[str] = Counter()
    output_rows = 0
    failure_examples: list[dict[str, Any]] = []

    def record(flow_id: str, error: str, **details: Any) -> None:
        nonlocal bad_embeddings
        bad_embeddings += 1
        if len(failure_examples) < 20:
            failure_examples.append({"flow_id": flow_id, "error": error, **details})

    for row in load_jsonl(embedding_index):
        output_rows += 1
        flow_id = str(row["flow_id"])
        if flow_id in seen_flows:
            duplicate_flows += 1
        seen_flows.add(flow_id)
        expected_ids = packet_ids.get(flow_id)
        if expected_ids is None:
            extra_flows += 1
            record(flow_id, "extra_flow")
            continue
        observed_ids = [int(meta["packet_id"]) for meta in row.get("packet_metas", [])]
        if observed_ids != expected_ids:
            packet_order_mismatches += 1
            record(
                flow_id,
                "packet_order_mismatch",
                expected_count=len(expected_ids),
                observed_count=len(observed_ids),
            )
        expected_label = labels[flow_id]
        observed_label = (str(row.get("label")), int(row.get("label_id", -1)))
        if observed_label != expected_label:
            label_mismatches += 1
            record(flow_id, "label_mismatch")

        embedding_path = Path(row["embedding_path"])
        if not embedding_path.is_file():
            record(flow_id, "missing_embedding", path=str(embedding_path))
            continue
        try:
            values = np.load(embedding_path, mmap_mode="r", allow_pickle=False)
        except (OSError, ValueError) as exc:
            record(flow_id, "unreadable_embedding", message=str(exc))
            continue
        if values.ndim != 2:
            record(flow_id, "embedding_not_2d", shape=list(values.shape))
            continue
        output_packets += int(values.shape[0])
        dimensions[int(values.shape[1])] += 1
        dtypes[str(values.dtype)] += 1
        declared_dim = int(row.get("embedding_dim", -1))
        if values.shape != (len(expected_ids), declared_dim):
            record(
                flow_id,
                "embedding_shape_mismatch",
                shape=list(values.shape),
                expected_shape=[len(expected_ids), declared_dim],
            )
        if require_finite and not bool(np.isfinite(values).all()):
            nonfinite_embeddings += 1
            record(flow_id, "nonfinite_embedding")

    missing_flows = len(set(packet_ids) - seen_flows)
    status = "pass" if not any(
        (
            duplicate_packet_keys,
            duplicate_flows,
            missing_flows,
            extra_flows,
            packet_order_mismatches,
            label_mismatches,
            bad_embeddings,
            nonfinite_embeddings,
            input_packets != output_packets,
        )
    ) else "fail"
    embedding_config = embedding_index.parent / "embedding_config.json"
    config_payload = {}
    if embedding_config.is_file():
        try:
            config_payload = json.loads(embedding_config.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            config_payload = {}
    model_provenance = verify_model_provenance(config_payload)
    if require_model_provenance and not model_provenance["verified"]:
        status = "fail"
    return {
        "schema": "flow_embedding_audit_v1",
        "status": status,
        "inputs": {
            "packet_index": str(packet_index),
            "packet_index_sha256": sha256_file(packet_index),
            "embedding_index": str(embedding_index),
            "embedding_index_sha256": sha256_file(embedding_index),
            "embedding_config": str(embedding_config) if embedding_config.is_file() else None,
            "embedding_config_sha256": (
                sha256_file(embedding_config) if embedding_config.is_file() else None
            ),
        },
        "counts": {
            "input_flows": len(packet_ids),
            "input_packets": input_packets,
            "output_rows": output_rows,
            "unique_output_flows": len(seen_flows),
            "output_packets": output_packets,
        },
        "integrity": {
            "duplicate_packet_keys": duplicate_packet_keys,
            "duplicate_flows": duplicate_flows,
            "missing_flows": missing_flows,
            "extra_flows": extra_flows,
            "packet_order_mismatches": packet_order_mismatches,
            "label_mismatches": label_mismatches,
            "bad_embeddings": bad_embeddings,
            "nonfinite_embeddings": nonfinite_embeddings,
        },
        "embedding_dimensions": dict(sorted(dimensions.items())),
        "embedding_dtypes": dict(sorted(dtypes.items())),
        "require_finite": require_finite,
        "require_model_provenance": require_model_provenance,
        "model_provenance": model_provenance,
        "failure_examples": failure_examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet_index", required=True)
    parser.add_argument("--flow_embedding_index", required=True)
    parser.add_argument("--output_json", default="")
    parser.add_argument("--skip_finite_check", action="store_true")
    parser.add_argument("--require_model_provenance", action="store_true")
    args = parser.parse_args()
    report = audit_embeddings(
        Path(args.packet_index),
        Path(args.flow_embedding_index),
        require_finite=not args.skip_finite_check,
        require_model_provenance=args.require_model_provenance,
    )
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
