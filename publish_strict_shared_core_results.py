#!/usr/bin/env python3
"""Publish fixed-consensus results only after all strict checkpoint audits pass."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any

from unified_framework_spec import FLOW_LEVEL_RESULTS, PACKET_LEVEL_RESULTS


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    staging.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    staging.replace(destination)


def canonical_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_copy(path: str | Path, destination: str | Path) -> None:
    source = Path(path)
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    shutil.copy2(source, staging)
    staging.replace(target)


def archive_hashed_evidence(
    evidence: dict[str, Any],
    *,
    label: str,
    destination: Path,
) -> dict[str, Any]:
    evidence_path = Path(str(evidence.get("path") or ""))
    evidence_hash = str(evidence.get("sha256") or "")
    if not evidence_path.is_file() or sha256_file(evidence_path) != evidence_hash:
        raise ValueError(f"frozen config {label} evidence hash mismatch")
    payload = load_json(evidence_path)
    if payload.get("test_labels_used") is True:
        raise ValueError(f"frozen config {label} evidence used test labels")
    atomic_copy(evidence_path, destination)
    return {
        "source_path": str(evidence_path),
        "source_sha256": evidence_hash,
        "archived_path": str(destination),
        "archived_sha256": sha256_file(destination),
    }


def archive_frozen_method_evidence(
    config_path: str | Path,
    *,
    expected_fingerprint: str,
    archive_root: str | Path,
) -> dict[str, Any]:
    source = Path(config_path)
    config = load_json(source)
    if config.get("schema") != "exact_shared_packet_core_v2":
        raise ValueError("frozen shared-core config has the wrong schema")
    if config.get("status") != "frozen_from_cross_dataset_validation":
        raise ValueError("shared-core config was not frozen from validation")
    recorded_fingerprint = str(config.get("config_sha256") or "")
    unsigned = dict(config)
    unsigned.pop("config_sha256", None)
    recomputed_fingerprint = canonical_sha256(unsigned)
    if not (
        recorded_fingerprint == expected_fingerprint == recomputed_fingerprint
    ):
        raise ValueError(
            "frozen shared-core config fingerprint does not match strict audits"
        )

    archive_root = Path(archive_root)
    archive_root.mkdir(parents=True, exist_ok=True)
    archived = {}
    selection = config.get("selection_evidence") or {}
    for key, filename in (
        ("balance", "balance_selection.json"),
        ("paired_invariance", "paired_selection.json"),
    ):
        archived[key] = archive_hashed_evidence(
            selection.get(key) or {},
            label=f"{key} selection",
            destination=archive_root / filename,
        )

    archived_method_selection = {}
    method_selection = config.get("method_selection")
    if method_selection is not None:
        if not isinstance(method_selection, dict):
            raise ValueError("frozen config method_selection must be an object")
        if method_selection.get("decision_status") != (
            "final_after_preregistered_validation"
        ):
            raise ValueError("frozen method selection is not final")
        if method_selection.get("test_labels_used") is not False:
            raise ValueError("frozen method selection is not validation-only")

        required = ["base_shared_core_config", "d1"]
        if method_selection.get("packet_selected_identity_safe_contrastive"):
            required.extend(
                ["d2_incremental", "d2_overall", "flow_noninferiority"]
            )
        for key in required:
            if not isinstance(method_selection.get(key), dict):
                raise ValueError(f"frozen method selection is missing {key} evidence")

        evidence_files = (
            ("base_shared_core_config", "base_shared_core_config.json"),
            ("d1", "d1_selection.json"),
            ("d2_incremental", "d2_incremental_selection.json"),
            ("d2_overall", "d2_overall_selection.json"),
            ("cross_scale_exposure", "cross_scale_exposure.json"),
            ("flow_noninferiority", "flow_noninferiority_selection.json"),
        )
        for key, filename in evidence_files:
            evidence = method_selection.get(key)
            if evidence is None:
                continue
            if not isinstance(evidence, dict):
                raise ValueError(f"frozen method selection {key} evidence is invalid")
            archived_method_selection[key] = archive_hashed_evidence(
                evidence,
                label=f"method selection {key}",
                destination=archive_root / filename,
            )

        base_evidence = method_selection["base_shared_core_config"]
        base_payload = load_json(base_evidence["path"])
        base_recorded = str(base_payload.get("config_sha256") or "")
        base_unsigned = dict(base_payload)
        base_unsigned.pop("config_sha256", None)
        if not (
            base_recorded
            == str(base_evidence.get("config_sha256") or "")
            == canonical_sha256(base_unsigned)
        ):
            raise ValueError("base shared-core config fingerprint mismatch")

        packet_selected_identity = bool(
            method_selection.get("packet_selected_identity_safe_contrastive")
        )
        packet_selected_cross_scale = bool(
            method_selection.get("packet_selected_availability_aware_cross_scale")
        )
        flow_gate_passed = method_selection.get("flow_noninferiority_passed")
        if packet_selected_identity and flow_gate_passed not in (True, False):
            raise ValueError("final method selection lacks a Flow gate decision")
        if not packet_selected_identity and flow_gate_passed is not None:
            raise ValueError("Flow gate decision is inapplicable when D1 was rejected")
        if method_selection.get("identity_safe_contrastive") and flow_gate_passed is not True:
            raise ValueError("identity-safe method was retained without passing Flow gate")
        if packet_selected_cross_scale and not isinstance(
            method_selection.get("cross_scale_exposure"), dict
        ):
            raise ValueError("cross-scale method lacks exposure evidence")

        d1_selected = load_json(method_selection["d1"]["path"]).get("selected")
        if (d1_selected == "candidate") != packet_selected_identity:
            raise ValueError("D1 evidence contradicts the frozen Packet decision")
        if packet_selected_identity:
            d2_incremental_selected = load_json(
                method_selection["d2_incremental"]["path"]
            ).get("selected")
            d2_overall_selected = load_json(
                method_selection["d2_overall"]["path"]
            ).get("selected")
            evidence_selected_cross_scale = (
                d2_incremental_selected == "candidate"
                and d2_overall_selected == "candidate"
            )
            if evidence_selected_cross_scale != packet_selected_cross_scale:
                raise ValueError("D2 evidence contradicts the frozen Packet decision")
            flow_selected = load_json(
                method_selection["flow_noninferiority"]["path"]
            ).get("selected")
            if (flow_selected == "candidate") != flow_gate_passed:
                raise ValueError("Flow evidence contradicts the frozen gate decision")

        final_identity = bool(method_selection.get("identity_safe_contrastive"))
        final_cross_scale = bool(
            method_selection.get("availability_aware_cross_scale")
        )
        if final_identity != (packet_selected_identity and flow_gate_passed is True):
            raise ValueError("final identity-safe objective contradicts selection evidence")
        if final_cross_scale != (packet_selected_cross_scale and final_identity):
            raise ValueError("final cross-scale objective contradicts selection evidence")
        expected_method = (
            "availability_aware_cross_scale"
            if final_cross_scale
            else "identity_safe_contrastive"
            if final_identity
            else "shared_core_v2_control"
        )
        if method_selection.get("selected_method") != expected_method:
            raise ValueError("selected_method contradicts the frozen objectives")

    config_destination = archive_root / "frozen_config.json"
    atomic_copy(source, config_destination)
    archived_config = {
        "source_path": str(source),
        "source_file_sha256": sha256_file(source),
        "archived_path": str(config_destination),
        "archived_file_sha256": sha256_file(config_destination),
        "config_sha256": recorded_fingerprint,
    }
    archive = {
        "schema": (
            "strict_shared_core_v2_method_archive_v2"
            if method_selection is not None
            else "strict_shared_core_v2_method_archive_v1"
        ),
        "status": "verified_and_archived",
        "shared_core_config": archived_config,
        "selection_evidence": archived,
        "method_selection_evidence": archived_method_selection,
    }
    archive_manifest = archive_root / "archive_manifest.json"
    atomic_write_json(archive_manifest, archive)
    archive["archive_manifest"] = str(archive_manifest)
    archive["archive_manifest_sha256"] = sha256_file(archive_manifest)
    return archive


def metrics(payload: dict[str, Any], task: str) -> tuple[float, float]:
    if task == "flow-level":
        values = (payload.get("metrics") or {}).get("flow_level") or {}
    else:
        values = payload.get("metrics") or payload.get("test_metrics") or {}
        values = values.get("packet_level") or values
    return float(values["accuracy"]), float(values["macro_f1"])


def validate_fixed_consensus(payload: dict[str, Any], task: str) -> None:
    method = payload.get("method")
    if task == "flow-level":
        config = payload.get("config") or {}
        method = config.get("selected_mode")
        if config.get("requested_mode") != "log_mean":
            raise ValueError("flow publication candidate did not request fixed log_mean")
    if method != "log_mean":
        raise ValueError(f"{task} publication candidate must use fixed log_mean")
    inputs = payload.get("inputs") or []
    if len(inputs) != 3:
        raise ValueError(f"{task} publication candidate must contain exactly three folds")
    paths = [str(row.get("path", "")) if isinstance(row, dict) else str(row) for row in inputs]
    if task == "packet-level":
        if any(not path.endswith("test_unified_packet_single_head.npz") for path in paths):
            raise ValueError("packet candidate contains a non-single-head input")
    elif any("test_seq_metrics_flow_" not in path or not path.endswith("_probs.json") for path in paths):
        raise ValueError("flow candidate contains a non-seq or non-probability input")


def validated_audits(dataset: str, audit_root: Path) -> tuple[list[Path], str]:
    paths: list[Path] = []
    fingerprints: set[str] = set()
    for fold in range(3):
        path = audit_root / dataset / f"fold{fold}" / "audit.json"
        payload = load_json(path)
        if (
            payload.get("status") != "pass"
            or payload.get("dataset") != dataset
            or int(payload.get("fold", -1)) != fold
        ):
            raise ValueError(f"strict checkpoint audit did not pass: {path}")
        if (
            payload.get("runtime_mechanism_evidence_required") is not True
            or payload.get("flow_native_extraction_evidence_required") is not True
            or payload.get("algorithm_source_evidence_required") is not True
            or (payload.get("runtime_mechanism_evidence") or {}).get("status")
            != "pass"
            or (payload.get("flow_native_extraction_evidence") or {}).get("status")
            != "pass"
            or payload.get("algorithm_source_evidence_verified") is not True
            or not payload.get("algorithm_source_fingerprint")
        ):
            raise ValueError(
                f"strict checkpoint audit has no passing runtime mechanism evidence: {path}"
            )
        fingerprint = str(
            payload.get("shared_core_method_sha256")
            or payload.get("shared_core_config_sha256")
            or ""
        )
        if not fingerprint:
            raise ValueError(f"strict checkpoint audit has no frozen fingerprint: {path}")
        fingerprints.add(fingerprint)
        paths.append(path)
    if len(fingerprints) != 1:
        raise ValueError(f"strict checkpoint audit fingerprints differ: {sorted(fingerprints)}")
    return paths, next(iter(fingerprints))


def validate_bootstrap_evidence(
    path: str, *, task: str, accuracy: float, macro_f1: float
) -> dict[str, Any]:
    payload = load_json(path)
    if payload.get("task") != task:
        raise ValueError(f"bootstrap task mismatch for {path}")
    if payload.get("method") != "class_stratified_flow_cluster_bootstrap":
        raise ValueError(f"bootstrap method mismatch for {path}")
    if int(payload.get("bootstrap_samples", 0)) < 2000:
        raise ValueError(f"bootstrap evidence needs at least 2000 draws: {path}")
    values = payload.get("metrics") or {}
    observed_accuracy = float(
        (values.get("accuracy") or {}).get("point_estimate", float("nan"))
    )
    observed_f1 = float(
        (values.get("macro_f1") or {}).get("point_estimate", float("nan"))
    )
    if (
        not math.isfinite(observed_accuracy)
        or not math.isfinite(observed_f1)
        or abs(observed_accuracy - accuracy) > 1e-8
        or abs(observed_f1 - macro_f1) > 1e-8
    ):
        raise ValueError(
            f"bootstrap point estimate does not match publication candidate: {path}"
        )
    if int(payload.get("num_flow_clusters", 0)) <= int(payload.get("num_classes", 0)):
        raise ValueError(f"bootstrap evidence has too few independent flow clusters: {path}")
    return payload


def validate_session_novelty_evidence(
    path: str, *, task: str, accuracy: float, macro_f1: float
) -> dict[str, Any]:
    payload = load_json(path)
    if payload.get("schema") != "session_novelty_evaluation_v1":
        raise ValueError(f"session-novelty schema mismatch for {path}")
    if payload.get("task") != task:
        raise ValueError(f"session-novelty task mismatch for {path}")
    if payload.get("selection_role") != "reporting_only_no_training_or_model_selection":
        raise ValueError(f"session-novelty evidence cannot participate in selection: {path}")
    if payload.get("group_definition_uses_test_labels") is not False:
        raise ValueError(f"session-novelty groups must be label independent: {path}")
    if payload.get("training_reference") != "union_of_all_supplied_training_signature_sets":
        raise ValueError(f"session-novelty evidence must use the three-fold training union: {path}")
    all_metrics = (((payload.get("groups") or {}).get("all") or {}).get("metrics") or {})
    observed_accuracy = float(all_metrics.get("accuracy", float("nan")))
    observed_f1 = float(all_metrics.get("macro_f1", float("nan")))
    if abs(observed_accuracy - accuracy) > 1e-8 or abs(observed_f1 - macro_f1) > 1e-8:
        raise ValueError(
            f"session-novelty overall metrics do not match publication candidate: {path}"
        )
    inputs = payload.get("inputs") or {}
    train_inputs = inputs.get("train_packet_indices") or []
    if len(train_inputs) != 3:
        raise ValueError(f"session-novelty evidence needs exactly three training folds: {path}")
    if len({str(row.get("path", "")) for row in train_inputs if isinstance(row, dict)}) != 3:
        raise ValueError(f"session-novelty evidence must contain three distinct training folds: {path}")
    for row in [*train_inputs, inputs.get("test_packet_index"), inputs.get("predictions"), inputs.get("label_map")]:
        if not isinstance(row, dict) or not row.get("path") or not row.get("sha256"):
            raise ValueError(f"session-novelty evidence has incomplete input hashes: {path}")
        if sha256_file(row["path"]) != row["sha256"]:
            raise ValueError(f"session-novelty input hash mismatch: {row['path']}")
    return payload


def publish_canonical_result(
    payload: dict[str, Any],
    destination: Path,
    *,
    fingerprint: str,
    audit_paths: list[Path],
    candidate: str,
    session_novelty: str,
    bootstrap_evidence: str,
    method_archive_manifest: str,
    method_archive_manifest_sha256: str,
) -> None:
    published = dict(payload)
    published["publication_provenance"] = {
        "status": "strict_shared_core_v2",
        "shared_core_method_sha256": fingerprint,
        "shared_core_config_sha256": fingerprint,
        "audit_paths": [str(path) for path in audit_paths],
        "audit_evidence": [
            {"path": str(path), "sha256": sha256_file(path)}
            for path in audit_paths
        ],
        "runtime_mechanism_evidence_required": True,
        "flow_native_extraction_evidence_required": True,
        "fixed_consensus": "equal_log_mean_three_folds",
        "candidate": candidate,
        "candidate_sha256": sha256_file(candidate),
        "session_novelty": session_novelty,
        "session_novelty_sha256": sha256_file(session_novelty),
        "bootstrap_evidence": bootstrap_evidence,
        "bootstrap_evidence_sha256": sha256_file(bootstrap_evidence),
        "method_archive_manifest": method_archive_manifest,
        "method_archive_manifest_sha256": method_archive_manifest_sha256,
    }
    atomic_write_json(destination, published)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["vpn-app", "tls-120"], required=True)
    parser.add_argument("--audit_root", default="reasoningDataset/shared-core-audits")
    parser.add_argument("--packet_manifest_root", required=True)
    parser.add_argument(
        "--shared_core_config",
        default="/tmp/two_tower_runs/shared_core_v2/frozen_config.json",
    )
    parser.add_argument(
        "--method_archive_root",
        default="reasoningDataset/shared-core-v2",
    )
    parser.add_argument("--packet_candidate", required=True)
    parser.add_argument("--flow_candidate", required=True)
    parser.add_argument("--packet_bootstrap", required=True)
    parser.add_argument("--flow_bootstrap", required=True)
    parser.add_argument("--packet_session_novelty", required=True)
    parser.add_argument("--flow_session_novelty", required=True)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    lock_handle = open(f"{output}.lock", "w", encoding="utf-8")
    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)

    audit_paths, fingerprint = validated_audits(args.dataset, Path(args.audit_root))
    frozen_method_evidence = archive_frozen_method_evidence(
        args.shared_core_config,
        expected_fingerprint=fingerprint,
        archive_root=args.method_archive_root,
    )
    packet_payload = load_json(args.packet_candidate)
    flow_payload = load_json(args.flow_candidate)
    validate_fixed_consensus(packet_payload, "packet-level")
    validate_fixed_consensus(flow_payload, "flow-level")
    packet_accuracy, packet_macro_f1 = metrics(packet_payload, "packet-level")
    flow_accuracy, flow_macro_f1 = metrics(flow_payload, "flow-level")
    packet_bootstrap = validate_bootstrap_evidence(
        args.packet_bootstrap,
        task="packet",
        accuracy=packet_accuracy,
        macro_f1=packet_macro_f1,
    )
    flow_bootstrap = validate_bootstrap_evidence(
        args.flow_bootstrap,
        task="flow",
        accuracy=flow_accuracy,
        macro_f1=flow_macro_f1,
    )
    packet_session_novelty = validate_session_novelty_evidence(
        args.packet_session_novelty,
        task="packet",
        accuracy=packet_accuracy,
        macro_f1=packet_macro_f1,
    )
    flow_session_novelty = validate_session_novelty_evidence(
        args.flow_session_novelty,
        task="flow",
        accuracy=flow_accuracy,
        macro_f1=flow_macro_f1,
    )
    packet_spec = PACKET_LEVEL_RESULTS[args.dataset]
    packet_target_pass = (
        packet_accuracy >= float(packet_spec.target_accuracy)
        and packet_macro_f1 >= float(packet_spec.target_macro_f1)
    )
    flow_spec = FLOW_LEVEL_RESULTS[args.dataset]
    flow_target_pass = (
        (flow_spec.target_accuracy is None or flow_accuracy >= flow_spec.target_accuracy)
        and (flow_spec.target_macro_f1 is None or flow_macro_f1 >= flow_spec.target_macro_f1)
    )

    canonical_packet = Path(packet_spec.path)
    canonical_packet_novelty = (
        canonical_packet.parent / "session_novelty_strict_shared_core_v2.json"
    )
    canonical_packet_bootstrap = (
        canonical_packet.parent / "bootstrap_strict_shared_core_v2.json"
    )
    copied_manifests = []
    copied_flow_manifests = []
    packet_manifest_copies = []
    flow_manifest_copies = []
    for fold in range(3):
        source = (
            Path(args.packet_manifest_root)
            / args.dataset
            / f"fold{fold}"
            / "packet_framework_manifest.json"
        )
        manifest = load_json(source)
        notes = ((manifest.get("framework") or {}).get("notes") or {})
        if (
            not notes.get("completed")
            or (
                notes.get("shared_core_method_sha256")
                or notes.get("shared_core_config_sha256")
            )
            != fingerprint
        ):
            raise ValueError(f"packet manifest is incomplete or has wrong fingerprint: {source}")
        destination = (
            Path("reasoningDataset/packet-level")
            / args.dataset
            / f"fold{fold}"
            / "packet_framework_manifest.json"
        )
        packet_manifest_copies.append((source, destination))
        audit_payload = load_json(audit_paths[fold])
        flow_source = Path((audit_payload.get("inputs") or {}).get("flow_manifest", ""))
        flow_manifest = load_json(flow_source)
        flow_notes = ((flow_manifest.get("framework") or {}).get("notes") or {})
        if (
            not flow_notes.get("completed")
            or (
                flow_notes.get("shared_core_method_sha256")
                or flow_notes.get("shared_core_config_sha256")
            )
            != fingerprint
        ):
            raise ValueError(
                f"flow manifest is incomplete or has wrong fingerprint: {flow_source}"
            )
        flow_destination = (
            Path("reasoningDataset")
            / args.dataset
            / f"stage8_flowaware_manifest_paper_unified_strict_shared_core_v2_fold{fold}.json"
        )
        flow_manifest_copies.append((flow_source, flow_destination))

    canonical_flow = Path(flow_spec.path)
    canonical_flow_novelty = (
        canonical_flow.parent / "session_novelty_strict_shared_core_v2.json"
    )
    canonical_flow_bootstrap = (
        canonical_flow.parent / "bootstrap_strict_shared_core_v2.json"
    )
    if packet_target_pass:
        canonical_packet_novelty.parent.mkdir(parents=True, exist_ok=True)
        atomic_copy(args.packet_session_novelty, canonical_packet_novelty)
        atomic_copy(args.packet_bootstrap, canonical_packet_bootstrap)
        publish_canonical_result(
            packet_payload,
            canonical_packet,
            fingerprint=fingerprint,
            audit_paths=audit_paths,
            candidate=args.packet_candidate,
            session_novelty=str(canonical_packet_novelty),
            bootstrap_evidence=str(canonical_packet_bootstrap),
            method_archive_manifest=frozen_method_evidence["archive_manifest"],
            method_archive_manifest_sha256=frozen_method_evidence[
                "archive_manifest_sha256"
            ],
        )
        for source, destination in packet_manifest_copies:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            copied_manifests.append(str(destination))
    if flow_target_pass:
        canonical_flow_novelty.parent.mkdir(parents=True, exist_ok=True)
        atomic_copy(args.flow_session_novelty, canonical_flow_novelty)
        atomic_copy(args.flow_bootstrap, canonical_flow_bootstrap)
        publish_canonical_result(
            flow_payload,
            canonical_flow,
            fingerprint=fingerprint,
            audit_paths=audit_paths,
            candidate=args.flow_candidate,
            session_novelty=str(canonical_flow_novelty),
            bootstrap_evidence=str(canonical_flow_bootstrap),
            method_archive_manifest=frozen_method_evidence["archive_manifest"],
            method_archive_manifest_sha256=frozen_method_evidence[
                "archive_manifest_sha256"
            ],
        )
        for source, destination in flow_manifest_copies:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            copied_flow_manifests.append(str(destination))

    if packet_target_pass and flow_target_pass:
        status = "published"
    elif packet_target_pass:
        status = "packet_published_flow_target_gap"
    elif flow_target_pass:
        status = "flow_published_packet_target_gap"
    else:
        status = "packet_and_flow_target_gap"

    report = {
        "status": status,
        "dataset": args.dataset,
        "shared_core_method_sha256": fingerprint,
        "shared_core_config_sha256": fingerprint,
        "audit_paths": [str(path) for path in audit_paths],
        "audit_evidence": [
            {"path": str(path), "sha256": sha256_file(path)}
            for path in audit_paths
        ],
        "frozen_method_evidence": frozen_method_evidence,
        "uncertainty_evidence": {
            "packet": {
                "path": args.packet_bootstrap,
                "sha256": sha256_file(args.packet_bootstrap),
                "method": packet_bootstrap["method"],
                "metrics": packet_bootstrap["metrics"],
            },
            "flow": {
                "path": args.flow_bootstrap,
                "sha256": sha256_file(args.flow_bootstrap),
                "method": flow_bootstrap["method"],
                "metrics": flow_bootstrap["metrics"],
            },
        },
        "session_novelty_evidence": {
            "packet": {
                "path": args.packet_session_novelty,
                "sha256": sha256_file(args.packet_session_novelty),
                "seen_minus_novel_gaps": packet_session_novelty[
                    "seen_minus_novel_gaps"
                ],
            },
            "flow": {
                "path": args.flow_session_novelty,
                "sha256": sha256_file(args.flow_session_novelty),
                "seen_minus_novel_gaps": flow_session_novelty[
                    "seen_minus_novel_gaps"
                ],
            },
        },
        "packet": {
            "published": packet_target_pass,
            "accuracy": packet_accuracy,
            "macro_f1": packet_macro_f1,
            "target_accuracy": packet_spec.target_accuracy,
            "target_macro_f1": packet_spec.target_macro_f1,
            "candidate": args.packet_candidate,
            "candidate_sha256": sha256_file(args.packet_candidate),
            "canonical": str(canonical_packet) if packet_target_pass else "",
            "canonical_sha256": (
                sha256_file(canonical_packet) if packet_target_pass else None
            ),
        },
        "flow": {
            "published": flow_target_pass,
            "accuracy": flow_accuracy,
            "macro_f1": flow_macro_f1,
            "target_accuracy": flow_spec.target_accuracy,
            "target_macro_f1": flow_spec.target_macro_f1,
            "candidate": args.flow_candidate,
            "candidate_sha256": sha256_file(args.flow_candidate),
            "canonical": str(canonical_flow) if flow_target_pass else "",
            "canonical_sha256": (
                sha256_file(canonical_flow) if flow_target_pass else None
            ),
        },
        "canonical_packet_manifests": copied_manifests,
        "canonical_flow_manifests": copied_flow_manifests,
    }
    atomic_write_json(output, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
