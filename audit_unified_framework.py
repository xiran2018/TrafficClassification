#!/usr/bin/env python3
"""Audit the current unified packet-to-flow paper contract."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple

from unified_framework_spec import (
    ABLATION_ONLY_MODULES,
    FLOW_LEVEL_RESULTS,
    PAPER_UNIFIED_SHARED_STATUS_ALIASES,
    FRAMEWORK_PROFILES,
    MODEL_SHARED_CORE_MODULES,
    PAPER_MAIN_MODULES,
    PACKET_LEVEL_RESULTS,
    SHARED_CORE_MODULES,
    SHARED_PROTOCOL_GUARDS,
    UNIFIED_CANDIDATE_EXPERTS,
    framework_profile_fingerprint,
    shared_module_overlap,
    task_modules,
)


REQUIRED_PROFILE = "paper_unified"
REQUIRED_SHARED_STATUS = FRAMEWORK_PROFILES[REQUIRED_PROFILE]["shared_module_status"]


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_metrics(data: Dict[str, Any], task: str) -> Tuple[float | None, float | None, int | None]:
    if task == "flow-level":
        metrics = data.get("metrics")
        if isinstance(metrics, dict):
            flow = metrics.get("flow_level")
            if isinstance(flow, dict):
                return flow.get("accuracy"), flow.get("macro_f1"), flow.get("num_flows")
        return None, None, None

    for key in ("metrics", "test_metrics", "packet_level"):
        metrics = data.get(key)
        if isinstance(metrics, dict):
            packet = metrics.get("packet_level")
            if isinstance(packet, dict):
                acc = packet.get("accuracy")
                f1 = packet.get("macro_f1")
                n = packet.get("num_samples") or packet.get("num_packets")
                if acc is not None or f1 is not None:
                    return acc, f1, n
            acc = metrics.get("accuracy")
            f1 = metrics.get("macro_f1")
            n = metrics.get("num_samples") or metrics.get("num_packets")
            if acc is not None or f1 is not None:
                return acc, f1, n
    return None, None, None


def canonical_result_path_status(spec) -> Dict[str, Any]:
    path = str(spec.path)
    if spec.task == "flow-level":
        expected = f"reasoningDataset/{spec.dataset}/<paper-result>.json"
        ok = path.startswith(f"reasoningDataset/{spec.dataset}/") and not path.startswith("/tmp/")
    elif spec.task == "packet-level":
        expected = f"reasoningDataset/packet-level/{spec.dataset}/paper_default_result.json"
        ok = path == expected
    else:
        expected = "reasoningDataset/<dataset>/<paper-result>.json"
        ok = path.startswith("reasoningDataset/") and not path.startswith("/tmp/")
    return {"ok": ok, "expected": expected, "actual": path}


def audit_result(spec) -> Dict[str, Any]:
    path = Path(spec.path)
    canonical_path = canonical_result_path_status(spec)
    row: Dict[str, Any] = {
        "dataset": spec.dataset,
        "task": spec.task,
        "path": spec.path,
        "exists": path.exists(),
        "target_accuracy": spec.target_accuracy,
        "target_macro_f1": spec.target_macro_f1,
        "required_framework_profile": spec.framework_profile,
        "framework_manifest_glob": spec.framework_manifest_glob,
        "note": spec.note,
        "canonical_result_path": canonical_path,
    }
    provenance = audit_result_provenance(spec)
    row["framework_provenance"] = provenance
    if not path.exists():
        row["metric_status"] = "missing"
        row["publication_status"] = "missing_result"
        row["status"] = row["publication_status"]
        return row
    data = load_json(spec.path)
    acc, f1, n = extract_metrics(data, spec.task)
    row.update({"accuracy": acc, "macro_f1": f1, "num_samples": n})
    if acc is None or f1 is None:
        row["metric_status"] = "metrics_missing"
        row["publication_status"] = "metrics_missing"
        row["status"] = row["publication_status"]
        return row
    point_ok = True
    if spec.target_accuracy is not None:
        point_ok = point_ok and float(acc) >= spec.target_accuracy
    if spec.target_macro_f1 is not None:
        point_ok = point_ok and float(f1) >= spec.target_macro_f1
    row["metric_status"] = "pass" if point_ok else "target_gap"
    if not point_ok:
        row["publication_status"] = "target_gap"
    elif not canonical_path["ok"]:
        row["publication_status"] = "noncanonical_result_path"
    elif provenance["status"] == "pass":
        row["publication_status"] = "pass"
    else:
        row["publication_status"] = "needs_paper_unified_repro"
    row["status"] = row["publication_status"]
    return row


def audit_result_provenance(spec) -> Dict[str, Any]:
    paths = sorted(Path().glob(spec.framework_manifest_glob)) if spec.framework_manifest_glob else []
    rows = [audit_manifest(path, spec.task) for path in paths]
    candidates = [
        row
        for row in rows
        if row.get("status") == "pass"
        and row.get("framework_profile") == spec.framework_profile
        and row.get("dataset") == spec.dataset
        and row.get("execution_status") == "candidate_executed"
    ]
    if spec.task == "flow-level":
        fold_rows = [
            row for row in candidates
            if row.get("stage") in {"all", "paper_unified"} and row.get("fold") in {0, 1, 2}
        ]
        completed_folds = sorted({int(row["fold"]) for row in fold_rows})
        result_bound_rows = [
            row for row in candidates
            if row.get("stage") == "paper_result_binding"
            and spec.path in set(row.get("published_result_paths") or [])
        ]
        matching = fold_rows + result_bound_rows
        if len(completed_folds) < 3:
            status = "insufficient_fold_manifests"
        elif not result_bound_rows:
            status = "missing_bound_result_manifest"
        else:
            status = "pass"
    elif spec.task == "packet-level":
        fold_rows = [row for row in candidates if manifest_fold(row.get("path", "")) is not None]
        completed_folds = sorted(
            {
                fold
                for row in fold_rows
                for fold in [manifest_fold(row.get("path", ""))]
                if fold is not None
            }
        )
        result_bound_rows = [
            row
            for row in candidates
            if row.get("stage") == "paper_result_binding"
            and spec.path in set(row.get("published_result_paths") or [])
        ]
        matching = fold_rows + result_bound_rows
        if len(completed_folds) < 3:
            status = "insufficient_fold_manifests"
        elif not result_bound_rows:
            status = "missing_bound_result_manifest"
        else:
            status = "pass"
    else:
        matching = candidates
        status = "pass" if matching else "missing_or_nonmatching_manifest"
    result = {
        "status": status,
        "glob": spec.framework_manifest_glob,
        "required_profile": spec.framework_profile,
        "matching_manifest_count": len(matching),
        "candidate_manifest_count": len(candidates),
        "manifest_count": len(rows),
        "matching_manifests": [row["path"] for row in matching],
        "required_result_path": spec.path if spec.task == "flow-level" else "",
    }
    if spec.task == "flow-level":
        result.update(
            {
                "fold_manifest_count": len(fold_rows),
                "completed_folds": completed_folds,
                "result_bound_manifest_count": len(result_bound_rows),
                "result_bound_manifests": [row["path"] for row in result_bound_rows],
            }
        )
    if spec.task == "packet-level":
        result.update(
            {
                "required_result_path": spec.path,
                "fold_manifest_count": len(fold_rows),
                "completed_folds": completed_folds,
                "result_bound_manifest_count": len(result_bound_rows),
                "result_bound_manifests": [row["path"] for row in result_bound_rows],
            }
        )
    return result


def manifest_fold(path: str) -> int | None:
    for part in Path(str(path)).parts:
        if part.startswith("fold") and part[4:].isdigit():
            return int(part[4:])
    return None


def audit_manifest(path: Path, task: str) -> Dict[str, Any]:
    row: Dict[str, Any] = {"path": str(path), "task": task, "exists": path.exists()}
    if not path.exists():
        row["status"] = "missing"
        return row
    data = load_json(str(path))
    framework = data.get("framework")
    if not isinstance(framework, dict):
        row["status"] = "legacy_framework_missing"
        return row
    shared = tuple(framework.get("shared_core_modules") or [])
    model_shared = tuple(framework.get("model_shared_core_modules") or [])
    protocol_guards = tuple(framework.get("shared_protocol_guards") or [])
    shared_status = framework.get("shared_module_status") or {}
    notes = framework.get("notes") or {}
    task_status = framework.get("task_module_status") or {}
    run_tag = data.get("run_tag") or notes.get("run_tag") or ""
    stage = framework.get("stage") or data.get("stage")
    expected_fingerprint = (
        framework_profile_fingerprint(REQUIRED_PROFILE, task)
        if (framework.get("framework_profile") or notes.get("framework_profile")) == REQUIRED_PROFILE
        else None
    )
    actual_fingerprint = framework.get("framework_profile_fingerprint")
    dry_run = bool(notes.get("dry_run") or data.get("dry_run"))
    completed = bool(notes.get("completed"))
    result_paths = notes.get("published_result_paths") or notes.get("result_paths") or []
    if isinstance(result_paths, str):
        result_paths = [result_paths]
    partial_stage = (
        (task == "flow-level" and stage not in {"all", "paper_unified", "paper_result_binding"})
        or (task == "packet-level" and stage not in {"paper_unified", "paper_result_binding"})
    )
    smoke = "smoke" in str(run_tag).lower() or "manifest" in str(run_tag).lower()
    execution_status = "candidate_executed"
    if dry_run:
        execution_status = "dry_run"
    elif not completed:
        execution_status = "not_completed"
    elif smoke or partial_stage:
        execution_status = "planned_or_partial"
    row.update(
        {
            "framework_name": framework.get("framework_name"),
            "framework_profile": framework.get("framework_profile") or notes.get("framework_profile"),
            "dataset": framework.get("dataset"),
            "stage": stage,
            "fold": notes.get("fold", data.get("fold")),
            "run_tag": run_tag,
            "dry_run": dry_run,
            "completed": completed,
            "execution_status": execution_status,
            "input_unit": framework.get("input_unit"),
            "published_result_paths": result_paths,
            "shared_core_modules": list(shared),
            "shared_module_status": shared_status,
            "task_module_status": task_status,
            "matches_shared_core": shared == SHARED_CORE_MODULES,
            "matches_model_shared_core": model_shared == MODEL_SHARED_CORE_MODULES,
            "matches_shared_protocol_guards": protocol_guards == SHARED_PROTOCOL_GUARDS,
            "matches_shared_status": shared_status_matches(shared_status),
            "framework_profile_fingerprint": actual_fingerprint,
            "expected_framework_profile_fingerprint": expected_fingerprint,
            "matches_framework_profile_fingerprint": (
                expected_fingerprint is None or actual_fingerprint == expected_fingerprint
            ),
            "shared_core_config": notes.get("shared_core_config", ""),
            "shared_core_method_sha256": notes.get("shared_core_method_sha256", "")
            or notes.get("shared_core_config_sha256", ""),
            "shared_core_config_sha256": notes.get("shared_core_config_sha256", ""),
        }
    )
    expected_packet_classifier = (
        "shared_representation_single_head_crossfold_consensus"
        if stage == "paper_result_binding"
        else "shared_representation_single_head"
    )
    packet_main_ok = not (
        task == "packet-level"
        and row.get("framework_profile") == REQUIRED_PROFILE
        and (
            task_status.get("packet_level_classifier") != expected_packet_classifier
            or notes.get("paper_main_experts") != []
            or notes.get("semantic_fusion_level") != "representation"
        )
    )
    row["matches_task_main_path"] = packet_main_ok
    if not row["matches_shared_core"] or not row["matches_model_shared_core"]:
        row["status"] = "shared_core_mismatch"
    elif not row["matches_shared_protocol_guards"]:
        row["status"] = "shared_protocol_mismatch"
    elif row.get("framework_profile") == REQUIRED_PROFILE and not row["matches_shared_status"]:
        row["status"] = "shared_status_mismatch"
    elif row.get("framework_profile") == REQUIRED_PROFILE and not actual_fingerprint:
        row["status"] = "profile_fingerprint_missing"
    elif row.get("framework_profile") == REQUIRED_PROFILE and not row["matches_framework_profile_fingerprint"]:
        row["status"] = "profile_fingerprint_mismatch"
    elif not packet_main_ok:
        row["status"] = "task_main_path_mismatch"
    else:
        row["status"] = "pass"
    return row


def shared_status_matches(shared_status: Dict[str, Any]) -> bool:
    for module, required in REQUIRED_SHARED_STATUS.items():
        accepted = PAPER_UNIFIED_SHARED_STATUS_ALIASES.get(module, (required,))
        if shared_status.get(module) not in accepted:
            return False
    return True


def discover_manifests() -> Dict[str, Any]:
    flow_candidates = sorted(Path("reasoningDataset").glob("*/stage8_flowaware_manifest_*.json"))
    packet_candidates = sorted(Path("reasoningDataset/packet-level").glob("*/*/packet_framework_manifest.json"))
    return {
        "flow_level": [audit_manifest(path, "flow-level") for path in flow_candidates],
        "packet_level": [audit_manifest(path, "packet-level") for path in packet_candidates],
    }


def audit_shared_core_v2_fingerprints(
    flow_rows: list[Dict[str, Any]],
    packet_rows: list[Dict[str, Any]],
) -> Dict[str, Any]:
    required_datasets = ("vpn-app", "tls-120")
    required_folds = {0, 1, 2}
    groups: Dict[str, Any] = {}
    all_fingerprints: set[str] = set()
    all_method_fingerprints: set[str] = set()
    for task, rows in (("flow-level", flow_rows), ("packet-level", packet_rows)):
        for dataset in required_datasets:
            selected = []
            for row in rows:
                fold = row.get("fold")
                if fold is None:
                    fold = manifest_fold(row.get("path", ""))
                if (
                    row.get("status") == "pass"
                    and row.get("execution_status") == "candidate_executed"
                    and row.get("framework_profile") == REQUIRED_PROFILE
                    and row.get("dataset") == dataset
                    and fold in required_folds
                    and (
                        row.get("shared_core_method_sha256")
                        or row.get("shared_core_config_sha256")
                    )
                ):
                    selected.append({**row, "resolved_fold": int(fold)})
            folds = {row["resolved_fold"] for row in selected}
            fingerprints = {str(row["shared_core_config_sha256"]) for row in selected}
            method_fingerprints = {
                str(
                    row.get("shared_core_method_sha256")
                    or row["shared_core_config_sha256"]
                )
                for row in selected
            }
            all_fingerprints.update(fingerprints)
            all_method_fingerprints.update(method_fingerprints)
            key = f"{task}:{dataset}"
            groups[key] = {
                "completed_folds": sorted(folds),
                "fingerprints": sorted(fingerprints),
                "method_fingerprints": sorted(method_fingerprints),
                "manifest_paths": [row["path"] for row in selected],
                "status": (
                    "pass"
                    if folds == required_folds and len(fingerprints) == 1
                    else "missing_or_inconsistent"
                ),
                "unified_method_status": (
                    "pass"
                    if folds == required_folds and len(method_fingerprints) == 1
                    else "missing_or_inconsistent"
                ),
            }
    all_groups_pass = all(row["status"] == "pass" for row in groups.values())
    all_method_groups_pass = all(
        row["unified_method_status"] == "pass" for row in groups.values()
    )
    cross_task_match = len(all_fingerprints) == 1
    cross_task_method_match = len(all_method_fingerprints) == 1
    return {
        "claim": "one_frozen_shared_core_config_across_datasets_tasks_and_folds",
        "status": "pass" if all_groups_pass and cross_task_match else "not_ready",
        "unified_method_status": (
            "pass"
            if all_method_groups_pass and cross_task_method_match
            else "not_ready"
        ),
        "required_datasets": list(required_datasets),
        "required_folds": sorted(required_folds),
        "groups": groups,
        "observed_fingerprints": sorted(all_fingerprints),
        "observed_method_fingerprints": sorted(all_method_fingerprints),
        "cross_task_cross_dataset_fingerprint_match": cross_task_match,
        "cross_task_cross_dataset_method_match": cross_task_method_match,
    }


def audit_cross_task_checkpoint_reports() -> Dict[str, Any]:
    required_datasets = ("vpn-app", "tls-120")
    required_folds = {0, 1, 2}
    groups: Dict[str, Any] = {}
    all_fingerprints: set[str] = set()
    all_method_fingerprints: set[str] = set()
    for dataset in required_datasets:
        rows = []
        for path in sorted(
            Path("reasoningDataset/shared-core-audits").glob(
                f"{dataset}/fold*/audit.json"
            )
        ):
            payload = load_json(str(path))
            rows.append({**payload, "path": str(path)})
        passing = [row for row in rows if row.get("status") == "pass"]
        folds = {int(row["fold"]) for row in passing if row.get("fold") is not None}
        fingerprints = {
            str(row["shared_core_config_sha256"])
            for row in passing
            if row.get("shared_core_config_sha256")
        }
        method_fingerprints = {
            str(row.get("shared_core_method_sha256") or row["shared_core_config_sha256"])
            for row in passing
            if row.get("shared_core_method_sha256")
            or row.get("shared_core_config_sha256")
        }
        all_fingerprints.update(fingerprints)
        all_method_fingerprints.update(method_fingerprints)
        groups[dataset] = {
            "completed_folds": sorted(folds),
            "fingerprints": sorted(fingerprints),
            "method_fingerprints": sorted(method_fingerprints),
            "report_paths": [row["path"] for row in passing],
            "status": (
                "pass"
                if folds == required_folds and len(fingerprints) == 1
                else "missing_or_inconsistent"
            ),
            "unified_method_status": (
                "pass"
                if folds == required_folds and len(method_fingerprints) == 1
                else "missing_or_inconsistent"
            ),
        }
    all_groups_pass = all(row["status"] == "pass" for row in groups.values())
    all_method_groups_pass = all(
        row["unified_method_status"] == "pass" for row in groups.values()
    )
    fingerprint_match = len(all_fingerprints) == 1
    method_fingerprint_match = len(all_method_fingerprints) == 1
    return {
        "claim": "checkpoint_proven_exact_shared_packet_module_for_every_dataset_fold",
        "status": "pass" if all_groups_pass and fingerprint_match else "not_ready",
        "unified_method_status": (
            "pass"
            if all_method_groups_pass and method_fingerprint_match
            else "not_ready"
        ),
        "required_datasets": list(required_datasets),
        "required_folds": sorted(required_folds),
        "groups": groups,
        "observed_fingerprints": sorted(all_fingerprints),
        "observed_method_fingerprints": sorted(all_method_fingerprints),
        "cross_dataset_fingerprint_match": fingerprint_match,
        "cross_dataset_method_match": method_fingerprint_match,
    }


def build_report() -> Dict[str, Any]:
    shared = shared_module_overlap()
    flow_modules = task_modules("flow-level")
    packet_modules = task_modules("packet-level")
    flow_rows = [audit_result(spec) for spec in FLOW_LEVEL_RESULTS.values()]
    packet_rows = [audit_result(spec) for spec in PACKET_LEVEL_RESULTS.values()]
    manifests = discover_manifests()
    exact_v2_configuration = audit_shared_core_v2_fingerprints(
        manifests["flow_level"], manifests["packet_level"]
    )
    exact_v2_checkpoints = audit_cross_task_checkpoint_reports()
    exact_v2 = {
        "status": (
            "pass"
            if exact_v2_configuration["status"] == "pass"
            and exact_v2_checkpoints["status"] == "pass"
            else "not_ready"
        ),
        "configuration_provenance": exact_v2_configuration,
        "checkpoint_provenance": exact_v2_checkpoints,
    }
    unified_method_v2 = {
        "status": (
            "pass"
            if exact_v2_configuration["unified_method_status"] == "pass"
            and exact_v2_checkpoints["unified_method_status"] == "pass"
            else "not_ready"
        ),
        "claim": (
            "one_shared_module_algorithm_and_training_protocol_with_"
            "independently_optimized_numeric_hyperparameters"
        ),
        "configuration_provenance": exact_v2_configuration,
        "checkpoint_provenance": exact_v2_checkpoints,
    }
    manifest_rows = manifests["flow_level"] + manifests["packet_level"]
    flow_manifest_passes = sum(1 for row in manifests["flow_level"] if row.get("status") == "pass")
    packet_manifest_passes = sum(1 for row in manifests["packet_level"] if row.get("status") == "pass")
    paper_unified_flow_passes = sum(
        1
        for row in manifests["flow_level"]
        if row.get("status") == "pass" and row.get("framework_profile") == "paper_unified"
        and row.get("execution_status") == "candidate_executed"
    )
    paper_unified_packet_passes = sum(
        1
        for row in manifests["packet_level"]
        if row.get("status") == "pass" and row.get("framework_profile") == "paper_unified"
        and row.get("execution_status") == "candidate_executed"
    )
    manifest_status_ok = (
        flow_manifest_passes > 0
        and packet_manifest_passes > 0
        and paper_unified_flow_passes > 0
        and paper_unified_packet_passes > 0
        and all(row.get("status") in {"pass", "legacy_framework_missing"} for row in manifest_rows)
    )
    return {
        "framework": {
            "name": "unified_shortcut_resistant_packet_to_flow",
            "paper_main_modules": list(PAPER_MAIN_MODULES),
            "shared_core_modules": list(SHARED_CORE_MODULES),
            "model_shared_core_modules": list(MODEL_SHARED_CORE_MODULES),
            "shared_protocol_guards": list(SHARED_PROTOCOL_GUARDS),
            "flow_modules": list(flow_modules),
            "packet_modules": list(packet_modules),
            "unified_candidate_experts": list(UNIFIED_CANDIDATE_EXPERTS),
            "ablation_only_modules": list(ABLATION_ONLY_MODULES),
            "expected_flow_profile_fingerprint": framework_profile_fingerprint(REQUIRED_PROFILE, "flow-level"),
            "expected_packet_profile_fingerprint": framework_profile_fingerprint(REQUIRED_PROFILE, "packet-level"),
            "shared_module_overlap": list(shared),
            "all_shared_modules_present_in_both_tasks": tuple(shared) == SHARED_CORE_MODULES,
            "flow_scope": list(FLOW_LEVEL_RESULTS),
            "packet_scope": list(PACKET_LEVEL_RESULTS),
            "flow_scope_excludes_ustc": "ustc-app" not in FLOW_LEVEL_RESULTS and "ustc-binary" not in FLOW_LEVEL_RESULTS,
            "flow_runner_manifest_passes": flow_manifest_passes,
            "packet_runner_manifest_passes": packet_manifest_passes,
            "paper_unified_flow_manifest_passes": paper_unified_flow_passes,
            "paper_unified_packet_manifest_passes": paper_unified_packet_passes,
        },
        "flow_level": flow_rows,
        "packet_level": packet_rows,
        "runner_manifests": manifests,
        "exact_shared_core_v2": exact_v2,
        "unified_method_v2": unified_method_v2,
        "status": "pass"
        if (
            tuple(shared) == SHARED_CORE_MODULES
            and all(row.get("publication_status") == "pass" for row in flow_rows)
            and all(row.get("publication_status") == "pass" for row in packet_rows)
            and manifest_status_ok
            and exact_v2["status"] == "pass"
        )
        else "review",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_json", default="reasoningDataset/unified_framework_audit.json")
    ap.add_argument("--output_md", default="reasoningDataset/unified_framework_audit.md")
    args = ap.parse_args()

    report = build_report()
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    lines = [
        "# Unified Framework Audit",
        "",
        f"Status: `{report['status']}`",
        f"Exact common-reference v2: `{report['exact_shared_core_v2']['status']}`",
        f"Unified method v2: `{report['unified_method_v2']['status']}`",
        "",
        "Expected profile fingerprints:",
        "",
        f"- flow-level: `{report['framework']['expected_flow_profile_fingerprint']}`",
        f"- packet-level: `{report['framework']['expected_packet_profile_fingerprint']}`",
        "",
        "## Shared Representation",
        "",
    ]
    for module in report["framework"]["model_shared_core_modules"]:
        lines.append(f"- `{module}`")
    lines += ["", "## Shared Protocol Guards", ""]
    for module in report["framework"]["shared_protocol_guards"]:
        lines.append(f"- `{module}`")
    lines += ["", "## Unified Candidate Experts", ""]
    for module in report["framework"]["unified_candidate_experts"]:
        lines.append(f"- `{module}`")
    lines += ["", "## Ablation-Only Modules", ""]
    for module in report["framework"]["ablation_only_modules"]:
        lines.append(f"- `{module}`")
    lines += [
        "",
        "## Flow-Level Results",
        "",
        "| Dataset | Accuracy | Macro-F1 | Metric Status | Publication Status | Canonical Path | Provenance | Path |",
        "| --- | ---: | ---: | --- | --- | --- | ---: | --- |",
    ]
    for row in report["flow_level"]:
        prov = row.get("framework_provenance") or {}
        canonical = row.get("canonical_result_path") or {}
        lines.append(
            f"| {row['dataset']} | {row.get('accuracy', '-')!s} | {row.get('macro_f1', '-')!s} | {row['metric_status']} | {row['publication_status']} | {canonical.get('ok')} | {prov.get('matching_manifest_count', 0)} | `{row['path']}` |"
        )
    lines += [
        "",
        "## Packet-Level Results",
        "",
        "| Dataset | Accuracy | Macro-F1 | Metric Status | Publication Status | Canonical Path | Provenance | Path |",
        "| --- | ---: | ---: | --- | --- | --- | ---: | --- |",
    ]
    for row in report["packet_level"]:
        prov = row.get("framework_provenance") or {}
        canonical = row.get("canonical_result_path") or {}
        lines.append(
            f"| {row['dataset']} | {row.get('accuracy', '-')!s} | {row.get('macro_f1', '-')!s} | {row['metric_status']} | {row['publication_status']} | {canonical.get('ok')} | {prov.get('matching_manifest_count', 0)} | `{row['path']}` |"
        )
    lines += ["", "## Runner Manifests", "", "| Task | Manifests | Passing | paper_unified Passing |", "| --- | ---: | ---: | ---: |"]
    for task in ("flow_level", "packet_level"):
        rows = report["runner_manifests"][task]
        passing = sum(1 for row in rows if row.get("status") == "pass")
        unified_passing = sum(
            1
            for row in rows
            if row.get("status") == "pass" and row.get("framework_profile") == "paper_unified"
            and row.get("execution_status") == "candidate_executed"
        )
        lines.append(f"| {task} | {len(rows)} | {passing} | {unified_passing} |")

    out_md = Path(args.output_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output_json": str(out_json), "output_md": str(out_md)}, indent=2))


if __name__ == "__main__":
    main()
