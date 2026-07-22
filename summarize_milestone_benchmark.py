#!/usr/bin/env python3
"""Audit and summarize a frozen-method development Test milestone."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from compare_sweet_reference import SWEET_REFERENCES, compare_metrics
from unified_framework_spec import FLOW_LEVEL_RESULTS, PACKET_LEVEL_RESULTS


DATASETS = ("vpn-app", "tls-120")
TAG = "milestone_dev_fold0"
ALLOWED_DECISIONS = {
    "base_frozen_pending_identity_cross_scale_validation",
    "final_after_preregistered_validation",
}


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_notes(manifest: dict[str, Any]) -> dict[str, Any]:
    notes = (manifest.get("framework") or {}).get("notes") or {}
    if not isinstance(notes, dict):
        raise ValueError("manifest framework.notes must be an object")
    return notes


def result_path(manifest: dict[str, Any], needle: str) -> Path:
    matches = [
        Path(str(path))
        for path in manifest_notes(manifest).get("result_paths") or []
        if needle in str(path) and str(path).endswith(".json")
    ]
    if len(matches) != 1 or not matches[0].is_file():
        raise ValueError(f"expected one existing result containing {needle!r}: {matches}")
    return matches[0].resolve()


def metric_summary(payload: dict[str, Any], task: str) -> dict[str, Any]:
    metrics = payload.get("metrics") or {}
    if task == "flow":
        metrics = metrics.get("flow_level") or {}
    num_samples = metrics.get("num_samples")
    if num_samples is None:
        num_samples = (metrics.get("calibration") or {}).get("num_samples")
    if num_samples is None and task == "flow":
        labels = payload.get("flow_y_true") or []
        num_samples = len(labels) if labels else None
    if num_samples is None:
        raise ValueError(f"{task} result does not report its sample count")
    return {
        "num_samples": int(num_samples),
        "accuracy": float(metrics["accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
    }


def require_stable_source(manifest: dict[str, Any], path: Path) -> str:
    evidence = manifest_notes(manifest).get("algorithm_source_evidence") or {}
    launch = str(evidence.get("launch_fingerprint") or "")
    if not (
        evidence.get("status") == "pass"
        and len(launch) == 64
        and evidence.get("completion_fingerprint") == launch
        and evidence.get("changed_paths") == []
    ):
        raise ValueError(f"algorithm source stability failed: {path}")
    return launch


def benchmark_comparison(
    dataset: str, task: str, metrics: dict[str, Any]
) -> dict[str, Any]:
    task_name = "packet-level" if task == "packet" else "flow-level"
    references = SWEET_REFERENCES[task_name][dataset]
    specification = (
        PACKET_LEVEL_RESULTS if task == "packet" else FLOW_LEVEL_RESULTS
    )[dataset]
    accuracy = float(metrics["accuracy"])
    macro_f1 = float(metrics["macro_f1"])
    target_accuracy = specification.target_accuracy
    target_macro_f1 = specification.target_macro_f1
    comparisons = {
        name: compare_metrics(accuracy, macro_f1, reference)
        for name, reference in references.items()
    }
    return {
        "predeclared_target": {
            "accuracy": target_accuracy,
            "macro_f1": target_macro_f1,
            "met": bool(
                target_accuracy is not None
                and target_macro_f1 is not None
                and accuracy >= target_accuracy
                and macro_f1 >= target_macro_f1
            ),
        },
        "sweet": comparisons,
        "headline_sweet_claim": (
            "exceeds_protocol_matched_end_to_end"
            if comparisons["end_to_end"]["exceeds_both"]
            else "does_not_exceed_protocol_matched_end_to_end"
        ),
    }


def validate_manifest(
    manifest: dict[str, Any],
    path: Path,
    *,
    dataset: str,
    task: str,
    config_sha: str,
) -> str:
    framework = manifest.get("framework") or {}
    notes = manifest_notes(manifest)
    if not (
        manifest.get("dataset") == dataset
        and int(manifest.get("fold", -1)) == 0
        and framework.get("task") == task
        and notes.get("completed") is True
        and notes.get("shared_core_config_sha256") == config_sha
    ):
        raise ValueError(f"mismatched or incomplete {task} manifest: {path}")
    return require_stable_source(manifest, path)


def summarize(
    root: Path, repo: Path, config_path: Path, *, tag: str = TAG
) -> dict[str, Any]:
    root = root.resolve()
    repo = repo.resolve()
    config_path = config_path.resolve()
    config = load_json(config_path)
    config_sha = str(config.get("config_sha256") or "")
    if not (
        config.get("method_selection", {}).get("decision_status")
        in ALLOWED_DECISIONS
        and config.get("selection_protocol", {}).get("test_evaluation_allowed")
        is True
        and config.get("method_selection", {}).get("test_labels_used") is False
        and len(config_sha) == 64
    ):
        raise ValueError("method config is not frozen from validation")

    report: dict[str, Any] = {
        "schema": "unified_milestone_development_benchmark_v1",
        "status": "pass",
        "evaluation_role": config.get("selection_protocol", {}).get(
            "test_evaluation_role", "development_benchmark_after_validation_freeze"
        ),
        "may_inform_future_method_design": True,
        "unbiased_final_claim_allowed": False,
        "test_labels_used_for_frozen_config_selection": False,
        "required_final_evaluation": (
            "new_outer_holdout_or_nested_cross_validation_if_feedback_is_used"
        ),
        "method_config": {
            "path": str(config_path),
            "file_sha256": file_sha256(config_path),
            "config_sha256": config_sha,
            "selected_method": config["method_selection"]["selected_method"],
        },
        "datasets": {},
    }
    source_fingerprints: set[str] = set()
    for dataset in DATASETS:
        audit_path = root / "audits" / dataset / "fold0" / "audit.json"
        audit = load_json(audit_path)
        if not (
            audit.get("status") == "pass"
            and audit.get("dataset") == dataset
            and int(audit.get("fold", -1)) == 0
            and audit.get("shared_core_config_sha256") == config_sha
        ):
            raise ValueError(f"cross-task audit failed or mismatched: {audit_path}")

        packet_manifest_path = (
            root / "packet_artifacts" / dataset / "fold0" / "packet_framework_manifest.json"
        )
        packet_manifest = load_json(packet_manifest_path)
        flow_matches = list(
            (repo / "reasoningDataset" / dataset).glob(
                f"stage8_flowaware_manifest_*{tag}*.json"
            )
        )
        if len(flow_matches) != 1:
            raise ValueError(f"expected one milestone Flow manifest: {flow_matches}")
        flow_manifest_path = flow_matches[0].resolve()
        flow_manifest = load_json(flow_manifest_path)
        source_fingerprints.add(
            validate_manifest(
                packet_manifest,
                packet_manifest_path,
                dataset=dataset,
                task="packet-level",
                config_sha=config_sha,
            )
        )
        source_fingerprints.add(
            validate_manifest(
                flow_manifest,
                flow_manifest_path,
                dataset=dataset,
                task="flow-level",
                config_sha=config_sha,
            )
        )

        packet_result = result_path(packet_manifest, "test_unified_packet_single_head")
        flow_result = result_path(flow_manifest, "test_seq_metrics")
        packet_metrics = metric_summary(load_json(packet_result), "packet")
        flow_metrics = metric_summary(load_json(flow_result), "flow")
        report["datasets"][dataset] = {
            "packet": {
                "metrics": packet_metrics,
                "comparison": benchmark_comparison(dataset, "packet", packet_metrics),
                "result": str(packet_result),
                "result_sha256": file_sha256(packet_result),
            },
            "flow": {
                "metrics": flow_metrics,
                "comparison": benchmark_comparison(dataset, "flow", flow_metrics),
                "result": str(flow_result),
                "result_sha256": file_sha256(flow_result),
            },
            "cross_task_audit": {
                "path": str(audit_path.resolve()),
                "sha256": file_sha256(audit_path),
            },
        }
    if len(source_fingerprints) != 1:
        raise ValueError(
            f"Packet/Flow algorithm source fingerprints differ: {sorted(source_fingerprints)}"
        )
    report["algorithm_source_fingerprint"] = next(iter(source_fingerprints))
    return report


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Unified Milestone Development Benchmark",
        "",
        "These Test results were produced after validation freeze, but may guide later method development. They are not an unbiased final paper claim.",
        "",
        "| Dataset | Task | Samples | Accuracy | Macro-F1 | Target | Sweet end-to-end |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        for task in ("packet", "flow"):
            metrics = report["datasets"][dataset][task]["metrics"]
            comparison = report["datasets"][dataset][task]["comparison"]
            lines.append(
                f"| {dataset} | {task} | {metrics['num_samples']} | "
                f"{metrics['accuracy'] * 100:.2f}% | "
                f"{metrics['macro_f1'] * 100:.2f}% | "
                f"{'PASS' if comparison['predeclared_target']['met'] else 'FAIL'} | "
                f"{'PASS' if comparison['sweet']['end_to_end']['exceeds_both'] else 'FAIL'} |"
            )
    lines.extend(
        [
            "",
            "If these results influence any later design decision, the final paper evaluation must use a new outer holdout or nested cross-validation.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", required=True)
    parser.add_argument("--tag", default=TAG)
    args = parser.parse_args()
    report = summarize(
        Path(args.root), Path(args.repo), Path(args.config), tag=args.tag
    )
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    output_md.write_text(markdown(report), encoding="utf-8")
    print(json.dumps({"status": "pass", "output_json": str(output_json)}))


if __name__ == "__main__":
    main()
