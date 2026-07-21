#!/usr/bin/env python3
"""Select one shared Packet-to-Flow evidence candidate on validation only."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


REQUIRED_DATASETS = {"vpn-app", "tls-120"}
MATCHED_TOWER2_KEYS = (
    "seed",
    "tower2_epochs",
    "tower2_batch_size",
    "tower2_select_metric",
    "hidden_dim",
    "num_layers",
    "num_heads",
    "dropout",
    "tower2_lr",
    "weight_decay",
    "window_size",
    "stride",
    "window_loss_weight",
    "class_weight_strength",
    "label_smoothing",
    "flow_contrastive_weight",
    "flow_temperature",
    "content_group_loss_reduction",
    "content_group_unique_batches",
    "split_group_key",
    "shared_packet_hidden_dim",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: str | Path) -> tuple[Path, dict[str, Any]]:
    resolved = Path(path)
    return resolved, json.loads(resolved.read_text(encoding="utf-8"))


def flow_metrics(payload: dict[str, Any]) -> dict[str, float]:
    metrics = (payload.get("metrics") or {}).get("flow_level") or {}
    required = {"accuracy", "macro_f1"}
    if not required.issubset(metrics):
        raise ValueError("prediction JSON is missing flow-level accuracy/macro_f1")
    return {key: float(metrics[key]) for key in sorted(required)}


def manifest_notes(payload: dict[str, Any]) -> dict[str, Any]:
    return (payload.get("framework") or {}).get("notes") or {}


def normalized_eval_splits(value: Any) -> list[str]:
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, list):
        return [str(item) for item in value]
    return []


def matched_tower2_contract(payload: dict[str, Any]) -> dict[str, Any]:
    missing = [key for key in MATCHED_TOWER2_KEYS if key not in payload]
    if missing:
        raise ValueError(f"manifest lacks matched Tower2 settings: {missing}")
    notes = manifest_notes(payload)
    for key in ("tower2_data_suffix", "paired_embedding_suffix"):
        if not notes.get(key):
            raise ValueError(f"manifest lacks matched Tower2 data setting: {key}")
    return {
        **{key: payload[key] for key in MATCHED_TOWER2_KEYS},
        "tower2_data_suffix": notes["tower2_data_suffix"],
        "paired_embedding_suffix": notes["paired_embedding_suffix"],
        "native_structural_suffix": payload.get("native_structural_suffix"),
        "flow_pooling": payload.get("flow_pooling"),
        "exact_shared_packet_encoder": payload.get("exact_shared_packet_encoder"),
    }


def validate_arm_manifest(
    payload: dict[str, Any], dataset: str, arm: str
) -> tuple[str, float]:
    if payload.get("dataset") != dataset:
        raise ValueError(f"candidate manifest dataset mismatch for {dataset}")
    if normalized_eval_splits(payload.get("eval_splits")) != ["valid"]:
        raise ValueError(f"{dataset} candidate must be evaluated on validation only")
    bound = float(payload.get("packet_evidence_max_weight", 0.0))
    is_control = payload.get("packet_evidence_ablation_control") is True
    if arm == "control" and (bound != 0.0 or not is_control):
        raise ValueError(f"{dataset} matched control is not an evidence-free control")
    if arm == "candidate" and (bound <= 0.0 or is_control):
        raise ValueError(f"{dataset} candidate did not enable packet evidence")
    if payload.get("flow_pooling") != "late_fusion":
        raise ValueError(f"{dataset} candidate must use late_fusion")
    if payload.get("exact_shared_packet_encoder") is not True:
        raise ValueError(f"{dataset} candidate must use the exact shared packet encoder")
    notes = manifest_notes(payload)
    if notes.get("packet_module_training_source") != "flow_task_train_split_packets":
        raise ValueError(f"{dataset} candidate has an invalid packet training source")
    if notes.get("cross_task_trained_weights_reused") is not False:
        raise ValueError(f"{dataset} candidate reused cross-task supervised weights")
    expected_label_source = (
        "disabled"
        if arm == "control"
        else "flow_train_split_labels_broadcast_to_member_packets"
    )
    if notes.get("packet_evidence_training_label_source") != expected_label_source:
        raise ValueError(f"{dataset} {arm} has an invalid weak-label source")
    if any("test_" in str(path) for path in notes.get("result_paths") or []):
        raise ValueError(f"{dataset} candidate manifest contains a test result path")
    fingerprint = str(payload.get("shared_core_config_sha256") or "")
    if len(fingerprint) != 64:
        raise ValueError(f"{dataset} candidate lacks a frozen shared-core fingerprint")
    return fingerprint, bound


def select_candidate(
    records: list[tuple[str, str, str, str, str, str]],
    *,
    min_macro_f1_gain: float,
    max_accuracy_drop: float,
    max_reference_macro_f1_drop: float,
) -> dict[str, Any]:
    if {record[0] for record in records} != REQUIRED_DATASETS or len(records) != 2:
        raise ValueError(f"records must contain exactly {sorted(REQUIRED_DATASETS)}")
    datasets: dict[str, Any] = {}
    fingerprints: set[str] = set()
    bounds: set[float] = set()
    for (
        dataset,
        reference_path_raw,
        control_path_raw,
        candidate_path_raw,
        control_manifest_path_raw,
        candidate_manifest_path_raw,
    ) in records:
        reference_path, reference = load_json(reference_path_raw)
        control_path, control = load_json(control_path_raw)
        candidate_path, candidate = load_json(candidate_path_raw)
        control_manifest_path, control_manifest = load_json(control_manifest_path_raw)
        candidate_manifest_path, candidate_manifest = load_json(
            candidate_manifest_path_raw
        )
        control_fingerprint, control_bound = validate_arm_manifest(
            control_manifest, dataset, "control"
        )
        candidate_fingerprint, bound = validate_arm_manifest(
            candidate_manifest, dataset, "candidate"
        )
        if control_fingerprint != candidate_fingerprint or control_bound != 0.0:
            raise ValueError(f"{dataset} matched arms do not share the same frozen core")
        control_contract = matched_tower2_contract(control_manifest)
        candidate_contract = matched_tower2_contract(candidate_manifest)
        if control_contract != candidate_contract:
            raise ValueError(f"{dataset} control/candidate Tower2 contracts differ")
        fingerprints.add(candidate_fingerprint)
        bounds.add(bound)
        for key in ("flow_ids", "flow_y_true"):
            if not (reference.get(key) == control.get(key) == candidate.get(key)):
                raise ValueError(f"{dataset} reference/control/candidate {key} are not aligned")
        reference_metrics = flow_metrics(reference)
        control_metrics = flow_metrics(control)
        candidate_metrics = flow_metrics(candidate)
        macro_gain = candidate_metrics["macro_f1"] - control_metrics["macro_f1"]
        accuracy_gain = candidate_metrics["accuracy"] - control_metrics["accuracy"]
        reference_macro_gain = (
            candidate_metrics["macro_f1"] - reference_metrics["macro_f1"]
        )
        reference_accuracy_gain = (
            candidate_metrics["accuracy"] - reference_metrics["accuracy"]
        )
        passed = bool(
            macro_gain >= min_macro_f1_gain
            and accuracy_gain >= -max_accuracy_drop
            and reference_macro_gain >= -max_reference_macro_f1_drop
            and reference_accuracy_gain >= -max_accuracy_drop
        )
        datasets[dataset] = {
            "strict_v2_reference": reference_metrics,
            "matched_late_fusion_control": control_metrics,
            "candidate": candidate_metrics,
            "candidate_minus_matched_control": {
                "macro_f1": macro_gain,
                "accuracy": accuracy_gain,
            },
            "candidate_minus_strict_v2_reference": {
                "macro_f1": reference_macro_gain,
                "accuracy": reference_accuracy_gain,
            },
            "passed": passed,
            "matched_tower2_contract": control_contract,
            "evidence": {
                "reference_path": str(reference_path),
                "reference_sha256": file_sha256(reference_path),
                "control_path": str(control_path),
                "control_sha256": file_sha256(control_path),
                "candidate_path": str(candidate_path),
                "candidate_sha256": file_sha256(candidate_path),
                "control_manifest_path": str(control_manifest_path),
                "control_manifest_sha256": file_sha256(control_manifest_path),
                "candidate_manifest_path": str(candidate_manifest_path),
                "candidate_manifest_sha256": file_sha256(candidate_manifest_path),
            },
        }
    if len(fingerprints) != 1:
        raise ValueError("VPN/TLS candidates do not share one frozen-core fingerprint")
    if len(bounds) != 1:
        raise ValueError("VPN/TLS candidates do not use the same packet evidence bound")
    promoted = all(row["passed"] for row in datasets.values())
    return {
        "schema": "shared_packet_evidence_validation_v2",
        "selection_scope": "heldout_validation_only",
        "metric": "flow_macro_f1",
        "selected": "candidate" if promoted else "baseline",
        "promotion_scope": "same_candidate_must_pass_every_dataset",
        "thresholds": {
            "min_macro_f1_gain": min_macro_f1_gain,
            "max_accuracy_drop": max_accuracy_drop,
            "max_reference_macro_f1_drop": max_reference_macro_f1_drop,
        },
        "shared_core_config_sha256": next(iter(fingerprints)),
        "packet_evidence_max_weight": next(iter(bounds)),
        "datasets": datasets,
        "test_labels_used": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--record",
        action="append",
        nargs=6,
        metavar=(
            "DATASET",
            "STRICT_V2_VALID",
            "CONTROL_VALID",
            "CANDIDATE_VALID",
            "CONTROL_MANIFEST",
            "CANDIDATE_MANIFEST",
        ),
        required=True,
    )
    parser.add_argument("--min_macro_f1_gain", type=float, default=0.005)
    parser.add_argument("--max_accuracy_drop", type=float, default=0.01)
    parser.add_argument("--max_reference_macro_f1_drop", type=float, default=0.01)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()
    if (
        args.min_macro_f1_gain < 0
        or args.max_accuracy_drop < 0
        or args.max_reference_macro_f1_drop < 0
    ):
        parser.error("selection thresholds must be non-negative")
    report = select_candidate(
        [tuple(record) for record in args.record],
        min_macro_f1_gain=args.min_macro_f1_gain,
        max_accuracy_drop=args.max_accuracy_drop,
        max_reference_macro_f1_drop=args.max_reference_macro_f1_drop,
    )
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
