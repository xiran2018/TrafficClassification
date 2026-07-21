#!/usr/bin/env python3
"""Gate final Test release on locked-Test Packet applicability evidence."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from freeze_shared_core_v2_config import canonical_sha256, file_sha256


REQUIRED_DATASETS = {
    "ustc-app",
    "ustc-binary",
    "vpn-binary",
    "vpn-service",
}
MIN_CHANCE_NORMALIZED_ACCURACY = 0.70
MIN_CHANCE_NORMALIZED_MACRO_F1 = 0.65


def parse_run(value: str) -> tuple[str, Path, Path]:
    parts = value.split("=", 1)
    if len(parts) != 2 or not parts[0]:
        raise ValueError(f"expected DATASET=MANIFEST,RESULT, got {value}")
    paths = parts[1].split(",", 1)
    if len(paths) != 2 or not all(paths):
        raise ValueError(f"expected DATASET=MANIFEST,RESULT, got {value}")
    return parts[0], Path(paths[0]), Path(paths[1])


def load_frozen_config(path: Path) -> tuple[dict[str, Any], str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    fingerprint = str(payload.get("config_sha256") or "")
    unsigned = {key: value for key, value in payload.items() if key != "config_sha256"}
    if len(fingerprint) != 64 or canonical_sha256(unsigned) != fingerprint:
        raise ValueError("invalid final shared-core config fingerprint")
    packet_scope = set(
        (payload.get("task_datasets") or {}).get(
            "packet-level-classification", []
        )
    )
    if not REQUIRED_DATASETS <= packet_scope:
        raise ValueError("final shared-core config does not cover the Packet application scope")
    if (payload.get("method_selection") or {}).get("decision_status") != (
        "final_after_preregistered_validation"
    ):
        raise ValueError("Packet applicability gate requires a validation-frozen method")
    return payload, fingerprint


def verify_file_binding(binding: dict[str, Any], name: str) -> None:
    path = Path(str(binding.get("path") or ""))
    expected = str(binding.get("sha256") or "")
    if not path.is_file() or len(expected) != 64 or file_sha256(path) != expected:
        raise ValueError(f"stale or missing {name} binding")


def chance_normalize(value: float, num_classes: int) -> float:
    chance = 1.0 / num_classes
    return (value - chance) / (1.0 - chance)


def evaluate_run(
    dataset: str,
    manifest_path: Path,
    result_path: Path,
    config_fingerprint: str,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if manifest.get("dataset") != dataset or manifest.get("fold") != 0:
        raise ValueError(f"{dataset}: applicability evidence must be its fold 0 run")
    if manifest.get("stage") != "paper_unified":
        raise ValueError(f"{dataset}: wrong Packet framework stage")
    notes = ((manifest.get("framework") or {}).get("notes") or {})
    if notes.get("completed") is not True:
        raise ValueError(f"{dataset}: Packet validation run is incomplete")
    if notes.get("executed_splits") != ["train", "valid"]:
        raise ValueError(f"{dataset}: Packet evidence is not exactly train,valid")
    if notes.get("test_labels_used") is not False:
        raise ValueError(f"{dataset}: Packet evidence used Test labels")
    if notes.get("shared_core_method_sha256") != config_fingerprint:
        raise ValueError(f"{dataset}: Packet evidence used a different shared method")

    if result.get("task") != "packet-level-classification":
        raise ValueError(f"{dataset}: wrong result task")
    if result.get("sample_unit") != "one_packet":
        raise ValueError(f"{dataset}: inference is not one Packet per sample")
    packet_index = Path(str(result.get("packet_index") or ""))
    if packet_index.parent.name != "valid":
        raise ValueError(f"{dataset}: result is not explicitly from the Valid split")
    for name, binding in (result.get("provenance") or {}).items():
        if isinstance(binding, dict) and {"path", "sha256"} <= set(binding):
            verify_file_binding(binding, f"{dataset} result {name}")

    metrics = result.get("metrics") or {}
    accuracy = float(metrics["accuracy"])
    macro_f1 = float(metrics["macro_f1"])
    if not all(
        math.isfinite(value) and 0.0 <= value <= 1.0
        for value in (accuracy, macro_f1)
    ):
        raise ValueError(f"{dataset}: invalid Valid metrics")
    per_class = metrics.get("per_class") or {}
    num_classes = len(per_class)
    if num_classes < 2:
        raise ValueError(f"{dataset}: could not verify the class count")
    normalized_accuracy = chance_normalize(accuracy, num_classes)
    normalized_macro_f1 = chance_normalize(macro_f1, num_classes)

    artifact_dir = manifest_path.parent
    forbidden = [
        artifact_dir / "test_unified_packet_single_head.json",
        artifact_dir / "test_unified_packet_single_head.npz",
    ]
    present_forbidden = [str(path.resolve()) for path in forbidden if path.exists()]
    if present_forbidden:
        raise ValueError(f"{dataset}: Test prediction artifacts already exist")

    accuracy_passes = normalized_accuracy >= MIN_CHANCE_NORMALIZED_ACCURACY
    macro_f1_passes = normalized_macro_f1 >= MIN_CHANCE_NORMALIZED_MACRO_F1
    return {
        "dataset": dataset,
        "fold": 0,
        "evaluation_split": "valid",
        "num_classes": num_classes,
        "metrics": {"accuracy": accuracy, "macro_f1": macro_f1},
        "chance_normalized_metrics": {
            "accuracy": normalized_accuracy,
            "macro_f1": normalized_macro_f1,
        },
        "accuracy_passes": accuracy_passes,
        "macro_f1_passes": macro_f1_passes,
        "passes": accuracy_passes and macro_f1_passes,
        "manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": file_sha256(manifest_path),
        },
        "result": {
            "path": str(result_path.resolve()),
            "sha256": file_sha256(result_path),
        },
        "test_prediction_artifacts_present": False,
    }


def evaluate(config_path: Path, run_values: list[str]) -> dict[str, Any]:
    _, fingerprint = load_frozen_config(config_path)
    parsed: dict[str, tuple[Path, Path]] = {}
    for value in run_values:
        dataset, manifest, result = parse_run(value)
        if dataset in parsed:
            raise ValueError(f"duplicate Packet applicability dataset: {dataset}")
        parsed[dataset] = (manifest, result)
    if set(parsed) != REQUIRED_DATASETS:
        raise ValueError(
            "Packet applicability gate requires exactly: "
            + ", ".join(sorted(REQUIRED_DATASETS))
        )
    datasets = {
        dataset: evaluate_run(dataset, *parsed[dataset], fingerprint)
        for dataset in sorted(parsed)
    }
    passed = all(row["passes"] for row in datasets.values())
    return {
        "schema": "packet_scope_validation_gate_v1",
        "selection_scope": "locked_test_fold0_validation_only",
        "purpose": "architecture_applicability_safety_gate_not_model_selection",
        "method_config": {
            "path": str(config_path.resolve()),
            "sha256": file_sha256(config_path),
            "config_sha256": fingerprint,
        },
        "required_datasets": sorted(REQUIRED_DATASETS),
        "thresholds": {
            "chance_normalized_accuracy_min": MIN_CHANCE_NORMALIZED_ACCURACY,
            "chance_normalized_macro_f1_min": MIN_CHANCE_NORMALIZED_MACRO_F1,
        },
        "datasets": datasets,
        "all_datasets_pass": passed,
        "test_evaluation_released": passed,
        "test_labels_used": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run", action="append", default=[], required=True)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()
    report = evaluate(Path(args.config), args.run)
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "all_datasets_pass": report["all_datasets_pass"],
                "datasets": {
                    name: row["chance_normalized_metrics"]
                    for name, row in report["datasets"].items()
                },
            },
            indent=2,
        )
    )
    if not report["all_datasets_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
