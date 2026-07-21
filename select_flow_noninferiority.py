#!/usr/bin/env python3
"""Apply the preregistered cross-dataset Flow validation non-inferiority gate."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from freeze_shared_core_v2_config import canonical_sha256, file_sha256


MAX_DROP = 0.003


def parse_named_paths(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"expected DATASET=PATH, got {value}")
        dataset, raw_path = value.split("=", 1)
        if not dataset or dataset in result:
            raise ValueError(f"invalid or duplicate dataset: {dataset}")
        result[dataset] = Path(raw_path)
    return result


def load_valid_metrics(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "flow_validation_metric_summary_v1":
        raise ValueError(f"flow selection input has wrong schema: {path}")
    if payload.get("evaluation_split") != "valid":
        raise ValueError(f"flow selection input is not explicitly valid-only: {path}")
    if payload.get("test_labels_used") is not False:
        raise ValueError(f"flow selection input does not exclude test labels: {path}")
    for binding_name in ("source", "framework_manifest", "shared_core_config"):
        binding = payload.get(binding_name) or {}
        bound_path = Path(str(binding.get("path") or ""))
        if not bound_path.is_file() or file_sha256(bound_path) != binding.get("sha256"):
            raise ValueError(f"stale {binding_name} binding in {path}")
    config_binding = payload["shared_core_config"]
    config_path = Path(config_binding["path"])
    config_payload = json.loads(config_path.read_text(encoding="utf-8"))
    config_fingerprint = config_payload.get("config_sha256")
    unsigned = {
        key: value for key, value in config_payload.items() if key != "config_sha256"
    }
    if not (
        config_fingerprint == canonical_sha256(unsigned)
        and config_binding.get("config_sha256") == config_fingerprint
    ):
        raise ValueError(f"invalid shared-core config fingerprint in {path}")
    metrics = payload.get("metrics") or {}
    result = {
        "accuracy": float(metrics["accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
    }
    if any(not math.isfinite(value) for value in result.values()):
        raise ValueError(f"non-finite flow validation metric: {path}")
    return {
        **result,
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "config_path": str(config_path.resolve()),
        "config_sha256": config_fingerprint,
    }


def flow_factorial_integrity(loaded: dict[str, dict[str, Any]]) -> dict[str, Any]:
    config_hashes = {
        arm: sorted({row["config_sha256"] for row in datasets.values()})
        for arm, datasets in loaded.items()
    }
    if any(len(values) != 1 for values in config_hashes.values()):
        raise ValueError("each Flow gate arm must use one config across both datasets")
    configs = {
        arm: json.loads(
            Path(next(iter(datasets.values()))["config_path"]).read_text(
                encoding="utf-8"
            )
        )
        for arm, datasets in loaded.items()
    }
    allowed_tower1 = {
        "identity_safe_contrastive",
        "cross_scale_weight",
        "cross_scale_temperature",
    }
    mismatched_sections = []
    for section in (
        "packet_core",
        "native_pretraining",
        "empirical_risk",
        "embedding_extraction",
    ):
        if configs["baseline"].get(section) != configs["candidate"].get(section):
            mismatched_sections.append(section)
    baseline_tower1 = {
        key: value
        for key, value in (configs["baseline"].get("tower1") or {}).items()
        if key not in allowed_tower1
    }
    candidate_tower1 = {
        key: value
        for key, value in (configs["candidate"].get("tower1") or {}).items()
        if key not in allowed_tower1
    }
    if baseline_tower1 != candidate_tower1:
        mismatched_sections.append("tower1_undeclared_fields")
    baseline_values = configs["baseline"].get("tower1") or {}
    candidate_values = configs["candidate"].get("tower1") or {}
    transitions_valid = bool(
        baseline_values.get("identity_safe_contrastive") is False
        and candidate_values.get("identity_safe_contrastive") is True
        and math.isclose(
            float(baseline_values.get("cross_scale_weight", -1.0)),
            0.0,
            abs_tol=1e-12,
        )
        and float(candidate_values.get("cross_scale_weight", -1.0)) in {0.0, 0.05}
        and math.isclose(
            float(baseline_values.get("cross_scale_temperature", -1.0)),
            0.07,
            abs_tol=1e-12,
        )
        and math.isclose(
            float(candidate_values.get("cross_scale_temperature", -1.0)),
            0.07,
            abs_tol=1e-12,
        )
    )
    passed = not mismatched_sections and transitions_valid
    return {
        "required": True,
        "status": "pass" if passed else "fail",
        "same_config_per_arm_across_datasets": all(
            len(values) == 1 for values in config_hashes.values()
        ),
        "config_sha256": {arm: values[0] for arm, values in config_hashes.items()},
        "allowed_tower1_fields": sorted(allowed_tower1),
        "mismatched_sections": mismatched_sections,
        "objective_transitions_valid": transitions_valid,
    }


def select(
    baseline_paths: dict[str, Path], candidate_paths: dict[str, Path]
) -> dict[str, Any]:
    if set(baseline_paths) != {"vpn-app", "tls-120"} or set(candidate_paths) != set(
        baseline_paths
    ):
        raise ValueError("flow gate requires exactly VPN and TLS-120 in both arms")
    datasets: dict[str, Any] = {}
    loaded = {"baseline": {}, "candidate": {}}
    for dataset in sorted(baseline_paths):
        baseline = load_valid_metrics(baseline_paths[dataset])
        candidate = load_valid_metrics(candidate_paths[dataset])
        loaded["baseline"][dataset] = baseline
        loaded["candidate"][dataset] = candidate
        delta_accuracy = candidate["accuracy"] - baseline["accuracy"]
        delta_macro_f1 = candidate["macro_f1"] - baseline["macro_f1"]
        accuracy_passes = delta_accuracy >= -MAX_DROP
        macro_f1_passes = delta_macro_f1 >= -MAX_DROP
        datasets[dataset] = {
            "baseline": baseline,
            "candidate": candidate,
            "delta_accuracy": delta_accuracy,
            "delta_macro_f1": delta_macro_f1,
            "accuracy_passes": accuracy_passes,
            "macro_f1_passes": macro_f1_passes,
            "passes": accuracy_passes and macro_f1_passes,
        }
    promoted = all(row["passes"] for row in datasets.values())
    factorial_integrity = flow_factorial_integrity(loaded)
    if factorial_integrity["status"] != "pass":
        raise ValueError("Flow gate arms violate factorial config integrity")
    return {
        "schema": "flow_noninferiority_selection_v1",
        "selection_scope": "heldout_validation_only",
        "metric": "accuracy_and_macro_f1_noninferiority",
        "promotion_scope": "same_candidate_must_pass_every_dataset",
        "thresholds": {
            "accuracy_max_drop": MAX_DROP,
            "macro_f1_max_drop": MAX_DROP,
        },
        "datasets": datasets,
        "candidate_promoted_for_all_datasets": promoted,
        "selected": "candidate" if promoted else "baseline",
        "test_labels_used": False,
        "factorial_config_integrity": factorial_integrity,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", action="append", required=True)
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()
    payload = select(
        parse_named_paths(args.baseline), parse_named_paths(args.candidate)
    )
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
