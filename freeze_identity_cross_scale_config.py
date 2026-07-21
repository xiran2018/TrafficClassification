#!/usr/bin/env python3
"""Freeze the preregistered D1/D2 decision into one shared Packet/Flow config."""
from __future__ import annotations

import argparse
import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any

from freeze_shared_core_v2_config import canonical_sha256, file_sha256
from shared_core_v2 import load_frozen_shared_core


REQUIRED_DATASETS = {"vpn-app", "tls-120"}
MIN_CROSS_SCALE_ACTIVE_ANCHOR_RATE = 0.5
D1_FACTORIAL_FIELDS = {"identity_safe_contrastive"}
D2_INCREMENTAL_FACTORIAL_FIELDS = {
    "cross_scale_weight",
    "cross_scale_temperature",
    "paired_packet_aux_jsonl",
}
D2_OVERALL_FACTORIAL_FIELDS = D1_FACTORIAL_FIELDS | D2_INCREMENTAL_FACTORIAL_FIELDS


def validate_selection(
    report: dict[str, Any],
    *,
    name: str,
    min_delta: float,
    factorial_fields: set[str],
) -> None:
    if report.get("selection_scope") != "heldout_validation_only":
        raise ValueError(f"{name} must use heldout validation only")
    if report.get("metric") != "macro_f1_with_accuracy_guard":
        raise ValueError(f"{name} has the wrong selection metric")
    if report.get("promotion_scope") != "same_candidate_must_pass_every_dataset":
        raise ValueError(f"{name} must require one candidate to pass both datasets")
    if set((report.get("datasets") or {}).keys()) != REQUIRED_DATASETS:
        raise ValueError(f"{name} must contain exactly VPN and TLS-120")
    if not math.isclose(float(report.get("min_delta", -1)), min_delta, abs_tol=1e-12):
        raise ValueError(f"{name} has the wrong Macro-F1 threshold")
    if not math.isclose(
        float(report.get("max_accuracy_drop", -1)), 0.005, abs_tol=1e-12
    ):
        raise ValueError(f"{name} has the wrong accuracy guard")
    promoted = all(bool(row.get("passes")) for row in report["datasets"].values())
    expected = "candidate" if promoted else "baseline"
    if report.get("selected") != expected:
        raise ValueError(f"{name} selected arm disagrees with dataset gates")
    if report.get("candidate_promoted_for_all_datasets") is not promoted:
        raise ValueError(f"{name} promotion flag disagrees with dataset gates")
    completion = report.get("training_completion_evidence") or {}
    if set(completion) != {"baseline", "candidate"} or any(
        arm.get("status") != "pass" for arm in completion.values()
    ):
        raise ValueError(f"{name} lacks complete two-arm training evidence")
    implementation = report.get("training_implementation_consistency") or {}
    if implementation.get("status") != "pass":
        raise ValueError(f"{name} does not prove one trainer source across arms")
    factorial = report.get("factorial_config_integrity") or {}
    if not (
        factorial.get("required") is True
        and factorial.get("status") == "pass"
        and set(factorial.get("declared_factorial_fields") or [])
        == factorial_fields
        and set((factorial.get("datasets") or {}).keys()) == REQUIRED_DATASETS
        and all(
            row.get("status") == "pass"
            for row in factorial["datasets"].values()
        )
    ):
        raise ValueError(f"{name} lacks matching factorial-integrity evidence")


def evidence(path: Path, report: dict[str, Any]) -> dict[str, Any]:
    result = {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "selected": report["selected"],
        "datasets": report["datasets"],
    }
    if "factorial_config_integrity" in report:
        result["factorial_config_integrity"] = report[
            "factorial_config_integrity"
        ]
    return result


def validate_flow_noninferiority(report: dict[str, Any]) -> None:
    if report.get("schema") != "flow_noninferiority_selection_v1":
        raise ValueError("flow gate has the wrong schema")
    if report.get("selection_scope") != "heldout_validation_only":
        raise ValueError("flow gate must use heldout validation only")
    if set((report.get("datasets") or {}).keys()) != REQUIRED_DATASETS:
        raise ValueError("flow gate must contain exactly VPN and TLS-120")
    thresholds = report.get("thresholds") or {}
    if not math.isclose(
        float(thresholds.get("macro_f1_max_drop", -1)), 0.003, abs_tol=1e-12
    ) or not math.isclose(
        float(thresholds.get("accuracy_max_drop", -1)), 0.003, abs_tol=1e-12
    ):
        raise ValueError("flow gate does not match preregistered thresholds")
    promoted = all(bool(row.get("passes")) for row in report["datasets"].values())
    if report.get("selected") != ("candidate" if promoted else "baseline"):
        raise ValueError("flow gate selected arm disagrees with dataset gates")
    if report.get("test_labels_used") is not False:
        raise ValueError("flow gate must explicitly exclude test labels")


def validate_cross_scale_exposure(report: dict[str, Any]) -> None:
    if report.get("schema") != "cross_scale_sampler_exposure_audit_v1":
        raise ValueError("cross-scale exposure audit has the wrong schema")
    if report.get("scope") != "training_inputs_and_exact_sampler_only":
        raise ValueError("cross-scale exposure audit must be training-input-only")
    if report.get("test_predictions_used") is not False:
        raise ValueError("cross-scale exposure audit must exclude test predictions")
    if report.get("identity_policy") != "exact_packet_uid_compaction":
        raise ValueError("cross-scale exposure audit has the wrong identity policy")
    if report.get("context_policy") != "leave_one_distinct_packet_out":
        raise ValueError("cross-scale exposure audit has the wrong context policy")
    reports = report.get("reports") or {}
    if set(reports) != REQUIRED_DATASETS:
        raise ValueError("cross-scale exposure audit must cover VPN and TLS-120")
    for dataset, dataset_report in reports.items():
        aggregate = dataset_report.get("aggregate") or {}
        input_bindings_valid = True
        for prefix in ("factual", "paired"):
            path = Path(str(dataset_report.get(f"{prefix}_path") or ""))
            expected_sha256 = dataset_report.get(f"{prefix}_sha256")
            input_bindings_valid = bool(
                input_bindings_valid
                and path.is_file()
                and isinstance(expected_sha256, str)
                and len(expected_sha256) == 64
                and file_sha256(path) == expected_sha256
            )
        if not (
            input_bindings_valid
            and int(dataset_report.get("source_packets", 0)) > 0
            and int(dataset_report.get("source_flows", 0)) > 0
            and int(dataset_report.get("epochs", 0)) == 8
            and int(dataset_report.get("batch_size", 0)) == 16
            and int(dataset_report.get("packets_per_flow", 0)) == 2
            and int(dataset_report.get("seed", -1)) == 42
            and math.isclose(
                float(aggregate.get("paired_identity_rate", -1.0)),
                1.0,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            and float(
                aggregate.get("bidirectional_valid_anchor_rate", -1.0)
            )
            >= MIN_CROSS_SCALE_ACTIVE_ANCHOR_RATE
            and float(
                aggregate.get("alias_only_false_context_anchor_rate", 0.0)
            )
            > 0.0
        ):
            raise ValueError(
                f"cross-scale exposure is insufficient, stale, or inconsistent for {dataset}"
            )


def freeze_method_config(
    base: dict[str, Any],
    *,
    base_path: Path,
    d1: dict[str, Any],
    d1_path: Path,
    d2_incremental: dict[str, Any] | None = None,
    d2_incremental_path: Path | None = None,
    d2_overall: dict[str, Any] | None = None,
    d2_overall_path: Path | None = None,
    cross_scale_exposure: dict[str, Any] | None = None,
    cross_scale_exposure_path: Path | None = None,
    flow_noninferiority: dict[str, Any] | None = None,
    flow_noninferiority_path: Path | None = None,
) -> dict[str, Any]:
    validate_selection(
        d1,
        name="D1 identity-safe",
        min_delta=0.005,
        factorial_fields=D1_FACTORIAL_FIELDS,
    )
    identity_safe = d1["selected"] == "candidate"
    cross_scale = False

    if identity_safe:
        if any(
            value is None
            for value in (
                d2_incremental,
                d2_incremental_path,
                d2_overall,
                d2_overall_path,
            )
        ):
            raise ValueError("promoted D1 requires both preregistered D2 gates")
        assert d2_incremental is not None and d2_incremental_path is not None
        assert d2_overall is not None and d2_overall_path is not None
        validate_selection(
            d2_incremental,
            name="D2 incremental",
            min_delta=0.002,
            factorial_fields=D2_INCREMENTAL_FACTORIAL_FIELDS,
        )
        validate_selection(
            d2_overall,
            name="D2 overall",
            min_delta=0.005,
            factorial_fields=D2_OVERALL_FACTORIAL_FIELDS,
        )
        if cross_scale_exposure is None or cross_scale_exposure_path is None:
            raise ValueError("D2 requires the exact-sampler exposure audit")
        validate_cross_scale_exposure(cross_scale_exposure)
        cross_scale = bool(
            d2_incremental["selected"] == "candidate"
            and d2_overall["selected"] == "candidate"
        )
    elif any(
        value is not None
        for value in (
            d2_incremental,
            d2_incremental_path,
            d2_overall,
            d2_overall_path,
            cross_scale_exposure,
            cross_scale_exposure_path,
        )
    ):
        raise ValueError("D2 evidence is forbidden when D1 was not promoted")

    packet_selected_identity_safe = identity_safe
    packet_selected_cross_scale = cross_scale
    flow_gate_passed: bool | None = None
    if identity_safe and flow_noninferiority is not None:
        if flow_noninferiority_path is None:
            raise ValueError("flow gate report path is required")
        validate_flow_noninferiority(flow_noninferiority)
        flow_gate_passed = flow_noninferiority["selected"] == "candidate"
        if not flow_gate_passed:
            identity_safe = False
            cross_scale = False
    elif identity_safe and flow_noninferiority_path is not None:
        raise ValueError("flow gate report content is required")
    elif not identity_safe and (
        flow_noninferiority is not None or flow_noninferiority_path is not None
    ):
        raise ValueError("flow gate is not applicable when Packet D1 failed")

    provisional = bool(packet_selected_identity_safe and flow_gate_passed is None)
    payload = deepcopy(base)
    base_fingerprint = str(payload.pop("config_sha256"))
    tower1 = payload["tower1"]
    tower1["identity_safe_contrastive"] = identity_safe
    tower1["cross_scale_weight"] = 0.05 if cross_scale else 0.0
    tower1["cross_scale_temperature"] = 0.07
    payload["method_selection"] = {
        "scope": "fold0_validation_only_before_test",
        "decision_status": (
            "packet_selected_pending_flow_noninferiority"
            if provisional
            else "final_after_preregistered_validation"
        ),
        "selected_method": (
            "availability_aware_cross_scale"
            if cross_scale
            else "identity_safe_contrastive"
            if identity_safe
            else "shared_core_v2_control"
        ),
        "identity_safe_contrastive": identity_safe,
        "availability_aware_cross_scale": cross_scale,
        "packet_selected_identity_safe_contrastive": packet_selected_identity_safe,
        "packet_selected_availability_aware_cross_scale": packet_selected_cross_scale,
        "flow_noninferiority_passed": flow_gate_passed,
        "base_shared_core_config": {
            "path": str(base_path.resolve()),
            "sha256": file_sha256(base_path),
            "config_sha256": base_fingerprint,
        },
        "d1": evidence(d1_path, d1),
        "d2_incremental": (
            evidence(d2_incremental_path, d2_incremental)
            if packet_selected_identity_safe
            and d2_incremental_path
            and d2_incremental
            else None
        ),
        "d2_overall": (
            evidence(d2_overall_path, d2_overall)
            if packet_selected_identity_safe and d2_overall_path and d2_overall
            else None
        ),
        "cross_scale_exposure": (
            {
                "path": str(cross_scale_exposure_path.resolve()),
                "sha256": file_sha256(cross_scale_exposure_path),
                "minimum_active_anchor_rate": MIN_CROSS_SCALE_ACTIVE_ANCHOR_RATE,
                "datasets": {
                    dataset: {
                        "bidirectional_valid_anchor_rate": row["aggregate"][
                            "bidirectional_valid_anchor_rate"
                        ],
                        "alias_only_false_context_anchor_rate": row["aggregate"][
                            "alias_only_false_context_anchor_rate"
                        ],
                    }
                    for dataset, row in cross_scale_exposure["reports"].items()
                },
            }
            if packet_selected_identity_safe
            and cross_scale_exposure_path
            and cross_scale_exposure
            else None
        ),
        "flow_noninferiority": (
            evidence(flow_noninferiority_path, flow_noninferiority)
            if flow_noninferiority_path and flow_noninferiority
            else None
        ),
        "test_labels_used": False,
    }
    payload["selection_protocol"]["test_evaluation_allowed"] = not provisional
    payload["task_contract"]["objective_activation_topology_must_match"] = True
    payload["task_contract"]["selected_tower1_objectives"] = {
        "identity_safe_contrastive": identity_safe,
        "availability_aware_cross_scale": cross_scale,
    }
    payload["config_sha256"] = canonical_sha256(payload)
    return payload


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_config", required=True)
    parser.add_argument("--d1_selection", required=True)
    parser.add_argument("--d2_incremental_selection", default="")
    parser.add_argument("--d2_overall_selection", default="")
    parser.add_argument("--cross_scale_exposure_audit", default="")
    parser.add_argument("--flow_noninferiority_selection", default="")
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    base_path = Path(args.base_config)
    d1_path = Path(args.d1_selection)
    d2_incremental_path = (
        Path(args.d2_incremental_selection)
        if args.d2_incremental_selection
        else None
    )
    d2_overall_path = (
        Path(args.d2_overall_selection) if args.d2_overall_selection else None
    )
    flow_path = (
        Path(args.flow_noninferiority_selection)
        if args.flow_noninferiority_selection
        else None
    )
    exposure_path = (
        Path(args.cross_scale_exposure_audit)
        if args.cross_scale_exposure_audit
        else None
    )
    payload = freeze_method_config(
        load_frozen_shared_core(base_path),
        base_path=base_path,
        d1=load_json(d1_path),
        d1_path=d1_path,
        d2_incremental=(
            load_json(d2_incremental_path) if d2_incremental_path else None
        ),
        d2_incremental_path=d2_incremental_path,
        d2_overall=load_json(d2_overall_path) if d2_overall_path else None,
        d2_overall_path=d2_overall_path,
        cross_scale_exposure=(load_json(exposure_path) if exposure_path else None),
        cross_scale_exposure_path=exposure_path,
        flow_noninferiority=load_json(flow_path) if flow_path else None,
        flow_noninferiority_path=flow_path,
    )
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "selected_method": payload["method_selection"]["selected_method"],
                "config_sha256": payload["config_sha256"],
                "output_json": str(output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
