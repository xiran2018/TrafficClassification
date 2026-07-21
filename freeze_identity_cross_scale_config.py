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


def validate_selection(
    report: dict[str, Any],
    *,
    name: str,
    min_delta: float,
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


def evidence(path: Path, report: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "selected": report["selected"],
        "datasets": report["datasets"],
    }


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
) -> dict[str, Any]:
    validate_selection(d1, name="D1 identity-safe", min_delta=0.005)
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
            d2_incremental, name="D2 incremental", min_delta=0.002
        )
        validate_selection(d2_overall, name="D2 overall", min_delta=0.005)
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
        )
    ):
        raise ValueError("D2 evidence is forbidden when D1 was not promoted")

    payload = deepcopy(base)
    base_fingerprint = str(payload.pop("config_sha256"))
    tower1 = payload["tower1"]
    tower1["identity_safe_contrastive"] = identity_safe
    tower1["cross_scale_weight"] = 0.05 if cross_scale else 0.0
    tower1["cross_scale_temperature"] = 0.07
    payload["method_selection"] = {
        "scope": "fold0_validation_only_before_test",
        "selected_method": (
            "availability_aware_cross_scale"
            if cross_scale
            else "identity_safe_contrastive"
            if identity_safe
            else "shared_core_v2_control"
        ),
        "identity_safe_contrastive": identity_safe,
        "availability_aware_cross_scale": cross_scale,
        "base_shared_core_config": {
            "path": str(base_path.resolve()),
            "sha256": file_sha256(base_path),
            "config_sha256": base_fingerprint,
        },
        "d1": evidence(d1_path, d1),
        "d2_incremental": (
            evidence(d2_incremental_path, d2_incremental)
            if identity_safe and d2_incremental_path and d2_incremental
            else None
        ),
        "d2_overall": (
            evidence(d2_overall_path, d2_overall)
            if identity_safe and d2_overall_path and d2_overall
            else None
        ),
        "test_labels_used": False,
    }
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
