#!/usr/bin/env python3
"""Materialize test-forbidden control/candidate configs for the Flow-valid gate."""
from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

from freeze_shared_core_v2_config import canonical_sha256, file_sha256
from shared_core_v2 import load_frozen_shared_core


def resign(payload):
    payload.pop("config_sha256", None)
    payload["config_sha256"] = canonical_sha256(payload)
    return payload


def make_configs(base, provisional, *, base_path: Path, provisional_path: Path):
    decision = provisional.get("method_selection") or {}
    if decision.get("decision_status") != "packet_selected_pending_flow_noninferiority":
        raise ValueError("candidate config is not pending the Flow non-inferiority gate")
    if (provisional.get("selection_protocol") or {}).get(
        "test_evaluation_allowed"
    ) is not False:
        raise ValueError("provisional candidate does not forbid test evaluation")

    control = deepcopy(base)
    control["tower1"]["identity_safe_contrastive"] = False
    control["tower1"]["cross_scale_weight"] = 0.0
    control["tower1"]["cross_scale_temperature"] = 0.07
    control["selection_protocol"]["test_evaluation_allowed"] = False
    control["method_selection"] = {
        "decision_status": "flow_noninferiority_control_only",
        "selected_method": "shared_core_v2_control",
        "paired_candidate_config": {
            "path": str(provisional_path.resolve()),
            "sha256": file_sha256(provisional_path),
            "config_sha256": provisional["config_sha256"],
        },
        "base_shared_core_config": {
            "path": str(base_path.resolve()),
            "sha256": file_sha256(base_path),
            "config_sha256": base["config_sha256"],
        },
        "test_labels_used": False,
    }
    return resign(control), deepcopy(provisional)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_config", required=True)
    parser.add_argument("--provisional_candidate_config", required=True)
    parser.add_argument("--control_output", required=True)
    parser.add_argument("--candidate_output", required=True)
    args = parser.parse_args()
    base_path = Path(args.base_config)
    candidate_path = Path(args.provisional_candidate_config)
    control, candidate = make_configs(
        load_frozen_shared_core(base_path),
        load_frozen_shared_core(candidate_path),
        base_path=base_path,
        provisional_path=candidate_path,
    )
    for path, payload in (
        (Path(args.control_output), control),
        (Path(args.candidate_output), candidate),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    print(json.dumps({"status": "flow_gate_configs_ready"}, indent=2))


if __name__ == "__main__":
    main()
