#!/usr/bin/env python3
"""Release a signed, development-only Test milestone from a locked config."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from freeze_shared_core_v2_config import canonical_sha256, file_sha256


ALLOWED_DECISIONS = {
    "base_frozen_pending_identity_cross_scale_validation",
    "final_after_preregistered_validation",
}


def release(base_path: Path) -> dict[str, Any]:
    base_path = base_path.resolve()
    base = json.loads(base_path.read_text(encoding="utf-8"))
    recorded_sha = str(base.get("config_sha256") or "")
    unsigned = {key: value for key, value in base.items() if key != "config_sha256"}
    if recorded_sha != canonical_sha256(unsigned):
        raise ValueError("base method config has an invalid canonical signature")
    decision = str(
        base.get("method_selection", {}).get("decision_status")
        or (
            "base_frozen_pending_identity_cross_scale_validation"
            if base.get("status") == "frozen_from_cross_dataset_validation"
            else ""
        )
    )
    if decision not in ALLOWED_DECISIONS:
        raise ValueError(f"unsupported frozen decision status: {decision}")
    if base.get("selection_protocol", {}).get("test_evaluation_allowed") is True:
        raise ValueError("base method config must not already release Test")
    if base.get("method_selection", {}).get(
        "test_labels_used", base.get("selection_protocol", {}).get("test_labels_used")
    ) is not False:
        raise ValueError("base method selection must not use Test labels")

    released = json.loads(json.dumps(base))
    released.pop("config_sha256", None)
    released["selection_protocol"]["test_evaluation_allowed"] = True
    released["selection_protocol"]["test_evaluation_role"] = (
        "development_benchmark_after_base_shared_core_freeze"
        if decision == "base_frozen_pending_identity_cross_scale_validation"
        else "development_benchmark_after_final_validation_freeze"
    )
    released["method_selection"] = {
        **released.get("method_selection", {}),
        "scope": released.get("method_selection", {}).get(
            "scope", "fold0_validation_only_before_test"
        ),
        "decision_status": decision,
        "selected_method": released.get("method_selection", {}).get(
            "selected_method", "shared_core_v2_base"
        ),
        "test_labels_used": False,
    }
    released["evaluation_release"] = {
        "source_config_path": str(base_path),
        "source_config_file_sha256": file_sha256(base_path),
        "source_config_sha256": recorded_sha,
        "may_inform_future_method_design": True,
        "unbiased_final_claim_allowed": False,
        "required_final_evaluation": (
            "new_outer_holdout_or_nested_cross_validation_if_feedback_is_used"
        ),
    }
    released["config_sha256"] = canonical_sha256(released)
    return released


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_config", required=True)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()
    output = Path(args.output_json)
    payload = release(Path(args.base_config))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "released", "output_json": str(output)}))


if __name__ == "__main__":
    main()
