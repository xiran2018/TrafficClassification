import json
from pathlib import Path

import pytest

from freeze_shared_core_v2_config import canonical_sha256, file_sha256
from make_development_milestone_config import release


def locked_config(path: Path, *, decision: str) -> dict:
    payload = {
        "schema": "exact_shared_packet_core_v2",
        "selection_protocol": {
            "test_evaluation_allowed": False,
            "test_labels_used": False,
        },
        "method_selection": {
            "decision_status": decision,
            "selected_method": "shared_core_v2_base",
            "test_labels_used": False,
        },
    }
    payload["config_sha256"] = canonical_sha256(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def test_release_marks_base_milestone_as_development_only(tmp_path):
    path = tmp_path / "frozen.json"
    base = locked_config(
        path, decision="base_frozen_pending_identity_cross_scale_validation"
    )
    released = release(path)
    assert released["selection_protocol"]["test_evaluation_allowed"] is True
    assert released["selection_protocol"]["test_evaluation_role"] == (
        "development_benchmark_after_base_shared_core_freeze"
    )
    assert released["evaluation_release"] == {
        "source_config_path": str(path.resolve()),
        "source_config_file_sha256": file_sha256(path),
        "source_config_sha256": base["config_sha256"],
        "may_inform_future_method_design": True,
        "unbiased_final_claim_allowed": False,
        "required_final_evaluation": (
            "new_outer_holdout_or_nested_cross_validation_if_feedback_is_used"
        ),
    }
    signature = released.pop("config_sha256")
    assert signature == canonical_sha256(released)


def test_release_rejects_unsigned_or_already_unlocked_config(tmp_path):
    path = tmp_path / "frozen.json"
    payload = locked_config(
        path, decision="base_frozen_pending_identity_cross_scale_validation"
    )
    payload["selection_protocol"]["test_evaluation_allowed"] = True
    payload["config_sha256"] = canonical_sha256(
        {key: value for key, value in payload.items() if key != "config_sha256"}
    )
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="must not already release Test"):
        release(path)


def test_release_accepts_signed_base_freeze_without_explicit_lock(tmp_path):
    path = tmp_path / "frozen.json"
    payload = {
        "schema": "exact_shared_packet_core_v2",
        "status": "frozen_from_cross_dataset_validation",
        "selection_protocol": {"test_labels_used": False},
    }
    payload["config_sha256"] = canonical_sha256(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")
    released = release(path)
    assert released["method_selection"]["decision_status"] == (
        "base_frozen_pending_identity_cross_scale_validation"
    )
    assert released["selection_protocol"]["test_evaluation_allowed"] is True
