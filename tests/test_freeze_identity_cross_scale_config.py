import json
from pathlib import Path

import pytest

from freeze_identity_cross_scale_config import freeze_method_config
from freeze_shared_core_v2_config import canonical_sha256


def selection(selected: str, min_delta: float):
    promoted = selected == "candidate"
    return {
        "selection_scope": "heldout_validation_only",
        "metric": "macro_f1_with_accuracy_guard",
        "promotion_scope": "same_candidate_must_pass_every_dataset",
        "min_delta": min_delta,
        "max_accuracy_drop": 0.005,
        "selected": selected,
        "candidate_promoted_for_all_datasets": promoted,
        "datasets": {
            name: {"passes": promoted} for name in ("vpn-app", "tls-120")
        },
        "training_completion_evidence": {
            "baseline": {"status": "pass"},
            "candidate": {"status": "pass"},
        },
        "training_implementation_consistency": {"status": "pass"},
    }


def write(path: Path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def finalize(tmp_path, d1, incremental=None, overall=None):
    base = {
        "schema": "exact_shared_packet_core_v2",
        "status": "frozen_from_cross_dataset_validation",
        "selection_protocol": {"test_labels_used": False},
        "tower1": {},
        "task_contract": {},
    }
    base["config_sha256"] = canonical_sha256(base)
    base_path = write(tmp_path / "base.json", base)
    d1_path = write(tmp_path / "d1.json", d1)
    incremental_path = (
        write(tmp_path / "incremental.json", incremental) if incremental else None
    )
    overall_path = write(tmp_path / "overall.json", overall) if overall else None
    return freeze_method_config(
        base,
        base_path=base_path,
        d1=d1,
        d1_path=d1_path,
        d2_incremental=incremental,
        d2_incremental_path=incremental_path,
        d2_overall=overall,
        d2_overall_path=overall_path,
    )


def flow_gate(selected="candidate"):
    promoted = selected == "candidate"
    return {
        "schema": "flow_noninferiority_selection_v1",
        "selection_scope": "heldout_validation_only",
        "thresholds": {
            "macro_f1_max_drop": 0.003,
            "accuracy_max_drop": 0.003,
        },
        "datasets": {
            name: {"passes": promoted} for name in ("vpn-app", "tls-120")
        },
        "selected": selected,
        "test_labels_used": False,
    }


def assert_fingerprint(payload):
    unsigned = {key: value for key, value in payload.items() if key != "config_sha256"}
    assert payload["config_sha256"] == canonical_sha256(unsigned)


def test_d1_failure_freezes_control_and_forbids_d2(tmp_path):
    payload = finalize(tmp_path, selection("baseline", 0.005))
    assert payload["method_selection"]["selected_method"] == "shared_core_v2_control"
    assert payload["tower1"]["identity_safe_contrastive"] is False
    assert payload["tower1"]["cross_scale_weight"] == 0.0
    assert_fingerprint(payload)

    with pytest.raises(ValueError, match="forbidden"):
        finalize(
            tmp_path,
            selection("baseline", 0.005),
            selection("candidate", 0.002),
            selection("candidate", 0.005),
        )


def test_d1_success_d2_failure_freezes_identity_safe_only(tmp_path):
    payload = finalize(
        tmp_path,
        selection("candidate", 0.005),
        selection("candidate", 0.002),
        selection("baseline", 0.005),
    )
    assert payload["method_selection"]["selected_method"] == "identity_safe_contrastive"
    assert payload["tower1"]["identity_safe_contrastive"] is True
    assert payload["tower1"]["cross_scale_weight"] == 0.0
    assert payload["method_selection"]["decision_status"] == (
        "packet_selected_pending_flow_noninferiority"
    )
    assert payload["selection_protocol"]["test_evaluation_allowed"] is False
    assert_fingerprint(payload)


def test_both_d2_gates_freeze_cross_scale_for_packet_and_flow(tmp_path):
    payload = finalize(
        tmp_path,
        selection("candidate", 0.005),
        selection("candidate", 0.002),
        selection("candidate", 0.005),
    )
    assert payload["method_selection"]["selected_method"] == "availability_aware_cross_scale"
    assert payload["tower1"]["identity_safe_contrastive"] is True
    assert payload["tower1"]["cross_scale_weight"] == 0.05
    assert payload["task_contract"]["selected_tower1_objectives"] == {
        "identity_safe_contrastive": True,
        "availability_aware_cross_scale": True,
    }
    assert_fingerprint(payload)


def test_flow_gate_promotes_or_rolls_back_packet_selected_candidate(tmp_path):
    d1 = selection("candidate", 0.005)
    incremental = selection("candidate", 0.002)
    overall = selection("candidate", 0.005)
    provisional = finalize(tmp_path, d1, incremental, overall)

    gate_path = write(tmp_path / "flow_gate.json", flow_gate("candidate"))
    promoted = freeze_method_config(
        provisional,
        base_path=write(tmp_path / "provisional.json", provisional),
        d1=d1,
        d1_path=tmp_path / "d1.json",
        d2_incremental=incremental,
        d2_incremental_path=tmp_path / "incremental.json",
        d2_overall=overall,
        d2_overall_path=tmp_path / "overall.json",
        flow_noninferiority=flow_gate("candidate"),
        flow_noninferiority_path=gate_path,
    )
    assert promoted["tower1"]["cross_scale_weight"] == 0.05
    assert promoted["selection_protocol"]["test_evaluation_allowed"] is True

    rejected_gate_path = write(tmp_path / "flow_gate_rejected.json", flow_gate("baseline"))
    rejected = freeze_method_config(
        provisional,
        base_path=tmp_path / "provisional.json",
        d1=d1,
        d1_path=tmp_path / "d1.json",
        d2_incremental=incremental,
        d2_incremental_path=tmp_path / "incremental.json",
        d2_overall=overall,
        d2_overall_path=tmp_path / "overall.json",
        flow_noninferiority=flow_gate("baseline"),
        flow_noninferiority_path=rejected_gate_path,
    )
    assert rejected["method_selection"]["selected_method"] == "shared_core_v2_control"
    assert rejected["tower1"]["identity_safe_contrastive"] is False
    assert rejected["tower1"]["cross_scale_weight"] == 0.0
    assert rejected["selection_protocol"]["test_evaluation_allowed"] is True


def test_promoted_d1_requires_complete_d2_decision(tmp_path):
    with pytest.raises(ValueError, match="requires both"):
        finalize(tmp_path, selection("candidate", 0.005))
