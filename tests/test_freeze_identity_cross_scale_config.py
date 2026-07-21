import hashlib
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


def cross_scale_exposure(tmp_path, active_rate=0.6):
    reports = {}
    for name in ("vpn-app", "tls-120"):
        factual = tmp_path / f"{name}_factual.jsonl"
        paired = tmp_path / f"{name}_paired.jsonl"
        factual.write_text(f"{name}:factual", encoding="utf-8")
        paired.write_text(f"{name}:paired", encoding="utf-8")
        reports[name] = {
            "source_packets": 100,
            "source_flows": 20,
            "epochs": 8,
            "batch_size": 16,
            "packets_per_flow": 2,
            "seed": 42,
            "factual_path": str(factual),
            "factual_sha256": hashlib.sha256(factual.read_bytes()).hexdigest(),
            "paired_path": str(paired),
            "paired_sha256": hashlib.sha256(paired.read_bytes()).hexdigest(),
            "aggregate": {
                "paired_identity_rate": 1.0,
                "bidirectional_valid_anchor_rate": active_rate,
                "alias_only_false_context_anchor_rate": 0.2,
            },
        }
    return {
        "schema": "cross_scale_sampler_exposure_audit_v1",
        "scope": "training_inputs_and_exact_sampler_only",
        "test_predictions_used": False,
        "identity_policy": "exact_packet_uid_compaction",
        "context_policy": "leave_one_distinct_packet_out",
        "reports": reports,
    }


def finalize(tmp_path, d1, incremental=None, overall=None, exposure=None):
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
    exposure = exposure if exposure is not None else (
        cross_scale_exposure(tmp_path) if incremental is not None else None
    )
    exposure_path = write(tmp_path / "exposure.json", exposure) if exposure else None
    return freeze_method_config(
        base,
        base_path=base_path,
        d1=d1,
        d1_path=d1_path,
        d2_incremental=incremental,
        d2_incremental_path=incremental_path,
        d2_overall=overall,
        d2_overall_path=overall_path,
        cross_scale_exposure=exposure,
        cross_scale_exposure_path=exposure_path,
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
    assert payload["method_selection"]["cross_scale_exposure"]["datasets"][
        "vpn-app"
    ]["bidirectional_valid_anchor_rate"] == 0.6
    assert_fingerprint(payload)


def test_flow_gate_promotes_or_rolls_back_packet_selected_candidate(tmp_path):
    d1 = selection("candidate", 0.005)
    incremental = selection("candidate", 0.002)
    overall = selection("candidate", 0.005)
    provisional = finalize(tmp_path, d1, incremental, overall)
    exposure = cross_scale_exposure(tmp_path)
    exposure_path = write(tmp_path / "exposure_flow.json", exposure)

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
        cross_scale_exposure=exposure,
        cross_scale_exposure_path=exposure_path,
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
        cross_scale_exposure=exposure,
        cross_scale_exposure_path=exposure_path,
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


def test_d2_rejects_insufficient_effective_sampler_exposure(tmp_path):
    with pytest.raises(ValueError, match="insufficient"):
        finalize(
            tmp_path,
            selection("candidate", 0.005),
            selection("candidate", 0.002),
            selection("candidate", 0.005),
            exposure=cross_scale_exposure(tmp_path, active_rate=0.49),
        )


def test_d2_rejects_stale_sampler_input_binding(tmp_path):
    exposure = cross_scale_exposure(tmp_path)
    Path(exposure["reports"]["vpn-app"]["factual_path"]).write_text(
        "mutated", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="stale"):
        finalize(
            tmp_path,
            selection("candidate", 0.005),
            selection("candidate", 0.002),
            selection("candidate", 0.005),
            exposure=exposure,
        )
