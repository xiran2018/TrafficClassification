from pathlib import Path

import pytest

from freeze_shared_core_v2_config import canonical_sha256
from preregister_paired_shared_core_v2 import build_paired_candidate


def base_payload():
    payload = {
        "status": "frozen_for_development_milestone",
        "tower1": {
            "paired_consistency_weight": 0.0,
            "paired_cls_weight": 0.0,
            "paired_logit_kl_weight": 0.5,
            "paired_raw_consistency_weight": 1.0,
        },
        "selection_evidence": {"balance": {"selected": "baseline"}},
    }
    payload["config_sha256"] = canonical_sha256(payload)
    return payload


def test_preregistered_candidate_changes_only_the_paired_method_factor():
    base = base_payload()
    candidate = build_paired_candidate(
        base,
        source_path=Path("base.json"),
        source_file_sha256="a" * 64,
    )

    assert base["tower1"]["paired_consistency_weight"] == 0.0
    assert candidate["tower1"]["paired_consistency_weight"] == 0.05
    assert candidate["tower1"]["paired_cls_weight"] == 0.2
    assert candidate["tower1"]["paired_validation_selection"] == (
        "worst_view_macro_f1"
    )
    assert candidate["selection_evidence"]["balance"] == (
        base["selection_evidence"]["balance"]
    )
    assert candidate["method_selection"]["test_labels_used"] is False
    fingerprint = candidate.pop("config_sha256")
    assert fingerprint == canonical_sha256(candidate)


def test_preregister_rejects_an_already_paired_source():
    base = base_payload()
    base["tower1"]["paired_consistency_weight"] = 0.05
    with pytest.raises(ValueError, match="disabled control"):
        build_paired_candidate(
            base,
            source_path=Path("base.json"),
            source_file_sha256="a" * 64,
        )
