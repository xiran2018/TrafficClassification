import json

from freeze_shared_core_v2_config import canonical_sha256
from make_flow_gate_configs import make_configs


def signed(payload):
    payload["config_sha256"] = canonical_sha256(payload)
    return payload


def test_flow_gate_configs_are_test_forbidden_and_change_only_candidate_objectives(tmp_path):
    base = signed(
        {
            "tower1": {},
            "selection_protocol": {"test_labels_used": False},
        }
    )
    provisional = signed(
        {
            "tower1": {
                "identity_safe_contrastive": True,
                "cross_scale_weight": 0.05,
                "cross_scale_temperature": 0.07,
            },
            "selection_protocol": {
                "test_labels_used": False,
                "test_evaluation_allowed": False,
            },
            "method_selection": {
                "decision_status": "packet_selected_pending_flow_noninferiority"
            },
        }
    )
    base_path = tmp_path / "base.json"
    candidate_path = tmp_path / "candidate.json"
    base_path.write_text(json.dumps(base), encoding="utf-8")
    candidate_path.write_text(json.dumps(provisional), encoding="utf-8")
    control, candidate = make_configs(
        base,
        provisional,
        base_path=base_path,
        provisional_path=candidate_path,
    )
    assert control["tower1"]["identity_safe_contrastive"] is False
    assert control["tower1"]["cross_scale_weight"] == 0.0
    assert candidate["tower1"]["identity_safe_contrastive"] is True
    assert candidate["tower1"]["cross_scale_weight"] == 0.05
    assert control["selection_protocol"]["test_evaluation_allowed"] is False
    assert candidate["selection_protocol"]["test_evaluation_allowed"] is False
