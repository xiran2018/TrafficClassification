import json

from audit_paper_candidate_promotion import audit_candidate, content_group_ci_status
from paper_framework_defaults import DEFAULT_UNIFIED_EXPERT_SLOTS


def _write_result(path):
    payload = {
        "accuracy": 0.9,
        "macro_f1": 0.8,
        "flow_ids": ["flow-a", "flow-b"],
        "flow_y_true": [0, 1],
        "flow_y_pred": [0, 1],
        "flow_prob": [[0.95, 0.05], [0.05, 0.95]],
        "feature_config": {"unified_expert_slots": list(DEFAULT_UNIFIED_EXPERT_SLOTS)},
        "selected": {
            "strategy": "always",
            "bootstrap_guard": {"enabled": True, "gain_quantile": 0.01},
            "target_shift_guard": {"prediction_change_rate": 0.0},
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_content_group_ci_status_requires_raw_candidate_specific_evidence():
    claim = {
        "content_group_ci_target_met": True,
        "content_group_accuracy_ci95": [0.8, 0.9],
        "content_group_macro_f1_ci95": [0.7, 0.8],
    }

    same = content_group_ci_status(
        {"raw_best_path": "same.json", "paper_safe_path": "same.json"},
        claim,
    )
    changed = content_group_ci_status(
        {"raw_best_path": "raw.json", "paper_safe_path": "safe.json"},
        claim,
    )

    assert same["status"] == "available"
    assert same["target_met"] is True
    assert changed["status"] == "missing_raw_candidate_group_ci"
    assert changed["target_met"] is False


def test_audit_candidate_blocks_changed_raw_without_content_group_ci(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    raw = tmp_path / "raw.json"
    safe = tmp_path / "safe.json"
    _write_result(raw)
    _write_result(safe)
    row = {
        "dataset": "demo",
        "raw_best_path": str(raw),
        "paper_safe_path": str(safe),
        "raw_minus_paper_safe_accuracy": 0.01,
        "raw_minus_paper_safe_macro_f1": 0.02,
    }
    claim = {"content_group_ci_target_met": True}

    audit = audit_candidate(row, 0.003, claim=claim, require_content_group_ci=True)

    assert audit["direct_promotable"] is False
    assert "raw_candidate_content_group_ci_not_ready:missing_raw_candidate_group_ci" in audit["blockers"]


def test_audit_candidate_allows_same_result_when_content_group_ci_passes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    raw = tmp_path / "same.json"
    _write_result(raw)
    row = {
        "dataset": "demo",
        "raw_best_path": str(raw),
        "paper_safe_path": str(raw),
        "raw_minus_paper_safe_accuracy": 0.01,
        "raw_minus_paper_safe_macro_f1": 0.02,
    }
    claim = {
        "content_group_ci_target_met": True,
        "content_group_accuracy_ci95": [0.8, 0.9],
        "content_group_macro_f1_ci95": [0.7, 0.8],
    }

    audit = audit_candidate(row, 0.003, claim=claim, require_content_group_ci=True)

    assert audit["direct_promotable"] is True
    assert audit["blockers"] == []


def test_audit_candidate_computes_raw_content_group_ci_when_index_is_supplied(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    raw = tmp_path / "raw.json"
    safe = tmp_path / "safe.json"
    _write_result(raw)
    _write_result(safe)
    pcap_a = tmp_path / "a.pcap"
    pcap_b = tmp_path / "b.pcap"
    pcap_a.write_bytes(b"packet-a")
    pcap_b.write_bytes(b"packet-b")
    index = tmp_path / "flow_embedding_index.jsonl"
    index.write_text(
        "\n".join(
            [
                json.dumps({"flow_id": "flow-a", "pcap_path": str(pcap_a)}),
                json.dumps({"flow_id": "flow-b", "pcap_path": str(pcap_b)}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    row = {
        "dataset": "demo",
        "raw_best_path": str(raw),
        "paper_safe_path": str(safe),
        "raw_minus_paper_safe_accuracy": 0.01,
        "raw_minus_paper_safe_macro_f1": 0.02,
    }
    claim = {
        "target_accuracy": 0.9,
        "target_macro_f1": 0.9,
        "content_group_ci_target_met": False,
    }

    audit = audit_candidate(
        row,
        0.003,
        claim=claim,
        require_content_group_ci=True,
        raw_content_group_index=str(index),
        content_group_bootstrap_samples=20,
        content_group_bootstrap_seed=7,
    )

    assert audit["content_group_ci"]["status"] == "computed"
    assert audit["content_group_ci"]["target_met"] is True
    assert audit["direct_promotable"] is True
    assert audit["blockers"] == []
