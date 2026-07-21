import json
from pathlib import Path

import make_paper_method_card as method_card
from make_paper_method_card import build_card, render_markdown
from unified_framework_spec import ResultSpec


def _strict_result(path, fingerprint):
    path.parent.mkdir(parents=True, exist_ok=True)
    novelty = path.parent / f"{path.stem}_session_novelty.json"
    novelty.write_text('{"schema":"session_novelty_evaluation_v1"}', encoding="utf-8")
    import hashlib

    novelty_sha = hashlib.sha256(novelty.read_bytes()).hexdigest()
    path.write_text(
        json.dumps(
            {
                "publication_provenance": {
                    "status": "strict_shared_core_v2",
                    "shared_core_config_sha256": fingerprint,
                    "audit_paths": ["fold0.json", "fold1.json", "fold2.json"],
                    "runtime_mechanism_evidence_required": True,
                    "flow_native_extraction_evidence_required": True,
                    "fixed_consensus": "equal_log_mean_three_folds",
                    "session_novelty": str(novelty),
                    "session_novelty_sha256": novelty_sha,
                }
            }
        ),
        encoding="utf-8",
    )


def _strict_specs(tmp_path, task):
    return {
        dataset: ResultSpec(
            dataset=dataset,
            task=task,
            path=str(tmp_path / task / f"{dataset}.json"),
        )
        for dataset in ("vpn-app", "tls-120")
    }


def test_strict_publication_status_requires_one_fingerprint_across_four_results(
    tmp_path, monkeypatch
):
    packet_specs = _strict_specs(tmp_path, "packet-level")
    flow_specs = _strict_specs(tmp_path, "flow-level")
    monkeypatch.setattr(method_card, "PACKET_LEVEL_RESULTS", packet_specs)
    monkeypatch.setattr(method_card, "FLOW_LEVEL_RESULTS", flow_specs)
    for specs in (packet_specs, flow_specs):
        for spec in specs.values():
            _strict_result(tmp_path / spec.path, "shared-sha")

    report = method_card.strict_shared_core_publication_status()

    assert report["ready"] is True
    assert report["single_shared_core_config"] is True
    assert report["all_session_novelty_evidence_verified"] is True
    assert report["shared_core_config_sha256"] == "shared-sha"
    assert len(report["canonical_results"]) == 4


def test_strict_publication_status_rejects_cross_task_fingerprint_mismatch(
    tmp_path, monkeypatch
):
    packet_specs = _strict_specs(tmp_path, "packet-level")
    flow_specs = _strict_specs(tmp_path, "flow-level")
    monkeypatch.setattr(method_card, "PACKET_LEVEL_RESULTS", packet_specs)
    monkeypatch.setattr(method_card, "FLOW_LEVEL_RESULTS", flow_specs)
    for spec in packet_specs.values():
        _strict_result(tmp_path / spec.path, "packet-sha")
    for spec in flow_specs.values():
        _strict_result(tmp_path / spec.path, "flow-sha")

    report = method_card.strict_shared_core_publication_status()

    assert report["all_results_have_strict_provenance"] is True
    assert report["single_shared_core_config"] is False
    assert report["ready"] is False


def test_strict_publication_status_requires_session_novelty_provenance(
    tmp_path, monkeypatch
):
    packet_specs = _strict_specs(tmp_path, "packet-level")
    flow_specs = _strict_specs(tmp_path, "flow-level")
    monkeypatch.setattr(method_card, "PACKET_LEVEL_RESULTS", packet_specs)
    monkeypatch.setattr(method_card, "FLOW_LEVEL_RESULTS", flow_specs)
    for specs in (packet_specs, flow_specs):
        for spec in specs.values():
            _strict_result(tmp_path / spec.path, "shared-sha")
    payload = json.loads(Path(packet_specs["vpn-app"].path).read_text())
    payload["publication_provenance"].pop("session_novelty")
    Path(packet_specs["vpn-app"].path).write_text(json.dumps(payload))

    report = method_card.strict_shared_core_publication_status()

    assert report["ready"] is False
    failed = next(
        row
        for row in report["canonical_results"]
        if row["task"] == "packet-level" and row["dataset"] == "vpn-app"
    )
    assert failed["passed"] is False


def test_method_card_surfaces_group_ci_risk_and_packet_evidence():
    evidence = {
        "claims": [
            {
                "dataset": "vpn-app",
                "accuracy": 0.751,
                "macro_f1": 0.752,
                "target_accuracy": 0.74,
                "target_macro_f1": 0.65,
                "point_target_met": True,
                "ci_target_met": False,
                "content_group_ci_target_met": False,
                "claim_strength": "point_pass_ci_mixed",
            }
        ],
        "ablations": [
            {"dataset": "vpn-app", "stage": "unsafe reliability fusion", "effect": "harmful"},
            {"dataset": "tls-120", "stage": "soft expert gate", "effect": "helpful"},
        ],
        "paper_audit_gates": {"paper_defaults_ok": True},
    }
    unified_audit = {
        "status": "pass",
        "exact_shared_core_v2": {"status": "not_ready"},
        "unified_method_v2": {"status": "pass"},
        "framework": {
            "flow_scope": ["vpn-app", "tls-120"],
            "packet_scope": ["vpn-app"],
            "shared_core_modules": ["per_flow_split_guard", "field_aware_header_policy"],
            "flow_modules": ["per_flow_split_guard", "field_aware_header_policy", "flow_level_classifier"],
            "packet_modules": ["per_flow_split_guard", "field_aware_header_policy", "packet_level_classifier"],
        },
    }
    defaults_audit = {
        "ok": True,
        "defaults": {"framework_profile": "paper_unified"},
        "packet_datasets": [
            {
                "dataset": "vpn-app",
                "accuracy": 0.9,
                "macro_f1": 0.8,
                "publication_status": "pass",
                "path": "reasoningDataset/packet-level/vpn-app/paper_default_result.json",
            }
        ],
    }

    card = build_card(
        evidence,
        unified_audit,
        defaults_audit,
        candidate_rankings=[
            {
                "dataset": "vpn-app",
                "num_candidates": 10,
                "num_ok": 8,
                "best_path": "reasoningDataset/vpn-app/test_crossfold_consensus_auto_confidence.json",
                "best_accuracy": 0.751,
                "best_macro_f1": 0.752,
                "best_content_group_accuracy_lower": 0.731,
                "best_content_group_macro_f1_lower": 0.710,
                "best_content_group_target_met": False,
            }
        ],
    )
    md = render_markdown(card)

    assert card["readiness"]["legacy_protocol_audit_ready"] is True
    assert card["readiness"]["exact_common_reference_v2_ready"] is False
    assert card["readiness"]["unified_method_v2_ready"] is True
    assert card["readiness"]["strict_shared_core_v2_ready"] is False
    assert card["readiness"]["session_novelty_evidence_ready"] is False
    assert card["readiness"]["paper_unified_profile_ready"] is False
    assert card["readiness"]["flow_point_targets_ready"] is True
    assert card["readiness"]["flow_content_group_ci_targets_ready"] is False
    assert card["readiness"]["packet_publication_defaults_ready"] is True
    assert card["readiness"]["ccf_a_risk_level"] == "high"
    assert "historical packet-level scores" in card["readiness"]["recommended_main_claim"]
    assert card["ablation_summary"]["helpful_count"] == 1
    assert card["ablation_summary"]["harmful_count"] == 1
    assert "content-grouped bootstrap" in md
    assert "paper_default_result.json" in md
    assert "Content-Group Candidate Scan" in md
    assert "test_crossfold_consensus_auto_confidence.json" in md
    assert "dataset-specific branches" in md
    assert "| unified method v2 | True |" in md
    contribution_text = " ".join(
        row["name"] + " " + row["claim"] for row in card["contributions"]
    ).lower()
    assert "counterfactual semantic-structural routing" in contribution_text
    assert "graph, sequence, statistics" not in contribution_text


def test_method_card_marks_low_risk_when_all_gates_pass():
    evidence = {
        "claims": [
            {
                "dataset": "tls-120",
                "accuracy": 0.846,
                "macro_f1": 0.829,
                "target_accuracy": 0.78,
                "target_macro_f1": 0.70,
                "point_target_met": True,
                "ci_target_met": True,
                "content_group_ci_target_met": True,
                "claim_strength": "strong",
            }
        ],
        "ablations": [],
        "paper_audit_gates": {"paper_defaults_ok": True},
    }
    unified_audit = {
        "status": "pass",
        "framework": {
            "flow_scope": ["tls-120"],
            "packet_scope": ["tls-120"],
            "shared_core_modules": ["per_flow_split_guard"],
            "flow_modules": ["per_flow_split_guard", "flow_level_classifier"],
            "packet_modules": ["per_flow_split_guard", "packet_level_classifier"],
        },
    }
    defaults_audit = {
        "ok": True,
        "defaults": {"framework_profile": "paper_unified"},
        "packet_datasets": [
            {
                "dataset": "tls-120",
                "accuracy": 0.87,
                "macro_f1": 0.84,
                "publication_status": "pass",
                "path": "reasoningDataset/packet-level/tls-120/paper_default_result.json",
            }
        ],
    }

    card = build_card(
        evidence,
        unified_audit,
        defaults_audit,
        strict_status={
            "ready": True,
            "canonical_results": [],
            "single_shared_core_config": True,
            "shared_core_config_sha256": "abc",
        },
    )

    assert card["readiness"]["ccf_a_risk_level"] == "moderate"
    assert card["readiness"]["ccf_a_risks"] == []
    assert "provenance gates satisfied" in card["readiness"]["recommended_main_claim"]


def test_method_card_does_not_promote_historical_packet_scores_without_provenance():
    evidence = {
        "claims": [{"point_target_met": True, "ci_target_met": False, "content_group_ci_target_met": False}],
        "ablations": [],
    }
    unified_audit = {
        "status": "review",
        "framework": {"flow_scope": ["vpn-app"], "packet_scope": ["vpn-app"]},
    }
    defaults_audit = {
        "ok": False,
        "defaults": {"framework_profile": "paper_unified"},
        "packet_datasets": [
            {
                "dataset": "vpn-app",
                "accuracy": 0.9,
                "macro_f1": 0.8,
                "publication_status": "needs_paper_unified_repro",
            }
        ],
    }

    card = build_card(evidence, unified_audit, defaults_audit)

    claim = card["readiness"]["recommended_main_claim"]
    assert "historical packet-level scores" in claim
    assert "remain non-headline evidence" in claim
