import json

from compare_sweet_reference import (
    compare_result,
    extract_metrics,
    verify_strict_provenance,
)


def test_extract_metrics_supports_packet_and_flow_shapes():
    assert extract_metrics(
        {"metrics": {"packet_level": {"accuracy": 0.9, "macro_f1": 0.8}}},
        "packet-level",
    ) == (0.9, 0.8)
    assert extract_metrics(
        {"metrics": {"flow_level": {"accuracy": 0.85, "macro_f1": 0.82}}},
        "flow-level",
    ) == (0.85, 0.82)


def test_tls_flow_does_not_claim_to_beat_sweet_end_to_end_at_85_percent(tmp_path):
    path = tmp_path / "tls_flow.json"
    path.write_text(
        json.dumps(
            {"metrics": {"flow_level": {"accuracy": 0.85, "macro_f1": 0.83}}}
        ),
        encoding="utf-8",
    )

    row = compare_result("flow-level", "tls-120", str(path))

    assert row["sweet"]["frozen_representation"]["exceeds_both"] is True
    assert row["sweet"]["end_to_end"]["exceeds_both"] is False
    assert row["strict_shared_core_v2_result"] is False
    assert row["headline_sweet_claim"] == "does_not_exceed_protocol_matched_end_to_end"


def test_vpn_packet_can_exceed_both_sweet_reference_tiers(tmp_path):
    path = tmp_path / "vpn_packet.json"
    path.write_text(
        json.dumps({"metrics": {"accuracy": 0.91, "macro_f1": 0.81}}),
        encoding="utf-8",
    )

    row = compare_result("packet-level", "vpn-app", str(path))

    assert row["sweet"]["frozen_representation"]["exceeds_both"] is True
    assert row["sweet"]["end_to_end"]["exceeds_both"] is True
    assert row["headline_sweet_claim"] == "exceeds_protocol_matched_end_to_end"


def test_strict_provenance_requires_hash_bound_method_and_novelty_evidence(tmp_path):
    method = tmp_path / "method_archive.json"
    novelty = tmp_path / "session_novelty.json"
    bootstrap = tmp_path / "bootstrap.json"
    method.write_text('{"status":"verified"}', encoding="utf-8")
    novelty.write_text('{"status":"reported"}', encoding="utf-8")
    bootstrap.write_text('{"method":"flow_cluster_bootstrap"}', encoding="utf-8")
    audits = []
    for fold in range(3):
        audit = tmp_path / f"fold{fold}_audit.json"
        audit.write_text(json.dumps({"fold": fold, "status": "pass"}), encoding="utf-8")
        audits.append(audit)

    import hashlib

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    provenance = {
        "status": "strict_shared_core_v2",
        "shared_core_config_sha256": "a" * 64,
        "fixed_consensus": "equal_log_mean_three_folds",
        "audit_evidence": [
            {"path": str(path), "sha256": digest(path)} for path in audits
        ],
        "method_archive_manifest": str(method),
        "method_archive_manifest_sha256": digest(method),
        "session_novelty": str(novelty),
        "session_novelty_sha256": digest(novelty),
        "bootstrap_evidence": str(bootstrap),
        "bootstrap_evidence_sha256": digest(bootstrap),
    }
    assert verify_strict_provenance(provenance)["status"] == "pass"

    audits[0].write_text('{"status":"mutated"}', encoding="utf-8")
    rejected_audit = verify_strict_provenance(provenance)
    assert rejected_audit["status"] == "fail"
    assert "audit_evidence_hash_mismatch" in rejected_audit["reasons"]
    audits[0].write_text(json.dumps({"fold": 0, "status": "pass"}), encoding="utf-8")

    method.write_text('{"status":"mutated"}', encoding="utf-8")
    rejected = verify_strict_provenance(provenance)
    assert rejected["status"] == "fail"
    assert "method_archive_manifest_hash_mismatch" in rejected["reasons"]


def test_status_string_alone_is_not_strict_provenance():
    result = verify_strict_provenance({"status": "strict_shared_core_v2"})
    assert result["status"] == "fail"
    assert "missing_shared_core_fingerprint" in result["reasons"]


def test_strict_provenance_accepts_explicit_shared_method_fingerprint(tmp_path):
    artifacts = []
    for index in range(6):
        path = tmp_path / f"artifact_{index}.json"
        path.write_text("{}", encoding="utf-8")
        artifacts.append(path)

    import hashlib

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    provenance = {
        "status": "strict_shared_core_v2",
        "shared_core_method_sha256": "b" * 64,
        "fixed_consensus": "equal_log_mean_three_folds",
        "audit_evidence": [
            {"path": str(path), "sha256": digest(path)} for path in artifacts[:3]
        ],
        "method_archive_manifest": str(artifacts[3]),
        "method_archive_manifest_sha256": digest(artifacts[3]),
        "session_novelty": str(artifacts[4]),
        "session_novelty_sha256": digest(artifacts[4]),
        "bootstrap_evidence": str(artifacts[5]),
        "bootstrap_evidence_sha256": digest(artifacts[5]),
    }

    result = verify_strict_provenance(provenance)

    assert result["status"] == "pass"
    assert result["shared_core_method_sha256"] == "b" * 64
