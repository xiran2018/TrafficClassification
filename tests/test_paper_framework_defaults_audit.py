import audit_paper_framework_defaults as audit_mod


def _fake_unified_report(packet_status: str = "pass"):
    return {
        "status": "pass",
        "framework": {
            "shared_core_modules": list(audit_mod.DEFAULT_SHARED_CORE_MODULES),
            "paper_unified_flow_manifest_passes": 1,
            "paper_unified_packet_manifest_passes": 1,
            "flow_scope": ["vpn-app", "tls-120"],
            "packet_scope": ["vpn-app", "tls-120"],
        },
        "packet_level": [
            {
                "dataset": "vpn-app",
                "task": "packet-level",
                "path": "packet.json",
                "exists": True,
                "accuracy": 0.9,
                "macro_f1": 0.8,
                "publication_status": packet_status,
                "framework_provenance": {"matching_manifest_count": 4},
            }
        ],
    }


def test_paper_defaults_audit_includes_packet_publication_gate(monkeypatch):
    monkeypatch.setattr(
        audit_mod,
        "default_framework_results",
        lambda: [("vpn-app", "flow.json", 0.74, 0.65)],
    )
    monkeypatch.setattr(
        audit_mod,
        "audit_dataset",
        lambda dataset, path, target: {
            "dataset": dataset,
            "path": path,
            "exists": True,
            "target": {"accuracy": 0.74, "macro_f1": 0.65},
            "ok": True,
            "accuracy": 0.75,
            "macro_f1": 0.70,
            "target_met": True,
            "slot_status": {"mode": "recorded", "matches_required": True},
            "errors": [],
        },
    )
    monkeypatch.setattr(audit_mod, "build_unified_framework_report", _fake_unified_report)

    audit = audit_mod.build_audit()

    assert audit["ok"] is True
    assert [row["dataset"] for row in audit["flow_datasets"]] == ["vpn-app"]
    assert [row["dataset"] for row in audit["packet_datasets"]] == ["vpn-app"]


def test_paper_defaults_audit_fails_when_packet_provenance_fails(monkeypatch):
    monkeypatch.setattr(
        audit_mod,
        "default_framework_results",
        lambda: [("vpn-app", "flow.json", 0.74, 0.65)],
    )
    monkeypatch.setattr(
        audit_mod,
        "audit_dataset",
        lambda dataset, path, target: {
            "dataset": dataset,
            "path": path,
            "exists": True,
            "target": {"accuracy": 0.74, "macro_f1": 0.65},
            "ok": True,
            "accuracy": 0.75,
            "macro_f1": 0.70,
            "target_met": True,
            "slot_status": {"mode": "recorded", "matches_required": True},
            "errors": [],
        },
    )
    monkeypatch.setattr(
        audit_mod,
        "build_unified_framework_report",
        lambda: _fake_unified_report("missing_bound_result_manifest"),
    )

    audit = audit_mod.build_audit()

    assert audit["ok"] is False
    assert audit["packet_datasets"][0]["publication_status"] == "missing_bound_result_manifest"
