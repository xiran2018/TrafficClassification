import json

from make_paper_evidence_pack import (
    attach_content_robustness_to_claims,
    content_unique_rows,
    paper_audit_gates,
    paper_positioning,
    render_markdown,
)


def test_content_unique_rows_and_markdown(tmp_path):
    result_path = tmp_path / "vpn_content_unique.json"
    result_path.write_text(
        json.dumps(
            {
                "metrics": {
                    "original_flow_level": {"accuracy": 0.75, "macro_f1": 0.74},
                    "content_unique_flow_level": {"accuracy": 0.76, "macro_f1": 0.755},
                },
                "audit": {
                    "input_flows": 100,
                    "unique_content_flows": 96,
                    "duplicate_content_groups": 4,
                    "duplicate_rows_removed": 4,
                },
                "content_unique_bootstrap": {
                    "samples": 2000,
                    "accuracy_95_ci": [0.72, 0.80],
                    "macro_f1_95_ci": [0.70, 0.79],
                },
                "content_group_bootstrap": {
                    "samples": 2000,
                    "num_groups": 96,
                    "num_rows": 100,
                    "accuracy_95_ci": [0.71, 0.81],
                    "macro_f1_95_ci": [0.69, 0.80],
                },
            }
        ),
        encoding="utf-8",
    )
    claims = [{"dataset": "vpn-app"}]

    rows = content_unique_rows(claims, {"vpn-app": str(result_path)})

    assert rows[0]["content_unique_accuracy"] == 0.76
    assert rows[0]["delta_macro_f1"] == 0.015000000000000013
    assert rows[0]["duplicate_content_groups"] == 4
    assert rows[0]["content_group_count"] == 96
    assert rows[0]["content_group_accuracy_ci95"] == [0.71, 0.81]

    md = render_markdown(
        {
            "framework_consistency": {"consistent": True},
            "paper_audit_gates": {},
            "claims": [],
            "ablations": [],
            "content_unique": rows,
            "recommendations": [],
            "paper_positioning": {},
        }
    )

    assert "Content-Unique Robustness" in md
    assert "96/100" in md
    assert "[0.7200, 0.8000]" in md
    assert "[0.7100, 0.8100]" in md


def test_content_grouped_ci_is_attached_to_claims_and_positioning():
    claims = [
        {
            "dataset": "vpn-app",
            "target_accuracy": 0.74,
            "target_macro_f1": 0.65,
            "point_target_met": True,
            "claim_strength": "point_pass_ci_mixed",
        }
    ]
    content = [
        {
            "dataset": "vpn-app",
            "content_unique_accuracy": 0.753,
            "content_unique_macro_f1": 0.757,
            "content_group_accuracy_ci95": [0.729, 0.773],
            "content_group_macro_f1_ci95": [0.717, 0.782],
            "content_group_count": 1645,
            "content_group_rows": 1672,
        }
    ]

    enriched = attach_content_robustness_to_claims(claims, content)
    positioning = paper_positioning(enriched, [], {"consistent": True})

    assert enriched[0]["content_group_ci_target_met"] is False
    assert enriched[0]["content_group_count"] == 1645
    assert positioning["content_group_ci_mixed_datasets"] == ["vpn-app"]
    assert any("content-grouped bootstrap" in risk for risk in positioning["reviewer_risks"])


def test_paper_audit_gates_surface_flow_and_packet_scopes_in_markdown():
    gates = paper_audit_gates(
        {
            "status": "pass",
            "framework": {
                "paper_unified_flow_manifest_passes": 2,
                "paper_unified_packet_manifest_passes": 6,
                "flow_scope": ["vpn-app", "tls-120"],
                "packet_scope": ["vpn-app", "tls-120", "ustc-app"],
            },
        },
        {
            "ok": True,
            "unified_framework": {"shared_core_matches_defaults": True},
            "flow_datasets": [{"ok": True}, {"ok": True}],
            "packet_datasets": [
                {"publication_status": "pass"},
                {"publication_status": "pass"},
                {"publication_status": "pass"},
            ],
        },
    )

    assert gates["unified_framework_status"] == "pass"
    assert gates["paper_defaults_ok"] is True
    assert gates["flow_defaults_pass"] is True
    assert gates["packet_defaults_pass"] is True

    md = render_markdown(
        {
            "framework_consistency": {"consistent": True},
            "paper_audit_gates": gates,
            "claims": [],
            "ablations": [],
            "content_unique": [],
            "recommendations": [],
            "paper_positioning": {},
        }
    )

    assert "Paper Audit Gates" in md
    assert "vpn-app, tls-120" in md
    assert "vpn-app, tls-120, ustc-app" in md
    assert "| pass | True | True | 2 | 6 | True/2 | True/3 | True |" in md


def test_paper_audit_gates_fail_when_packet_default_not_publication_ready():
    gates = paper_audit_gates(
        {"status": "pass", "framework": {}},
        {
            "ok": False,
            "unified_framework": {"shared_core_matches_defaults": True},
            "flow_datasets": [{"ok": True}],
            "packet_datasets": [{"publication_status": "missing_bound_result_manifest"}],
        },
    )

    assert gates["paper_defaults_ok"] is False
    assert gates["flow_defaults_pass"] is True
    assert gates["packet_defaults_pass"] is False


def test_paper_audit_gates_do_not_replace_strict_zero_counts_with_stale_defaults():
    gates = paper_audit_gates(
        {
            "status": "review",
            "framework": {
                "paper_unified_flow_manifest_passes": 0,
                "paper_unified_packet_manifest_passes": 0,
                "flow_scope": [],
                "packet_scope": [],
            },
        },
        {
            "ok": True,
            "unified_framework": {
                "shared_core_matches_defaults": True,
                "paper_unified_flow_manifest_passes": 4,
                "paper_unified_packet_manifest_passes": 24,
                "flow_scope": ["stale-flow"],
                "packet_scope": ["stale-packet"],
            },
            "flow_datasets": [{"ok": True}],
            "packet_datasets": [{"publication_status": "pass"}],
        },
    )

    assert gates["paper_unified_flow_manifest_passes"] == 0
    assert gates["paper_unified_packet_manifest_passes"] == 0
    assert gates["flow_scope"] == []
    assert gates["packet_scope"] == []
    assert gates["strict_reproduction_complete"] is False
