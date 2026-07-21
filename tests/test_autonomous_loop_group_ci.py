from types import SimpleNamespace

from run_autonomous_research_loop import ci_targets_ready, content_unique_eval_cmd, report_commands


def test_ci_targets_ready_requires_content_group_gate_by_default():
    evidence = {
        "claims": [
            {
                "dataset": "vpn-app",
                "ci_target_met": True,
                "content_group_ci_target_met": False,
            }
        ]
    }

    assert not ci_targets_ready(evidence, ["vpn-app"], required=True)
    assert ci_targets_ready(evidence, ["vpn-app"], required=True, require_content_group=False)


def test_ci_targets_ready_passes_when_standard_and_group_ci_pass():
    evidence = {
        "claims": [
            {
                "dataset": "tls-120",
                "ci_target_met": True,
                "content_group_ci_target_met": True,
            }
        ]
    }

    assert ci_targets_ready(evidence, ["tls-120"], required=True)


def test_report_commands_refresh_content_grouped_metrics_when_index_exists(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    index = tmp_path / "reasoningDataset" / "vpn-app" / "test_embeddings_rawproj_change_weight"
    index.mkdir(parents=True)
    (index / "flow_embedding_index.jsonl").write_text("", encoding="utf-8")
    args = SimpleNamespace(
        final_selector_unified_expert_slots="base,graph,seq",
        content_unique_bootstrap_samples=17,
    )

    command = content_unique_eval_cmd(args, "vpn-app")

    assert command is not None
    assert "evaluate_content_unique_predictions.py" in command
    assert command[command.index("--bootstrap_samples") + 1] == "17"

    commands = report_commands(args, ["vpn-app"])
    names = [cmd[1] for cmd in commands]
    assert "evaluate_content_unique_predictions.py" in names
    assert "audit_unified_framework.py" in names
    assert "audit_paper_framework_defaults.py" in names
    assert "audit_paper_candidate_promotion.py" in names
    assert names.index("evaluate_content_unique_predictions.py") < names.index("make_paper_evidence_pack.py")
    assert names.index("audit_unified_framework.py") < names.index("make_paper_evidence_pack.py")
    assert names.index("audit_paper_framework_defaults.py") < names.index("make_paper_evidence_pack.py")
    assert names.index("make_paper_evidence_pack.py") < names.index("make_paper_method_card.py")
    promotion = commands[names.index("audit_paper_candidate_promotion.py")]
    assert promotion[promotion.index("--content_group_bootstrap_samples") + 1] == "17"
    assert promotion[promotion.index("--raw_content_group_index") + 1].endswith(
        "reasoningDataset/vpn-app/test_embeddings_rawproj_change_weight/flow_embedding_index.jsonl"
    )
