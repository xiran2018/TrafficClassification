from types import SimpleNamespace

from make_next_experiment_plan import (
    build_commands,
    classify_row,
    distill_coverage_audit_status,
)


def test_ci_gap_with_negative_fresh_paired_recommends_content_grouped_path():
    claim = {
        "dataset": "vpn-app",
        "target_accuracy": 0.74,
        "target_macro_f1": 0.65,
        "accuracy": 0.751,
        "macro_f1": 0.752,
        "accuracy_ci95": [0.728, 0.772],
        "macro_f1_ci95": [0.718, 0.783],
        "claim_strength": "point_pass_ci_mixed",
        "point_target_met": True,
        "ci_target_met": False,
    }
    rec = {
        "recommendation": "Fresh flow-aware paired-view probes are also negative. Do not increase Tower-2 paired IP/port-randomization weight.",
        "raw_minus_paper_safe_accuracy": 0.0,
        "raw_minus_paper_safe_macro_f1": 0.0,
    }

    row = classify_row(claim, rec)

    assert row["gap_kind"] == "ci_lower_bound_gap"
    assert "content-grouped" in row["next_action"]
    assert "native structural pretraining as a negative ablation" in row["next_action"]
    assert "distillation" in row["next_action"]


def test_content_grouped_ci_gap_takes_priority_after_group_evidence_exists():
    claim = {
        "dataset": "vpn-app",
        "target_accuracy": 0.74,
        "target_macro_f1": 0.65,
        "accuracy": 0.751,
        "macro_f1": 0.752,
        "accuracy_ci95": [0.732, 0.774],
        "macro_f1_ci95": [0.724, 0.783],
        "content_group_accuracy_ci95": [0.729, 0.773],
        "content_group_macro_f1_ci95": [0.717, 0.782],
        "claim_strength": "point_pass_ci_mixed",
        "point_target_met": True,
        "ci_target_met": False,
    }

    row = classify_row(claim, {"raw_minus_paper_safe_accuracy": 0.0, "raw_minus_paper_safe_macro_f1": 0.0})

    assert row["gap_kind"] == "content_grouped_ci_lower_bound_gap"
    assert row["content_group_ci_accuracy_gap"] == 0.01100000000000001
    assert "Content-grouped robustness evidence is available" in row["next_action"]
    assert "expert-slot interface" in row["next_action"]


def test_default_distillation_plan_is_coverage_audited_before_flow_id_kl():
    args = SimpleNamespace(
        datasets="vpn-app,tls-120",
        goal_datasets="vpn-app,tls-120",
        max_iters=1,
        run_tag="paper_unified_next",
        run_tag_template="{run_tag}_iter{iteration:02d}",
        loop_output_json="reasoningDataset/autonomous_loop/research_loop_next.json",
    )

    commands = build_commands(
        args,
        {"status": "negative_target_gap", "accuracy": 0.57, "macro_f1": 0.54},
        {"status": "incomplete"},
        {"status": "incomplete"},
    )
    names = [item["name"] for item in commands]

    assert "vpn_distillation_coverage_audit" in names
    assert "tls120_distillation_coverage_audit" in names
    assert "native_structural_pretraining_pilot" not in names

    candidate = next(item for item in commands if item["name"] == "coverage_audited_calibrated_distillation_ablation")
    cmd = candidate["cmd"]
    assert cmd[cmd.index("--framework_profile") + 1] == "paper_unified"
    assert cmd[cmd.index("--distill_weight") + 1] == "0.02"
    assert cmd[cmd.index("--distill_class_prior_weight") + 1] != "0.0"
    assert cmd[cmd.index("--distill_temperature") + 1] == "6.0"
    assert cmd[cmd.index("--distill_max_confidence") + 1] == "0.85"
    assert cmd[cmd.index("--distill_confidence_power") + 1] == "0.0"
    assert "consensus_distill_student" in cmd

    vpn_audit = next(item for item in commands if item["name"] == "vpn_distillation_coverage_audit")
    assert any(
        "train_tower2_rawproj_flowaware_change_weight_split2_retrain/seq_dataset.pt" in item
        for item in vpn_audit["cmd"]
    )
    assert vpn_audit["cmd"].count("--gate_dataset") == 2


def test_completed_distillation_audits_do_not_recommend_stale_seeded_suite():
    args = SimpleNamespace(
        datasets="vpn-app,tls-120",
        goal_datasets="vpn-app,tls-120",
        max_iters=1,
        run_tag="paper_unified_next",
        run_tag_template="{run_tag}_iter{iteration:02d}",
        loop_output_json="reasoningDataset/autonomous_loop/research_loop_next.json",
    )

    commands = build_commands(
        args,
        {"status": "negative_target_gap", "accuracy": 0.57, "macro_f1": 0.54},
        {"status": "complete"},
        {"status": "complete"},
    )

    assert "coverage_audited_calibrated_distillation_ablation" not in [item["name"] for item in commands]
    assert "vpn_distillation_coverage_audit" not in [item["name"] for item in commands]
    assert "tls120_distillation_coverage_audit" not in [item["name"] for item in commands]
    assert "group_ci_distillation_stability_suite" not in [item["name"] for item in commands]


def test_default_plan_can_include_native_structural_when_not_already_negative():
    args = SimpleNamespace(
        datasets="vpn-app,tls-120",
        goal_datasets="vpn-app,tls-120",
        max_iters=1,
        run_tag="paper_unified_next",
        run_tag_template="{run_tag}_iter{iteration:02d}",
        loop_output_json="reasoningDataset/autonomous_loop/research_loop_next.json",
    )

    commands = build_commands(args, {"status": "not_run"}, {"status": "incomplete"}, {"status": "incomplete"})
    native = next(item for item in commands if item["name"] == "native_structural_pretraining_pilot")

    assert "run_stage8_flowaware_pipeline.py" in native["cmd"]
    assert native["cmd"][native["cmd"].index("--stage") + 1] == "paper_unified"
    assert native["cmd"][native["cmd"].index("--framework_profile") + 1] == "paper_unified"
    assert native["cmd"][native["cmd"].index("--native_structural_suffix") + 1] == "struct_pilot"


def test_distillation_coverage_audit_status_complete_when_both_safe(tmp_path):
    paths = {}
    for dataset in ("vpn-app", "tls-120"):
        path = tmp_path / f"{dataset}.json"
        path.write_text(
            """
{
  "recommendation": "flow_id_distillation_safe",
  "min_coverage": 0.5,
  "datasets": [{"coverage": 0.75}]
}
""".strip(),
            encoding="utf-8",
        )
        paths[dataset] = path

    status = distill_coverage_audit_status(paths)

    assert status["status"] == "complete"
    assert all(row["audit_status"] == "passed" for row in status["results"])


def test_distillation_coverage_audit_status_incomplete_when_missing(tmp_path):
    paths = {
        "vpn-app": tmp_path / "missing_vpn.json",
        "tls-120": tmp_path / "missing_tls.json",
    }

    status = distill_coverage_audit_status(paths)

    assert status["status"] == "incomplete"
    assert all(row["audit_status"] == "missing" for row in status["results"])
