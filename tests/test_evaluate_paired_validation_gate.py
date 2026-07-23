import json

import pytest

from evaluate_paired_validation_gate import (
    build_gate_payload,
    compare_metrics,
    mechanism_diagnostics,
)


def write_metrics(path, task, accuracy, macro_f1):
    metrics = {"accuracy": accuracy, "macro_f1": macro_f1}
    if task == "flow":
        metrics = {"flow_level": metrics}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"metrics": metrics}), encoding="utf-8")
    return path


def test_compare_metrics_applies_preregistered_macro_f1_and_accuracy_guards():
    passing = compare_metrics(
        {"accuracy": 0.80, "macro_f1": 0.70},
        {"accuracy": 0.795, "macro_f1": 0.705},
        min_macro_f1_delta=0.005,
        max_accuracy_drop=0.005,
    )
    assert passing["pass"] is True

    failing = compare_metrics(
        {"accuracy": 0.80, "macro_f1": 0.70},
        {"accuracy": 0.79, "macro_f1": 0.71},
        min_macro_f1_delta=0.005,
        max_accuracy_drop=0.005,
    )
    assert failing["macro_f1_improvement_pass"] is True
    assert failing["accuracy_guard_pass"] is False
    assert failing["pass"] is False


def test_mechanism_diagnostics_supports_legacy_and_standardized_names(tmp_path):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    values = {
        "packet": {"factual_only": (0.80, 0.79), "intervened_only": (0.12, 0.08)},
        "flow": {"factual_only": (0.59, 0.58), "intervened_only": (0.11, 0.07)},
    }
    improved = {
        "packet": {"factual_only": (0.81, 0.80), "intervened_only": (0.62, 0.60)},
        "flow": {"factual_only": (0.60, 0.59), "intervened_only": (0.45, 0.43)},
    }
    for task in ("packet", "flow"):
        for view in ("factual_only", "intervened_only"):
            write_metrics(
                baseline / f"{task}_valid_{view}.json",
                task,
                *values[task][view],
            )
            write_metrics(
                candidate / f"vpn-app_fold0_valid_{task}_{view}.json",
                task,
                *improved[task][view],
            )

    diagnostics = mechanism_diagnostics(baseline, candidate, "vpn-app")
    assert diagnostics["packet"]["intervened_only"]["delta"]["macro_f1"] == pytest.approx(0.52)
    assert diagnostics["flow"]["intervention_gap"]["macro_f1"]["reduction"] == pytest.approx(0.35)
    assert "pass" not in diagnostics["packet"]["intervened_only"]


def test_gate_requires_both_tasks_and_keeps_sensitivity_diagnostic_only(tmp_path):
    baseline_packet = write_metrics(tmp_path / "base_packet.json", "packet", 0.80, 0.70)
    candidate_packet = write_metrics(tmp_path / "candidate_packet.json", "packet", 0.80, 0.71)
    baseline_flow = write_metrics(tmp_path / "base_flow.json", "flow", 0.60, 0.59)
    candidate_flow = write_metrics(tmp_path / "candidate_flow.json", "flow", 0.59, 0.60)

    payload = build_gate_payload(
        dataset="vpn-app",
        baseline_packet=baseline_packet,
        candidate_packet=candidate_packet,
        baseline_flow=baseline_flow,
        candidate_flow=candidate_flow,
    )
    assert payload["tasks"]["packet"]["pass"] is True
    assert payload["tasks"]["flow"]["macro_f1_improvement_pass"] is True
    assert payload["tasks"]["flow"]["accuracy_guard_pass"] is False
    assert payload["strict_pass"] is False
    assert payload["test_metrics_used"] is False


def test_gate_rejects_only_one_sensitivity_directory(tmp_path):
    packet = write_metrics(tmp_path / "packet.json", "packet", 0.8, 0.7)
    flow = write_metrics(tmp_path / "flow.json", "flow", 0.6, 0.5)
    with pytest.raises(ValueError, match="supplied together"):
        build_gate_payload(
            dataset="vpn-app",
            baseline_packet=packet,
            candidate_packet=packet,
            baseline_flow=flow,
            candidate_flow=flow,
            baseline_sensitivity_dir=tmp_path,
        )
