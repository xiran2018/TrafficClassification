import pytest

from analyze_tower1_validation_history import (
    aggregate_runs,
    compare_matched_histories,
    summarize_history,
)


def row(step, accuracy, macro_f1, alpha_f1, beta_f1):
    return {
        "step": step,
        "metrics": {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "per_class": {
                "alpha": {"f1": alpha_f1, "recall": alpha_f1, "support": 10},
                "beta": {"f1": beta_f1, "recall": beta_f1, "support": 100},
            },
        },
    }


def test_validation_summary_selects_macro_f1_and_tracks_regression():
    summary = summarize_history(
        [
            row(10, 0.50, 0.40, 0.10, 0.70),
            row(20, 0.61, 0.55, 0.35, 0.75),
            row(30, 0.65, 0.51, 0.30, 0.72),
        ]
    )

    assert summary["best"]["step"] == 20
    assert summary["validation_regression_after_best"] == pytest.approx(0.04)
    assert summary["recovered_classes"] == ["alpha"]
    assert summary["persistently_weak_classes"] == []


def test_validation_summary_and_aggregate_report_persistent_weak_class():
    first = summarize_history([row(10, 0.5, 0.4, 0.1, 0.7)])
    second = summarize_history([row(10, 0.7, 0.6, 0.2, 0.8)])
    aggregate = aggregate_runs({"fold0": first, "fold1": second})

    assert first["persistently_weak_classes"] == ["alpha"]
    assert aggregate["num_runs"] == 2
    assert aggregate["best_accuracy_mean"] == 0.6
    assert aggregate["best_macro_f1_mean"] == 0.5


def test_matched_history_comparison_reports_long_tail_gain_without_mixing_steps():
    baseline = [
        row(10, 0.50, 0.40, 0.10, 0.70),
        row(20, 0.60, 0.50, 0.20, 0.80),
    ]
    candidate = [
        row(10, 0.55, 0.50, 0.30, 0.70),
        row(20, 0.58, 0.48, 0.18, 0.78),
        row(30, 0.70, 0.60, 0.40, 0.80),
    ]

    comparison = compare_matched_histories(baseline, candidate)

    assert comparison["matched_steps"] == [10, 20]
    latest = comparison["latest_matched"]
    assert latest["accuracy_delta"] == pytest.approx(-0.02)
    assert latest["macro_f1_delta"] == pytest.approx(-0.02)
    curve = comparison["matched_curve_summary"]
    assert curve["accuracy"]["candidate_wins"] == 1
    assert curve["accuracy"]["candidate_losses"] == 1
    assert curve["accuracy"]["mean_delta"] == pytest.approx(0.015)
    assert curve["macro_f1"]["best_matched_step"] == 10
    assert curve["macro_f1"]["worst_matched_step"] == 20
    phase = curve["macro_f1"]["phase_dynamics"]
    assert phase["early_steps"] == [10]
    assert phase["late_steps"] == [20]
    assert phase["early_mean_delta"] == pytest.approx(0.10)
    assert phase["late_mean_delta"] == pytest.approx(-0.02)
    assert phase["late_minus_early_mean_delta"] == pytest.approx(-0.12)
    assert phase["first_to_latest_delta_change"] == pytest.approx(-0.12)
    assert phase["late_phase"]["candidate_losses"] == 1
    assert curve["selection_role"] == "descriptive_only"
    first = comparison["comparisons"][0]
    assert first["positive_f1_classes"] == 1
    assert first["support_gain_spearman"] == pytest.approx(-1.0)
    assert first["top_f1_gains"][0]["label"] == "alpha"


def test_matched_history_comparison_rejects_changed_validation_support():
    baseline = [row(10, 0.50, 0.40, 0.10, 0.70)]
    candidate = [row(10, 0.55, 0.50, 0.30, 0.70)]
    candidate[0]["metrics"]["per_class"]["alpha"]["support"] = 11

    with pytest.raises(ValueError, match="support differs"):
        compare_matched_histories(baseline, candidate)
