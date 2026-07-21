import pytest

from analyze_tower1_exposure_outcomes import (
    analyze_exposure_history,
    analyze_exposure_outcomes,
)


def metric_row(label_id, f1, recall):
    return {"label_id": label_id, "f1": f1, "recall": recall, "support": 10}


def test_exposure_outcome_audit_reports_rank_association_without_causal_claim():
    sampling = {"flow_counts": {"0": 2, "1": 8, "2": 32}}
    validation = {
        "metrics": {
            "per_class": {
                "low": metric_row(0, 0.1, 0.2),
                "mid": metric_row(1, 0.5, 0.6),
                "high": metric_row(2, 0.9, 1.0),
            }
        }
    }
    result = analyze_exposure_outcomes(sampling, validation)
    assert result["causal_claim"] is False
    assert result["association"]["validation_f1"]["rho"] == pytest.approx(1.0)
    assert result["association"]["validation_recall"]["rho"] == pytest.approx(1.0)
    assert [row["class"] for row in result["classes"]] == ["low", "mid", "high"]


def test_exposure_outcome_audit_requires_identical_label_sets():
    sampling = {"flow_counts": {"0": 2, "1": 8}}
    validation = {
        "metrics": {"per_class": {"only": metric_row(0, 0.5, 0.5)}}
    }
    with pytest.raises(ValueError, match="missing from validation"):
        analyze_exposure_outcomes(sampling, validation)


def test_exposure_history_reports_every_checkpoint_without_selecting_one():
    sampling = {"flow_counts": {"0": 2, "1": 8, "2": 32}}
    history = [
        {
            "step": step,
            "metrics": {
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "per_class": {
                    "low": metric_row(0, *low),
                    "mid": metric_row(1, *mid),
                    "high": metric_row(2, *high),
                },
            },
        }
        for step, accuracy, macro_f1, low, mid, high in [
            (20, 0.6, 0.5, (0.1, 0.2), (0.5, 0.6), (0.9, 1.0)),
            (10, 0.5, 0.4, (0.9, 1.0), (0.5, 0.6), (0.1, 0.2)),
        ]
    ]

    result = analyze_exposure_history(sampling, history)

    assert [row["step"] for row in result["trajectory"]] == [10, 20]
    assert result["trajectory"][0]["association"]["validation_f1"]["rho"] == pytest.approx(-1.0)
    assert result["trajectory"][1]["association"]["validation_f1"]["rho"] == pytest.approx(1.0)
    assert result["trajectory_summary"]["selection_role"] == "descriptive_only"
