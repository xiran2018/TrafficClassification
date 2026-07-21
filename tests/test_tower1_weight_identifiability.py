import pytest

from analyze_tower1_weight_identifiability import build_report


def test_weight_identifiability_report_aligns_classes_and_reports_associations():
    matched = {
        "matched_comparison": {
            "latest_matched": {
                "step": 20,
                "per_class": [
                    {"label": "alpha", "support": 10, "f1_delta": -0.2, "recall_delta": -0.1},
                    {"label": "beta", "support": 20, "f1_delta": 0.0, "recall_delta": 0.0},
                    {"label": "gamma", "support": 30, "f1_delta": 0.2, "recall_delta": 0.1},
                ],
            }
        }
    }
    identifiability = {
        "levels": {
            "session": {
                "train": {
                    "per_class": {
                        "0": {"conflicting_sample_rate": 0.9},
                        "1": {"conflicting_sample_rate": 0.5},
                        "2": {"conflicting_sample_rate": 0.1},
                    }
                },
                "test": {
                    "per_class": {
                        "0": {"conflicting_sample_rate": 0.8},
                        "1": {"conflicting_sample_rate": 0.4},
                        "2": {"conflicting_sample_rate": 0.0},
                    }
                },
            }
        }
    }
    sampling = {
        "flow_count_weights": {
            "0.5": {"0": 3.0, "1": 2.0, "2": 1.0}
        }
    }

    report = build_report(
        matched,
        identifiability,
        sampling,
        {"alpha": 0, "beta": 1, "gamma": 2},
    )

    assert report["selection_role"] == "descriptive_only"
    assert report["causal_claim"] is False
    assert report["matched_step"] == 20
    associations = report["association_with_f1_delta"]
    assert associations["flow_count_weight"]["rho"] == pytest.approx(-1.0)
    assert associations["train_conflicting_sample_rate"]["rho"] == pytest.approx(-1.0)
    assert associations["validation_conflicting_sample_rate"]["rho"] == pytest.approx(-1.0)


def test_weight_identifiability_report_rejects_misaligned_labels():
    matched = {
        "matched_comparison": {
            "latest_matched": {
                "step": 1,
                "per_class": [
                    {"label": "alpha", "support": 1, "f1_delta": 0.0, "recall_delta": 0.0}
                ],
            }
        }
    }
    identifiability = {
        "levels": {
            "session": {
                "train": {"per_class": {"0": {"conflicting_sample_rate": 0.0}}},
                "test": {"per_class": {"0": {"conflicting_sample_rate": 0.0}}},
            }
        }
    }
    sampling = {"flow_count_weights": {"0.5": {"0": 1.0}}}

    with pytest.raises(ValueError, match="do not align"):
        build_report(matched, identifiability, sampling, {"different": 0})
