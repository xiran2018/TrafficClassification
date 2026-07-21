import pytest

from analyze_prediction_domain_shift import compare_payloads


def payload(y_true, y_pred, effective_gate, bounds_satisfied=True):
    return {
        "label_map": {"a": 0, "b": 1},
        "flow_y_true": y_true,
        "flow_y_pred": y_pred,
        "metrics": {
            "eval_config": {
                "learned_gate_diagnostics": {
                    "dual_channel_gate": {
                        "effective_mean": effective_gate,
                        "theoretical_bounds": {
                            "semantic": [0.75, 1.0],
                            "content": [0.0, 0.25],
                        },
                        "bounds_satisfied": bounds_satisfied,
                    }
                }
            }
        },
    }


def test_domain_shift_report_tracks_class_and_prior_changes():
    valid = payload([0, 0, 1, 1], [0, 0, 1, 1], [0.8, 0.2])
    test = payload([0, 1, 1, 1], [1, 1, 1, 0], [0.7, 0.3])
    report = compare_payloads(valid, test, level="flow", top_k=3)
    assert report["summary"]["valid_accuracy"] == 1.0
    assert report["summary"]["test_accuracy"] == 0.5
    assert report["summary"]["true_prior_total_variation"] == 0.25
    assert report["per_class_shift"][0]["delta_f1"] < 0
    assert report["top_test_confusions"][0]["count"] == 1
    assert report["gate_diagnostics"]["test"]["dual_channel_gate"]["effective_mean"] == [0.7, 0.3]
    assert report["gate_diagnostics"]["test"]["dual_channel_gate"]["bounds_satisfied"] is True


def test_domain_shift_requires_matching_label_maps():
    valid = payload([0], [0], [1.0, 0.0])
    test = payload([0], [0], [1.0, 0.0])
    test["label_map"] = {"x": 0, "b": 1}
    with pytest.raises(ValueError, match="label maps differ"):
        compare_payloads(valid, test, level="flow", top_k=1)
