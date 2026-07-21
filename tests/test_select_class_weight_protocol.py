import pytest

from select_class_weight_protocol import choose_protocol


def candidate(passes, minimum_f1, mean_f1, minimum_accuracy):
    return {
        "passes_all_datasets": passes,
        "ranking_key": {
            "minimum_macro_f1_delta": minimum_f1,
            "mean_macro_f1_delta": mean_f1,
            "minimum_accuracy_delta": minimum_accuracy,
        },
    }


def test_selects_full_when_both_pass_and_full_has_stronger_worst_dataset():
    selected, comparison, ranked = choose_protocol(
        {
            "flow_sqrt": candidate(True, 0.006, 0.010, -0.001),
            "flow_full": candidate(True, 0.012, 0.014, -0.003),
        }
    )
    assert selected == "flow_full"
    assert comparison == "flow_full"
    assert ranked == ["flow_full", "flow_sqrt"]


def test_selects_only_cross_dataset_eligible_arm():
    selected, comparison, _ = choose_protocol(
        {
            "flow_sqrt": candidate(True, 0.006, 0.007, 0.0),
            "flow_full": candidate(False, 0.020, 0.030, 0.01),
        }
    )
    assert selected == "flow_sqrt"
    assert comparison == "flow_full"


def test_falls_back_to_packet_control_when_no_flow_arm_passes():
    selected, comparison, ranked = choose_protocol(
        {
            "flow_sqrt": candidate(False, -0.002, 0.003, -0.001),
            "flow_full": candidate(False, -0.010, 0.020, 0.005),
        }
    )
    assert selected == "packet_full"
    assert comparison == "flow_sqrt"
    assert ranked[0] == "flow_sqrt"


def test_rejects_unregistered_arm_set():
    with pytest.raises(ValueError, match="expected candidate arms"):
        choose_protocol({"flow_sqrt": candidate(True, 0.1, 0.1, 0.1)})
