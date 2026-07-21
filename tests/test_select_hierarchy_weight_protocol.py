import pytest

from select_hierarchy_weight_protocol import select_eta


def test_selects_best_eligible_eta_with_accuracy_guard():
    selected = select_eta(
        {
            0.0: {"accuracy": 0.80, "macro_f1": 0.70},
            0.25: {"accuracy": 0.798, "macro_f1": 0.71},
            0.5: {"accuracy": 0.79, "macro_f1": 0.73},
            1.0: {"accuracy": 0.801, "macro_f1": 0.708},
        },
        min_delta=0.005,
        max_accuracy_drop=0.005,
    )

    assert selected["selected_eta"] == 0.25
    assert selected["arms"]["0.5"]["eligible"] is False


def test_uses_smaller_eta_as_final_exact_tie_break():
    selected = select_eta(
        {
            0.0: {"accuracy": 0.80, "macro_f1": 0.70},
            0.25: {"accuracy": 0.81, "macro_f1": 0.72},
            0.5: {"accuracy": 0.81, "macro_f1": 0.72},
        },
        min_delta=0.005,
        max_accuracy_drop=0.005,
    )

    assert selected["selected_eta"] == 0.25


def test_requires_eta_zero_reference():
    with pytest.raises(ValueError, match="eta=0 reference"):
        select_eta(
            {0.5: {"accuracy": 0.8, "macro_f1": 0.7}},
            min_delta=0.005,
            max_accuracy_drop=0.005,
        )
