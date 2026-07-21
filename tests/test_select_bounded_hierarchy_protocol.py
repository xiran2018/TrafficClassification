import pytest

from select_bounded_hierarchy_protocol import noninferiority_decision


def test_promotes_only_when_every_dataset_is_noninferior():
    reference = {
        "vpn-app": {"accuracy": 0.80, "macro_f1": 0.75},
        "tls-120": {"accuracy": 0.85, "macro_f1": 0.83},
    }
    candidate = {
        "vpn-app": {"accuracy": 0.799, "macro_f1": 0.748},
        "tls-120": {"accuracy": 0.86, "macro_f1": 0.84},
    }

    result = noninferiority_decision(reference, candidate)

    assert result["selected"] == "candidate"
    assert result["candidate_promoted_for_all_datasets"] is True


@pytest.mark.parametrize("metric", ["accuracy", "macro_f1"])
def test_rejects_when_either_metric_exceeds_drop_on_one_dataset(metric):
    reference = {
        "vpn-app": {"accuracy": 0.80, "macro_f1": 0.75},
        "tls-120": {"accuracy": 0.85, "macro_f1": 0.83},
    }
    candidate = {dataset: dict(metrics) for dataset, metrics in reference.items()}
    candidate["tls-120"][metric] -= 0.0031

    result = noninferiority_decision(reference, candidate)

    assert result["selected"] == "reference"
    assert result["datasets"]["tls-120"][f"{metric}_passes"] is False


def test_requires_matching_dataset_sets():
    with pytest.raises(ValueError, match="datasets must match"):
        noninferiority_decision(
            {"vpn-app": {"accuracy": 1.0, "macro_f1": 1.0}},
            {"tls-120": {"accuracy": 1.0, "macro_f1": 1.0}},
        )
