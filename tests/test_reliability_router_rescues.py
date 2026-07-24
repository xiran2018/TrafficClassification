import pytest

from analyze_reliability_router_rescues import analyze


def payload(flow_ids, labels, predictions):
    return {
        "flow_ids": flow_ids,
        "flow_y_true": labels,
        "flow_y_pred": predictions,
    }


def test_analyze_counts_router_rescues_and_harms_after_id_alignment():
    semantic = payload(["b", "a", "c"], [1, 0, 1], [0, 0, 1])
    structural = payload(["c", "b", "a"], [1, 1, 0], [0, 1, 1])
    router = payload(["a", "b", "c"], [0, 1, 1], [0, 1, 0])
    router["flow_structural_gate"] = [[0.1, 0.3], [0.8, 1.0], [0.4, 0.6]]

    report = analyze(semantic, structural, router)

    assert report["accuracy"] == pytest.approx(
        {"semantic": 2 / 3, "structural": 1 / 3, "router": 2 / 3}
    )
    assert report["conditions"]["router_rescue"] == pytest.approx(
        {"count": 1, "gate_mean": 0.9}
    )
    assert report["conditions"]["router_harm"] == pytest.approx(
        {"count": 1, "gate_mean": 0.5}
    )
    assert report["net_rescues"] == 0


def test_analyze_rejects_misaligned_labels():
    semantic = payload(["a"], [0], [0])
    structural = payload(["a"], [1], [1])
    router = payload(["a"], [0], [0])
    router["flow_structural_gate"] = [[0.5]]

    with pytest.raises(ValueError, match="labels do not align"):
        analyze(semantic, structural, router)
