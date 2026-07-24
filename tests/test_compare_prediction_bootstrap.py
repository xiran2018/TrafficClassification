import numpy as np
import pytest

from compare_prediction_bootstrap import (
    align_packet_predictions,
    align_predictions,
    paired_bootstrap,
)


def payload(ids, labels, probs):
    return {"flow_ids": ids, "flow_y_true": labels, "flow_prob": probs}


def test_alignment_is_by_flow_id_and_rejects_label_mismatch():
    baseline = payload(["b", "a"], [1, 0], [[0.1, 0.9], [0.8, 0.2]])
    candidate = payload(["a", "b"], [0, 1], [[0.9, 0.1], [0.2, 0.8]])
    y, base, other, ids = align_predictions(baseline, candidate)
    assert list(ids) == ["a", "b"]
    assert y.tolist() == [0, 1]
    candidate["flow_y_true"][0] = 1
    with pytest.raises(ValueError, match="labels differ"):
        align_predictions(baseline, candidate)


def test_paired_bootstrap_reports_clear_candidate_gain():
    y = np.asarray([0, 0, 1, 1])
    base = np.asarray([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.1, 0.9]])
    candidate = np.asarray([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]])
    report = paired_bootstrap(y, base, candidate, samples=100, seed=3)
    assert report["delta"]["accuracy"]["delta"] == 0.5
    assert report["mcnemar"]["candidate_only_correct"] == 2
    assert report["mcnemar"]["baseline_only_correct"] == 0


def test_packet_alignment_uses_uid_and_preserves_flow_clusters():
    baseline = {
        "packet_uids": np.asarray(["a", "b", "c"]),
        "flow_ids": np.asarray([10, 10, 20]),
        "y_true": np.asarray([0, 0, 1]),
        "probabilities": np.asarray([[0.8, 0.2], [0.7, 0.3], [0.4, 0.6]]),
    }
    candidate = {
        "packet_uids": np.asarray(["c", "a", "b"]),
        "flow_ids": np.asarray([20, 10, 10]),
        "y_true": np.asarray([1, 0, 0]),
        "probabilities": np.asarray([[0.1, 0.9], [0.9, 0.1], [0.6, 0.4]]),
    }
    y, base, other, groups = align_packet_predictions(baseline, candidate)
    assert y.tolist() == [0, 0, 1]
    assert groups.tolist() == ["10", "10", "20"]
    assert other.argmax(axis=1).tolist() == [0, 0, 1]
    assert base.shape == other.shape


def test_cluster_bootstrap_rejects_flows_that_span_classes():
    y = np.asarray([0, 1, 1])
    prob = np.asarray([[0.9, 0.1], [0.2, 0.8], [0.1, 0.9]])
    with pytest.raises(ValueError, match="spans labels"):
        paired_bootstrap(
            y,
            prob,
            prob,
            samples=10,
            group_ids=np.asarray(["same", "same", "other"]),
        )


def test_cluster_bootstrap_reports_independent_flow_count():
    y = np.asarray([0, 0, 0, 1, 1])
    base = np.asarray(
        [[0.9, 0.1], [0.9, 0.1], [0.2, 0.8], [0.8, 0.2], [0.1, 0.9]]
    )
    candidate = np.asarray(
        [[0.9, 0.1], [0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]]
    )
    report = paired_bootstrap(
        y,
        base,
        candidate,
        samples=50,
        seed=7,
        group_ids=np.asarray(["a", "a", "b", "c", "d"]),
    )
    assert report["resampling_unit"] == "flow_cluster"
    assert report["num_rows"] == 5
    assert report["num_flow_clusters"] == 4
    assert report["clusters_per_class"] == {"0": 2, "1": 2}
