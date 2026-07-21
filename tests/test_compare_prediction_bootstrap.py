import numpy as np
import pytest

from compare_prediction_bootstrap import align_predictions, paired_bootstrap


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
