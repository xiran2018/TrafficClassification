import json

import numpy as np
import pytest

from bootstrap_classification_metrics import (
    cluster_bootstrap,
    load_fixed_predictions,
)


def test_packet_loader_requires_and_uses_flow_clusters(tmp_path):
    path = tmp_path / "packet.npz"
    np.savez_compressed(
        path,
        y_true=np.asarray([0, 0, 1, 1]),
        probabilities=np.asarray(
            [[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.7, 0.3]]
        ),
        flow_ids=np.asarray([10, 10, 20, 21]),
    )
    labels, predictions, groups = load_fixed_predictions(str(path), "packet")
    report = cluster_bootstrap(labels, predictions, groups, samples=50, seed=7)
    assert report["method"] == "class_stratified_flow_cluster_bootstrap"
    assert report["num_samples"] == 4
    assert report["num_flow_clusters"] == 3
    assert report["metrics"]["accuracy"]["point_estimate"] == 0.75


def test_flow_loader_treats_each_flow_as_a_cluster(tmp_path):
    path = tmp_path / "flow.json"
    path.write_text(
        json.dumps(
            {
                "flow_y_true": [0, 0, 1, 1],
                "flow_prob": [[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]],
                "flow_ids": ["a", "b", "c", "d"],
            }
        ),
        encoding="utf-8",
    )
    labels, predictions, groups = load_fixed_predictions(str(path), "flow")
    report = cluster_bootstrap(labels, predictions, groups, samples=20, seed=11)
    assert report["num_flow_clusters"] == 4
    assert report["metrics"]["macro_f1"]["point_estimate"] == 1.0


def test_cluster_bootstrap_rejects_flow_ids_spanning_classes():
    with pytest.raises(ValueError, match="spans labels"):
        cluster_bootstrap(
            np.asarray([0, 1]),
            np.asarray([0, 1]),
            np.asarray([3, 3]),
            samples=10,
            seed=1,
        )
