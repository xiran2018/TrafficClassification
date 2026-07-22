import json

import numpy as np
import pytest

from analyze_counterfactual_gate_alignment import (
    _load_flow,
    _load_packet,
    analyze,
)


def test_packet_gate_alignment_detects_positive_counterfactual_association(tmp_path):
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    ids = np.asarray(["p0", "p1", "p2", "p3"])
    factual_prob = np.asarray(
        [[0.95, 0.05], [0.80, 0.20], [0.55, 0.45], [0.20, 0.80]]
    )
    intervened_prob = np.asarray(
        [[0.60, 0.40], [0.70, 0.30], [0.10, 0.90], [0.05, 0.95]]
    )
    gate = np.asarray(
        [[0.70, 0.30], [0.60, 0.40], [0.35, 0.65], [0.30, 0.70]]
    )

    paths = {}
    for name, probabilities in {
        "full": factual_prob,
        "factual": factual_prob,
        "intervened": intervened_prob,
    }.items():
        path = tmp_path / f"{name}.npz"
        values = {
            "y_true": labels,
            "probabilities": probabilities,
            "packet_uids": ids,
        }
        if name == "full":
            values["effective_intervention_view_gate"] = gate
        np.savez_compressed(path, **values)
        paths[name] = path

    report = analyze(
        _load_packet(str(paths["full"])),
        _load_packet(str(paths["factual"])),
        _load_packet(str(paths["intervened"])),
        bootstrap_samples=0,
        seed=7,
    )

    assert report["status"] == "positive_association"
    assert report["association"]["pearson"] > 0
    assert report["association"]["spearman"] > 0
    assert report["association"]["top_minus_bottom"] > 0


def test_flow_loader_aggregates_window_gates_in_flow_order(tmp_path):
    payload = {
        "flow_y_true": [0, 1],
        "flow_prob": [[0.8, 0.2], [0.1, 0.9]],
        "flow_ids": ["flow-b", "flow-a"],
        "window_flow_ids": ["flow-a", "flow-b", "flow-a"],
        "window_effective_gate_values": {
            "intervention_view_gate": [
                [0.2, 0.8],
                [0.7, 0.3],
                [0.4, 0.6],
            ]
        },
    }
    path = tmp_path / "flow.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = _load_flow(str(path))

    np.testing.assert_allclose(loaded["gate"], [[0.7, 0.3], [0.3, 0.7]])


def test_gate_alignment_rejects_sample_identity_mismatch():
    reference = {
        "ids": np.asarray(["a", "b"]),
        "y_true": np.asarray([0, 1]),
        "probabilities": np.asarray([[0.9, 0.1], [0.1, 0.9]]),
        "gate": np.asarray([[0.6, 0.4], [0.4, 0.6]]),
    }
    candidate = dict(reference)
    candidate["ids"] = np.asarray(["b", "a"])

    with pytest.raises(ValueError, match="sample identities"):
        analyze(
            reference,
            candidate,
            reference,
            bootstrap_samples=0,
            seed=0,
        )
