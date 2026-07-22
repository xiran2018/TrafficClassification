import json

import numpy as np
import pytest

from analyze_counterfactual_gate_alignment import (
    _bootstrap_pearson,
    _file_evidence,
    _load_flow,
    _load_packet,
    analyze,
)


def test_prediction_input_evidence_binds_content(tmp_path):
    path = tmp_path / "prediction.bin"
    path.write_bytes(b"counterfactual-prediction")

    evidence = _file_evidence(path)

    assert evidence["path"] == str(path.resolve())
    assert evidence["sha256"] == (
        "06a33732c8cc54345e574ef0e3edfa3d62e40d73a4d2c262a6fa57026d3bbef3"
    )
    assert evidence["size_bytes"] == 25


def test_packet_gate_alignment_detects_positive_counterfactual_association(tmp_path):
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    ids = np.asarray(["p0", "p1", "p2", "p3"])
    flow_ids = np.asarray(["f0", "f0", "f1", "f1"])
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
            "flow_ids": flow_ids,
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
        "groups": np.asarray(["flow-a", "flow-b"]),
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


def test_tiny_positive_association_does_not_pass_effect_size_gate():
    count = 200
    labels = np.zeros(count, dtype=np.int64)
    advantage = np.linspace(-1.0, 1.0, count)
    factual_true_probability = np.full(count, 0.5)
    intervened_true_probability = factual_true_probability * np.exp(-advantage)
    intervened_true_probability = np.clip(intervened_true_probability, 0.01, 0.99)
    factual = np.c_[factual_true_probability, 1.0 - factual_true_probability]
    intervened = np.c_[intervened_true_probability, 1.0 - intervened_true_probability]
    weak_preference = 0.005 * advantage
    gate = np.c_[0.5 + weak_preference, 0.5 - weak_preference]
    base = {
        "ids": np.asarray([f"p-{index}" for index in range(count)]),
        "groups": np.asarray([f"flow-{index}" for index in range(count)]),
        "y_true": labels,
    }
    full = {**base, "probabilities": factual, "gate": gate}

    report = analyze(
        full,
        {**base, "probabilities": factual},
        {**base, "probabilities": intervened},
        bootstrap_samples=0,
        seed=0,
    )

    assert report["association"]["pearson"] > 0
    assert report["association"]["top_minus_bottom"] < 0.02
    assert report["status"] == "not_demonstrated"


def test_packet_bootstrap_resamples_flow_clusters():
    report = _bootstrap_pearson(
        np.asarray([-0.4, -0.3, 0.3, 0.4]),
        np.asarray([-2.0, -1.0, 1.0, 2.0]),
        np.asarray(["flow-a", "flow-a", "flow-b", "flow-b"]),
        samples=20,
        seed=11,
    )

    assert report["resampling_unit"] == "flow_cluster"
    assert report["num_clusters"] == 2
    assert report["positive_fraction"] == 1.0
