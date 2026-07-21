import json

import numpy as np
import pytest

from evaluate_session_novelty import (
    evaluate_session_novelty,
    load_predictions,
    session_signatures,
)
from train_tower1_multitask import stable_flow_id


def row(flow_id, label, src, sport, dst, dport):
    return {
        "flow_id": flow_id,
        "label_id": label,
        "meta": {
            "l4": "TCP",
            "src_ip": src,
            "sport": sport,
            "dst_ip": dst,
            "dport": dport,
        },
    }


def test_session_signature_is_direction_invariant():
    forward = row("a", 0, "10.0.0.1", 1234, "20.0.0.1", 443)
    reverse = row("a", 0, "20.0.0.1", 443, "10.0.0.1", 1234)

    assert session_signatures(forward) == session_signatures(reverse)


def test_packet_session_novelty_reports_seen_and_novel_groups():
    train = [row("train", 0, "10.0.0.1", 1000, "20.0.0.1", 443)]
    test = [
        row("seen", 0, "20.0.0.1", 443, "10.0.0.1", 1000),
        row("novel", 1, "30.0.0.1", 2000, "40.0.0.1", 8443),
    ]
    probabilities = np.asarray([[0.9, 0.1], [0.7, 0.3]])
    flow_ids = np.asarray([stable_flow_id("seen"), stable_flow_id("novel")])

    report = evaluate_session_novelty(
        train,
        test,
        np.asarray([0, 1]),
        probabilities,
        flow_ids,
        "packet",
        ["zero", "one"],
    )

    assert report["groups"]["session_seen"]["num_samples"] == 1
    assert report["groups"]["session_seen"]["metrics"]["accuracy"] == 1.0
    assert report["groups"]["session_novel"]["metrics"]["accuracy"] == 0.0
    assert report["group_definition_uses_test_labels"] is False
    assert report["conditional_gaps_are_not_causal_effects"] is True
    assert report["seen_minus_novel_gaps"]["session"][
        "accuracy_seen_minus_novel"
    ] == 1.0


def test_flow_session_novelty_requires_one_tuple_per_flow():
    train = [row("train", 0, "10.0.0.1", 1000, "20.0.0.1", 443)]
    test = [
        row("flow", 0, "10.0.0.1", 1000, "20.0.0.1", 443),
        row("flow", 0, "10.0.0.2", 1001, "20.0.0.2", 443),
    ]

    with pytest.raises(ValueError, match="multiple session tuples"):
        evaluate_session_novelty(
            train,
            test,
            np.asarray([0]),
            np.asarray([[1.0, 0.0]]),
            np.asarray(["flow"]),
            "flow",
            ["zero", "one"],
        )


def test_flow_prediction_loader_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "flow.json"
    path.write_text(
        json.dumps(
            {
                "flow_y_true": [0, 1],
                "flow_prob": [[0.8, 0.2], [0.1, 0.9]],
                "flow_ids": ["same", "same"],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate flow_ids"):
        load_predictions(path, "flow")
