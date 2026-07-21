import json
from pathlib import Path

import numpy as np
import pytest

from analyze_packet_crossfold_disagreement import build_report


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_fixture(tmp_path: Path):
    label_map = tmp_path / "label_map.json"
    write_json(label_map, {"a": 0, "b": 1})
    index = tmp_path / "packet_index.jsonl"
    rows = [
        {"packet_uid": f"p{i}", "label": "a" if label == 0 else "b", "label_id": label}
        for i, label in enumerate([0, 0, 1, 1])
    ]
    index.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    inputs = []
    sources = []
    predictions = {
        "fold0": [0, 1, 1, 0],
        "fold1": [0, 0, 0, 1],
        "fold2": [1, 0, 1, 1],
    }
    for name, predicted in predictions.items():
        probabilities = np.full((4, 2), 0.1, dtype=np.float32)
        probabilities[np.arange(4), predicted] = 0.9
        npz = tmp_path / f"{name}.npz"
        np.savez(npz, y_true=np.asarray([0, 0, 1, 1]), probabilities=probabilities)
        source = tmp_path / f"{name}.json"
        write_json(source, {"config": {"test_index": str(index.resolve())}})
        inputs.append((name, npz))
        sources.append((name, source))
    return inputs, sources, index, label_map


def test_report_verifies_alignment_and_complementarity(tmp_path: Path) -> None:
    inputs, sources, index, label_map = write_fixture(tmp_path)
    report = build_report(inputs, sources, index, label_map)

    assert report["alignment"]["status"].startswith("shared_source_index")
    assert report["alignment"]["num_packets"] == 4
    assert report["summary"]["oracle_any_fold_accuracy"] == 1.0
    assert report["summary"]["all_folds_wrong_rate"] == 0.0
    assert report["correct_fold_count"]["2"]["count"] == 4
    assert report["analysis_contract"]["selection_role"] == "none"


def test_report_rejects_source_bound_to_another_index(tmp_path: Path) -> None:
    inputs, sources, index, label_map = write_fixture(tmp_path)
    wrong_index = tmp_path / "wrong.jsonl"
    wrong_index.write_text(index.read_text(encoding="utf-8"), encoding="utf-8")
    write_json(sources[1][1], {"config": {"test_index": str(wrong_index.resolve())}})

    with pytest.raises(ValueError, match="does not match"):
        build_report(inputs, sources, index, label_map)


def test_report_rejects_npz_label_order_mismatch(tmp_path: Path) -> None:
    inputs, sources, index, label_map = write_fixture(tmp_path)
    with np.load(inputs[0][1]) as payload:
        probabilities = payload["probabilities"]
    np.savez(inputs[0][1], y_true=np.asarray([0, 1, 0, 1]), probabilities=probabilities)

    with pytest.raises(ValueError, match="labels do not match"):
        build_report(inputs, sources, index, label_map)


def test_report_accepts_evaluator_packet_index_binding(tmp_path: Path) -> None:
    inputs, sources, index, label_map = write_fixture(tmp_path)
    write_json(sources[0][1], {"packet_index": str(index.resolve())})

    report = build_report(inputs, sources, index, label_map)

    assert report["alignment"]["num_packets"] == 4


def test_report_verifies_npz_packet_uids(tmp_path: Path) -> None:
    inputs, sources, index, label_map = write_fixture(tmp_path)
    packet_uids = np.asarray(["p0", "p1", "p2", "p3"])
    for _, path in inputs:
        with np.load(path) as payload:
            y_true = payload["y_true"]
            probabilities = payload["probabilities"]
        np.savez(
            path,
            y_true=y_true,
            probabilities=probabilities,
            packet_uids=packet_uids,
        )

    report = build_report(inputs, sources, index, label_map)

    assert report["alignment"]["npz_packet_uids"] == "exact_packet_index_row_match"


def test_report_rejects_npz_packet_uid_order_mismatch(tmp_path: Path) -> None:
    inputs, sources, index, label_map = write_fixture(tmp_path)
    for input_index, (_, path) in enumerate(inputs):
        with np.load(path) as payload:
            y_true = payload["y_true"]
            probabilities = payload["probabilities"]
        packet_uids = np.asarray(["p0", "p1", "p2", "p3"])
        if input_index == 1:
            packet_uids = packet_uids[::-1]
        np.savez(
            path,
            y_true=y_true,
            probabilities=probabilities,
            packet_uids=packet_uids,
        )

    with pytest.raises(ValueError, match="packet_uids do not match"):
        build_report(inputs, sources, index, label_map)
