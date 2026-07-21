import json
import math

import pytest

from analyze_class_sampling_hierarchy import analyze


def write_inputs(tmp_path):
    label_map = tmp_path / "label_map.json"
    label_map.write_text(json.dumps({"a": 0, "b": 1}), encoding="utf-8")
    rows = [
        {"flow_id": "a1", "label_id": 0},
        {"flow_id": "a1", "label_id": 0},
        {"flow_id": "a2", "label_id": 0},
        {"flow_id": "b1", "label_id": 1},
        {"flow_id": "b1", "label_id": 1},
        {"flow_id": "b1", "label_id": 1},
        {"flow_id": "b1", "label_id": 1},
    ]
    packets = tmp_path / "packets.jsonl"
    packets.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    return packets, label_map


def test_reports_packet_flow_prior_divergence_from_train_only_data(tmp_path):
    packets, label_map = write_inputs(tmp_path)
    report = analyze(packets, label_map)
    summary = report["summary"]
    assert report["selection_role"] == "train_only_reporting_not_model_selection"
    assert report["test_labels_used"] is False
    assert summary["num_packets"] == 7
    assert summary["num_flows"] == 3
    assert summary["packet_flow_prior_total_variation"] > 0
    classes = {row["label"]: row for row in report["classes"]}
    assert classes["a"]["flow_count"] == 2
    assert classes["b"]["flow_count"] == 1
    assert math.isclose(classes["a"]["packets_per_flow"], 1.5)
    assert classes["b"]["packets_per_flow"] == 4.0
    assert classes["b"]["flow_to_packet_weight_ratio"] > 1.0


def test_rejects_conflicting_flow_labels(tmp_path):
    packets, label_map = write_inputs(tmp_path)
    packets.write_text(
        json.dumps({"flow_id": "same", "label_id": 0})
        + "\n"
        + json.dumps({"flow_id": "same", "label_id": 1})
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="conflicting labels"):
        analyze(packets, label_map)


def test_marks_correlation_undefined_when_packet_classes_are_balanced(tmp_path):
    label_map = tmp_path / "label_map.json"
    label_map.write_text(json.dumps({"a": 0, "b": 1}), encoding="utf-8")
    rows = [
        {"flow_id": "a1", "label_id": 0},
        {"flow_id": "a1", "label_id": 0},
        {"flow_id": "a2", "label_id": 0},
        {"flow_id": "b1", "label_id": 1},
        {"flow_id": "b1", "label_id": 1},
        {"flow_id": "b1", "label_id": 1},
    ]
    packets = tmp_path / "balanced_packets.jsonl"
    packets.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    summary = analyze(packets, label_map)["summary"]

    assert summary["minimum_packet_class_count"] == 3
    assert summary["maximum_packet_class_count"] == 3
    assert summary["minimum_flow_class_count"] == 1
    assert summary["maximum_flow_class_count"] == 2
    assert summary["log_count_pearson"] is None
    assert summary["effective_weight_log_pearson"] is None
    assert summary["packet_flow_prior_total_variation"] > 0


def test_rejects_class_without_training_flow(tmp_path):
    packets, label_map = write_inputs(tmp_path)
    packets.write_text(
        json.dumps({"flow_id": "a", "label_id": 0}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="classes without training flows"):
        analyze(packets, label_map)
