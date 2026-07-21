import json

import pytest

from analyze_flow_length_performance import build_report


def write_inputs(tmp_path):
    predictions = tmp_path / "predictions.json"
    predictions.write_text(
        json.dumps(
            {
                "flow_ids": ["a", "b", "c", "d"],
                "flow_y_true": [0, 0, 1, 1],
                "flow_prob": [[0.9, 0.1], [0.2, 0.8], [0.1, 0.9], [0.8, 0.2]],
            }
        ),
        encoding="utf-8",
    )
    index = tmp_path / "flow_embedding_index.jsonl"
    rows = [
        {"flow_id": "a", "packet_metas": [{}] * 4},
        {"flow_id": "b", "packet_metas": [{}] * 12},
        {"flow_id": "c", "packet_metas": [{}] * 24},
        {"flow_id": "d", "packet_metas": [{}] * 40},
    ]
    index.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return predictions, index


def test_flow_length_report_aligns_ids_and_reports_strata(tmp_path):
    predictions, index = write_inputs(tmp_path)
    report = build_report(predictions, index)
    assert report["status"] == "reporting_only"
    assert report["num_flows"] == 4
    assert report["packet_count_summary"]["median"] == 18.0
    assert [row["num_flows"] for row in report["fixed_packet_count_strata"]] == [1, 1, 1, 1, 0]
    assert report["fixed_packet_count_strata"][0]["accuracy"] == 1.0
    assert report["fixed_packet_count_strata"][3]["accuracy"] == 0.0
    assert report["associations"]["spearman_packet_count_correctness"] < 0
    conditional = report["associations"]["within_class_correctness"]
    assert conditional["num_eligible_classes"] == 2
    assert conditional["fraction_negative"] == 1.0


def test_flow_length_report_rejects_missing_or_duplicate_flow_ids(tmp_path):
    predictions, index = write_inputs(tmp_path)
    rows = index.read_text(encoding="utf-8").splitlines()
    index.write_text("\n".join(rows[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="misses 1"):
        build_report(predictions, index)

    index.write_text(rows[0] + "\n" + rows[0] + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate flow_id"):
        build_report(predictions, index)
