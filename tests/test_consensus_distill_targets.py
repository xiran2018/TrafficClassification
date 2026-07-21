import json
import sys

import pytest

from build_consensus_distill_targets import main


def write_prediction(path, flow_ids, labels, probabilities):
    path.write_text(
        json.dumps(
            {
                "valid_flow_ids": flow_ids,
                "valid_y_true": labels,
                "valid_prob": probabilities,
            }
        ),
        encoding="utf-8",
    )


def run_builder(monkeypatch, first, second, output, min_teachers):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_consensus_distill_targets.py",
            "--input",
            "fold0",
            str(first),
            "--input",
            "fold1",
            str(second),
            "--split",
            "valid",
            "--align",
            "union",
            "--mode",
            "mean",
            "--min_teachers_per_flow",
            str(min_teachers),
            "--output_json",
            str(output),
        ],
    )
    main()


def test_union_filters_to_true_multi_teacher_flows(monkeypatch, tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    output = tmp_path / "teacher.json"
    write_prediction(first, ["a", "b"], [0, 1], [[0.9, 0.1], [0.2, 0.8]])
    write_prediction(second, ["b", "c"], [1, 0], [[0.4, 0.6], [0.7, 0.3]])

    run_builder(monkeypatch, first, second, output, min_teachers=2)

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["flow_ids"] == ["b"]
    assert payload["flow_y_true"] == [1]
    assert payload["teacher_multiplicity"]["flow_ids"] == ["b"]
    assert payload["teacher_multiplicity"]["teacher_counts"] == [2]
    assert payload["teacher_multiplicity"]["all_output_flows_multi_teacher"] is True
    assert payload["teacher_multiplicity"]["oof_exclusion_proven"] is False
    assert payload["config"]["min_teachers_per_flow"] == 2


def test_union_rejects_empty_teacher_count_filter(monkeypatch, tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    output = tmp_path / "teacher.json"
    write_prediction(first, ["a"], [0], [[0.9, 0.1]])
    write_prediction(second, ["b"], [1], [[0.2, 0.8]])

    with pytest.raises(ValueError, match="No teacher targets remain"):
        run_builder(monkeypatch, first, second, output, min_teachers=2)
