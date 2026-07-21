import json
import sys
from types import SimpleNamespace

import pytest
import torch

from build_consensus_distill_targets import main
from test_tower2 import prediction_provenance
from train_tower2 import tower2_training_input_evidence
from write_oof_teacher_evidence import verify_oof_teacher


def write_prediction(path, flow_ids, labels, probabilities):
    path.write_text(
        json.dumps(
            {
                "valid_flow_ids": flow_ids,
                "valid_y_true": labels,
                "valid_prob": probabilities,
                "flow_ids": flow_ids,
                "flow_y_true": labels,
                "flow_prob": probabilities,
            }
        ),
        encoding="utf-8",
    )


def write_oof_evidence(path, prediction_path, flow_ids):
    train_dataset = path.with_name(path.stem + "_train.pt")
    evaluation_dataset = path.with_name(path.stem + "_valid.pt")
    checkpoint = path.with_name(path.stem + "_best.pt")
    torch.save([{"flow_id": f"train-{path.stem}"}], train_dataset)
    torch.save([{"flow_id": flow_id} for flow_id in flow_ids], evaluation_dataset)
    args = SimpleNamespace(
        dataset=str(train_dataset),
        valid_dataset=str(evaluation_dataset),
        paired_view_dataset="",
        paired_valid_dataset="",
        distill_targets_json="",
    )
    training_evidence = tower2_training_input_evidence(args)
    torch.save({"training_input_evidence": training_evidence}, checkpoint)
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    prediction["provenance"] = prediction_provenance(
        str(checkpoint),
        str(evaluation_dataset),
        "",
        {"training_input_evidence": training_evidence},
    )
    prediction_path.write_text(json.dumps(prediction), encoding="utf-8")
    evidence = verify_oof_teacher(
        prediction_path, checkpoint, train_dataset, evaluation_dataset
    )
    path.write_text(json.dumps(evidence), encoding="utf-8")


def run_builder(
    monkeypatch,
    first,
    second,
    output,
    min_teachers,
    evidence=None,
    require_oof=False,
):
    argv = [
        "build_consensus_distill_targets.py",
        "--input",
        "fold0",
        str(first),
        "--input",
        "fold1",
        str(second),
        "--split",
        "heldout" if require_oof else "valid",
        "--align",
        "union",
        "--mode",
        "mean",
        "--min_teachers_per_flow",
        str(min_teachers),
        "--output_json",
        str(output),
    ]
    for name, path in evidence or []:
        argv.extend(["--oof_evidence", name, str(path)])
    if require_oof:
        argv.append("--require_oof_exclusion_proof")
    monkeypatch.setattr(
        sys,
        "argv",
        argv,
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


def test_builder_proves_only_fully_bound_oof_consensus(monkeypatch, tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first_evidence = tmp_path / "first_oof.json"
    second_evidence = tmp_path / "second_oof.json"
    output = tmp_path / "teacher.json"
    write_prediction(first, ["a", "b"], [0, 1], [[0.9, 0.1], [0.2, 0.8]])
    write_prediction(second, ["a", "b"], [0, 1], [[0.8, 0.2], [0.3, 0.7]])
    write_oof_evidence(first_evidence, first, ["a", "b"])
    write_oof_evidence(second_evidence, second, ["a", "b"])

    run_builder(
        monkeypatch,
        first,
        second,
        output,
        min_teachers=2,
        evidence=[("fold0", first_evidence), ("fold1", second_evidence)],
        require_oof=True,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    contract = payload["teacher_multiplicity"]
    assert contract["teacher_counts"] == [2, 2]
    assert contract["oof_teacher_counts"] == [2, 2]
    assert contract["oof_exclusion_proven"] is True
    assert contract["oof_multi_teacher_consensus_proven"] is True


def test_builder_rejects_partially_proven_oof_consensus(monkeypatch, tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first_evidence = tmp_path / "first_oof.json"
    output = tmp_path / "teacher.json"
    write_prediction(first, ["a"], [0], [[0.9, 0.1]])
    write_prediction(second, ["a"], [0], [[0.8, 0.2]])
    write_oof_evidence(first_evidence, first, ["a"])

    with pytest.raises(ValueError, match="Not every contributing teacher"):
        run_builder(
            monkeypatch,
            first,
            second,
            output,
            min_teachers=2,
            evidence=[("fold0", first_evidence)],
            require_oof=True,
        )
