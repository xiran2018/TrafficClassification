import json
from types import SimpleNamespace

import pytest
import torch

from test_tower2 import prediction_provenance
from train_tower2 import file_evidence, tower2_training_input_evidence
from write_oof_teacher_evidence import validate_oof_evidence, verify_oof_teacher


def write_dataset(path, rows):
    torch.save(rows, path)


def build_bound_prediction(tmp_path, train_rows, evaluation_rows):
    train_dataset = tmp_path / "train.pt"
    evaluation_dataset = tmp_path / "valid.pt"
    checkpoint_path = tmp_path / "best.pt"
    prediction_path = tmp_path / "valid_predictions.json"
    write_dataset(train_dataset, train_rows)
    write_dataset(evaluation_dataset, evaluation_rows)
    args = SimpleNamespace(
        dataset=str(train_dataset),
        valid_dataset=str(evaluation_dataset),
        paired_view_dataset="",
        paired_valid_dataset="",
        distill_targets_json="",
    )
    training_evidence = tower2_training_input_evidence(args)
    torch.save({"training_input_evidence": training_evidence}, checkpoint_path)
    provenance = prediction_provenance(
        str(checkpoint_path), str(evaluation_dataset), "", {"training_input_evidence": training_evidence}
    )
    prediction_path.write_text(
        json.dumps(
            {
                "flow_ids": [str(row["flow_id"]) for row in evaluation_rows],
                "flow_prob": [[0.8, 0.2] for _ in evaluation_rows],
                "provenance": provenance,
            }
        ),
        encoding="utf-8",
    )
    return prediction_path, checkpoint_path, train_dataset, evaluation_dataset


def test_oof_evidence_proves_checkpoint_bound_disjoint_flows(tmp_path):
    paths = build_bound_prediction(
        tmp_path,
        [{"flow_id": "train-a"}, {"flow_id": "train-b"}],
        [{"flow_id": "valid-a"}, {"flow_id": "valid-b"}],
    )

    evidence = verify_oof_teacher(*paths)

    assert evidence["oof_exclusion_proven"] is True
    assert evidence["disjointness"]["overlap_count"] == 0
    assert evidence["bindings"]["checkpoint_binds_training_dataset"] is True


def test_oof_evidence_rejects_train_evaluation_overlap(tmp_path):
    paths = build_bound_prediction(
        tmp_path,
        [{"flow_id": "shared"}],
        [{"flow_id": "shared"}],
    )

    with pytest.raises(ValueError, match="flow overlap"):
        verify_oof_teacher(*paths)


def test_oof_evidence_rejects_unbound_checkpoint(tmp_path):
    paths = list(
        build_bound_prediction(
            tmp_path,
            [{"flow_id": "train"}],
            [{"flow_id": "valid"}],
        )
    )
    replacement_train = tmp_path / "replacement.pt"
    write_dataset(replacement_train, [{"flow_id": "different"}])
    paths[2] = replacement_train

    with pytest.raises(ValueError, match="does not bind"):
        verify_oof_teacher(*paths)


def test_file_evidence_hash_is_content_bound(tmp_path):
    path = tmp_path / "value.bin"
    path.write_bytes(b"first")
    first = file_evidence(path)
    path.write_bytes(b"second")
    second = file_evidence(path)

    assert first["sha256"] != second["sha256"]


def test_oof_evidence_is_rechecked_after_checkpoint_mutation(tmp_path):
    paths = build_bound_prediction(
        tmp_path,
        [{"flow_id": "train"}],
        [{"flow_id": "valid"}],
    )
    prediction, checkpoint, _, _ = paths
    evidence_path = tmp_path / "oof_evidence.json"
    evidence_path.write_text(
        json.dumps(verify_oof_teacher(*paths)), encoding="utf-8"
    )

    validate_oof_evidence(evidence_path, prediction, ["valid"])
    torch.save({"replacement": True}, checkpoint)

    with pytest.raises(ValueError, match="checkpoint changed"):
        validate_oof_evidence(evidence_path, prediction, ["valid"])
