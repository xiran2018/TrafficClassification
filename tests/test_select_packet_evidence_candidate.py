import json
from pathlib import Path

import pytest

from select_packet_evidence_candidate import select_candidate


def write_json(path: Path, payload: dict) -> str:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def prediction(f1: float, accuracy: float) -> dict:
    return {
        "metrics": {"flow_level": {"macro_f1": f1, "accuracy": accuracy}},
        "flow_ids": ["a", "b"],
        "flow_y_true": [0, 1],
    }


def manifest(dataset: str, *, eval_splits="valid", bound=0.4, control=False) -> dict:
    return {
        "dataset": dataset,
        "eval_splits": eval_splits,
        "packet_evidence_max_weight": bound,
        "packet_evidence_ablation_control": control,
        "flow_pooling": "late_fusion",
        "exact_shared_packet_encoder": True,
        "seed": 42,
        "tower2_epochs": 30,
        "tower2_batch_size": 16,
        "tower2_select_metric": "content_group_macro_f1",
        "hidden_dim": 256,
        "num_layers": 2,
        "num_heads": 4,
        "dropout": 0.15,
        "tower2_lr": 1e-4,
        "weight_decay": 0.03,
        "window_size": 32,
        "stride": 16,
        "window_loss_weight": 0.3,
        "class_weight_strength": 0.6,
        "label_smoothing": 0.05,
        "flow_contrastive_weight": 0.03,
        "flow_temperature": 0.07,
        "content_group_loss_reduction": "group_mean",
        "content_group_unique_batches": True,
        "split_group_key": "content_group_id",
        "shared_packet_hidden_dim": 128,
        "native_structural_suffix": "shared_content_fold0",
        "shared_core_config_sha256": "a" * 64,
        "framework": {
            "notes": {
                "packet_module_training_source": "flow_task_train_split_packets",
                "cross_task_trained_weights_reused": False,
                "packet_evidence_training_label_source": (
                    "disabled"
                    if control
                    else "flow_train_split_labels_broadcast_to_member_packets"
                ),
                "result_paths": [f"reasoningDataset/{dataset}/valid_seq_metrics.json"],
                "tower2_data_suffix": "primary_fold0",
                "paired_embedding_suffix": "intervention_fold0",
            }
        },
    }


def records(tmp_path: Path, tls_gain: float = 0.01):
    output = []
    for dataset, gain in (("vpn-app", 0.02), ("tls-120", tls_gain)):
        prefix = dataset.replace("-", "_")
        output.append(
            (
                dataset,
                write_json(tmp_path / f"{prefix}_reference.json", prediction(0.70, 0.75)),
                write_json(tmp_path / f"{prefix}_control.json", prediction(0.70, 0.75)),
                write_json(
                    tmp_path / f"{prefix}_candidate.json",
                    prediction(0.70 + gain, 0.755),
                ),
                write_json(
                    tmp_path / f"{prefix}_control_manifest.json",
                    manifest(dataset, bound=0.0, control=True),
                ),
                write_json(
                    tmp_path / f"{prefix}_candidate_manifest.json",
                    manifest(dataset),
                ),
            )
        )
    return output


def test_packet_evidence_candidate_requires_both_datasets_to_improve(tmp_path):
    report = select_candidate(
        records(tmp_path),
        min_macro_f1_gain=0.005,
        max_accuracy_drop=0.01,
        max_reference_macro_f1_drop=0.01,
    )

    assert report["selected"] == "candidate"
    assert report["test_labels_used"] is False
    assert all(row["passed"] for row in report["datasets"].values())


def test_packet_evidence_candidate_falls_back_when_tls_does_not_improve(tmp_path):
    report = select_candidate(
        records(tmp_path, tls_gain=0.001),
        min_macro_f1_gain=0.005,
        max_accuracy_drop=0.01,
        max_reference_macro_f1_drop=0.01,
    )

    assert report["selected"] == "baseline"
    assert report["datasets"]["tls-120"]["passed"] is False


def test_packet_evidence_candidate_rejects_test_evaluation(tmp_path):
    candidate_records = records(tmp_path)
    manifest_path = Path(candidate_records[0][5])
    payload = json.loads(manifest_path.read_text())
    payload["eval_splits"] = "valid,test"
    manifest_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="validation only"):
        select_candidate(
            candidate_records,
            min_macro_f1_gain=0.005,
            max_accuracy_drop=0.01,
            max_reference_macro_f1_drop=0.01,
        )


def test_packet_evidence_candidate_rejects_missing_matched_control(tmp_path):
    candidate_records = records(tmp_path)
    manifest_path = Path(candidate_records[0][4])
    payload = json.loads(manifest_path.read_text())
    payload["packet_evidence_ablation_control"] = False
    manifest_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="evidence-free control"):
        select_candidate(
            candidate_records,
            min_macro_f1_gain=0.005,
            max_accuracy_drop=0.01,
            max_reference_macro_f1_drop=0.01,
        )


def test_packet_evidence_candidate_rejects_tower2_hyperparameter_mismatch(tmp_path):
    candidate_records = records(tmp_path)
    manifest_path = Path(candidate_records[0][5])
    payload = json.loads(manifest_path.read_text())
    payload["seed"] = 43
    manifest_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="Tower2 contracts differ"):
        select_candidate(
            candidate_records,
            min_macro_f1_gain=0.005,
            max_accuracy_drop=0.01,
            max_reference_macro_f1_drop=0.01,
        )
