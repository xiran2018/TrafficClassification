import json
from pathlib import Path

import pytest

from evaluate_packet_scope_validation import evaluate
from freeze_shared_core_v2_config import canonical_sha256, file_sha256


DATASETS = ("vpn-binary", "vpn-service", "ustc-app", "ustc-binary")


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def config(tmp_path):
    payload = {
        "task_datasets": {"packet-level-classification": list(DATASETS)},
        "method_selection": {
            "decision_status": "final_after_preregistered_validation"
        },
        "selection_protocol": {"test_evaluation_allowed": True},
    }
    payload["config_sha256"] = canonical_sha256(payload)
    return write_json(tmp_path / "final.json", payload)


def run(tmp_path, dataset, accuracy=0.9, macro_f1=0.85, *, test_used=False):
    root = tmp_path / dataset / "fold0"
    index = write_json(root / "valid" / "packet_index.jsonl", {})
    label_map = write_json(root / "train" / "label_map.json", {})
    checkpoint = write_json(root / "checkpoint.pt", {})
    prediction = write_json(root / "valid_unified_packet_single_head.npz", {})
    result = write_json(
        root / "valid_unified_packet_single_head.json",
        {
            "task": "packet-level-classification",
            "sample_unit": "one_packet",
            "packet_index": str(index),
            "provenance": {
                "checkpoint": {
                    "path": str(checkpoint),
                    "sha256": file_sha256(checkpoint),
                },
                "packet_index": {"path": str(index), "sha256": file_sha256(index)},
                "label_map": {"path": str(label_map), "sha256": file_sha256(label_map)},
                "prediction_npz": {
                    "path": str(prediction),
                    "sha256": file_sha256(prediction),
                },
            },
            "metrics": {
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "per_class": {str(index): {} for index in range(4)},
            },
        },
    )
    manifest = write_json(
        root / "packet_framework_manifest.json",
        {
            "dataset": dataset,
            "fold": 0,
            "stage": "paper_unified",
            "framework": {
                "notes": {
                    "completed": True,
                    "executed_splits": ["train", "valid"],
                    "test_labels_used": test_used,
                    "shared_core_method_sha256": "placeholder",
                }
            },
        },
    )
    return manifest, result


def inputs(tmp_path, config_path, **overrides):
    fingerprint = json.loads(config_path.read_text())["config_sha256"]
    values = []
    paths = {}
    for dataset in DATASETS:
        manifest, result = run(tmp_path, dataset, **overrides.get(dataset, {}))
        payload = json.loads(manifest.read_text())
        payload["framework"]["notes"]["shared_core_method_sha256"] = fingerprint
        write_json(manifest, payload)
        values.append(f"{dataset}={manifest},{result}")
        paths[dataset] = (manifest, result)
    return values, paths


def test_all_packet_scope_valid_runs_release_test(tmp_path):
    cfg = config(tmp_path)
    values, _ = inputs(tmp_path, cfg)
    report = evaluate(cfg, values)
    assert report["all_datasets_pass"] is True
    assert report["test_evaluation_released"] is True
    assert report["test_labels_used"] is False
    assert set(report["datasets"]) == set(DATASETS)


def test_one_low_valid_result_blocks_test_release(tmp_path):
    cfg = config(tmp_path)
    values, _ = inputs(
        tmp_path,
        cfg,
        **{"vpn-service": {"accuracy": 0.7, "macro_f1": 0.6}},
    )
    report = evaluate(cfg, values)
    assert report["all_datasets_pass"] is False
    assert report["datasets"]["vpn-service"]["passes"] is False


def test_test_labels_or_test_prediction_artifact_are_rejected(tmp_path):
    cfg = config(tmp_path)
    values, paths = inputs(tmp_path, cfg)
    manifest, _ = paths["vpn-binary"]
    payload = json.loads(manifest.read_text())
    payload["framework"]["notes"]["test_labels_used"] = True
    write_json(manifest, payload)
    with pytest.raises(ValueError, match="Test labels"):
        evaluate(cfg, values)

    values, paths = inputs(tmp_path / "second", cfg)
    manifest, _ = paths["ustc-app"]
    write_json(manifest.parent / "test_unified_packet_single_head.json", {})
    with pytest.raises(ValueError, match="Test prediction artifacts"):
        evaluate(cfg, values)


def test_stale_provenance_and_wrong_method_are_rejected(tmp_path):
    cfg = config(tmp_path)
    values, paths = inputs(tmp_path, cfg)
    _, result = paths["ustc-binary"]
    payload = json.loads(result.read_text())
    Path(payload["provenance"]["checkpoint"]["path"]).write_text("changed")
    with pytest.raises(ValueError, match="stale or missing"):
        evaluate(cfg, values)

    values, paths = inputs(tmp_path / "second", cfg)
    manifest, _ = paths["vpn-service"]
    payload = json.loads(manifest.read_text())
    payload["framework"]["notes"]["shared_core_method_sha256"] = "0" * 64
    write_json(manifest, payload)
    with pytest.raises(ValueError, match="different shared method"):
        evaluate(cfg, values)
