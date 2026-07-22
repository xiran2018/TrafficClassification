import json
from pathlib import Path

import numpy as np
import pytest

from summarize_milestone_benchmark import DATASETS, summarize


CONFIG_SHA = "c" * 64
SOURCE_SHA = "s" * 64


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def result(path: Path, task: str) -> Path:
    metrics = {"num_samples": 10, "accuracy": 0.8, "macro_f1": 0.7}
    flow_metrics = {
        "accuracy": 0.8,
        "macro_f1": 0.7,
        "calibration": {"num_samples": 10},
    }
    if task == "packet":
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "task": "packet-level-classification",
            "sample_unit": "one_packet",
            "metrics": metrics,
        }
        np.savez_compressed(
            path.with_suffix(".npz"),
            y_true=np.arange(10) % 2,
            probabilities=np.full((10, 2), 0.5),
            flow_ids=np.asarray([f"flow-{index // 2}" for index in range(10)]),
            packet_uids=np.asarray([f"packet-{index}" for index in range(10)]),
        )
    else:
        payload = {
            "metrics": {"flow_level": flow_metrics},
            "flow_y_true": [0] * 10,
            "flow_y_pred": [0] * 10,
            "flow_prob": [[1.0, 0.0]] * 10,
            "flow_ids": [f"flow-{index}" for index in range(10)],
        }
    return write_json(path, payload)


def manifest(dataset: str, task: str, result_path: Path, *, source=SOURCE_SHA) -> dict:
    return {
        "dataset": dataset,
        "fold": 0,
        "framework": {
            "task": task,
            "notes": {
                "completed": True,
                "shared_core_config_sha256": CONFIG_SHA,
                "result_paths": [str(result_path)],
                "algorithm_source_evidence": {
                    "status": "pass",
                    "launch_fingerprint": source,
                    "completion_fingerprint": source,
                    "changed_paths": [],
                },
            },
        },
    }


def build_tree(
    tmp_path: Path,
    *,
    decision: str = "final_after_preregistered_validation",
    tag: str = "milestone_dev_fold0",
) -> tuple[Path, Path, Path]:
    root = tmp_path / "root"
    repo = tmp_path / "repo"
    config = write_json(
        tmp_path / "final.json",
        {
            "config_sha256": CONFIG_SHA,
            "method_selection": {
                "decision_status": decision,
                "test_labels_used": False,
                "selected_method": "shared_core_v2",
            },
            "selection_protocol": {"test_evaluation_allowed": True},
        },
    )
    for dataset in DATASETS:
        packet_result = result(
            root
            / "packet_artifacts"
            / dataset
            / "fold0"
            / "test_unified_packet_single_head.json",
            "packet",
        )
        flow_result = result(
            repo
            / "reasoningDataset"
            / dataset
            / f"test_seq_metrics_flow_{tag}_probs.json",
            "flow",
        )
        write_json(
            root
            / "packet_artifacts"
            / dataset
            / "fold0"
            / "packet_framework_manifest.json",
            manifest(dataset, "packet-level", packet_result),
        )
        write_json(
            repo
            / "reasoningDataset"
            / dataset
            / f"stage8_flowaware_manifest_{tag}.json",
            manifest(dataset, "flow-level", flow_result),
        )
        write_json(
            root / "audits" / dataset / "fold0" / "audit.json",
            {
                "status": "pass",
                "dataset": dataset,
                "fold": 0,
                "shared_core_config_sha256": CONFIG_SHA,
            },
        )
    return root, repo, config


def test_summarize_marks_test_as_development_benchmark(tmp_path):
    root, repo, config = build_tree(tmp_path)
    report = summarize(root, repo, config)
    assert report["status"] == "pass"
    assert report["may_inform_future_method_design"] is True
    assert report["unbiased_final_claim_allowed"] is False
    assert report["algorithm_source_fingerprint"] == SOURCE_SHA
    assert report["datasets"]["vpn-app"]["packet"]["metrics"] == {
        "num_samples": 10,
        "accuracy": 0.8,
        "macro_f1": 0.7,
    }
    assert report["datasets"]["vpn-app"]["packet"]["sample_unit_audit"][
        "num_unique_packet_uids"
    ] == 10
    assert report["datasets"]["vpn-app"]["flow"]["sample_unit_audit"] == {
        "status": "pass",
        "sample_unit": "one_flow",
        "num_samples": 10,
        "num_unique_flow_ids": 10,
        "prediction_path": str(
            (
                repo
                / "reasoningDataset"
                / "vpn-app"
                / "test_seq_metrics_flow_milestone_dev_fold0_probs.json"
            ).resolve()
        ),
        "prediction_sha256": report["datasets"]["vpn-app"]["flow"][
            "result_sha256"
        ],
    }
    comparison = report["datasets"]["vpn-app"]["packet"]["comparison"]
    assert comparison["predeclared_target"]["met"] is False
    assert comparison["sweet"]["end_to_end"]["delta_accuracy"] == pytest.approx(
        -0.056
    )


def test_summarize_accepts_base_frozen_milestone_with_unique_tag(tmp_path):
    tag = "base_milestone_dev_fold0"
    root, repo, config = build_tree(
        tmp_path,
        decision="base_frozen_pending_identity_cross_scale_validation",
        tag=tag,
    )
    payload = json.loads(config.read_text(encoding="utf-8"))
    payload["selection_protocol"]["test_evaluation_role"] = (
        "development_benchmark_after_base_shared_core_freeze"
    )
    write_json(config, payload)
    report = summarize(root, repo, config, tag=tag)
    assert report["evaluation_role"] == (
        "development_benchmark_after_base_shared_core_freeze"
    )


def test_summarize_rejects_cross_task_source_drift(tmp_path):
    root, repo, config = build_tree(tmp_path)
    path = (
        repo
        / "reasoningDataset"
        / "tls-120"
        / "stage8_flowaware_manifest_milestone_dev_fold0.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    evidence = payload["framework"]["notes"]["algorithm_source_evidence"]
    evidence["launch_fingerprint"] = "x" * 64
    evidence["completion_fingerprint"] = "x" * 64
    write_json(path, payload)
    with pytest.raises(ValueError, match="source fingerprints differ"):
        summarize(root, repo, config)


def test_summarize_rejects_unfrozen_config(tmp_path):
    root, repo, config = build_tree(tmp_path)
    payload = json.loads(config.read_text(encoding="utf-8"))
    payload["method_selection"]["decision_status"] = "provisional"
    write_json(config, payload)
    with pytest.raises(ValueError, match="not frozen"):
        summarize(root, repo, config)


def test_summarize_rejects_duplicate_packet_or_flow_samples(tmp_path):
    root, repo, config = build_tree(tmp_path)
    packet_npz = (
        root
        / "packet_artifacts"
        / "vpn-app"
        / "fold0"
        / "test_unified_packet_single_head.npz"
    )
    with np.load(packet_npz) as original:
        arrays = {key: original[key] for key in original.files}
    arrays["packet_uids"][-1] = arrays["packet_uids"][0]
    np.savez_compressed(packet_npz, **arrays)
    with pytest.raises(ValueError, match="one unique packet"):
        summarize(root, repo, config)

    root, repo, config = build_tree(tmp_path / "flow_case")
    flow_result = (
        repo
        / "reasoningDataset"
        / "vpn-app"
        / "test_seq_metrics_flow_milestone_dev_fold0_probs.json"
    )
    payload = json.loads(flow_result.read_text(encoding="utf-8"))
    payload["flow_ids"][-1] = payload["flow_ids"][0]
    write_json(flow_result, payload)
    with pytest.raises(ValueError, match="one unique flow"):
        summarize(root, repo, config)
