import json
from pathlib import Path

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
    payload = {
        "metrics": metrics if task == "packet" else {"flow_level": flow_metrics}
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


def build_tree(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / "root"
    repo = tmp_path / "repo"
    config = write_json(
        tmp_path / "final.json",
        {
            "config_sha256": CONFIG_SHA,
            "method_selection": {
                "decision_status": "final_after_preregistered_validation",
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
            / "test_seq_metrics_flow_milestone_dev_fold0_probs.json",
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
            / "stage8_flowaware_manifest_milestone_dev_fold0.json",
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
