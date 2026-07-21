import json

import pytest

from freeze_shared_core_v2_config import canonical_sha256, file_sha256
from select_flow_noninferiority import select


def metric(path, accuracy, macro_f1, split="valid", candidate=False):
    source = path.with_name(path.stem + "_source.json")
    manifest = path.with_name(path.stem + "_manifest.json")
    config = path.with_name(path.stem + "_config.json")
    source.write_text("{}", encoding="utf-8")
    manifest.write_text("{}", encoding="utf-8")
    config_payload = {
        "packet_core": {"encoder": "shared"},
        "native_pretraining": {"protocol": "shared"},
        "empirical_risk": {"reduction": "group_mean"},
        "embedding_extraction": {"mode": "concat"},
        "tower1": {
            "identity_safe_contrastive": candidate,
            "cross_scale_weight": 0.05 if candidate else 0.0,
            "cross_scale_temperature": 0.07,
            "lr": 1e-5,
        },
        "selection_protocol": {"test_evaluation_allowed": False},
    }
    config_payload["config_sha256"] = canonical_sha256(config_payload)
    config.write_text(json.dumps(config_payload), encoding="utf-8")
    path.write_text(
        json.dumps(
            {
                "schema": "flow_validation_metric_summary_v1",
                "evaluation_split": split,
                "metrics": {"accuracy": accuracy, "macro_f1": macro_f1},
                "source": {
                    "path": str(source),
                    "sha256": file_sha256(source),
                },
                "framework_manifest": {
                    "path": str(manifest),
                    "sha256": file_sha256(manifest),
                },
                "shared_core_config": {
                    "path": str(config),
                    "sha256": file_sha256(config),
                    "config_sha256": config_payload["config_sha256"],
                },
                "test_labels_used": False,
            }
        ),
        encoding="utf-8",
    )
    return path


def paths(tmp_path, prefix, vpn, tls):
    candidate = prefix == "candidate"
    return {
        "vpn-app": metric(
            tmp_path / f"{prefix}_vpn.json", *vpn, candidate=candidate
        ),
        "tls-120": metric(
            tmp_path / f"{prefix}_tls.json", *tls, candidate=candidate
        ),
    }


def test_candidate_must_be_noninferior_on_both_metrics_and_datasets(tmp_path):
    baseline = paths(tmp_path, "base", (0.75, 0.70), (0.84, 0.82))
    candidate = paths(tmp_path, "candidate", (0.748, 0.699), (0.839, 0.818))
    report = select(baseline, candidate)
    assert report["selected"] == "candidate"
    assert all(row["passes"] for row in report["datasets"].values())
    assert report["test_labels_used"] is False
    assert report["factorial_config_integrity"]["status"] == "pass"


def test_one_dataset_or_metric_failure_rejects_candidate(tmp_path):
    baseline = paths(tmp_path, "base", (0.75, 0.70), (0.84, 0.82))
    candidate = paths(tmp_path, "candidate", (0.7469, 0.71), (0.85, 0.83))
    report = select(baseline, candidate)
    assert report["selected"] == "baseline"
    assert report["datasets"]["vpn-app"]["accuracy_passes"] is False


def test_test_split_input_is_rejected(tmp_path):
    baseline = paths(tmp_path, "base", (0.75, 0.70), (0.84, 0.82))
    candidate = paths(tmp_path, "candidate", (0.76, 0.71), (0.85, 0.83))
    metric(candidate["vpn-app"], 0.76, 0.71, split="test", candidate=True)
    with pytest.raises(ValueError, match="valid-only"):
        select(baseline, candidate)


def test_stale_bound_config_is_rejected(tmp_path):
    baseline = paths(tmp_path, "base", (0.75, 0.70), (0.84, 0.82))
    candidate = paths(tmp_path, "candidate", (0.76, 0.71), (0.85, 0.83))
    summary = json.loads(candidate["vpn-app"].read_text(encoding="utf-8"))
    config = summary["shared_core_config"]["path"]
    with open(config, "a", encoding="utf-8") as handle:
        handle.write(" ")
    with pytest.raises(ValueError, match="stale shared_core_config"):
        select(baseline, candidate)
