import json

import pytest

from select_flow_noninferiority import select


def metric(path, accuracy, macro_f1, split="valid"):
    path.write_text(
        json.dumps(
            {
                "evaluation_split": split,
                "metrics": {"accuracy": accuracy, "macro_f1": macro_f1},
            }
        ),
        encoding="utf-8",
    )
    return path


def paths(tmp_path, prefix, vpn, tls):
    return {
        "vpn-app": metric(tmp_path / f"{prefix}_vpn.json", *vpn),
        "tls-120": metric(tmp_path / f"{prefix}_tls.json", *tls),
    }


def test_candidate_must_be_noninferior_on_both_metrics_and_datasets(tmp_path):
    baseline = paths(tmp_path, "base", (0.75, 0.70), (0.84, 0.82))
    candidate = paths(tmp_path, "candidate", (0.748, 0.699), (0.839, 0.818))
    report = select(baseline, candidate)
    assert report["selected"] == "candidate"
    assert all(row["passes"] for row in report["datasets"].values())
    assert report["test_labels_used"] is False


def test_one_dataset_or_metric_failure_rejects_candidate(tmp_path):
    baseline = paths(tmp_path, "base", (0.75, 0.70), (0.84, 0.82))
    candidate = paths(tmp_path, "candidate", (0.7469, 0.71), (0.85, 0.83))
    report = select(baseline, candidate)
    assert report["selected"] == "baseline"
    assert report["datasets"]["vpn-app"]["accuracy_passes"] is False


def test_test_split_input_is_rejected(tmp_path):
    baseline = paths(tmp_path, "base", (0.75, 0.70), (0.84, 0.82))
    candidate = paths(tmp_path, "candidate", (0.76, 0.71), (0.85, 0.83))
    metric(candidate["vpn-app"], 0.76, 0.71, split="test")
    with pytest.raises(ValueError, match="valid-only"):
        select(baseline, candidate)
