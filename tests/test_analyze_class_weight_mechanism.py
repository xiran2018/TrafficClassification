import json

import pytest

from analyze_class_weight_mechanism import analyze


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def history(path, step, a_f1, b_f1, accuracy, macro_f1):
    return write_jsonl(
        path,
        [
            {
                "step": step,
                "metrics": {
                    "accuracy": accuracy,
                    "macro_f1": macro_f1,
                    "per_class": {"a": {"f1": a_f1}, "b": {"f1": b_f1}},
                },
            }
        ],
    )


def inputs(tmp_path):
    train = write_jsonl(
        tmp_path / "train.jsonl",
        [
            {"label_id": 0, "label": "a", "flow_id": "a1"},
            {"label_id": 0, "label": "a", "flow_id": "a1"},
            {"label_id": 0, "label": "a", "flow_id": "a1"},
            {"label_id": 0, "label": "a", "flow_id": "a2"},
            {"label_id": 1, "label": "b", "flow_id": "b1"},
            {"label_id": 1, "label": "b", "flow_id": "b1"},
        ],
    )
    baseline = history(tmp_path / "base.jsonl", 10, 0.4, 0.8, 0.6, 0.6)
    candidate = history(tmp_path / "candidate.jsonl", 10, 0.6, 0.7, 0.65, 0.65)
    return train, baseline, candidate


def test_reports_hashed_validation_only_mechanism_evidence(tmp_path):
    train, baseline, candidate = inputs(tmp_path)
    report = analyze(
        train,
        baseline,
        candidate,
        step=10,
        baseline_basis="packet",
        candidate_basis="flow",
        baseline_strength=1.0,
        candidate_strength=0.5,
        beta=0.9999,
    )
    assert report["test_labels_used"] is False
    assert report["metrics"]["delta"]["accuracy"] == pytest.approx(0.05)
    assert report["metrics"]["delta"]["macro_f1"] == pytest.approx(0.05)
    assert report["num_classes_improved"] == 1
    assert report["num_classes_degraded"] == 1
    assert len(report["inputs"]["train_jsonl"]["sha256"]) == 64
    assert report["weighting"]["candidate"] == {"basis": "flow", "strength": 0.5}


def test_rejects_unmatched_step_or_label_contract(tmp_path):
    train, baseline, candidate = inputs(tmp_path)
    with pytest.raises(ValueError, match="must exist in both"):
        analyze(
            train,
            baseline,
            candidate,
            step=11,
            baseline_basis="packet",
            candidate_basis="flow",
            baseline_strength=1.0,
            candidate_strength=0.5,
            beta=0.9999,
        )

    history(candidate, 10, 0.6, 0.7, 0.65, 0.65)
    payload = json.loads(candidate.read_text())
    payload["metrics"]["per_class"] = {"a": {"f1": 0.6}, "c": {"f1": 0.7}}
    candidate.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="do not match"):
        analyze(
            train,
            baseline,
            candidate,
            step=10,
            baseline_basis="packet",
            candidate_basis="flow",
            baseline_strength=1.0,
            candidate_strength=0.5,
            beta=0.9999,
        )
