import json

import numpy as np

from rank_content_group_candidates import candidate_paths, evaluate_candidate, fast_metrics_from_pred, rank_rows


def _write_prediction(path, prob):
    path.write_text(
        json.dumps(
            {
                "flow_ids": ["a", "b"],
                "flow_y_true": [0, 1],
                "flow_prob": prob,
            }
        ),
        encoding="utf-8",
    )


def test_rank_rows_prefers_higher_content_group_lower_bound():
    rows = rank_rows(
        [
            {
                "status": "ok",
                "accuracy": 0.9,
                "macro_f1": 0.9,
                "content_group_accuracy_ci95": [0.7, 0.9],
                "content_group_macro_f1_ci95": [0.6, 0.9],
            },
            {
                "status": "ok",
                "accuracy": 0.8,
                "macro_f1": 0.8,
                "content_group_accuracy_ci95": [0.8, 0.9],
                "content_group_macro_f1_ci95": [0.7, 0.9],
            },
        ]
    )

    assert rows[0]["content_group_accuracy_ci95"][0] == 0.8


def test_fast_metrics_from_pred_uses_observed_label_union():
    acc, macro_f1 = fast_metrics_from_pred(
        np.asarray([0, 1, 1]),
        np.asarray([0, 0, 1]),
        num_classes=3,
    )

    assert acc == 2 / 3
    assert round(macro_f1, 4) == 0.6667


def test_evaluate_candidate_computes_group_ci(tmp_path):
    good = tmp_path / "test_good.json"
    _write_prediction(good, [[0.95, 0.05], [0.05, 0.95]])
    row = evaluate_candidate(
        good,
        flow_to_hash={"a": "ha", "b": "hb"},
        samples=20,
        seed=7,
        target_accuracy=0.9,
        target_macro_f1=0.9,
    )

    assert row["status"] == "ok"
    assert row["content_group_target_met"] is True
    assert row["content_group_count"] == 2


def test_evaluate_candidate_reports_missing_fields(tmp_path):
    bad = tmp_path / "test_bad.json"
    bad.write_text(json.dumps({"flow_ids": []}), encoding="utf-8")

    row = evaluate_candidate(
        bad,
        flow_to_hash={},
        samples=20,
        seed=7,
        target_accuracy=None,
        target_macro_f1=None,
    )

    assert row["status"] == "missing_fields"
    assert "flow_y_true" in row["missing"]


def test_candidate_paths_respects_max_files(tmp_path):
    for idx in range(3):
        (tmp_path / f"test_{idx}.json").write_text("{}", encoding="utf-8")

    paths = candidate_paths(tmp_path, ["test*.json"], max_files=2)

    assert len(paths) == 2
