import json

import pytest

from analyze_crossfold_disagreement import build_report


def write_prediction(path, flow_ids, y_true, y_pred):
    path.write_text(
        json.dumps(
            {
                "flow_ids": flow_ids,
                "flow_y_true": y_true,
                "flow_y_pred": y_pred,
                "label_map": {"a": 0, "b": 1},
            }
        ),
        encoding="utf-8",
    )


def test_crossfold_report_aligns_ids_and_measures_complementarity(tmp_path):
    fold0 = tmp_path / "fold0.json"
    fold1 = tmp_path / "fold1.json"
    consensus = tmp_path / "consensus.json"
    write_prediction(fold0, ["a", "b", "c", "d"], [0, 0, 1, 1], [0, 1, 1, 0])
    write_prediction(fold1, ["d", "c", "b", "a"], [1, 1, 0, 0], [1, 0, 0, 1])
    write_prediction(consensus, ["a", "b", "c", "d"], [0, 0, 1, 1], [0, 0, 1, 1])

    report = build_report(
        [("fold0", fold0), ("fold1", fold1)], ("consensus", consensus)
    )

    assert report["analysis_contract"]["selection_role"] == "none"
    assert report["alignment"]["status"] == "exact_flow_id_and_true_label_match"
    assert report["summary"]["oracle_any_fold_accuracy"] == 1.0
    assert report["correct_fold_count"]["1"]["count"] == 4
    assert report["pairwise"][0]["error_jaccard"] == 0.0
    assert report["consensus"]["metrics"]["accuracy"] == 1.0


def test_crossfold_report_rejects_label_mismatch(tmp_path):
    fold0 = tmp_path / "fold0.json"
    fold1 = tmp_path / "fold1.json"
    write_prediction(fold0, ["a", "b"], [0, 1], [0, 1])
    write_prediction(fold1, ["b", "a"], [0, 1], [0, 1])

    with pytest.raises(ValueError, match="true label mismatch"):
        build_report([("fold0", fold0), ("fold1", fold1)])


def test_crossfold_report_rejects_different_flow_sets(tmp_path):
    fold0 = tmp_path / "fold0.json"
    fold1 = tmp_path / "fold1.json"
    write_prediction(fold0, ["a", "b"], [0, 1], [0, 1])
    write_prediction(fold1, ["a", "c"], [0, 1], [0, 1])

    with pytest.raises(ValueError, match="flow ID set differs"):
        build_report([("fold0", fold0), ("fold1", fold1)])
