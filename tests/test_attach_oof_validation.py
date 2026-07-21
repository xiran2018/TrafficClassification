import pytest

from attach_oof_validation import attach_oof


def test_attach_oof_prefixes_fold_local_ids():
    consensus = {"flow_prob": [[0.7, 0.3]], "flow_y_true": [0], "flow_ids": ["test"]}
    payloads = [
        ("fold0", "fold0.json", {"valid_flow_ids": ["same"], "valid_y_true": [0], "valid_prob": [[0.8, 0.2]]}),
        ("fold1", "fold1.json", {"valid_flow_ids": ["same"], "valid_y_true": [1], "valid_prob": [[0.1, 0.9]]}),
    ]

    output = attach_oof(consensus, payloads)

    assert output["valid_flow_ids"] == ["fold0::same", "fold1::same"]
    assert output["valid_y_true"] == [0, 1]
    assert output["oof_validation"]["cross_fold_validation_ensemble"] is False


def test_attach_oof_rejects_wrong_class_dimension():
    consensus = {"flow_prob": [[0.7, 0.3]]}
    payloads = [
        ("fold0", "fold0.json", {"valid_flow_ids": ["a"], "valid_y_true": [0], "valid_prob": [[1.0]]}),
    ]

    with pytest.raises(ValueError, match="unaligned OOF"):
        attach_oof(consensus, payloads)
