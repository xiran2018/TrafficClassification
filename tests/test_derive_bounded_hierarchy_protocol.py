import json

import pytest

from derive_bounded_hierarchy_protocol import derive


def hierarchy_report(path, flow_counts, *, test_labels_used=False):
    packet_count = 100
    raw = {label: 1.0 / count for label, count in enumerate(flow_counts)}
    mean = sum(raw.values()) / len(raw)
    rows = [
        {
            "label_id": label,
            "packet_count": packet_count,
            "flow_count": count,
            "flow_effective_weight": raw[label] / mean,
        }
        for label, count in enumerate(flow_counts)
    ]
    path.write_text(
        json.dumps(
            {
                "schema": "class_sampling_hierarchy_analysis_v1",
                "selection_role": "train_only_reporting_not_model_selection",
                "test_labels_used": test_labels_used,
                "summary": {"effective_number_beta": 0.9999},
                "classes": rows,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_derives_dataset_specific_numbers_with_one_shared_ratio_rule(tmp_path):
    vpn = hierarchy_report(tmp_path / "vpn.json", [100, 25])
    tls = hierarchy_report(tmp_path / "tls.json", [100, 4])

    result = derive(
        {"vpn-app": vpn, "tls-120": tls},
        max_weight_ratio=2.0,
        beta=0.9999,
    )

    assert result["test_labels_used"] is False
    assert result["shared_algorithm"] == (
        "largest_flow_risk_power_subject_to_max_min_ratio"
    )
    assert set(result["datasets"]) == {"vpn-app", "tls-120"}
    for row in result["datasets"].values():
        assert row["bounded_effective_weight_ratio"] == pytest.approx(2.0)
        assert 0.0 < row["class_weight_strength"] < 1.0
        assert len(row["input"]["sha256"]) == 64


def test_rejects_report_that_may_have_used_test_labels(tmp_path):
    report = hierarchy_report(
        tmp_path / "invalid.json", [10, 2], test_labels_used=True
    )

    with pytest.raises(ValueError, match="train-only"):
        derive({"vpn-app": report}, max_weight_ratio=4.0, beta=0.9999)


def test_rejects_beta_mismatch(tmp_path):
    report = hierarchy_report(tmp_path / "input.json", [10, 2])

    with pytest.raises(ValueError, match="beta disagrees"):
        derive({"vpn-app": report}, max_weight_ratio=4.0, beta=0.9)
