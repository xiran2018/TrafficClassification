import json

import pytest

from audit_cross_scale_availability import summarize


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_availability_audit_uses_distinct_packet_identity(tmp_path):
    path = tmp_path / "packets.jsonl"
    write_rows(
        path,
        [
            {"flow_id": "a", "packet_uid": "a-0", "label_id": 0},
            {"flow_id": "a", "packet_uid": "a-0", "label_id": 0},
            {"flow_id": "b", "packet_uid": "b-0", "label_id": 1},
            {"flow_id": "b", "packet_uid": "b-1", "label_id": 1},
        ],
    )
    result = summarize(path)
    assert result["rows"] == 4
    assert result["unique_packet_identities"] == 3
    assert result["duplicate_identity_rows"] == 1
    assert result["singleton_flows"] == 1
    assert result["context_eligible_flow_rate"] == 0.5
    assert result["context_eligible_packet_identity_rate"] == pytest.approx(2 / 3)


def test_availability_audit_rejects_conflicting_flow_labels(tmp_path):
    path = tmp_path / "packets.jsonl"
    write_rows(
        path,
        [
            {"flow_id": "a", "packet_uid": "a-0", "label_id": 0},
            {"flow_id": "a", "packet_uid": "a-1", "label_id": 1},
        ],
    )
    with pytest.raises(ValueError, match="conflicting labels"):
        summarize(path)
