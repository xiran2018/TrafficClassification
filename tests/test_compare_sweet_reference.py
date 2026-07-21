import json

from compare_sweet_reference import compare_result, extract_metrics


def test_extract_metrics_supports_packet_and_flow_shapes():
    assert extract_metrics(
        {"metrics": {"packet_level": {"accuracy": 0.9, "macro_f1": 0.8}}},
        "packet-level",
    ) == (0.9, 0.8)
    assert extract_metrics(
        {"metrics": {"flow_level": {"accuracy": 0.85, "macro_f1": 0.82}}},
        "flow-level",
    ) == (0.85, 0.82)


def test_tls_flow_does_not_claim_to_beat_sweet_end_to_end_at_85_percent(tmp_path):
    path = tmp_path / "tls_flow.json"
    path.write_text(
        json.dumps(
            {"metrics": {"flow_level": {"accuracy": 0.85, "macro_f1": 0.83}}}
        ),
        encoding="utf-8",
    )

    row = compare_result("flow-level", "tls-120", str(path))

    assert row["sweet"]["frozen_representation"]["exceeds_both"] is True
    assert row["sweet"]["end_to_end"]["exceeds_both"] is False
    assert row["strict_shared_core_v2_result"] is False
    assert row["headline_sweet_claim"] == "does_not_exceed_protocol_matched_end_to_end"


def test_vpn_packet_can_exceed_both_sweet_reference_tiers(tmp_path):
    path = tmp_path / "vpn_packet.json"
    path.write_text(
        json.dumps({"metrics": {"accuracy": 0.91, "macro_f1": 0.81}}),
        encoding="utf-8",
    )

    row = compare_result("packet-level", "vpn-app", str(path))

    assert row["sweet"]["frozen_representation"]["exceeds_both"] is True
    assert row["sweet"]["end_to_end"]["exceeds_both"] is True
    assert row["headline_sweet_claim"] == "exceeds_protocol_matched_end_to_end"
