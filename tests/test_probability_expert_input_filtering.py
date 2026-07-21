import json

from train_expert_gate import filter_compatible_inputs as filter_gate_inputs
from train_expert_gate import load_named_prob_payloads as load_gate_inputs
from train_prediction_stacker import filter_compatible_inputs as filter_stacker_inputs
from train_prediction_stacker import load_named_prob_payloads as load_stacker_inputs
from validation_gated_selector import load_named_payloads as load_selector_inputs
from validation_gated_selector import select_compatible_input_group


def _write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def _complete_payload():
    return {
        "valid_flow_ids": ["v0", "v1", "v2", "v3"],
        "valid_y_true": [0, 1, 0, 1],
        "valid_prob": [[0.8, 0.2], [0.3, 0.7], [0.6, 0.4], [0.2, 0.8]],
        "flow_ids": ["t0", "t1"],
        "flow_y_true": [0, 1],
        "flow_prob": [[0.7, 0.3], [0.4, 0.6]],
    }


def _complete_payload_with_prefix(prefix):
    payload = _complete_payload()
    payload["valid_flow_ids"] = [f"{prefix}_{fid}" for fid in payload["valid_flow_ids"]]
    payload["flow_ids"] = [f"{prefix}_{fid}" for fid in payload["flow_ids"]]
    return payload


def _test_only_payload():
    return {
        "flow_ids": ["t0", "t1"],
        "flow_y_true": [0, 1],
        "flow_prob": [[0.9, 0.1], [0.2, 0.8]],
    }


def test_stacker_skips_test_only_probability_inputs(tmp_path):
    test_only = _write_json(tmp_path / "base.json", _test_only_payload())
    complete = _write_json(tmp_path / "paired.json", _complete_payload())

    loaded, skipped = load_stacker_inputs([["base", test_only], ["paired", complete]])

    assert [name for name, _, _ in loaded] == ["paired"]
    assert skipped[0]["name"] == "base"
    assert "valid_flow_ids" in skipped[0]["reason"]


def test_expert_gate_skips_test_only_probability_inputs(tmp_path):
    test_only = _write_json(tmp_path / "base.json", _test_only_payload())
    complete = _write_json(tmp_path / "paired.json", _complete_payload())

    loaded, skipped = load_gate_inputs([["base", test_only], ["paired", complete]])

    assert [name for name, _, _ in loaded] == ["paired"]
    assert skipped[0]["name"] == "base"
    assert "valid_flow_ids" in skipped[0]["reason"]


def test_stacker_filters_incompatible_flow_id_groups(tmp_path):
    first = _write_json(tmp_path / "first.json", _complete_payload_with_prefix("a"))
    second = _write_json(tmp_path / "second.json", _complete_payload_with_prefix("b"))
    loaded, skipped = load_stacker_inputs([["first", first], ["second", second]])

    compatible, incompatible = filter_stacker_inputs(loaded)

    assert [name for name, _, _ in compatible] == ["first"]
    assert skipped == []
    assert incompatible[0]["name"] == "second"
    assert "valid_overlap=0" in incompatible[0]["reason"]


def test_expert_gate_filters_incompatible_flow_id_groups(tmp_path):
    first = _write_json(tmp_path / "first.json", _complete_payload_with_prefix("a"))
    second = _write_json(tmp_path / "second.json", _complete_payload_with_prefix("b"))
    loaded, skipped = load_gate_inputs([["first", first], ["second", second]])

    compatible, incompatible = filter_gate_inputs(loaded)

    assert [name for name, _, _ in compatible] == ["first"]
    assert skipped == []
    assert incompatible[0]["name"] == "second"
    assert "valid_overlap=0" in incompatible[0]["reason"]


def test_selector_uses_largest_compatible_flow_id_group(tmp_path):
    paired = _write_json(tmp_path / "paired.json", _complete_payload_with_prefix("paired"))
    stacker = _write_json(tmp_path / "stacker.json", _complete_payload_with_prefix("shared"))
    soft_gate = _write_json(tmp_path / "soft_gate.json", _complete_payload_with_prefix("shared"))
    loaded, skipped = load_selector_inputs([["paired", paired], ["slot_stacker", stacker], ["soft_gate", soft_gate]])

    compatible, incompatible = select_compatible_input_group(loaded)

    assert [name for name, _, _ in compatible] == ["slot_stacker", "soft_gate"]
    assert skipped == []
    assert incompatible[0]["name"] == "paired"
    assert "valid_overlap=0" in incompatible[0]["reason"]
