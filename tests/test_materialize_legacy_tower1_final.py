import json

import pytest

from materialize_legacy_tower1_final import materialize_final


def make_run(tmp_path, points=8):
    output = tmp_path / "run"
    output.mkdir()
    rows = [
        {
            "step": index + 1,
            "select_metric": "macro_f1",
            "metrics": {"macro_f1": index / 10, "accuracy": index / 10},
        }
        for index in range(points)
    ]
    (output / "packet_validation_history.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    (output / "best_packet_validation_metrics.json").write_text(
        json.dumps(rows[-1]), encoding="utf-8"
    )
    (output / "tower1_heads.pt").write_bytes(b"heads")
    (output / "tower1_config.json").write_text("{}", encoding="utf-8")
    (output / "adapter").mkdir()
    (output / "adapter" / "adapter_model.safetensors").write_bytes(b"adapter")
    (output / "tokenizer.json").write_text("{}", encoding="utf-8")
    return output


def test_materializes_completed_legacy_root_checkpoint(tmp_path):
    output = make_run(tmp_path)

    result = materialize_final(output)

    assert result["status"] == "materialized"
    assert (output / "final" / "tower1_heads.pt").read_bytes() == b"heads"
    assert (output / "final" / "adapter" / "adapter_model.safetensors").is_file()
    assert (output / "legacy_final_materialization.json").is_file()


def test_rejects_incomplete_or_contaminated_history(tmp_path):
    output = make_run(tmp_path, points=7)
    with pytest.raises(ValueError, match="exactly 8"):
        materialize_final(output)


def test_rejects_stale_best_metrics(tmp_path):
    output = make_run(tmp_path)
    stale = {
        "step": 1,
        "select_metric": "macro_f1",
        "metrics": {"macro_f1": 0.0, "accuracy": 0.0},
    }
    (output / "best_packet_validation_metrics.json").write_text(
        json.dumps(stale), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="do not match"):
        materialize_final(output)
