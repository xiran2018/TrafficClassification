import json

from summarize_flow_valid_metrics import summarize


def test_summary_uses_flow_level_metrics_and_binds_inputs(tmp_path):
    source = tmp_path / "valid.json"
    manifest = tmp_path / "manifest.json"
    config = tmp_path / "config.json"
    source.write_text(
        json.dumps(
            {
                "metrics": {
                    "window_level": {"accuracy": 0.1, "macro_f1": 0.2},
                    "flow_level": {"accuracy": 0.75, "macro_f1": 0.70},
                }
            }
        ),
        encoding="utf-8",
    )
    manifest.write_text("{}", encoding="utf-8")
    config.write_text("{}", encoding="utf-8")
    result = summarize(source, manifest, config)
    assert result["evaluation_split"] == "valid"
    assert result["metrics"] == {"accuracy": 0.75, "macro_f1": 0.70}
    assert result["test_labels_used"] is False
    assert len(result["source"]["sha256"]) == 64
