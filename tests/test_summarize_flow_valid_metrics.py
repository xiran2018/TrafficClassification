import json

from freeze_shared_core_v2_config import canonical_sha256
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
    config_payload = {
        "selection_protocol": {
            "test_labels_used": False,
            "test_evaluation_allowed": False,
        }
    }
    config_payload["config_sha256"] = canonical_sha256(config_payload)
    config.write_text(json.dumps(config_payload), encoding="utf-8")
    manifest.write_text(
        json.dumps(
            {
                "splits": ["train", "valid"],
                "eval_splits": "valid",
                "framework": {
                    "notes": {
                        "shared_core_method_sha256": config_payload[
                            "config_sha256"
                        ],
                        "shared_core_config": str(config.resolve()),
                        "result_paths": [str(source.resolve())],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    result = summarize(source, manifest, config)
    assert result["evaluation_split"] == "valid"
    assert result["metrics"] == {"accuracy": 0.75, "macro_f1": 0.70}
    assert result["test_labels_used"] is False
    assert len(result["source"]["sha256"]) == 64
    assert result["shared_core_config"]["config_sha256"] == config_payload[
        "config_sha256"
    ]
