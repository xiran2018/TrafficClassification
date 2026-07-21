#!/usr/bin/env python3
"""Create a compact, hash-bound Flow-valid metric input for method selection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from freeze_shared_core_v2_config import canonical_sha256, file_sha256


def _split_set(value) -> set[str]:
    if isinstance(value, str):
        return {item.strip() for item in value.split(",") if item.strip()}
    return {str(item) for item in (value or [])}


def summarize(source: Path, manifest: Path, config: Path):
    payload = json.loads(source.read_text(encoding="utf-8"))
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    config_payload = json.loads(config.read_text(encoding="utf-8"))
    config_fingerprint = config_payload.get("config_sha256")
    unsigned_config = {
        key: value for key, value in config_payload.items() if key != "config_sha256"
    }
    if config_fingerprint != canonical_sha256(unsigned_config):
        raise ValueError("Flow-valid shared-core config fingerprint mismatch")
    if (config_payload.get("selection_protocol") or {}).get(
        "test_evaluation_allowed"
    ) is not False:
        raise ValueError("Flow-valid config must explicitly forbid test evaluation")
    if _split_set(manifest_payload.get("splits")) != {"train", "valid"}:
        raise ValueError("Flow-valid manifest must contain train and valid only")
    if _split_set(manifest_payload.get("eval_splits")) != {"valid"}:
        raise ValueError("Flow-valid manifest must evaluate valid only")
    notes = ((manifest_payload.get("framework") or {}).get("notes") or {})
    if notes.get("shared_core_method_sha256") != config_fingerprint:
        raise ValueError("Flow-valid manifest is not bound to the supplied config")
    configured_path = Path(str(notes.get("shared_core_config") or "")).resolve()
    if configured_path != config.resolve():
        raise ValueError("Flow-valid manifest references a different config path")
    result_paths = {
        Path(str(path)).resolve() for path in (notes.get("result_paths") or [])
    }
    if source.resolve() not in result_paths:
        raise ValueError("Flow-valid metric source is absent from manifest results")
    metrics = (payload.get("metrics") or {}).get("flow_level") or {}
    return {
        "schema": "flow_validation_metric_summary_v1",
        "evaluation_split": "valid",
        "metrics": {
            "accuracy": float(metrics["accuracy"]),
            "macro_f1": float(metrics["macro_f1"]),
        },
        "source": {"path": str(source.resolve()), "sha256": file_sha256(source)},
        "framework_manifest": {
            "path": str(manifest.resolve()),
            "sha256": file_sha256(manifest),
        },
        "shared_core_config": {
            "path": str(config.resolve()),
            "sha256": file_sha256(config),
            "config_sha256": config_fingerprint,
        },
        "test_labels_used": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_json", required=True)
    parser.add_argument("--framework_manifest", required=True)
    parser.add_argument("--shared_core_config", required=True)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = summarize(
        Path(args.source_json),
        Path(args.framework_manifest),
        Path(args.shared_core_config),
    )
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
