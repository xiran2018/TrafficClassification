#!/usr/bin/env python3
"""Create a compact, hash-bound Flow-valid metric input for method selection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from freeze_shared_core_v2_config import file_sha256


def summarize(source: Path, manifest: Path, config: Path):
    payload = json.loads(source.read_text(encoding="utf-8"))
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
