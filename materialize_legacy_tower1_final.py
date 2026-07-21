#!/usr/bin/env python3
"""Materialize the pre-contract Tower1 root checkpoint as an explicit final snapshot."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_completed_history(output_dir: Path, expected_points: int) -> dict[str, Any]:
    history_path = output_dir / "packet_validation_history.jsonl"
    rows = [
        json.loads(line)
        for line in history_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != expected_points:
        raise ValueError(
            f"expected exactly {expected_points} validation points, got {len(rows)}"
        )
    best_path = output_dir / "best_packet_validation_metrics.json"
    best = load_json(best_path)
    history_best = max(
        rows,
        key=lambda row: (
            float(row["metrics"]["macro_f1"]),
            float(row["metrics"]["macro_f1"]),
            float(row["metrics"]["accuracy"]),
        ),
    )
    if (
        best.get("select_metric") != "macro_f1"
        or int(best["step"]) != int(history_best["step"])
        or float(best["metrics"]["macro_f1"])
        != float(history_best["metrics"]["macro_f1"])
    ):
        raise ValueError("best metrics do not match the completed validation history")
    return {
        "validation_points": len(rows),
        "history_sha256": file_sha256(history_path),
        "best_metrics_sha256": file_sha256(best_path),
        "best_step": int(best["step"]),
        "best_macro_f1": float(best["metrics"]["macro_f1"]),
    }


def materialize_final(output_dir: str | Path, expected_points: int = 8) -> dict[str, Any]:
    output = Path(output_dir)
    history = validate_completed_history(output, expected_points)
    required = (output / "tower1_heads.pt", output / "tower1_config.json", output / "adapter")
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"legacy root checkpoint is incomplete: {missing}")

    final_dir = output / "final"
    if final_dir.exists():
        final_heads = final_dir / "tower1_heads.pt"
        if not final_heads.is_file():
            raise ValueError(f"existing final directory is incomplete: {final_dir}")
        return {
            "status": "already_materialized",
            "output_dir": str(output.resolve()),
            "final_heads_sha256": file_sha256(final_heads),
            "history": history,
        }

    staging = output / ".final.materializing"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    shutil.copytree(output / "adapter", staging / "adapter")
    for path in output.iterdir():
        if not path.is_file():
            continue
        if path.name == "tower1_heads.pt" or path.name == "tower1_config.json":
            shutil.copy2(path, staging / path.name)
        elif path.name.startswith("tokenizer") or path.name == "chat_template.jinja":
            shutil.copy2(path, staging / path.name)
    staging.rename(final_dir)

    payload = {
        "schema": "legacy_tower1_final_materialization_v1",
        "status": "materialized",
        "reason": "legacy_trainer_wrote_the_terminal_checkpoint_to_output_root",
        "output_dir": str(output.resolve()),
        "source_heads_sha256": file_sha256(output / "tower1_heads.pt"),
        "final_heads_sha256": file_sha256(final_dir / "tower1_heads.pt"),
        "source_config_sha256": file_sha256(output / "tower1_config.json"),
        "final_config_sha256": file_sha256(final_dir / "tower1_config.json"),
        "history": history,
    }
    manifest = output / "legacy_final_materialization.json"
    manifest.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--expected_validation_points", type=int, default=8)
    args = parser.parse_args()
    result = materialize_final(args.output_dir, args.expected_validation_points)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
