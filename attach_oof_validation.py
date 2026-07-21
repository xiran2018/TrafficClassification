#!/usr/bin/env python3
"""Attach fold-local held-out predictions to a fixed cross-fold consensus."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


REQUIRED_VALID_FIELDS = ("valid_flow_ids", "valid_y_true", "valid_prob")


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def attach_oof(consensus: dict, named_payloads: list[tuple[str, str, dict]]) -> dict:
    expected_classes = np.asarray(consensus["flow_prob"]).shape[1]
    oof_ids: list[str] = []
    oof_labels: list[int] = []
    oof_prob: list[list[float]] = []
    sources = []
    for name, path, payload in named_payloads:
        missing = [field for field in REQUIRED_VALID_FIELDS if field not in payload]
        if missing:
            raise ValueError(f"{path} is missing OOF fields: {missing}")
        ids = [str(value) for value in payload["valid_flow_ids"]]
        labels = [int(value) for value in payload["valid_y_true"]]
        prob = np.asarray(payload["valid_prob"], dtype=np.float64)
        if len(ids) != len(labels) or prob.shape != (len(ids), expected_classes):
            raise ValueError(f"unaligned OOF payload: {path}")
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate validation flow IDs within {path}")
        oof_ids.extend(f"{name}::{flow_id}" for flow_id in ids)
        oof_labels.extend(labels)
        oof_prob.extend(prob.tolist())
        sources.append({"name": name, "path": path, "num_samples": len(ids)})

    output = dict(consensus)
    output.update(
        {
            "valid_flow_ids": oof_ids,
            "valid_y_true": oof_labels,
            "valid_prob": oof_prob,
            "oof_validation": {
                "scope": "concatenated_fold_local_held_out_predictions",
                "cross_fold_validation_ensemble": False,
                "num_samples": len(oof_ids),
                "sources": sources,
            },
        }
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--consensus_json", required=True)
    parser.add_argument("--input", nargs=2, action="append", metavar=("NAME", "JSON"), required=True)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    consensus = load_json(args.consensus_json)
    named_payloads = [(name, path, load_json(path)) for name, path in args.input]
    output = attach_oof(consensus, named_payloads)
    destination = Path(args.output_json)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {destination} with {len(output['valid_y_true'])} OOF validation samples")


if __name__ == "__main__":
    main()
