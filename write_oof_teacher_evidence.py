#!/usr/bin/env python3
"""Create checkpoint-bound proof that teacher prediction flows were held out."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from audit_distillation_teacher_coverage import dataset_flow_ids, load_pt
from train_tower2 import file_evidence


def canonical_ids_sha256(flow_ids: list[str]) -> str:
    canonical = "\n".join(sorted(set(map(str, flow_ids)))) + "\n"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_checkpoint(path: Path) -> dict[str, Any]:
    payload = load_pt(str(path))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: checkpoint payload is not a dictionary")
    return payload


def validate_oof_evidence(
    evidence_path: Path, prediction_path: Path, prediction_ids: list[str]
) -> dict[str, Any]:
    """Recheck an OOF proof and every content-bound artifact it references."""
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if evidence.get("schema") != "oof_teacher_evidence_v1":
        raise ValueError(f"{evidence_path}: unsupported OOF evidence schema")
    if evidence.get("oof_exclusion_proven") is not True:
        raise ValueError(f"{evidence_path}: OOF exclusion is not proven")
    if not prediction_ids or len(set(prediction_ids)) != len(prediction_ids):
        raise ValueError("prediction flow_ids must be non-empty and unique")

    prediction = evidence.get("prediction") or {}
    current_prediction = file_evidence(prediction_path)
    if prediction.get("path") != current_prediction["path"]:
        raise ValueError("OOF evidence references a different prediction path")
    if prediction.get("sha256") != current_prediction["sha256"]:
        raise ValueError("OOF prediction changed after evidence was written")
    if prediction.get("flow_ids_sha256") != canonical_ids_sha256(prediction_ids):
        raise ValueError("OOF evidence flow IDs do not match the prediction")

    for name in ("checkpoint", "training_dataset", "evaluation_dataset"):
        recorded = evidence.get(name) or {}
        recorded_path = recorded.get("path")
        if not recorded_path:
            raise ValueError(f"OOF evidence lacks {name} path")
        current = file_evidence(recorded_path)
        if current["path"] != recorded_path or current["sha256"] != recorded.get("sha256"):
            raise ValueError(f"OOF {name} changed after evidence was written")

    disjointness = evidence.get("disjointness") or {}
    bindings = evidence.get("bindings") or {}
    if disjointness.get("overlap_count") != 0:
        raise ValueError("OOF evidence reports training/evaluation overlap")
    if disjointness.get("prediction_matches_evaluation_flow_set") is not True:
        raise ValueError("OOF prediction does not match the evaluation flow set")
    required_bindings = (
        "checkpoint_binds_training_dataset",
        "prediction_binds_checkpoint",
        "prediction_binds_evaluation_dataset",
    )
    if not all(bindings.get(key) is True for key in required_bindings):
        raise ValueError("OOF evidence lacks required content bindings")
    recomputed = verify_oof_teacher(
        prediction_path,
        Path(evidence["checkpoint"]["path"]),
        Path(evidence["training_dataset"]["path"]),
        Path(evidence["evaluation_dataset"]["path"]),
    )
    if recomputed != evidence:
        raise ValueError("OOF evidence does not match a fresh exclusion proof")
    return evidence


def verify_oof_teacher(
    prediction_path: Path,
    checkpoint_path: Path,
    train_dataset_path: Path,
    evaluation_dataset_path: Path,
) -> dict[str, Any]:
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    prediction_ids = [str(value) for value in prediction.get("flow_ids", [])]
    if not prediction_ids or len(set(prediction_ids)) != len(prediction_ids):
        raise ValueError("prediction flow_ids must be non-empty and unique")

    checkpoint = load_checkpoint(checkpoint_path)
    checkpoint_evidence = file_evidence(checkpoint_path)
    train_evidence = file_evidence(train_dataset_path)
    evaluation_evidence = file_evidence(evaluation_dataset_path)
    embedded_training = checkpoint.get("training_input_evidence", {})
    embedded_train_dataset = embedded_training.get("train_dataset", {})
    if embedded_train_dataset.get("sha256") != train_evidence["sha256"]:
        raise ValueError("checkpoint does not bind the supplied training dataset SHA-256")

    provenance = prediction.get("provenance", {})
    if provenance.get("schema") != "tower2_prediction_provenance_v1":
        raise ValueError("prediction JSON lacks tower2_prediction_provenance_v1")
    if provenance.get("checkpoint", {}).get("sha256") != checkpoint_evidence["sha256"]:
        raise ValueError("prediction JSON does not bind the supplied checkpoint SHA-256")
    if provenance.get("evaluation_dataset", {}).get("sha256") != evaluation_evidence["sha256"]:
        raise ValueError("prediction JSON does not bind the supplied evaluation dataset SHA-256")
    if (
        provenance.get("checkpoint_training_input_evidence", {})
        .get("train_dataset", {})
        .get("sha256")
        != train_evidence["sha256"]
    ):
        raise ValueError("prediction provenance does not retain checkpoint training evidence")

    train_ids = dataset_flow_ids(str(train_dataset_path))
    evaluation_ids = dataset_flow_ids(str(evaluation_dataset_path))
    if set(prediction_ids) != set(evaluation_ids):
        raise ValueError("prediction flow IDs do not equal evaluation dataset flow IDs")
    overlap = sorted(set(train_ids) & set(evaluation_ids))
    if overlap:
        raise ValueError(
            f"training/evaluation flow overlap prevents OOF proof: count={len(overlap)} sample={overlap[:5]}"
        )

    return {
        "schema": "oof_teacher_evidence_v1",
        "oof_exclusion_proven": True,
        "proof_scope": "all_prediction_flows_excluded_from_checkpoint_bound_training_dataset",
        "prediction": {
            **file_evidence(prediction_path),
            "flow_count": len(prediction_ids),
            "flow_ids_sha256": canonical_ids_sha256(prediction_ids),
        },
        "checkpoint": checkpoint_evidence,
        "training_dataset": {
            **train_evidence,
            "unique_flow_count": len(train_ids),
            "flow_ids_sha256": canonical_ids_sha256(train_ids),
        },
        "evaluation_dataset": {
            **evaluation_evidence,
            "unique_flow_count": len(evaluation_ids),
            "flow_ids_sha256": canonical_ids_sha256(evaluation_ids),
        },
        "disjointness": {
            "overlap_count": 0,
            "prediction_matches_evaluation_flow_set": True,
        },
        "bindings": {
            "checkpoint_binds_training_dataset": True,
            "prediction_binds_checkpoint": True,
            "prediction_binds_evaluation_dataset": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction_json", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train_dataset", type=Path, required=True)
    parser.add_argument("--evaluation_dataset", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    args = parser.parse_args()
    evidence = verify_oof_teacher(
        args.prediction_json,
        args.checkpoint,
        args.train_dataset,
        args.evaluation_dataset,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(evidence, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_json": str(args.output_json), **evidence["disjointness"]}, indent=2))


if __name__ == "__main__":
    main()
