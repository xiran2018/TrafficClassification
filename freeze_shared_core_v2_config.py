#!/usr/bin/env python3
"""Freeze one cross-dataset packet-core configuration from validation screens."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


REQUIRED_DATASETS = {"vpn-app", "tls-120"}
SELECTION_SCOPE = "heldout_validation_only"
SELECTION_METRIC = "macro_f1_with_accuracy_guard"
MIN_MACRO_F1_DELTA = 0.005
MAX_ACCURACY_DROP = 0.005


def valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        char in "0123456789abcdef" for char in value
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_selection(report: dict[str, Any], name: str) -> None:
    if report.get("selection_scope") != SELECTION_SCOPE:
        raise ValueError(f"{name} must use {SELECTION_SCOPE}")
    datasets = set((report.get("datasets") or {}).keys())
    if datasets != REQUIRED_DATASETS:
        raise ValueError(
            f"{name} must contain exactly {sorted(REQUIRED_DATASETS)}, got {sorted(datasets)}"
        )
    if report.get("selected") not in {"baseline", "candidate"}:
        raise ValueError(f"{name} has no valid selected configuration")
    if report.get("promotion_scope") != "same_candidate_must_pass_every_dataset":
        raise ValueError(f"{name} does not enforce cross-dataset promotion")
    if report.get("metric") != SELECTION_METRIC:
        raise ValueError(f"{name} must use {SELECTION_METRIC}")
    if float(report.get("min_delta", -1.0)) != MIN_MACRO_F1_DELTA:
        raise ValueError(
            f"{name} must require macro-F1 delta {MIN_MACRO_F1_DELTA}"
        )
    if float(report.get("max_accuracy_drop", -1.0)) != MAX_ACCURACY_DROP:
        raise ValueError(
            f"{name} must cap accuracy drop at {MAX_ACCURACY_DROP}"
        )
    recomputed_passes = {}
    for dataset, row in (report.get("datasets") or {}).items():
        macro_passes = float(row["delta_macro_f1"]) >= MIN_MACRO_F1_DELTA
        accuracy_passes = float(row["delta_accuracy"]) >= -MAX_ACCURACY_DROP
        recomputed = macro_passes and accuracy_passes
        if not (
            row.get("macro_f1_passes") is macro_passes
            and row.get("accuracy_guard_passes") is accuracy_passes
            and row.get("passes") is recomputed
        ):
            raise ValueError(f"{name} {dataset} has inconsistent promotion flags")
        recomputed_passes[dataset] = recomputed
    promoted = bool(recomputed_passes) and all(recomputed_passes.values())
    if not (
        report.get("candidate_promoted_for_all_datasets") is promoted
        and report.get("selected") == ("candidate" if promoted else "baseline")
    ):
        raise ValueError(f"{name} selected field disagrees with its dual-metric gate")
    completion = report.get("training_completion_evidence") or {}
    if set(completion) != {"baseline", "candidate"}:
        raise ValueError(f"{name} is missing baseline/candidate training completion evidence")
    for role, evidence in completion.items():
        if evidence.get("required") is not True or evidence.get("status") != "pass":
            raise ValueError(f"{name} {role} training completion evidence did not pass")
        if int(evidence.get("required_validation_points", 0)) < 8:
            raise ValueError(f"{name} {role} used fewer than eight validation points")
        if evidence.get("required_packet_batch_scheduler") != (
            "epoch_resampled_dataloader_v1"
        ):
            raise ValueError(
                f"{name} {role} did not require the strict epoch-resampled scheduler"
            )
        rows = evidence.get("datasets") or {}
        if set(rows) != REQUIRED_DATASETS or not all(
            row.get("passed") is True
            and row.get("best_metric_matches_history") is True
            and row.get("provenance_verified") is True
            and row.get("packet_batch_scheduler")
            == "epoch_resampled_dataloader_v1"
            and int(row.get("validation_points", 0))
            == int(evidence.get("required_validation_points", 0))
            for row in rows.values()
        ):
            raise ValueError(
                f"{name} {role} completion evidence does not cover VPN/TLS "
                "with history-consistent best metrics"
            )
        required_hashes = (
            "metric_sha256",
            "final_checkpoint_sha256",
            "validation_history_sha256",
            "provenance_sha256",
        )
        if any(
            not all(valid_sha256(row.get(field)) for field in required_hashes)
            for row in rows.values()
        ):
            raise ValueError(f"{name} {role} completion evidence is missing SHA-256 hashes")
        artifact_bindings = (
            ("metric_path", "metric_sha256"),
            ("final_checkpoint_path", "final_checkpoint_sha256"),
            ("validation_history_path", "validation_history_sha256"),
            ("provenance_path", "provenance_sha256"),
        )
        for dataset, row in rows.items():
            for path_field, hash_field in artifact_bindings:
                raw_path = row.get(path_field)
                if not isinstance(raw_path, str) or not raw_path:
                    raise ValueError(
                        f"{name} {role} {dataset} is missing {path_field}"
                    )
                path = Path(raw_path)
                if not path.is_file() or file_sha256(path) != row[hash_field]:
                    raise ValueError(
                        f"{name} {role} {dataset} {path_field} no longer "
                        "matches its recorded SHA-256"
                    )

    implementation = report.get("training_implementation_consistency") or {}
    expected_runs = len(REQUIRED_DATASETS) * 2
    trainer_sha256 = implementation.get("trainer_source_sha256")
    if not (
        implementation.get("required") is True
        and implementation.get("status") == "pass"
        and int(implementation.get("num_runs", 0)) == expected_runs
        and implementation.get("all_runs_stable_through_completion") is True
        and valid_sha256(trainer_sha256)
    ):
        raise ValueError(
            f"{name} does not prove one stable trainer source across all runs"
        )
    completion_rows = [
        row
        for evidence in completion.values()
        for row in (evidence.get("datasets") or {}).values()
    ]
    if not all(
        row.get("trainer_source_stable_through_completion") is True
        and row.get("trainer_source_sha256") == trainer_sha256
        and row.get("completion_trainer_source_sha256") == trainer_sha256
        for row in completion_rows
    ):
        raise ValueError(
            f"{name} completion rows disagree with the stable trainer source"
        )


def training_config(row: dict[str, Any], context: str) -> dict[str, Any]:
    path = Path(str(row.get("provenance_path") or ""))
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        config = payload["training_config"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{context} has no readable Tower1 training config") from exc
    if payload.get("schema") != "tower1_training_contract_v1":
        raise ValueError(f"{context} is not a Tower1 training contract")
    return config


def require_config(
    config: dict[str, Any], expected: dict[str, Any], context: str
) -> None:
    for key, wanted in expected.items():
        observed = config.get(key)
        if isinstance(wanted, float):
            matches = isinstance(observed, (int, float)) and math.isclose(
                float(observed), wanted, rel_tol=0.0, abs_tol=1e-12
            )
        else:
            matches = observed == wanted
        if not matches:
            raise ValueError(
                f"{context} training config mismatch for {key}: "
                f"observed={observed!r}, expected={wanted!r}"
            )


def validate_experiment_chain(
    balance_report: dict[str, Any], paired_report: dict[str, Any]
) -> None:
    common = {
        "packet_batch_scheduler": "epoch_resampled_dataloader_v1",
        "class_weighting": "effective",
        "disable_packet_information_weights": True,
        "flow_balanced_packet_batches": True,
        "packets_per_flow": 2,
        "no_sft": True,
    }
    balance_completion = balance_report["training_completion_evidence"]
    expected_balance = {
        "baseline": {
            **common,
            "class_weight_basis": "packet",
            "class_weight_strength": 1.0,
            "paired_consistency_weight": 0.0,
            "paired_cls_weight": 0.0,
        },
        "candidate": {
            **common,
            "class_weight_basis": "flow",
            "class_weight_strength": 0.5,
            "paired_consistency_weight": 0.0,
            "paired_cls_weight": 0.0,
        },
    }
    for role, expected in expected_balance.items():
        for dataset, row in balance_completion[role]["datasets"].items():
            require_config(
                training_config(row, f"balance {role} {dataset}"),
                expected,
                f"balance {role} {dataset}",
            )

    selected_role = balance_report["selected"]
    selected_rows = balance_completion[selected_role]["datasets"]
    paired_completion = paired_report["training_completion_evidence"]
    paired_baseline_rows = paired_completion["baseline"]["datasets"]
    identity_fields = (
        "metric_sha256",
        "final_checkpoint_sha256",
        "validation_history_sha256",
        "provenance_sha256",
    )
    for dataset in sorted(REQUIRED_DATASETS):
        if any(
            paired_baseline_rows[dataset].get(field)
            != selected_rows[dataset].get(field)
            for field in identity_fields
        ):
            raise ValueError(
                "paired selection baseline is not the artifact-identical "
                f"balance-selected run for {dataset}"
            )

    selected_basis = "flow" if selected_role == "candidate" else "packet"
    selected_strength = 0.5 if selected_role == "candidate" else 1.0
    paired_expected = {
        **common,
        "class_weight_basis": selected_basis,
        "class_weight_strength": selected_strength,
        "paired_consistency_weight": 0.05,
        "paired_cls_weight": 0.2,
        "paired_logit_kl_weight": 0.5,
        "paired_raw_consistency_weight": 1.0,
    }
    for dataset, row in paired_completion["candidate"]["datasets"].items():
        config = training_config(row, f"paired candidate {dataset}")
        require_config(config, paired_expected, f"paired candidate {dataset}")
        if not config.get("paired_packet_aux_jsonl"):
            raise ValueError(
                f"paired candidate {dataset} has no paired intervention input"
            )


def freeze_config(
    balance_report: dict[str, Any],
    paired_report: dict[str, Any],
    *,
    balance_path: Path,
    paired_path: Path,
) -> dict[str, Any]:
    validate_selection(balance_report, "balance selection")
    validate_selection(paired_report, "paired selection")
    validate_experiment_chain(balance_report, paired_report)

    use_flow_weights = balance_report["selected"] == "candidate"
    use_paired_invariance = paired_report["selected"] == "candidate"
    tower1 = {
        "base_model": "Qwen/Qwen2.5-7B-Instruct",
        "epochs": 8,
        "max_steps": 0,
        "packet_batch_size": 16,
        "gradient_accumulation_steps": 1,
        "gradient_checkpointing": True,
        "max_packet_length": 1024,
        "projection_dim": 256,
        "cls_weight": 1.0,
        "contrastive_weight": 0.1,
        "same_flow_positive_weight": 1.0,
        "same_label_positive_weight": 1.0,
        "flow_proto_weight": 0.0,
        "flow_proto_positive": "same_class",
        "flow_proto_context": "inclusive",
        "temperature": 0.07,
        "learning_rate": 1e-5,
        "head_learning_rate": 1e-4,
        "weight_decay": 0.01,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "dtype": "float16",
        "seed": 42,
        "use_sft": False,
        "disable_packet_information_weights": True,
        "flow_balanced_packet_batches": True,
        "packets_per_flow": 2,
        "packet_batch_scheduler": "epoch_resampled_dataloader_v1",
        "class_weighting": "effective",
        "class_weight_beta": 0.9999,
        "class_weight_basis": "flow" if use_flow_weights else "packet",
        "class_weight_strength": 0.5 if use_flow_weights else 1.0,
        "paired_consistency_weight": 0.05 if use_paired_invariance else 0.0,
        "paired_cls_weight": 0.2 if use_paired_invariance else 0.0,
        "paired_logit_kl_weight": 0.5,
        "paired_raw_consistency_weight": 1.0,
        "identity_safe_contrastive": False,
        "cross_scale_weight": 0.0,
        "cross_scale_temperature": 0.07,
        "early_stop_patience": 0,
        "init_checkpoint_dir": "",
        "init_adapter_only": False,
    }
    payload: dict[str, Any] = {
        "schema": "exact_shared_packet_core_v2",
        "status": "frozen_from_cross_dataset_validation",
        "datasets": sorted(REQUIRED_DATASETS),
        "tasks": ["packet-level-classification", "flow-level-classification"],
        "selection_protocol": {
            "scope": SELECTION_SCOPE,
            "metric": SELECTION_METRIC,
            "min_macro_f1_delta": MIN_MACRO_F1_DELTA,
            "max_accuracy_drop": MAX_ACCURACY_DROP,
            "same_candidate_required_on_every_dataset": True,
            "test_labels_used": False,
        },
        "packet_core": {
            "encoder_class": "ProtocolAwarePacketContentEncoder",
            "max_bytes": 128,
            "hidden_dim": 128,
            "num_layers": 2,
            "num_heads": 4,
            "dropout": 0.1,
            "num_field_types": 9,
            "representation_encoder": "SharedPacketRepresentationEncoder",
            "semantic_packet_context_policy": "single_packet",
            "mask_protocol_session_fields": True,
            "structural_dim": 13,
            "channel_fusion_base_mode": "semantic_anchor",
            "channel_fusion_max_weight": 0.25,
            "intervention_view_base_mode": "symmetric_mean",
            "intervention_max_residual_weight": 0.25,
        },
        "native_pretraining": {
            "protocol": "native_flow_multitask_v1",
            "max_packets": 64,
            "flow_layers": 2,
            "projection_dim": 128,
            "epochs": 20,
            "batch_size": 8,
            "eval_batch_size": 16,
            "num_workers": 0,
            "learning_rate": 0.0003,
            "weight_decay": 0.01,
            "field_mask_probability": 0.2,
            "payload_dropout_probability": 0.5,
            "session_mask_probability": 1.0,
            "masked_byte_weight": 1.0,
            "relative_order_weight": 0.25,
            "same_flow_weight": 0.25,
            "next_length_weight": 0.2,
            "next_iat_weight": 0.2,
            "direction_weight": 0.1,
            "packet_consistency_weight": 0.25,
            "flow_contrastive_weight": 0.25,
            "temperature": 0.1,
            "patience": 4,
            "seed": 42,
        },
        "empirical_risk": {
            "content_group_loss_reduction": "group_mean",
        },
        "tower1": tower1,
        "embedding_extraction": {
            "scheduler": "cross_flow_length_bucketed_v1",
            "embedding_mode": "concat",
            "batch_size": 8,
            "flow_batch_packets": 128,
        },
        "task_contract": {
            "packet": "shared_packet_core_plus_packet_head",
            "flow": "shared_packet_core_plus_sequence_window_aggregator_and_flow_head",
            "shared_packet_module_reuse": "architecture_and_representation_contract_only",
            "packet_task_training_source": "packet_task_train_split_packets",
            "flow_task_training_source": "flow_task_train_split_packets",
            "cross_task_supervised_weights_reused": False,
            "dataset_specific_manual_options": False,
            "separately_learned_model_gate_and_expert_weights": True,
            "shared_method_signature_required": True,
            "independent_numeric_training_hyperparameters_allowed": True,
            "objective_activation_topology_must_match": True,
        },
        "selection_evidence": {
            "balance": {
                "path": str(balance_path),
                "sha256": file_sha256(balance_path),
                "selected": balance_report["selected"],
                "datasets": balance_report["datasets"],
            },
            "paired_invariance": {
                "path": str(paired_path),
                "sha256": file_sha256(paired_path),
                "selected": paired_report["selected"],
                "datasets": paired_report["datasets"],
            },
        },
    }
    payload["config_sha256"] = canonical_sha256(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--balance_selection", required=True)
    parser.add_argument("--paired_selection", required=True)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    balance_path = Path(args.balance_selection)
    paired_path = Path(args.paired_selection)
    balance = json.loads(balance_path.read_text(encoding="utf-8"))
    paired = json.loads(paired_path.read_text(encoding="utf-8"))
    payload = freeze_config(
        balance,
        paired,
        balance_path=balance_path,
        paired_path=paired_path,
    )
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "tower1": payload["tower1"],
                "config_sha256": payload["config_sha256"],
                "output_json": str(output),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
