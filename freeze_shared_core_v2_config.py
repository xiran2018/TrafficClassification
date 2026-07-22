#!/usr/bin/env python3
"""Freeze one cross-dataset packet-core configuration from validation screens."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


SELECTION_DATASETS = {"vpn-app", "tls-120"}
PACKET_TASK_DATASETS = {
    "vpn-app",
    "vpn-binary",
    "vpn-service",
    "tls-120",
    "ustc-app",
    "ustc-binary",
}
FLOW_TASK_DATASETS = {"vpn-app", "tls-120"}
METHOD_DATASETS = PACKET_TASK_DATASETS | FLOW_TASK_DATASETS
# Backward-compatible name for selection-report validators.
REQUIRED_DATASETS = SELECTION_DATASETS
PAIRED_FACTORIAL_FIELDS = {
    "paired_packet_aux_jsonl",
    "paired_consistency_weight",
    "paired_cls_weight",
}
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


def paired_factorial_integrity(
    paired_report: dict[str, Any],
) -> dict[str, Any]:
    """Verify that paired A/B differs only by its declared intervention factors."""
    completion = paired_report["training_completion_evidence"]
    evidence: dict[str, Any] = {}
    passed = True
    missing = object()
    for dataset in sorted(REQUIRED_DATASETS):
        baseline = training_config(
            completion["baseline"]["datasets"][dataset],
            f"paired baseline {dataset}",
        )
        candidate = training_config(
            completion["candidate"]["datasets"][dataset],
            f"paired candidate {dataset}",
        )
        keys = sorted((set(baseline) | set(candidate)) - PAIRED_FACTORIAL_FIELDS)
        mismatched = [
            key
            for key in keys
            if baseline.get(key, missing) != candidate.get(key, missing)
        ]
        dataset_passed = not mismatched
        passed = passed and dataset_passed
        evidence[dataset] = {
            "status": "pass" if dataset_passed else "fail",
            "mismatched_fields": mismatched,
        }
    return {
        "required": True,
        "status": "pass" if passed else "fail",
        "declared_factorial_fields": sorted(PAIRED_FACTORIAL_FIELDS),
        "datasets": evidence,
    }
def selected_class_weight_config(balance_report: dict[str, Any]) -> dict[str, Any]:
    selected_role = balance_report["selected"]
    rows = balance_report["training_completion_evidence"][selected_role]["datasets"]
    observed = {
        (
            training_config(row, f"balance {selected_role} {dataset}").get(
                "class_weight_basis"
            ),
            float(
                training_config(row, f"balance {selected_role} {dataset}").get(
                    "class_weight_strength", -1.0
                )
            ),
        )
        for dataset, row in rows.items()
    }
    if len(observed) != 1:
        raise ValueError("balance-selected class-weight protocol differs by dataset")
    basis, strength = observed.pop()
    allowed = {("packet", 1.0), ("flow", 0.5), ("flow", 1.0)}
    if (basis, strength) not in allowed:
        raise ValueError(
            f"unsupported balance-selected class-weight protocol: {(basis, strength)}"
        )
    multi_arm = balance_report.get("multi_arm_selection")
    if multi_arm is not None:
        factorial_integrity = multi_arm.get("factorial_config_integrity") or {}
        if not (
            factorial_integrity.get("required") is True
            and factorial_integrity.get("status") == "pass"
        ):
            raise ValueError(
                "multi-arm class-weight selection lacks passing factorial integrity"
            )
    declared = (multi_arm or {}).get("selected_config")
    if declared is not None and not (
        declared.get("class_weight_basis") == basis
        and math.isclose(
            float(declared.get("class_weight_strength", -1.0)),
            strength,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise ValueError("multi-arm selected_config disagrees with selected artifacts")
    return {"class_weight_basis": basis, "class_weight_strength": strength}


def validate_hierarchy_selection(
    hierarchy_report: dict[str, Any],
    balance_report: dict[str, Any],
    *,
    hierarchy_path: Path,
    balance_path: Path,
) -> dict[str, dict[str, Any]]:
    shared_algorithm = hierarchy_report.get("shared_algorithm")
    bounded_algorithm = shared_algorithm == (
        "bounded_effective_flow_class_risk_power_eta"
    )
    if not (
        hierarchy_report.get("schema")
        == "hierarchy_adaptive_class_weight_selection_v1"
        and hierarchy_report.get("status")
        == "frozen_numeric_hyperparameters_from_validation"
        and hierarchy_report.get("selection_scope") == SELECTION_SCOPE
        and hierarchy_report.get("test_labels_used") is False
        and shared_algorithm
        in {
            "normalized_effective_flow_class_risk_power_eta",
            "bounded_effective_flow_class_risk_power_eta",
        }
    ):
        raise ValueError("invalid hierarchy class-weight selection")
    if bounded_algorithm:
        numeric = hierarchy_report.get("numeric_protocol_selection") or {}
        decisions = numeric.get("datasets") or {}
        if not (
            numeric.get("schema") == "bounded_hierarchy_risk_selection_v1"
            and numeric.get("selected") == "candidate"
            and numeric.get("candidate_promoted_for_all_datasets") is True
            and numeric.get("promotion_scope")
            == "same_bounded_risk_algorithm_must_pass_every_dataset"
            and math.isclose(
                float(numeric.get("maximum_drop", -1.0)),
                0.003,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            and set(decisions) == REQUIRED_DATASETS
            and all(row.get("passes") is True for row in decisions.values())
        ):
            raise ValueError("bounded hierarchy rule did not pass its global gate")
    class_input = (hierarchy_report.get("inputs") or {}).get(
        "class_weight_selection"
    ) or {}
    if not (
        Path(str(class_input.get("path") or "")).resolve() == balance_path.resolve()
        and class_input.get("sha256") == file_sha256(balance_path)
    ):
        raise ValueError("hierarchy selection is not bound to balance evidence")
    datasets = hierarchy_report.get("datasets") or {}
    if set(datasets) != REQUIRED_DATASETS:
        raise ValueError("hierarchy selection must cover exactly VPN and TLS-120")
    expected_trainer = balance_report["multi_arm_selection"][
        "all_arm_training_implementation_consistency"
    ]["trainer_source_sha256"]
    output: dict[str, dict[str, Any]] = {}
    for dataset, row in datasets.items():
        eta = float(row.get("selected_eta", -1.0))
        if bounded_algorithm:
            eta_supported = 0.0 <= eta <= 1.0
        else:
            eta_supported = eta in {0.0, 0.25, 0.5, 1.0}
        if not eta_supported:
            raise ValueError(f"unsupported hierarchy eta for {dataset}: {eta}")
        parameters = row.get("training_hyperparameters") or {}
        if not (
            parameters.get("class_weight_basis") == "flow"
            and math.isclose(
                float(parameters.get("class_weight_strength", -1.0)),
                eta,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError(f"hierarchy parameters disagree for {dataset}")
        arms = row.get("arms") or {}
        if bounded_algorithm:
            selected_arm = arms.get("bounded_risk") or {}
            decision = hierarchy_report["numeric_protocol_selection"]["datasets"][
                dataset
            ]
            if not (
                selected_arm.get("eligible") is True
                and selected_arm.get("noninferiority_passed") is True
                and decision.get("passes") is True
                and math.isclose(
                    float(selected_arm.get("eta", -1.0)),
                    eta,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            ):
                raise ValueError(
                    f"bounded hierarchy eta is not globally eligible for {dataset}"
                )
        else:
            selected_arm = arms.get(str(eta)) or {}
            if selected_arm.get("eligible") is not True:
                raise ValueError(f"selected hierarchy eta is ineligible for {dataset}")
            eligible = [item for item in arms.values() if item.get("eligible") is True]
            recomputed = max(
                eligible,
                key=lambda item: (
                    float(item["macro_f1"]),
                    float(item["accuracy"]),
                    -float(item["eta"]),
                ),
            )
            if not math.isclose(float(recomputed["eta"]), eta, abs_tol=1e-12):
                raise ValueError(f"hierarchy selection ranking disagrees for {dataset}")
        evidence = row.get("selected_training_evidence") or {}
        metric = row.get("selected_validation_metric") or {}
        if not (
            evidence.get("trainer_source_stable_through_completion") is True
            and evidence.get("trainer_source_sha256") == expected_trainer
            and evidence.get("completion_trainer_source_sha256") == expected_trainer
            and evidence.get("metric_path") == metric.get("path")
            and evidence.get("metric_sha256") == metric.get("sha256")
        ):
            raise ValueError(f"hierarchy training evidence disagrees for {dataset}")
        for path_field, hash_field in (
            ("metric_path", "metric_sha256"),
            ("final_checkpoint_path", "final_checkpoint_sha256"),
            ("validation_history_path", "validation_history_sha256"),
            ("provenance_path", "provenance_sha256"),
        ):
            path = Path(str(evidence.get(path_field) or ""))
            if not path.is_file() or file_sha256(path) != evidence.get(hash_field):
                raise ValueError(
                    f"hierarchy {dataset} evidence no longer matches {path_field}"
                )
        observed_config = training_config(
            evidence, f"hierarchy selected evidence {dataset}"
        )
        expected_basis = "packet" if eta == 0.0 and not bounded_algorithm else "flow"
        expected_strength = 1.0 if eta == 0.0 and not bounded_algorithm else eta
        require_config(
            observed_config,
            {
                "class_weight_basis": expected_basis,
                "class_weight_strength": expected_strength,
            },
            f"hierarchy selected evidence {dataset}",
        )
        output[dataset] = {
            "class_weight_basis": "flow",
            "class_weight_strength": eta,
            "selected_training_evidence": evidence,
        }
    return output


def validate_task_hierarchy_derivation(
    derivation: dict[str, Any],
    *,
    derivation_path: Path,
    required_datasets: set[str],
    task_label: str,
) -> dict[str, dict[str, Any]]:
    """Validate task-specific numeric values derived only from task Train counts."""
    if not (
        derivation.get("schema") == "bounded_hierarchy_risk_protocol_v1"
        and derivation.get("status") == "derived_from_training_counts_only"
        and derivation.get("selection_role")
        == "candidate_numeric_derivation_not_model_selection"
        and derivation.get("test_labels_used") is False
        and derivation.get("shared_algorithm")
        == "largest_flow_risk_power_subject_to_max_min_ratio"
        and math.isclose(
            float(derivation.get("max_weight_ratio", -1.0)),
            4.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and math.isclose(
            float(derivation.get("effective_number_beta", -1.0)),
            0.9999,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise ValueError(f"invalid {task_label} bounded hierarchy derivation")
    datasets = derivation.get("datasets") or {}
    if set(datasets) != required_datasets:
        raise ValueError(
            f"{task_label} hierarchy derivation must cover exactly "
            f"{sorted(required_datasets)}"
        )
    output: dict[str, dict[str, Any]] = {}
    for dataset, row in datasets.items():
        strength = float(row.get("class_weight_strength", -1.0))
        realized_ratio = float(row.get("bounded_effective_weight_ratio", -1.0))
        source = row.get("input") or {}
        source_path = Path(str(source.get("path") or ""))
        if not (
            row.get("class_weight_basis") == "flow"
            and 0.0 <= strength <= 1.0
            and 0.0 < realized_ratio <= 4.0 + 1e-9
            and source_path.is_file()
            and file_sha256(source_path) == source.get("sha256")
        ):
            raise ValueError(f"invalid {task_label} hierarchy derivation for {dataset}")
        source_payload = json.loads(source_path.read_text(encoding="utf-8"))
        if not (
            source_payload.get("schema") == "class_sampling_hierarchy_analysis_v1"
            and source_payload.get("selection_role")
            == "train_only_reporting_not_model_selection"
            and source_payload.get("test_labels_used") is False
        ):
            raise ValueError(
                f"{task_label} hierarchy source is not train-only for {dataset}"
            )
        output[dataset] = {
            "class_weight_basis": "flow",
            "class_weight_strength": strength,
        }
    if not derivation_path.is_file():
        raise ValueError(f"{task_label} hierarchy derivation evidence is missing")
    return output


def validate_experiment_chain(
    balance_report: dict[str, Any],
    paired_report: dict[str, Any],
    hierarchy_weights: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    common = {
        "packet_batch_scheduler": "epoch_resampled_dataloader_v1",
        "class_weighting": "effective",
        "disable_packet_information_weights": True,
        "flow_balanced_packet_batches": True,
        "packets_per_flow": 2,
        "no_sft": True,
    }
    balance_completion = balance_report["training_completion_evidence"]
    candidate_configs = {
        (
            training_config(row, f"balance candidate {dataset}").get(
                "class_weight_basis"
            ),
            float(
                training_config(row, f"balance candidate {dataset}").get(
                    "class_weight_strength", -1.0
                )
            ),
        )
        for dataset, row in balance_completion["candidate"]["datasets"].items()
    }
    if len(candidate_configs) != 1:
        raise ValueError("balance candidate class-weight protocol differs by dataset")
    candidate_basis, candidate_strength = candidate_configs.pop()
    if candidate_basis != "flow" or candidate_strength not in {0.5, 1.0}:
        raise ValueError("balance candidate must be a declared flow-weight protocol")
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
            "class_weight_basis": candidate_basis,
            "class_weight_strength": candidate_strength,
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

    paired_completion = paired_report["training_completion_evidence"]
    paired_baseline_rows = paired_completion["baseline"]["datasets"]
    if hierarchy_weights is None:
        selected_role = balance_report["selected"]
        selected_rows = balance_completion[selected_role]["datasets"]
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
        selected = selected_class_weight_config(balance_report)
        hierarchy_weights = {dataset: selected for dataset in REQUIRED_DATASETS}

    for dataset in sorted(REQUIRED_DATASETS):
        selected = hierarchy_weights[dataset]
        base_expected = {
            **common,
            "class_weight_basis": selected["class_weight_basis"],
            "class_weight_strength": selected["class_weight_strength"],
            "paired_consistency_weight": 0.0,
            "paired_cls_weight": 0.0,
        }
        require_config(
            training_config(paired_baseline_rows[dataset], f"paired baseline {dataset}"),
            base_expected,
            f"paired baseline {dataset}",
        )
        paired_expected = {
            **base_expected,
            "paired_consistency_weight": 0.05,
            "paired_cls_weight": 0.2,
            "paired_logit_kl_weight": 0.5,
            "paired_raw_consistency_weight": 1.0,
        }
        row = paired_completion["candidate"]["datasets"][dataset]
        config = training_config(row, f"paired candidate {dataset}")
        require_config(config, paired_expected, f"paired candidate {dataset}")
        if not config.get("paired_packet_aux_jsonl"):
            raise ValueError(
                f"paired candidate {dataset} has no paired intervention input"
            )
    factorial_integrity = paired_factorial_integrity(paired_report)
    if factorial_integrity["status"] != "pass":
        raise ValueError(
            "paired A/B differs outside the declared intervention factors"
        )
    return factorial_integrity


def freeze_config(
    balance_report: dict[str, Any],
    paired_report: dict[str, Any] | None,
    *,
    balance_path: Path,
    paired_path: Path | None,
    hierarchy_report: dict[str, Any] | None = None,
    hierarchy_path: Path | None = None,
    packet_hierarchy_derivation: dict[str, Any] | None = None,
    packet_hierarchy_derivation_path: Path | None = None,
    flow_hierarchy_derivation: dict[str, Any] | None = None,
    flow_hierarchy_derivation_path: Path | None = None,
) -> dict[str, Any]:
    validate_selection(balance_report, "balance selection")
    if paired_report is not None:
        if paired_path is None:
            raise ValueError("paired_path is required with paired_report")
        validate_selection(paired_report, "paired selection")
    hierarchy_weights = None
    if hierarchy_report is not None:
        if hierarchy_path is None:
            raise ValueError("hierarchy_path is required with hierarchy_report")
        hierarchy_weights = validate_hierarchy_selection(
            hierarchy_report,
            balance_report,
            hierarchy_path=hierarchy_path,
            balance_path=balance_path,
        )
        if hierarchy_report.get("shared_algorithm") == (
            "bounded_effective_flow_class_risk_power_eta"
        ) and (
            packet_hierarchy_derivation is None
            or flow_hierarchy_derivation is None
        ):
            raise ValueError(
                "bounded hierarchy requires Train-only derivations for all Packet "
                "and Flow task datasets"
            )
    packet_hierarchy_weights = hierarchy_weights
    flow_hierarchy_weights = hierarchy_weights
    if hierarchy_weights is not None:
        balance_selected = selected_class_weight_config(balance_report)
        fallback_strength = (
            0.0
            if balance_selected["class_weight_basis"] == "packet"
            else float(balance_selected["class_weight_strength"])
        )
        packet_hierarchy_weights = {
            dataset: (
                hierarchy_weights[dataset]
                if dataset in hierarchy_weights
                else {
                    "class_weight_basis": "flow",
                    "class_weight_strength": fallback_strength,
                    "numeric_parameter_source": (
                        "cross_dataset_balance_gate_global_fallback"
                    ),
                }
            )
            for dataset in PACKET_TASK_DATASETS
        }
    if packet_hierarchy_derivation is not None:
        if hierarchy_report is None or hierarchy_report.get("shared_algorithm") != (
            "bounded_effective_flow_class_risk_power_eta"
        ):
            raise ValueError(
                "Packet-task bounded derivation requires the promoted bounded hierarchy rule"
            )
        if packet_hierarchy_derivation_path is None:
            raise ValueError("packet_hierarchy_derivation_path is required")
        packet_hierarchy_weights = validate_task_hierarchy_derivation(
            packet_hierarchy_derivation,
            derivation_path=packet_hierarchy_derivation_path,
            required_datasets=PACKET_TASK_DATASETS,
            task_label="Packet-task",
        )
        for dataset in sorted(SELECTION_DATASETS):
            if not math.isclose(
                float(packet_hierarchy_weights[dataset]["class_weight_strength"]),
                float(hierarchy_weights[dataset]["class_weight_strength"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    f"Packet-task train-only eta disagrees with validated eta for {dataset}"
                )
    if flow_hierarchy_derivation is not None:
        if hierarchy_report is None or hierarchy_report.get("shared_algorithm") != (
            "bounded_effective_flow_class_risk_power_eta"
        ):
            raise ValueError(
                "Flow-task bounded derivation requires the promoted bounded hierarchy rule"
            )
        if flow_hierarchy_derivation_path is None:
            raise ValueError("flow_hierarchy_derivation_path is required")
        flow_hierarchy_weights = validate_task_hierarchy_derivation(
            flow_hierarchy_derivation,
            derivation_path=flow_hierarchy_derivation_path,
            required_datasets=FLOW_TASK_DATASETS,
            task_label="Flow-task",
        )
    paired_factorial = (
        validate_experiment_chain(balance_report, paired_report, hierarchy_weights)
        if paired_report is not None
        else None
    )

    selected_weight = (
        {"class_weight_basis": "flow", "class_weight_strength": 1.0}
        if hierarchy_weights is not None
        else selected_class_weight_config(balance_report)
    )
    use_paired_invariance = bool(
        paired_report is not None and paired_report["selected"] == "candidate"
    )
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
        "class_weight_basis": selected_weight["class_weight_basis"],
        "class_weight_strength": selected_weight["class_weight_strength"],
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
        "datasets": sorted(METHOD_DATASETS),
        "task_datasets": {
            "packet-level-classification": sorted(PACKET_TASK_DATASETS),
            "flow-level-classification": sorted(FLOW_TASK_DATASETS),
        },
        "tasks": ["packet-level-classification", "flow-level-classification"],
        "selection_protocol": {
            "scope": SELECTION_SCOPE,
            "selection_datasets": sorted(SELECTION_DATASETS),
            "application_datasets_by_task": {
                "packet-level-classification": sorted(PACKET_TASK_DATASETS),
                "flow-level-classification": sorted(FLOW_TASK_DATASETS),
            },
            "metric": SELECTION_METRIC,
            "min_macro_f1_delta": MIN_MACRO_F1_DELTA,
            "max_accuracy_drop": MAX_ACCURACY_DROP,
            "same_candidate_required_on_every_dataset": True,
            "dataset_specific_numeric_hyperparameters_allowed": True,
            "dataset_specific_algorithm_choices_allowed": False,
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
        "dataset_numeric_hyperparameter_overrides": (
            {
                "packet-level-classification": {
                    dataset: {
                        "class_weight_strength": row["class_weight_strength"]
                    }
                    for dataset, row in sorted(packet_hierarchy_weights.items())
                },
                "flow-level-classification": {
                    dataset: {
                        "class_weight_strength": row["class_weight_strength"]
                    }
                    for dataset, row in sorted(flow_hierarchy_weights.items())
                },
            }
            if hierarchy_weights is not None
            else {}
        ),
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
            "paired_invariance": (
                {
                    "path": str(paired_path),
                    "sha256": file_sha256(paired_path),
                    "selected": paired_report["selected"],
                    "datasets": paired_report["datasets"],
                    "factorial_config_integrity": paired_factorial,
                }
                if paired_report is not None and paired_path is not None
                else {
                    "selected": "disabled",
                    "reason": "excluded_from_minimal_shared_core",
                    "validation_role": "future_optional_ablation",
                }
            ),
        },
    }
    if hierarchy_report is not None and hierarchy_path is not None:
        payload["selection_evidence"]["hierarchy_class_weight"] = {
            "path": str(hierarchy_path),
            "sha256": file_sha256(hierarchy_path),
            "shared_algorithm": hierarchy_report["shared_algorithm"],
            "datasets": hierarchy_report["datasets"],
        }
    if (
        packet_hierarchy_derivation is not None
        and packet_hierarchy_derivation_path is not None
    ):
        payload["selection_evidence"]["packet_task_hierarchy_derivation"] = {
            "path": str(packet_hierarchy_derivation_path),
            "sha256": file_sha256(packet_hierarchy_derivation_path),
            "shared_algorithm": packet_hierarchy_derivation["shared_algorithm"],
            "datasets": packet_hierarchy_derivation["datasets"],
            "test_labels_used": False,
        }
    if (
        flow_hierarchy_derivation is not None
        and flow_hierarchy_derivation_path is not None
    ):
        payload["selection_evidence"]["flow_task_hierarchy_derivation"] = {
            "path": str(flow_hierarchy_derivation_path),
            "sha256": file_sha256(flow_hierarchy_derivation_path),
            "shared_algorithm": flow_hierarchy_derivation["shared_algorithm"],
            "datasets": flow_hierarchy_derivation["datasets"],
            "test_labels_used": False,
        }
    payload["config_sha256"] = canonical_sha256(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--balance_selection", required=True)
    parser.add_argument("--paired_selection", default="")
    parser.add_argument("--hierarchy_selection", default="")
    parser.add_argument("--packet_hierarchy_derivation", default="")
    parser.add_argument("--flow_hierarchy_derivation", default="")
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    balance_path = Path(args.balance_selection)
    paired_path = Path(args.paired_selection) if args.paired_selection else None
    balance = json.loads(balance_path.read_text(encoding="utf-8"))
    paired = (
        json.loads(paired_path.read_text(encoding="utf-8"))
        if paired_path is not None
        else None
    )
    hierarchy_path = Path(args.hierarchy_selection) if args.hierarchy_selection else None
    hierarchy = (
        json.loads(hierarchy_path.read_text(encoding="utf-8"))
        if hierarchy_path is not None
        else None
    )
    flow_derivation_path = (
        Path(args.flow_hierarchy_derivation)
        if args.flow_hierarchy_derivation
        else None
    )
    flow_derivation = (
        json.loads(flow_derivation_path.read_text(encoding="utf-8"))
        if flow_derivation_path is not None
        else None
    )
    packet_derivation_path = (
        Path(args.packet_hierarchy_derivation)
        if args.packet_hierarchy_derivation
        else None
    )
    packet_derivation = (
        json.loads(packet_derivation_path.read_text(encoding="utf-8"))
        if packet_derivation_path is not None
        else None
    )
    payload = freeze_config(
        balance,
        paired,
        balance_path=balance_path,
        paired_path=paired_path,
        hierarchy_report=hierarchy,
        hierarchy_path=hierarchy_path,
        packet_hierarchy_derivation=packet_derivation,
        packet_hierarchy_derivation_path=packet_derivation_path,
        flow_hierarchy_derivation=flow_derivation,
        flow_hierarchy_derivation_path=flow_derivation_path,
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
