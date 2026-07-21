import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_named_paths(values: list[str]) -> Dict[str, Path]:
    result: Dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected DATASET=PATH, got {value}")
        dataset, path = value.split("=", 1)
        if not dataset or dataset in result:
            raise ValueError(f"Invalid or duplicate dataset key: {dataset}")
        result[dataset] = Path(path)
    return result


def load_macro_f1(path: Path) -> float:
    return load_validation_metrics(path)["macro_f1"]


def load_validation_metrics(path: Path) -> dict[str, float]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    metrics = {
        "accuracy": float(payload["metrics"]["accuracy"]),
        "macro_f1": float(payload["metrics"]["macro_f1"]),
    }
    for name, value in metrics.items():
        if not math.isfinite(value):
            raise ValueError(f"Non-finite {name} in {path}: {value}")
    return metrics


def training_completion_evidence(
    metric_paths: Dict[str, Path],
    required_validation_points: int,
    required_packet_batch_scheduler: str = "",
) -> dict:
    if required_validation_points <= 0:
        raise ValueError("required_validation_points must be positive")
    datasets = {}
    for dataset, metric_path in sorted(metric_paths.items()):
        output_dir = metric_path.parent
        final_checkpoint = output_dir / "final" / "tower1_heads.pt"
        history_path = output_dir / "packet_validation_history.jsonl"
        validation_points = 0
        best_metric_matches_history = False
        best_history_step = None
        best_history_macro_f1 = None
        provenance_kind = None
        provenance_path = None
        provenance_sha256 = None
        provenance_verified = False
        packet_batch_scheduler = None
        trainer_source_sha256 = None
        completion_trainer_source_sha256 = None
        trainer_source_stable_through_completion = False
        if history_path.is_file():
            try:
                with history_path.open("r", encoding="utf-8") as handle:
                    history = [json.loads(line) for line in handle if line.strip()]
                validation_points = len(history)
                best_payload = json.loads(metric_path.read_text(encoding="utf-8"))
                best_step = int(best_payload["step"])
                select_metric = str(best_payload["select_metric"])
                if select_metric not in {"macro_f1", "accuracy"}:
                    raise ValueError(f"Unsupported select metric: {select_metric}")

                def trainer_selection_key(row: dict) -> tuple[float, float, float]:
                    metrics = row["metrics"]
                    return (
                        float(metrics[select_metric]),
                        float(metrics["macro_f1"]),
                        float(metrics["accuracy"]),
                    )

                best_history_row = max(history, key=trainer_selection_key)
                best_history_step = int(best_history_row["step"])
                best_history_macro_f1 = float(
                    best_history_row["metrics"]["macro_f1"]
                )
                best_metrics = best_payload["metrics"]
                best_metric_matches_history = bool(
                    best_step == best_history_step
                    and all(
                        math.isfinite(float(best_metrics[name]))
                        and math.isclose(
                            float(best_metrics[name]),
                            float(best_history_row["metrics"][name]),
                            rel_tol=0.0,
                            abs_tol=1e-12,
                        )
                        for name in ("macro_f1", "accuracy")
                    )
                )
            except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
                best_metric_matches_history = False
        if final_checkpoint.is_file() and history_path.is_file():
            final_sha256 = file_sha256(final_checkpoint)
            history_sha256 = file_sha256(history_path)
            contract_path = output_dir / "tower1_training_contract.json"
            legacy_path = output_dir / "legacy_final_materialization.json"
            try:
                if contract_path.is_file():
                    contract = json.loads(contract_path.read_text(encoding="utf-8"))
                    artifacts = contract.get("completed_artifacts") or {}
                    trainer_source_sha256 = (contract.get("trainer_source") or {}).get(
                        "sha256"
                    )
                    completion_trainer_source_sha256 = (
                        contract.get("completion_observed_trainer_source") or {}
                    ).get("sha256")
                    trainer_source_stable_through_completion = bool(
                        trainer_source_sha256
                        and completion_trainer_source_sha256
                        and trainer_source_sha256 == completion_trainer_source_sha256
                    )
                    packet_batch_scheduler = (contract.get("training_config") or {}).get(
                        "packet_batch_scheduler"
                    )
                    provenance_verified = bool(
                        contract.get("schema") == "tower1_training_contract_v1"
                        and contract.get("status") == "complete"
                        and artifacts.get("final_heads", {}).get("sha256")
                        == final_sha256
                        and artifacts.get("validation_history", {}).get("sha256")
                        == history_sha256
                        and trainer_source_stable_through_completion
                        and (
                            not required_packet_batch_scheduler
                            or packet_batch_scheduler
                            == required_packet_batch_scheduler
                        )
                    )
                    provenance_kind = "training_contract_v1"
                    provenance_path = contract_path
                elif legacy_path.is_file():
                    legacy = json.loads(legacy_path.read_text(encoding="utf-8"))
                    provenance_verified = bool(
                        legacy.get("schema")
                        == "legacy_tower1_final_materialization_v1"
                        and legacy.get("status") == "materialized"
                        and legacy.get("final_heads_sha256") == final_sha256
                        and legacy.get("history", {}).get("history_sha256")
                        == history_sha256
                        and not required_packet_batch_scheduler
                    )
                    provenance_kind = "legacy_final_materialization_v1"
                    provenance_path = legacy_path
                if provenance_path is not None:
                    provenance_sha256 = file_sha256(provenance_path)
            except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
                provenance_verified = False
        passed = bool(
            metric_path.is_file()
            and final_checkpoint.is_file()
            and validation_points == required_validation_points
            and best_metric_matches_history
            and provenance_verified
        )
        datasets[dataset] = {
            "metric_path": str(metric_path),
            "metric_sha256": file_sha256(metric_path) if metric_path.is_file() else None,
            "final_checkpoint_path": str(final_checkpoint),
            "final_checkpoint_sha256": (
                file_sha256(final_checkpoint) if final_checkpoint.is_file() else None
            ),
            "validation_history_path": str(history_path),
            "validation_history_sha256": (
                file_sha256(history_path) if history_path.is_file() else None
            ),
            "validation_points": validation_points,
            "required_validation_points": required_validation_points,
            "best_metric_matches_history": best_metric_matches_history,
            "best_history_step": best_history_step,
            "best_history_macro_f1": best_history_macro_f1,
            "provenance_kind": provenance_kind,
            "provenance_path": str(provenance_path) if provenance_path else None,
            "provenance_sha256": provenance_sha256,
            "provenance_verified": provenance_verified,
            "packet_batch_scheduler": packet_batch_scheduler,
            "trainer_source_sha256": trainer_source_sha256,
            "completion_trainer_source_sha256": completion_trainer_source_sha256,
            "trainer_source_stable_through_completion": (
                trainer_source_stable_through_completion
            ),
            "required_packet_batch_scheduler": (
                required_packet_batch_scheduler or None
            ),
            "passed": passed,
        }
    return {
        "required": True,
        "required_validation_points": required_validation_points,
        "required_packet_batch_scheduler": (
            required_packet_batch_scheduler or None
        ),
        "status": "pass" if datasets and all(row["passed"] for row in datasets.values()) else "fail",
        "datasets": datasets,
    }


def training_implementation_consistency_evidence(completion: dict) -> dict:
    """Require every selected arm and dataset to use one stable trainer source."""
    rows = [
        row
        for arm in ("baseline", "candidate")
        for row in (completion.get(arm, {}).get("datasets") or {}).values()
    ]
    source_hashes = sorted(
        {
            str(row.get("trainer_source_sha256"))
            for row in rows
            if row.get("trainer_source_sha256")
        }
    )
    all_stable = bool(rows) and all(
        row.get("trainer_source_stable_through_completion") is True for row in rows
    )
    passed = bool(all_stable and len(source_hashes) == 1)
    return {
        "required": True,
        "status": "pass" if passed else "fail",
        "num_runs": len(rows),
        "trainer_source_sha256": source_hashes[0] if len(source_hashes) == 1 else None,
        "observed_trainer_source_sha256": source_hashes,
        "all_runs_stable_through_completion": all_stable,
    }


def training_factorial_integrity_evidence(
    completion: dict,
    allowed_difference_fields: set[str],
) -> dict[str, Any]:
    """Require every A/B pair to differ only in preregistered config fields."""
    if not allowed_difference_fields:
        raise ValueError("factorial integrity requires declared difference fields")
    datasets = sorted(completion["baseline"]["datasets"])
    evidence: dict[str, Any] = {}
    passed = True
    missing = object()
    for dataset in datasets:
        configs = {}
        for arm in ("baseline", "candidate"):
            row = completion[arm]["datasets"][dataset]
            path = Path(str(row.get("provenance_path") or ""))
            try:
                contract = json.loads(path.read_text(encoding="utf-8"))
                if contract.get("schema") != "tower1_training_contract_v1":
                    raise ValueError("not a Tower1 training contract")
                configs[arm] = contract["training_config"]
            except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"factorial integrity requires readable training config for "
                    f"{arm} {dataset}"
                ) from exc
        keys = sorted(
            (set(configs["baseline"]) | set(configs["candidate"]))
            - allowed_difference_fields
        )
        mismatched = [
            key
            for key in keys
            if configs["baseline"].get(key, missing)
            != configs["candidate"].get(key, missing)
        ]
        dataset_passed = not mismatched
        passed = passed and dataset_passed
        evidence[dataset] = {
            "status": "pass" if dataset_passed else "fail",
            "mismatched_fields": mismatched,
            "declared_values": {
                field: {
                    "baseline": configs["baseline"].get(field),
                    "candidate": configs["candidate"].get(field),
                }
                for field in sorted(allowed_difference_fields)
            },
        }
    return {
        "required": True,
        "status": "pass" if passed else "fail",
        "declared_factorial_fields": sorted(allowed_difference_fields),
        "datasets": evidence,
    }


def select_candidate(
    baseline_paths: Dict[str, Path],
    candidate_paths: Dict[str, Path],
    min_delta: float,
    max_accuracy_drop: float = 0.005,
) -> dict:
    if set(baseline_paths) != set(candidate_paths):
        raise ValueError("Baseline and candidate datasets must match")
    if len(baseline_paths) < 2:
        raise ValueError("Unified candidate selection requires at least two datasets")
    if not math.isfinite(min_delta) or min_delta < 0.0:
        raise ValueError("min_delta must be finite and non-negative")
    if not math.isfinite(max_accuracy_drop) or max_accuracy_drop < 0.0:
        raise ValueError("max_accuracy_drop must be finite and non-negative")
    datasets = {}
    for dataset in sorted(baseline_paths):
        baseline = load_validation_metrics(baseline_paths[dataset])
        candidate = load_validation_metrics(candidate_paths[dataset])
        macro_f1_delta = candidate["macro_f1"] - baseline["macro_f1"]
        accuracy_delta = candidate["accuracy"] - baseline["accuracy"]
        macro_f1_passes = macro_f1_delta >= min_delta
        accuracy_guard_passes = accuracy_delta >= -max_accuracy_drop
        datasets[dataset] = {
            "baseline_accuracy": baseline["accuracy"],
            "candidate_accuracy": candidate["accuracy"],
            "delta_accuracy": accuracy_delta,
            "baseline_macro_f1": baseline["macro_f1"],
            "candidate_macro_f1": candidate["macro_f1"],
            "delta_macro_f1": macro_f1_delta,
            "macro_f1_passes": macro_f1_passes,
            "accuracy_guard_passes": accuracy_guard_passes,
            "passes": macro_f1_passes and accuracy_guard_passes,
        }
    promoted = bool(datasets) and all(item["passes"] for item in datasets.values())
    return {
        "selection_scope": "heldout_validation_only",
        "metric": "macro_f1_with_accuracy_guard",
        "promotion_scope": "same_candidate_must_pass_every_dataset",
        "num_datasets": len(datasets),
        "min_delta": min_delta,
        "max_accuracy_drop": max_accuracy_drop,
        "candidate_promoted_for_all_datasets": promoted,
        "selected": "candidate" if promoted else "baseline",
        "datasets": datasets,
    }


def training_dynamics_evidence(
    baseline_paths: Dict[str, Path], candidate_paths: Dict[str, Path]
) -> dict:
    """Describe complete validation curves without changing candidate selection."""
    if set(baseline_paths) != set(candidate_paths):
        raise ValueError("Baseline and candidate datasets must match")

    def curve(metric_path: Path) -> dict[int, float]:
        history_path = metric_path.parent / "packet_validation_history.jsonl"
        with history_path.open("r", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        values = {
            int(row["step"]): float(row["metrics"]["macro_f1"])
            for row in rows
        }
        if not values or any(not math.isfinite(value) for value in values.values()):
            raise ValueError(f"Invalid validation curve: {history_path}")
        return values

    datasets = {}
    for dataset in sorted(baseline_paths):
        baseline = curve(baseline_paths[dataset])
        candidate = curve(candidate_paths[dataset])
        common_steps = sorted(set(baseline) & set(candidate))
        if not common_steps:
            raise ValueError(f"No matched validation steps for {dataset}")
        deltas = [candidate[step] - baseline[step] for step in common_steps]
        split_index = max(1, len(common_steps) // 2)
        early_steps = common_steps[:split_index]
        late_steps = common_steps[split_index:] or common_steps[-1:]
        early_deltas = [candidate[step] - baseline[step] for step in early_steps]
        late_deltas = [candidate[step] - baseline[step] for step in late_steps]
        baseline_best_step = max(baseline, key=baseline.__getitem__)
        candidate_best_step = max(candidate, key=candidate.__getitem__)
        datasets[dataset] = {
            "baseline": {
                "best_step": baseline_best_step,
                "best_macro_f1": baseline[baseline_best_step],
                "final_step": max(baseline),
                "final_macro_f1": baseline[max(baseline)],
                "regression_after_best": baseline[baseline_best_step]
                - baseline[max(baseline)],
            },
            "candidate": {
                "best_step": candidate_best_step,
                "best_macro_f1": candidate[candidate_best_step],
                "final_step": max(candidate),
                "final_macro_f1": candidate[max(candidate)],
                "regression_after_best": candidate[candidate_best_step]
                - candidate[max(candidate)],
            },
            "matched_curve": {
                "steps": common_steps,
                "candidate_wins": sum(delta > 0.0 for delta in deltas),
                "ties": sum(delta == 0.0 for delta in deltas),
                "candidate_losses": sum(delta < 0.0 for delta in deltas),
                "mean_macro_f1_delta": statistics.fmean(deltas),
                "median_macro_f1_delta": statistics.median(deltas),
                "phase_dynamics": {
                    "early_steps": early_steps,
                    "late_steps": late_steps,
                    "early_mean_macro_f1_delta": statistics.fmean(early_deltas),
                    "late_mean_macro_f1_delta": statistics.fmean(late_deltas),
                    "late_minus_early_mean_delta": statistics.fmean(late_deltas)
                    - statistics.fmean(early_deltas),
                    "first_to_latest_delta_change": deltas[-1] - deltas[0],
                    "late_candidate_wins": sum(delta > 0.0 for delta in late_deltas),
                    "late_ties": sum(delta == 0.0 for delta in late_deltas),
                    "late_candidate_losses": sum(delta < 0.0 for delta in late_deltas),
                },
            },
        }
    return {
        "scope": "heldout_validation_only",
        "selection_role": "descriptive_only",
        "datasets": datasets,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", action="append", required=True, help="DATASET=metrics.json")
    ap.add_argument("--candidate", action="append", required=True, help="DATASET=metrics.json")
    ap.add_argument("--min_delta", type=float, default=0.005)
    ap.add_argument("--max_accuracy_drop", type=float, default=0.005)
    ap.add_argument("--required_validation_points", type=int, default=8)
    ap.add_argument(
        "--required_packet_batch_scheduler",
        default="",
        help="Require a completed Tower1 contract with this batch scheduler.",
    )
    ap.add_argument(
        "--factorial_field",
        action="append",
        default=[],
        help=(
            "Training-config field allowed to differ between A/B arms. Supplying "
            "one or more fields enables strict factorial-integrity validation."
        ),
    )
    ap.add_argument("--output_json", required=True)
    args = ap.parse_args()
    baseline_paths = parse_named_paths(args.baseline)
    candidate_paths = parse_named_paths(args.candidate)
    completion = {
        "baseline": training_completion_evidence(
            baseline_paths,
            args.required_validation_points,
            args.required_packet_batch_scheduler,
        ),
        "candidate": training_completion_evidence(
            candidate_paths,
            args.required_validation_points,
            args.required_packet_batch_scheduler,
        ),
    }
    if any(item["status"] != "pass" for item in completion.values()):
        raise ValueError(
            "Unified Tower1 selection requires completed baseline and candidate "
            "training histories; refusing intermediate best metrics"
        )
    implementation = training_implementation_consistency_evidence(completion)
    if implementation["status"] != "pass":
        raise ValueError(
            "Unified Tower1 selection requires one trainer source SHA across "
            "every completed baseline/candidate run"
        )
    factorial_integrity = None
    if args.factorial_field:
        declared_fields = set(args.factorial_field)
        if len(declared_fields) != len(args.factorial_field):
            raise ValueError("duplicate --factorial_field declaration")
        factorial_integrity = training_factorial_integrity_evidence(
            completion, declared_fields
        )
        if factorial_integrity["status"] != "pass":
            raise ValueError(
                "baseline/candidate differ outside declared factorial fields"
            )
    payload = select_candidate(
        baseline_paths,
        candidate_paths,
        min_delta=args.min_delta,
        max_accuracy_drop=args.max_accuracy_drop,
    )
    payload["training_completion_evidence"] = completion
    payload["training_implementation_consistency"] = implementation
    if factorial_integrity is not None:
        payload["factorial_config_integrity"] = factorial_integrity
    payload["training_dynamics_evidence"] = training_dynamics_evidence(
        baseline_paths, candidate_paths
    )
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
