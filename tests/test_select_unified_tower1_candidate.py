import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from select_unified_tower1_candidate import (
    parse_named_paths,
    select_candidate,
    training_completion_evidence,
    training_dynamics_evidence,
    training_implementation_consistency_evidence,
)


ROOT = Path(__file__).resolve().parents[1]


def write_metric(path, value, accuracy=None):
    if accuracy is None:
        accuracy = value
    path.write_text(
        json.dumps({"metrics": {"macro_f1": value, "accuracy": accuracy}}),
        encoding="utf-8",
    )
    return path


def write_completion_artifacts(output, values, *, best_index=None):
    output.mkdir(parents=True, exist_ok=True)
    if best_index is None:
        best_index = max(range(len(values)), key=values.__getitem__)
    rows = [
        {
            "step": index + 1,
            "metrics": {"macro_f1": value, "accuracy": value + 0.05},
        }
        for index, value in enumerate(values)
    ]
    (output / "packet_validation_history.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    best = rows[best_index]
    metric = output / "best_packet_validation_metrics.json"
    metric.write_text(
        json.dumps(
            {
                "step": best["step"],
                "select_metric": "macro_f1",
                "metrics": best["metrics"],
            }
        ),
        encoding="utf-8",
    )
    (output / "final").mkdir(exist_ok=True)
    final_heads = output / "final" / "tower1_heads.pt"
    final_heads.write_bytes(b"checkpoint")
    sha256 = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    (output / "tower1_training_contract.json").write_text(
        json.dumps(
            {
                "schema": "tower1_training_contract_v1",
                "status": "complete",
                "training_config": {
                    "packet_batch_scheduler": "epoch_resampled_dataloader_v1"
                },
                "trainer_source": {"sha256": "a" * 64},
                "completion_observed_trainer_source": {"sha256": "a" * 64},
                "completed_artifacts": {
                    "final_heads": {"sha256": sha256(final_heads)},
                    "validation_history": {
                        "sha256": sha256(output / "packet_validation_history.jsonl")
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return metric


def test_candidate_requires_shared_gain_on_every_dataset(tmp_path):
    vpn_base = write_metric(tmp_path / "vpn_base.json", 0.60)
    tls_base = write_metric(tmp_path / "tls_base.json", 0.70)
    vpn_new = write_metric(tmp_path / "vpn_new.json", 0.61)
    tls_new = write_metric(tmp_path / "tls_new.json", 0.702)
    result = select_candidate(
        {"vpn": vpn_base, "tls": tls_base},
        {"vpn": vpn_new, "tls": tls_new},
        min_delta=0.005,
    )
    assert result["selected"] == "baseline"
    assert result["datasets"]["vpn"]["passes"] is True
    assert result["datasets"]["tls"]["passes"] is False


def test_candidate_is_promoted_when_all_datasets_pass(tmp_path):
    paths = {
        "vpn_base": write_metric(tmp_path / "vpn_base.json", 0.60),
        "tls_base": write_metric(tmp_path / "tls_base.json", 0.70),
        "vpn_new": write_metric(tmp_path / "vpn_new.json", 0.61),
        "tls_new": write_metric(tmp_path / "tls_new.json", 0.71),
    }
    result = select_candidate(
        {"vpn": paths["vpn_base"], "tls": paths["tls_base"]},
        {"vpn": paths["vpn_new"], "tls": paths["tls_new"]},
        min_delta=0.005,
    )
    assert result["candidate_promoted_for_all_datasets"] is True
    assert result["selected"] == "candidate"


def test_candidate_macro_f1_gain_cannot_hide_large_accuracy_drop(tmp_path):
    paths = {
        "vpn_base": write_metric(tmp_path / "vpn_base.json", 0.60, 0.80),
        "tls_base": write_metric(tmp_path / "tls_base.json", 0.70, 0.85),
        "vpn_new": write_metric(tmp_path / "vpn_new.json", 0.62, 0.79),
        "tls_new": write_metric(tmp_path / "tls_new.json", 0.72, 0.86),
    }
    result = select_candidate(
        {"vpn": paths["vpn_base"], "tls": paths["tls_base"]},
        {"vpn": paths["vpn_new"], "tls": paths["tls_new"]},
        min_delta=0.005,
        max_accuracy_drop=0.005,
    )

    assert result["selected"] == "baseline"
    assert result["datasets"]["vpn"]["macro_f1_passes"] is True
    assert result["datasets"]["vpn"]["accuracy_guard_passes"] is False
    assert result["datasets"]["tls"]["passes"] is True


def test_dataset_keys_must_match(tmp_path):
    metric = write_metric(tmp_path / "metric.json", 0.5)
    with pytest.raises(ValueError, match="must match"):
        select_candidate({"vpn": metric}, {"tls": metric}, min_delta=0.0)


def test_single_dataset_cannot_promote_a_unified_candidate(tmp_path):
    baseline = write_metric(tmp_path / "base.json", 0.5)
    candidate = write_metric(tmp_path / "candidate.json", 0.9)
    with pytest.raises(ValueError, match="at least two datasets"):
        select_candidate({"vpn": baseline}, {"vpn": candidate}, min_delta=0.0)


def test_candidate_rejects_negative_or_nonfinite_threshold(tmp_path):
    metric = write_metric(tmp_path / "metric.json", 0.5)
    paths = {"vpn": metric, "tls": metric}
    with pytest.raises(ValueError, match="finite and non-negative"):
        select_candidate(paths, paths, min_delta=-0.1)
    with pytest.raises(ValueError, match="max_accuracy_drop"):
        select_candidate(paths, paths, min_delta=0.0, max_accuracy_drop=-0.1)


def test_named_paths_reject_duplicate_dataset_keys():
    with pytest.raises(ValueError, match="duplicate"):
        parse_named_paths(["vpn=a.json", "vpn=b.json"])


def test_training_completion_rejects_intermediate_best_metric(tmp_path):
    output = tmp_path / "run"
    metric = write_completion_artifacts(output, [0.5, 0.6, 0.7])

    report = training_completion_evidence({"vpn-app": metric}, 8)

    assert report["status"] == "fail"
    assert report["datasets"]["vpn-app"]["validation_points"] == 3


def test_training_completion_requires_final_checkpoint_and_full_history(tmp_path):
    output = tmp_path / "run"
    (output / "final").mkdir(parents=True)
    metric = write_completion_artifacts(output, [0.1, 0.2, 0.3, 0.4, 0.7, 0.6, 0.5, 0.4])
    (output / "final" / "tower1_heads.pt").write_bytes(b"checkpoint")

    report = training_completion_evidence({"vpn-app": metric}, 8)

    assert report["status"] == "pass"
    assert report["datasets"]["vpn-app"]["passed"] is True
    assert len(report["datasets"]["vpn-app"]["metric_sha256"]) == 64
    assert len(report["datasets"]["vpn-app"]["final_checkpoint_sha256"]) == 64
    assert len(report["datasets"]["vpn-app"]["validation_history_sha256"]) == 64
    assert report["datasets"]["vpn-app"]["best_metric_matches_history"] is True
    assert report["datasets"]["vpn-app"]["best_history_step"] == 5
    assert report["datasets"]["vpn-app"]["provenance_verified"] is True


def test_training_completion_matches_trainer_accuracy_tiebreak(tmp_path):
    output = tmp_path / "run"
    metric = write_completion_artifacts(
        output,
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.8],
        best_index=7,
    )
    history_path = output / "packet_validation_history.jsonl"
    rows = [
        json.loads(line)
        for line in history_path.read_text(encoding="utf-8").splitlines()
    ]
    rows[6]["metrics"]["accuracy"] = 0.80
    rows[7]["metrics"]["accuracy"] = 0.85
    history_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    best = json.loads(metric.read_text(encoding="utf-8"))
    best["metrics"] = rows[7]["metrics"]
    metric.write_text(json.dumps(best), encoding="utf-8")
    contract_path = output / "tower1_training_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["completed_artifacts"]["validation_history"]["sha256"] = (
        hashlib.sha256(history_path.read_bytes()).hexdigest()
    )
    contract_path.write_text(json.dumps(contract), encoding="utf-8")

    report = training_completion_evidence({"vpn-app": metric}, 8)

    row = report["datasets"]["vpn-app"]
    assert report["status"] == "pass"
    assert row["best_metric_matches_history"] is True
    assert row["best_history_step"] == 8


def test_training_completion_can_require_epoch_resampled_scheduler(tmp_path):
    output = tmp_path / "run"
    metric = write_completion_artifacts(output, [index / 10 for index in range(8)])

    accepted = training_completion_evidence(
        {"vpn-app": metric}, 8, "epoch_resampled_dataloader_v1"
    )
    assert accepted["status"] == "pass"
    assert accepted["datasets"]["vpn-app"]["packet_batch_scheduler"] == (
        "epoch_resampled_dataloader_v1"
    )

    contract_path = output / "tower1_training_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["training_config"]["packet_batch_scheduler"] = "cached_first_epoch_v0"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    rejected = training_completion_evidence(
        {"vpn-app": metric}, 8, "epoch_resampled_dataloader_v1"
    )
    assert rejected["status"] == "fail"
    assert rejected["datasets"]["vpn-app"]["provenance_verified"] is False


def test_training_completion_rejects_trainer_source_drift(tmp_path):
    output = tmp_path / "run"
    metric = write_completion_artifacts(output, [index / 10 for index in range(8)])
    contract_path = output / "tower1_training_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["completion_observed_trainer_source"]["sha256"] = "b" * 64
    contract_path.write_text(json.dumps(contract), encoding="utf-8")

    report = training_completion_evidence(
        {"vpn-app": metric}, 8, "epoch_resampled_dataloader_v1"
    )

    row = report["datasets"]["vpn-app"]
    assert report["status"] == "fail"
    assert row["trainer_source_stable_through_completion"] is False
    assert row["provenance_verified"] is False


def test_unified_selection_requires_one_trainer_source_across_all_runs():
    def arm(source):
        return {
            "status": "pass",
            "datasets": {
                "vpn-app": {
                    "trainer_source_sha256": source,
                    "trainer_source_stable_through_completion": True,
                },
                "tls-120": {
                    "trainer_source_sha256": source,
                    "trainer_source_stable_through_completion": True,
                },
            },
        }

    accepted = training_implementation_consistency_evidence(
        {"baseline": arm("a" * 64), "candidate": arm("a" * 64)}
    )
    assert accepted["status"] == "pass"
    assert accepted["trainer_source_sha256"] == "a" * 64

    rejected = training_implementation_consistency_evidence(
        {"baseline": arm("a" * 64), "candidate": arm("b" * 64)}
    )
    assert rejected["status"] == "fail"
    assert rejected["trainer_source_sha256"] is None


def test_cli_emits_dual_metric_and_trainer_consistency_evidence(tmp_path):
    baseline = {
        dataset: write_completion_artifacts(
            tmp_path / f"{dataset}_baseline",
            [0.40 + 0.01 * index for index in range(8)],
        )
        for dataset in ("vpn-app", "tls-120")
    }
    candidate = {
        dataset: write_completion_artifacts(
            tmp_path / f"{dataset}_candidate",
            [0.41 + 0.01 * index for index in range(8)],
        )
        for dataset in ("vpn-app", "tls-120")
    }
    output = tmp_path / "selection.json"
    command = [
        sys.executable,
        str(ROOT / "select_unified_tower1_candidate.py"),
    ]
    for dataset in ("vpn-app", "tls-120"):
        command.extend(["--baseline", f"{dataset}={baseline[dataset]}"])
        command.extend(["--candidate", f"{dataset}={candidate[dataset]}"])
    command.extend(
        [
            "--required_packet_batch_scheduler",
            "epoch_resampled_dataloader_v1",
            "--min_delta",
            "0.005",
            "--max_accuracy_drop",
            "0.005",
            "--output_json",
            str(output),
        ]
    )

    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["selected"] == "candidate"
    assert report["metric"] == "macro_f1_with_accuracy_guard"
    assert report["max_accuracy_drop"] == 0.005
    assert report["training_implementation_consistency"]["status"] == "pass"
    assert report["training_implementation_consistency"][
        "trainer_source_sha256"
    ] == "a" * 64
    assert all(
        row["accuracy_guard_passes"] is True
        for row in report["datasets"].values()
    )


def test_training_completion_rejects_stale_best_metric(tmp_path):
    output = tmp_path / "run"
    (output / "final").mkdir(parents=True)
    metric = write_completion_artifacts(
        output,
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        best_index=6,
    )
    (output / "final" / "tower1_heads.pt").write_bytes(b"checkpoint")

    report = training_completion_evidence({"vpn-app": metric}, 8)

    assert report["status"] == "fail"
    assert report["datasets"]["vpn-app"]["best_metric_matches_history"] is False


def test_training_completion_rejects_appended_history_from_reused_directory(tmp_path):
    output = tmp_path / "run"
    (output / "final").mkdir(parents=True)
    metric = write_completion_artifacts(output, [0.1] * 8 + [0.9])
    (output / "final" / "tower1_heads.pt").write_bytes(b"checkpoint")

    report = training_completion_evidence({"vpn-app": metric}, 8)

    assert report["status"] == "fail"
    assert report["datasets"]["vpn-app"]["validation_points"] == 9


def test_training_completion_accepts_verified_legacy_materialization(tmp_path):
    output = tmp_path / "run"
    metric = write_completion_artifacts(output, [index / 10 for index in range(8)])
    (output / "tower1_training_contract.json").unlink()
    final_heads = output / "final" / "tower1_heads.pt"
    history = output / "packet_validation_history.jsonl"
    (output / "legacy_final_materialization.json").write_text(
        json.dumps(
            {
                "schema": "legacy_tower1_final_materialization_v1",
                "status": "materialized",
                "final_heads_sha256": hashlib.sha256(final_heads.read_bytes()).hexdigest(),
                "history": {
                    "history_sha256": hashlib.sha256(history.read_bytes()).hexdigest()
                },
            }
        ),
        encoding="utf-8",
    )

    report = training_completion_evidence({"vpn-app": metric}, 8)

    row = report["datasets"]["vpn-app"]
    assert report["status"] == "pass"
    assert row["provenance_kind"] == "legacy_final_materialization_v1"


def test_training_completion_rejects_provenance_hash_mismatch(tmp_path):
    output = tmp_path / "run"
    metric = write_completion_artifacts(output, [index / 10 for index in range(8)])
    contract_path = output / "tower1_training_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["completed_artifacts"]["final_heads"]["sha256"] = "0" * 64
    contract_path.write_text(json.dumps(contract), encoding="utf-8")

    report = training_completion_evidence({"vpn-app": metric}, 8)

    assert report["status"] == "fail"
    assert report["datasets"]["vpn-app"]["provenance_verified"] is False


def test_training_dynamics_reports_instability_without_changing_selection(tmp_path):
    baseline = write_completion_artifacts(
        tmp_path / "baseline", [0.40, 0.50, 0.60, 0.62]
    )
    candidate = write_completion_artifacts(
        tmp_path / "candidate", [0.45, 0.56, 0.64, 0.58]
    )

    report = training_dynamics_evidence(
        {"vpn-app": baseline}, {"vpn-app": candidate}
    )

    row = report["datasets"]["vpn-app"]
    assert report["selection_role"] == "descriptive_only"
    assert row["candidate"]["best_step"] == 3
    assert row["candidate"]["regression_after_best"] == pytest.approx(0.06)
    assert row["matched_curve"]["candidate_wins"] == 3
    assert row["matched_curve"]["candidate_losses"] == 1
    assert row["matched_curve"]["mean_macro_f1_delta"] == pytest.approx(0.0275)
    phase = row["matched_curve"]["phase_dynamics"]
    assert phase["early_steps"] == [1, 2]
    assert phase["late_steps"] == [3, 4]
    assert phase["early_mean_macro_f1_delta"] == pytest.approx(0.055)
    assert phase["late_mean_macro_f1_delta"] == pytest.approx(0.0)
    assert phase["late_minus_early_mean_delta"] == pytest.approx(-0.055)
    assert phase["first_to_latest_delta_change"] == pytest.approx(-0.09)
    assert phase["late_candidate_wins"] == 1
    assert phase["late_candidate_losses"] == 1
