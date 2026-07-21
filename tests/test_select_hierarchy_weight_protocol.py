import hashlib
import json
import sys
from pathlib import Path

import pytest

from select_class_weight_protocol import main as select_class_main
from select_hierarchy_weight_protocol import main as select_hierarchy_main, select_eta


TRAINER_SHA = "b" * 64


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_run(
    root: Path,
    *,
    arm: str,
    dataset: str,
    basis: str,
    strength: float,
    accuracy: float,
    macro_f1: float,
) -> Path:
    output = root / arm / dataset
    final = output / "final"
    final.mkdir(parents=True)
    heads = final / "tower1_heads.pt"
    heads.write_bytes(f"{arm}:{dataset}:heads".encode())
    history = output / "packet_validation_history.jsonl"
    rows = [
        {
            "step": index * 100,
            "metrics": {
                "accuracy": accuracy - 0.07 + 0.07 * index / 8,
                "macro_f1": macro_f1 - 0.07 + 0.07 * index / 8,
            },
        }
        for index in range(1, 9)
    ]
    history.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    best = output / "best_packet_validation_metrics.json"
    best.write_text(
        json.dumps(
            {
                "step": 800,
                "select_metric": "macro_f1",
                "metrics": {"accuracy": accuracy, "macro_f1": macro_f1},
            }
        ),
        encoding="utf-8",
    )
    contract = output / "tower1_training_contract.json"
    contract.write_text(
        json.dumps(
            {
                "schema": "tower1_training_contract_v1",
                "status": "complete",
                "training_config": {
                    "packet_batch_scheduler": "epoch_resampled_dataloader_v1",
                    "class_weight_basis": basis,
                    "class_weight_strength": strength,
                },
                "trainer_source": {"sha256": TRAINER_SHA},
                "completion_observed_trainer_source": {"sha256": TRAINER_SHA},
                "completed_artifacts": {
                    "final_heads": {"sha256": sha256(heads)},
                    "validation_history": {"sha256": sha256(history)},
                },
            }
        ),
        encoding="utf-8",
    )
    return best


def test_selects_best_eligible_eta_with_accuracy_guard():
    selected = select_eta(
        {
            0.0: {"accuracy": 0.80, "macro_f1": 0.70},
            0.25: {"accuracy": 0.798, "macro_f1": 0.71},
            0.5: {"accuracy": 0.79, "macro_f1": 0.73},
            1.0: {"accuracy": 0.801, "macro_f1": 0.708},
        },
        min_delta=0.005,
        max_accuracy_drop=0.005,
    )

    assert selected["selected_eta"] == 0.25
    assert selected["arms"]["0.5"]["eligible"] is False


def test_uses_smaller_eta_as_final_exact_tie_break():
    selected = select_eta(
        {
            0.0: {"accuracy": 0.80, "macro_f1": 0.70},
            0.25: {"accuracy": 0.81, "macro_f1": 0.72},
            0.5: {"accuracy": 0.81, "macro_f1": 0.72},
        },
        min_delta=0.005,
        max_accuracy_drop=0.005,
    )

    assert selected["selected_eta"] == 0.25


def test_requires_eta_zero_reference():
    with pytest.raises(ValueError, match="eta=0 reference"):
        select_eta(
            {0.5: {"accuracy": 0.8, "macro_f1": 0.7}},
            min_delta=0.005,
            max_accuracy_drop=0.005,
        )


def test_cli_freezes_dataset_numeric_eta_from_hash_bound_validation(
    tmp_path, monkeypatch
):
    metrics = {arm: {} for arm in ("packet_full", "flow_sqrt", "flow_full", "eta025")}
    settings = {
        "vpn-app": {
            "packet_full": ("packet", 1.0, 0.800, 0.700),
            "flow_sqrt": ("flow", 0.5, 0.790, 0.730),
            "flow_full": ("flow", 1.0, 0.780, 0.720),
            "eta025": ("flow", 0.25, 0.798, 0.712),
        },
        "tls-120": {
            "packet_full": ("packet", 1.0, 0.830, 0.810),
            "flow_sqrt": ("flow", 0.5, 0.850, 0.840),
            "flow_full": ("flow", 1.0, 0.820, 0.800),
            "eta025": ("flow", 0.25, 0.840, 0.825),
        },
    }
    for dataset, arms in settings.items():
        for arm, (basis, strength, accuracy, macro_f1) in arms.items():
            metrics[arm][dataset] = make_run(
                tmp_path,
                arm=arm,
                dataset=dataset,
                basis=basis,
                strength=strength,
                accuracy=accuracy,
                macro_f1=macro_f1,
            )

    class_prereg = tmp_path / "class_prereg.json"
    class_prereg.write_text(
        json.dumps(
            {
                "schema": "cross_dataset_class_weight_protocol_preregistration_v1",
                "scope": "heldout_fold0_validation_only",
                "test_access": "forbidden",
                "datasets": ["vpn-app", "tls-120"],
                "fixed_factors": {
                    "trainer_source_sha256": TRAINER_SHA,
                    "packet_batch_scheduler": "epoch_resampled_dataloader_v1",
                    "required_validation_points": 8,
                },
                "promotion_gate": {
                    "minimum_macro_f1_delta_per_dataset": 0.005,
                    "maximum_accuracy_drop_per_dataset": 0.005,
                    "same_arm_must_pass_every_dataset": True,
                },
            }
        ),
        encoding="utf-8",
    )
    class_output = tmp_path / "class_selection.json"
    argv = ["select_class_weight_protocol.py"]
    for flag, arm in (
        ("--packet", "packet_full"),
        ("--flow_sqrt", "flow_sqrt"),
        ("--flow_full", "flow_full"),
    ):
        for dataset, path in metrics[arm].items():
            argv.extend([flag, f"{dataset}={path}"])
    argv.extend(
        ["--preregistration", str(class_prereg), "--output_json", str(class_output)]
    )
    monkeypatch.setattr(sys, "argv", argv)
    select_class_main()

    hierarchy_prereg = tmp_path / "hierarchy_prereg.json"
    hierarchy_prereg.write_text(
        json.dumps(
            {
                "schema": "hierarchy_adaptive_class_weight_preregistration_v1",
                "status": "preregistered_before_complete_validation_histories",
                "launch_gate": {
                    "required_datasets": ["vpn-app", "tls-120"],
                    "eligibility": {
                        "minimum_macro_f1_gain_over_packet_full": 0.005,
                        "maximum_accuracy_drop_from_packet_full": 0.005,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    gate = tmp_path / "gate.json"
    gate.write_text(
        json.dumps(
            {
                "schema": "hierarchy_adaptive_class_weight_gate_v1",
                "status": "launch",
                "launch": True,
                "selection_scope": "heldout_validation_only",
                "test_labels_used": False,
                "inputs": {
                    "preregistration": {
                        "path": str(hierarchy_prereg.resolve()),
                        "sha256": sha256(hierarchy_prereg),
                    },
                    "class_weight_selection": {
                        "path": str(class_output.resolve()),
                        "sha256": sha256(class_output),
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "hierarchy_selection.json"
    argv = [
        "select_hierarchy_weight_protocol.py",
        "--gate",
        str(gate),
        "--preregistration",
        str(hierarchy_prereg),
        "--class_weight_selection",
        str(class_output),
    ]
    for dataset, path in metrics["eta025"].items():
        argv.extend(["--eta025", f"{dataset}={path}"])
    argv.extend(["--output_json", str(output)])
    monkeypatch.setattr(sys, "argv", argv)
    select_hierarchy_main()

    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["datasets"]["vpn-app"]["selected_eta"] == 0.25
    assert result["datasets"]["tls-120"]["selected_eta"] == 0.5
    assert result["datasets"]["vpn-app"]["selected_validation_metric"][
        "sha256"
    ] == sha256(metrics["eta025"]["vpn-app"])
    assert result["eta025_training_completion_evidence"]["status"] == "pass"
    assert result["test_labels_used"] is False
