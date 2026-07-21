import hashlib
import json
import sys
from pathlib import Path

from select_class_weight_protocol import main


TRAINER_SHA = "a" * 64


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
    rows = []
    for index in range(1, 9):
        fraction = index / 8
        rows.append(
            {
                "step": index * 100,
                "metrics": {
                    "accuracy": accuracy - 0.07 + 0.07 * fraction,
                    "macro_f1": macro_f1 - 0.07 + 0.07 * fraction,
                },
            }
        )
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


def test_complete_three_arm_artifacts_select_one_shared_full_protocol(
    tmp_path, monkeypatch
):
    metrics = {}
    settings = {
        "packet_full": ("packet", 1.0, 0.80, 0.70),
        "flow_sqrt": ("flow", 0.5, 0.801, 0.706),
        "flow_full": ("flow", 1.0, 0.802, 0.711),
    }
    for arm, (basis, strength, accuracy, macro_f1) in settings.items():
        metrics[arm] = {}
        for offset, dataset in enumerate(("vpn-app", "tls-120")):
            metrics[arm][dataset] = make_run(
                tmp_path,
                arm=arm,
                dataset=dataset,
                basis=basis,
                strength=strength,
                accuracy=accuracy - 0.01 * offset,
                macro_f1=macro_f1 - 0.01 * offset,
            )

    preregistration = tmp_path / "preregistration.json"
    preregistration.write_text(
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
    output = tmp_path / "selection.json"
    argv = ["select_class_weight_protocol.py"]
    for flag, arm in (
        ("--packet", "packet_full"),
        ("--flow_sqrt", "flow_sqrt"),
        ("--flow_full", "flow_full"),
    ):
        for dataset, path in metrics[arm].items():
            argv.extend([flag, f"{dataset}={path}"])
    argv.extend(
        [
            "--preregistration",
            str(preregistration),
            "--output_json",
            str(output),
        ]
    )
    monkeypatch.setattr(sys, "argv", argv)
    main()

    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["selected"] == "candidate"
    assert result["candidate_promoted_for_all_datasets"] is True
    assert result["multi_arm_selection"]["selected_protocol"] == "flow_full"
    assert result["multi_arm_selection"]["selected_config"] == {
        "class_weight_basis": "flow",
        "class_weight_strength": 1.0,
    }
    assert result["multi_arm_selection"]["eligible_flow_arms"] == [
        "flow_sqrt",
        "flow_full",
    ]
    assert result["training_completion_evidence"]["candidate"]["status"] == "pass"
    assert result["training_implementation_consistency"]["num_runs"] == 4
    assert result["multi_arm_selection"][
        "all_arm_training_implementation_consistency"
    ]["num_runs"] == 6
    assert result["multi_arm_selection"]["factorial_config_integrity"][
        "status"
    ] == "pass"
