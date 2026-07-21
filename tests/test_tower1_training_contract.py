import json
from types import SimpleNamespace

import pytest

import train_tower1_multitask as trainer


def args_with_inputs(tmp_path):
    inputs = {}
    for name in ("label_map", "packet", "valid", "paired"):
        path = tmp_path / f"{name}.jsonl"
        path.write_text(f"{name}\n", encoding="utf-8")
        inputs[name] = str(path)
    return SimpleNamespace(
        label_map=inputs["label_map"],
        packet_aux_jsonl=inputs["packet"],
        valid_packet_aux_jsonl=inputs["valid"],
        paired_packet_aux_jsonl=inputs["paired"],
        sft_jsonl=[],
    )


def test_completion_preserves_launch_source_identity(tmp_path, monkeypatch):
    args = args_with_inputs(tmp_path)
    monkeypatch.setattr(trainer, "tower1_training_config", lambda _: {"epochs": 8})
    output = tmp_path / "run"
    contract_path = trainer.write_training_contract(output, args, status="launched")
    launched = json.loads(contract_path.read_text(encoding="utf-8"))
    launched["trainer_source"]["sha256"] = "launch-sha"
    contract_path.write_text(json.dumps(launched), encoding="utf-8")

    trainer.write_training_contract(
        output,
        args,
        status="complete",
        completed_artifacts={"final_heads": {"sha256": "final-sha"}},
    )
    completed = json.loads(contract_path.read_text(encoding="utf-8"))

    assert completed["status"] == "complete"
    assert completed["trainer_source"]["sha256"] == "launch-sha"
    assert completed["completion_observed_trainer_source"]["sha256"]
    assert completed["completed_artifacts"]["final_heads"]["sha256"] == "final-sha"


def test_completion_rejects_config_drift(tmp_path, monkeypatch):
    args = args_with_inputs(tmp_path)
    config = {"epochs": 8}
    monkeypatch.setattr(trainer, "tower1_training_config", lambda _: config.copy())
    output = tmp_path / "run"
    trainer.write_training_contract(output, args, status="launched")
    config["epochs"] = 9

    with pytest.raises(ValueError, match="differs"):
        trainer.write_training_contract(output, args, status="complete")
