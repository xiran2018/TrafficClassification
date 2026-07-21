import hashlib
import json

from unified_framework_spec import tower1_execution_evidence


def declared_contract():
    return {
        "trainer": "train_tower1_multitask.py",
        "packet_context_policy": "single_packet",
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
        "class_weighting": "effective",
        "class_weight_beta": 0.9999,
        "class_weight_basis": "packet",
        "class_weight_strength": 1.0,
        "paired_consistency_weight": 0.0,
        "paired_cls_weight": 0.0,
        "paired_logit_kl_weight": 0.5,
        "paired_raw_consistency_weight": 1.0,
        "use_sft": False,
        "disable_packet_information_weights": True,
        "flow_balanced_packet_batches": True,
        "packets_per_flow": 2,
        "packet_batch_scheduler": "epoch_resampled_dataloader_v1",
        "select_metric": "macro_f1",
        "early_stop_patience": 0,
        "init_checkpoint_dir": "",
        "init_adapter_only": False,
    }


def write_execution(output, *, scheduler="epoch_resampled_dataloader_v1"):
    (output / "final").mkdir(parents=True)
    heads = output / "final" / "tower1_heads.pt"
    history = output / "packet_validation_history.jsonl"
    heads.write_bytes(b"heads")
    history.write_text('{"step": 1}\n', encoding="utf-8")
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    config = declared_contract()
    config.pop("trainer")
    config.pop("packet_context_policy")
    config["lr"] = config.pop("learning_rate")
    config["head_lr"] = config.pop("head_learning_rate")
    config["no_sft"] = not config.pop("use_sft")
    config["packet_batch_scheduler"] = scheduler
    source_sha = "a" * 64
    (output / "tower1_training_contract.json").write_text(
        json.dumps(
            {
                "schema": "tower1_training_contract_v1",
                "status": "complete",
                "training_config": config,
                "trainer_source": {"sha256": source_sha},
                "completion_observed_trainer_source": {"sha256": source_sha},
                "completed_artifacts": {
                    "final_heads": {"sha256": digest(heads)},
                    "validation_history": {"sha256": digest(history)},
                },
            }
        ),
        encoding="utf-8",
    )


def test_execution_evidence_binds_actual_artifacts_and_declared_method(tmp_path):
    write_execution(tmp_path)
    evidence = tower1_execution_evidence(tmp_path, declared_contract())

    assert evidence["verified"] is True
    assert evidence["declared_contract_match"] is True
    assert evidence["artifacts_verified"] is True
    assert evidence["method_config"]["packet_batch_scheduler"] == (
        "epoch_resampled_dataloader_v1"
    )
    assert evidence["shared_protocol_signature"]["objectives"][
        "supervised_contrastive"
    ] is True
    assert evidence["shared_protocol_signature"]["initialization"] == {
        "base_model_only": True,
        "adapter_only": False,
    }


def test_execution_evidence_rejects_old_scheduler_or_artifact_tampering(tmp_path):
    write_execution(tmp_path, scheduler="cached_first_epoch_v0")
    evidence = tower1_execution_evidence(tmp_path, declared_contract())
    assert evidence["verified"] is False
    assert evidence["declared_contract_match"] is False

    (tmp_path / "final" / "tower1_heads.pt").write_bytes(b"tampered")
    evidence = tower1_execution_evidence(tmp_path, declared_contract())
    assert evidence["artifacts_verified"] is False
