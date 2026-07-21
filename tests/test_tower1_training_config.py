from argparse import Namespace

from train_tower1_multitask import tower1_training_config


def test_training_config_records_sampler_and_paired_objective():
    values = {
        "base_model": "Qwen/Qwen2.5-7B-Instruct",
        "label_map": "label_map.json",
        "packet_aux_jsonl": "train.jsonl",
        "paired_packet_aux_jsonl": "masked.jsonl",
        "valid_packet_aux_jsonl": "valid.jsonl",
        "sft_jsonl": [],
        "epochs": 8,
        "max_steps": 0,
        "packet_batch_size": 16,
        "valid_batch_size": 16,
        "valid_packets_per_flow": 2,
        "max_packet_length": 1024,
        "lr": 1e-5,
        "head_lr": 1e-4,
        "weight_decay": 0.01,
        "class_weighting": "effective",
        "class_weight_beta": 0.9999,
        "class_weight_basis": "flow",
        "class_weight_strength": 0.5,
        "disable_packet_information_weights": True,
        "cls_weight": 1.0,
        "contrastive_weight": 0.1,
        "temperature": 0.07,
        "same_flow_positive_weight": 1.0,
        "same_label_positive_weight": 1.0,
        "identity_safe_contrastive": True,
        "flow_proto_weight": 0.0,
        "flow_proto_positive": "same_class",
        "flow_proto_context": "inclusive",
        "paired_consistency_weight": 0.05,
        "paired_cls_weight": 0.2,
        "paired_logit_kl_weight": 0.5,
        "paired_raw_consistency_weight": 1.0,
        "flow_balanced_packet_batches": True,
        "packets_per_flow": 2,
        "projection_dim": 256,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "gradient_accumulation_steps": 1,
        "gradient_checkpointing": True,
        "dtype": "float16",
        "local_files_only": True,
        "init_checkpoint_dir": "",
        "init_adapter_only": False,
        "select_metric": "macro_f1",
        "early_stop_patience": 0,
        "no_sft": True,
        "seed": 42,
    }
    config = tower1_training_config(Namespace(**values))
    assert config == {
        **values,
        "packet_batch_scheduler": "epoch_resampled_dataloader_v1",
    }
    assert config["class_weight_basis"] == "flow"
    assert config["paired_consistency_weight"] == 0.05
    assert config["paired_raw_consistency_weight"] == 1.0
    assert config["identity_safe_contrastive"] is True
