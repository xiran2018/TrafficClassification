from argparse import Namespace
import json

import pytest

from train_tower1_multitask import (
    PacketAuxDataset,
    tower1_training_config,
    validate_aligned_validation_views,
    validation_selection_key,
)


def write_jsonl(path, rows):
    path.write_text("".join(f"{json.dumps(row)}\n" for row in rows), encoding="utf-8")


def test_training_config_records_sampler_and_paired_objective():
    values = {
        "base_model": "Qwen/Qwen2.5-7B-Instruct",
        "label_map": "label_map.json",
        "packet_aux_jsonl": "train.jsonl",
        "paired_packet_aux_jsonl": "masked.jsonl",
        "valid_packet_aux_jsonl": "valid.jsonl",
        "valid_paired_packet_aux_jsonl": "valid_masked.jsonl",
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
        "flow_proto_weight": 0.0,
        "flow_proto_positive": "same_class",
        "flow_proto_context": "inclusive",
        "paired_consistency_weight": 0.05,
        "paired_cls_weight": 0.2,
        "paired_logit_kl_weight": 0.5,
        "paired_raw_consistency_weight": 1.0,
        "paired_consistency_mode": "factual_teacher",
        "paired_group_dro": True,
        "paired_group_dro_eta": 0.05,
        "paired_num_groups": 2,
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
        "paired_validation_selection": "worst_view_macro_f1",
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
    assert config["paired_validation_selection"] == "worst_view_macro_f1"


def test_paired_validation_views_require_exact_packet_and_label_alignment():
    factual = [
        {"packet_uid": "a", "label_id": 0},
        {"packet_uid": "b", "label_id": 1},
    ]
    validate_aligned_validation_views(
        factual,
        [
            {"packet_uid": "b", "label_id": 1},
            {"packet_uid": "a", "label_id": 0},
        ],
    )
    with pytest.raises(ValueError, match="not aligned"):
        validate_aligned_validation_views(
            factual,
            [
                {"packet_uid": "a", "label_id": 1},
                {"packet_uid": "c", "label_id": 1},
            ],
        )


def test_paired_training_views_require_complete_alignment(tmp_path):
    factual_path = tmp_path / "factual.jsonl"
    paired_path = tmp_path / "paired.jsonl"
    factual = [
        {"packet_uid": "a", "label_id": 0, "prompt": "factual-a"},
        {"packet_uid": "b", "label_id": 1, "prompt": "factual-b"},
    ]
    paired = [
        {"packet_uid": "b", "label_id": 1, "prompt": "paired-b"},
        {"packet_uid": "a", "label_id": 0, "prompt": "paired-a"},
    ]
    write_jsonl(factual_path, factual)
    write_jsonl(paired_path, paired)

    dataset = PacketAuxDataset(str(factual_path), show_progress=False, paired_path=str(paired_path))

    assert dataset.paired_rows == 2
    assert [row["paired_prompt"] for row in dataset.rows] == ["paired-a", "paired-b"]


@pytest.mark.parametrize(
    ("paired", "message"),
    [
        ([{"packet_uid": "a", "label_id": 0, "prompt": "paired-a"}], "missing"),
        (
            [
                {"packet_uid": "a", "label_id": 0, "prompt": "paired-a"},
                {"packet_uid": "b", "label_id": 0, "prompt": "paired-b"},
            ],
            "label_mismatches",
        ),
        (
            [
                {"packet_uid": "a", "label_id": 0, "prompt": "paired-a"},
                {"packet_uid": "b", "label_id": 1, "prompt": ""},
            ],
            "empty_prompts",
        ),
    ],
)
def test_paired_training_views_reject_partial_or_invalid_pairs(tmp_path, paired, message):
    factual_path = tmp_path / "factual.jsonl"
    paired_path = tmp_path / "paired.jsonl"
    write_jsonl(
        factual_path,
        [
            {"packet_uid": "a", "label_id": 0, "prompt": "factual-a"},
            {"packet_uid": "b", "label_id": 1, "prompt": "factual-b"},
        ],
    )
    write_jsonl(paired_path, paired)

    with pytest.raises(ValueError, match=message):
        PacketAuxDataset(str(factual_path), show_progress=False, paired_path=str(paired_path))


def test_paired_training_views_reject_duplicate_uids(tmp_path):
    factual_path = tmp_path / "factual.jsonl"
    paired_path = tmp_path / "paired.jsonl"
    write_jsonl(factual_path, [{"packet_uid": "a", "label_id": 0, "prompt": "factual"}])
    write_jsonl(
        paired_path,
        [
            {"packet_uid": "a", "label_id": 0, "prompt": "paired-1"},
            {"packet_uid": "a", "label_id": 0, "prompt": "paired-2"},
        ],
    )

    with pytest.raises(ValueError, match="duplicate paired packet_uid"):
        PacketAuxDataset(str(factual_path), show_progress=False, paired_path=str(paired_path))


def test_worst_view_checkpoint_selection_prioritizes_robust_view():
    factual = {"macro_f1": 0.90, "accuracy": 0.92}
    intervened = {"macro_f1": 0.60, "accuracy": 0.70}
    key, summary = validation_selection_key(
        factual,
        intervened,
        select_metric="macro_f1",
        paired_mode="worst_view_macro_f1",
    )
    assert key == (0.60, 0.75, 0.70, 0.81, 0.90, 0.92)
    assert summary["score"] == 0.60
    assert summary["mean_view_macro_f1"] == 0.75


def test_paired_selection_rejects_missing_intervened_metrics():
    with pytest.raises(ValueError, match="requires intervened"):
        validation_selection_key(
            {"macro_f1": 0.9, "accuracy": 0.9},
            None,
            select_metric="macro_f1",
            paired_mode="worst_view_macro_f1",
        )
