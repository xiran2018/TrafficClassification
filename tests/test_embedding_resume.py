from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import run_stage8_flowaware_pipeline as stage8
from extract_packet_embeddings_qwen import (
    CROSS_FLOW_SCHEDULER,
    acquire_cuda_capacity_lock,
    completed_flow_ids,
    iter_flow_batches,
    last_nonpadding_indices,
    logical_cuda_index,
    packet_index_policies,
    peft_device_kwargs,
    physical_cuda_token,
    select_cuda_device,
    write_flow_batch_embeddings,
)
from run_stage8_flowaware_pipeline import embedding_audit_cmd, embedding_cmd


def test_packet_index_policies_scan_every_row(tmp_path):
    index = tmp_path / "packet_index.jsonl"
    rows = [
        {
            "embedding_header_policy": "mask_ip_port",
            "packet_context_policy": "single_packet",
        },
        {
            "embedding_header_policy": "mask_ip_port",
            "packet_context_policy": "single_packet",
        },
    ]
    index.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    assert packet_index_policies(index) == ("mask_ip_port", "single_packet")


def test_packet_index_policies_reject_mixed_context(tmp_path):
    index = tmp_path / "packet_index.jsonl"
    index.write_text(
        '{"embedding_header_policy":"full","packet_context_policy":"single_packet"}\n'
        '{"embedding_header_policy":"full","packet_context_policy":"flow_context"}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="mixes embedding policies"):
        packet_index_policies(index)


def test_completed_flow_ids_requires_existing_embedding_file(tmp_path):
    emb = tmp_path / "packet_embeddings" / "flow-a.npy"
    emb.parent.mkdir()
    np.save(emb, np.zeros((1, 2), dtype="float32"))
    index = tmp_path / "flow_embedding_index.jsonl"
    index.write_text(
        "\n".join(
            [
                '{"flow_id":"flow-a","embedding_path":"' + str(emb) + '"}',
                '{"flow_id":"flow-b","embedding_path":"' + str(tmp_path / "missing.npy") + '"}',
                '{"flow_id":"flow-c"}',
                '{bad json',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert completed_flow_ids(index) == {"flow-a"}


def test_cross_flow_batches_preserve_flow_and_packet_order():
    flows = [
        ("a", [{"packet_id": 1}, {"packet_id": 0}]),
        ("b", [{"packet_id": 0}]),
        ("c", [{"packet_id": 0}, {"packet_id": 1}]),
    ]

    batches = list(iter_flow_batches(flows, max_packets=3))

    assert [[flow_id for flow_id, _rows in batch] for batch in batches] == [
        ["a", "b"],
        ["c"],
    ]
    assert [row["packet_id"] for row in batches[0][0][1]] == [0, 1]


def test_last_nonpadding_indices_support_left_and_right_padding():
    masks = torch.tensor(
        [
            [1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1],
            [0, 1, 1, 0, 0],
        ]
    )

    assert last_nonpadding_indices(masks).tolist() == [2, 4, 2]


def test_last_nonpadding_indices_rejects_empty_sequence():
    with pytest.raises(ValueError, match="at least one token"):
        last_nonpadding_indices(torch.zeros((1, 4), dtype=torch.long))


def test_frozen_cross_flow_scheduler_name_is_stable():
    assert CROSS_FLOW_SCHEDULER == "cross_flow_length_bucketed_v1"


def test_cross_flow_embedding_restores_per_flow_artifacts(tmp_path, monkeypatch):
    calls = []

    def fake_embed(_model, _tokenizer, texts, _max_length, _mode, projection_head=None):
        del projection_head
        calls.append(list(texts))
        return np.asarray([[float(text[1:]), 1.0] for text in texts], dtype="float32")

    monkeypatch.setattr("extract_packet_embeddings_qwen.embed_batch", fake_embed)
    rows = lambda flow_id, packet_ids: [
        {
            "packet_id": packet_id,
            "prompt": f"p{packet_id}",
            "label": flow_id,
            "label_id": 0,
            "pcap_path": f"{flow_id}.pcap",
            "meta": {"packet_id": packet_id},
        }
        for packet_id in packet_ids
    ]
    flow_batch = [("a", rows("a", [0, 1])), ("b", rows("b", [2]))]
    output = tmp_path / "flow_embedding_index.jsonl"
    emb_dir = tmp_path / "packet_embeddings"
    emb_dir.mkdir()

    with output.open("w", encoding="utf-8") as handle:
        write_flow_batch_embeddings(
            handle,
            emb_dir,
            object(),
            object(),
            None,
            flow_batch,
            batch_size=2,
            max_length=16,
            embedding_mode="raw",
        )

    index_rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert calls == [["p0", "p1"], ["p2"]]
    assert [row["flow_id"] for row in index_rows] == ["a", "b"]
    assert np.load(index_rows[0]["embedding_path"]).shape == (2, 2)
    assert np.load(index_rows[1]["embedding_path"]).tolist() == [[2.0, 1.0]]


def test_cross_flow_embedding_length_buckets_then_restores_order(tmp_path, monkeypatch):
    calls = []

    def fake_embed(_model, _tokenizer, texts, _max_length, _mode, projection_head=None):
        del projection_head
        calls.append(list(texts))
        return np.asarray([[float(text.split(":", 1)[0])] for text in texts], dtype="float32")

    monkeypatch.setattr("extract_packet_embeddings_qwen.embed_batch", fake_embed)
    flow_rows = [
        {
            "packet_id": packet_id,
            "prompt": prompt,
            "label": "a",
            "label_id": 0,
            "pcap_path": "a.pcap",
            "meta": {"packet_id": packet_id},
        }
        for packet_id, prompt in enumerate(
            ["0:" + "x" * 30, "1:x", "2:" + "x" * 10]
        )
    ]
    output = tmp_path / "flow_embedding_index.jsonl"
    emb_dir = tmp_path / "packet_embeddings"
    emb_dir.mkdir()

    with output.open("w", encoding="utf-8") as handle:
        write_flow_batch_embeddings(
            handle,
            emb_dir,
            object(),
            object(),
            None,
            [("a", flow_rows)],
            batch_size=2,
            max_length=64,
            embedding_mode="raw",
        )

    row = json.loads(output.read_text(encoding="utf-8"))
    assert calls == [["1:x", "2:" + "x" * 10], ["0:" + "x" * 30]]
    assert np.load(row["embedding_path"]).reshape(-1).tolist() == [0.0, 1.0, 2.0]


def test_stage8_embedding_cmd_enables_resume_existing():
    args = SimpleNamespace(
        dataset="vpn-app",
        output_suffix="flowaware_ipport_rand_change_weight",
        embedding_suffix="rawproj_flowaware_ipport_rand_change_weight",
        base_model="Qwen/Qwen2.5-7B-Instruct",
        tower1_output_dir="checkpoints/tower1_qwen_multitask_vpn_app_flowaware_change_weight_split2_retrain",
        embedding_mode="concat",
        embedding_batch_size=8,
        embedding_max_length=1024,
        embedding_device="auto",
        local_files_only=True,
        embedding_resume_existing=True,
        embedding_num_shards=1,
        no_progress=True,
    )

    cmd = embedding_cmd(args, "train")

    assert "--resume_existing" in cmd
    assert cmd[cmd.index("--flow_batch_packets") + 1] == "128"
    assert "checkpoints/tower1_qwen_multitask_vpn_app_flowaware_change_weight_split2_retrain/adapter" in cmd


def test_stage8_embedding_audit_uses_same_split_contract():
    args = SimpleNamespace(
        dataset="vpn-app",
        output_suffix="strict",
        embedding_suffix="strict",
    )

    cmd = embedding_audit_cmd(args, "valid")

    assert cmd[1] == "audit_flow_embeddings.py"
    assert "reasoningDataset/vpn-app/valid_tower1_strict/packet_index.jsonl" in cmd
    assert "reasoningDataset/vpn-app/valid_embeddings_strict/flow_embedding_index.jsonl" in cmd
    assert "reasoningDataset/vpn-app/valid_embeddings_strict/embedding_audit.json" in cmd


def test_embedding_gpu_lock_uses_physical_visible_device_identity():
    assert logical_cuda_index("cuda") == 0
    assert logical_cuda_index("cuda:2") == 2
    assert logical_cuda_index("cpu") is None
    assert physical_cuda_token("cuda:1", "5,7") == "7"
    assert physical_cuda_token("cuda:0", "GPU-ab/cd") == "GPU-ab_cd"


def test_select_cuda_device_sets_requested_logical_index(monkeypatch):
    selected = []
    monkeypatch.setattr("extract_packet_embeddings_qwen.torch.cuda.set_device", selected.append)

    assert select_cuda_device("cuda:5") == 5
    assert selected == [5]


def test_select_cuda_device_ignores_non_cuda(monkeypatch):
    selected = []
    monkeypatch.setattr("extract_packet_embeddings_qwen.torch.cuda.set_device", selected.append)

    assert select_cuda_device("cpu") is None
    assert selected == []


def test_peft_device_kwargs_bind_explicit_shard_device():
    assert peft_device_kwargs("cuda:6") == {"torch_device": "cuda:6"}
    assert peft_device_kwargs("cpu") == {"torch_device": "cpu"}
    assert peft_device_kwargs("auto") == {}


def test_embedding_gpu_capacity_lock_wait_contract(tmp_path, monkeypatch):
    monkeypatch.setattr("extract_packet_embeddings_qwen.torch.cuda.is_available", lambda: True)
    monkeypatch.setattr(
        "extract_packet_embeddings_qwen.torch.cuda.mem_get_info",
        lambda _index: (24 * 1024 ** 3, 80 * 1024 ** 3),
    )

    handle = acquire_cuda_capacity_lock(
        "cuda:0",
        min_free_gb=20.0,
        poll_seconds=1.0,
        lock_dir=tmp_path,
    )

    assert handle is not None
    assert (tmp_path / "qwen_embedding_gpu_0.lock").exists()
    handle.close()


def test_post_tower1_resumes_both_views_and_shared_tower2(monkeypatch):
    events = []
    args = SimpleNamespace(
        splits="train,valid",
        dry_run=False,
        native_structural_suffix="",
        native_checkpoint="",
        paper_unified_stages="model",
    )
    intervention_args = SimpleNamespace(view="intervened")

    monkeypatch.setattr(
        stage8,
        "run_embedding_stage",
        lambda current_args, split: events.append(
            ("embedding", "intervened" if current_args is intervention_args else "factual", split)
        ),
    )
    monkeypatch.setattr(
        stage8,
        "tower1_preprocess_cmd",
        lambda _args, split: ["intervention_preprocess", split],
    )
    monkeypatch.setattr(
        stage8,
        "tower2_preprocess_cmd",
        lambda current_args, split: [
            "tower2_preprocess",
            "intervened" if current_args is intervention_args else "factual",
            split,
        ],
    )
    monkeypatch.setattr(
        stage8,
        "selected_paper_unified_stages",
        lambda _value: ["tower2_preprocess", "tower2_train"],
    )
    monkeypatch.setattr(stage8, "commands", lambda _args: [["tower2_train"]])
    monkeypatch.setattr(
        stage8,
        "run",
        lambda cmd, dry_run=False: events.append(tuple(cmd)),
    )

    stage8.run_post_tower1_pipeline(args, intervention_args)

    assert events[:2] == [
        ("embedding", "factual", "train"),
        ("embedding", "factual", "valid"),
    ]
    assert ("embedding", "intervened", "train") in events
    assert ("embedding", "intervened", "valid") in events
    assert ("tower2_preprocess", "factual", "train") in events
    assert ("tower2_preprocess", "intervened", "valid") in events
    assert ("tower2_train",) in events
    assert not any(event and event[0] == "tower1_train" for event in events)
