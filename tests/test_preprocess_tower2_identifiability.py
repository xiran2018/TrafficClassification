import numpy as np
import pytest
import torch

from preprocess_tower2 import (
    attach_content_groups,
    apply_metadata_reference,
    build_identifiability_profile,
    build_samples_from_flow,
    packet_identifiability,
)
from train_tower2 import (
    FlowAggregationHead,
    content_group_mean_loss,
    configure_identifiability_adapter_only,
    content_group_metrics_from_lists,
    flow_stat_aux_loss,
    selected_metric_value,
)
from models.flow_transformer import AttentionPooling, FlowTransformerClassifier
from test_tower2 import checkpoint_flow_stat_meta_dim


ACK_PACKET = (
    "45 00 00 28 12 34 40 00 40 06 00 00 0a 00 00 01 0a 00 00 02 "
    "c3 50 01 bb 00 00 00 01 00 00 00 02 50 10 10 00 00 00 00 00"
)


def packet_meta(hex_prefix=ACK_PACKET, payload_len=0, flags="A"):
    return {
        "l3_hex_prefix": hex_prefix,
        "l3": "IPv4",
        "l4": "TCP",
        "packet_len": 40 + payload_len,
        "payload_len": payload_len,
        "payload_entropy": 0.0,
        "tcp_flags": flags,
        "direction": "C2S",
        "time": 0.0,
        "iat": 0.0,
        "seq": 1,
        "ack": 2,
    }


def test_training_profile_marks_cross_label_signature_as_unidentifiable():
    rows = [
        {"label_id": 0, "packet_metas": [packet_meta()]},
        {"label_id": 1, "packet_metas": [packet_meta()]},
    ]

    profile = build_identifiability_profile(rows)
    reliability, support, source = packet_identifiability(packet_meta(), profile)

    assert reliability == pytest.approx(0.0)
    assert support == 2
    assert source == "session"


def test_paired_metadata_reference_requires_aligned_label_and_packet_count(tmp_path):
    embedding_path = tmp_path / "flow.npy"
    np.save(embedding_path, np.zeros((2, 4), dtype=np.float32))
    paired = {
        "flow_id": "flow-a",
        "label_id": 3,
        "embedding_path": str(embedding_path),
        "packet_metas": [{"reduced": True}, {"reduced": True}],
    }
    reference = {
        "flow-a": {
            "flow_id": "flow-a",
            "label_id": 3,
            "packet_metas": [packet_meta(), packet_meta(payload_len=1)],
        }
    }

    output = apply_metadata_reference(paired, reference)
    assert output["packet_metas"] == reference["flow-a"]["packet_metas"]
    assert paired["packet_metas"] != output["packet_metas"]

    bad_label = {"flow-a": {**reference["flow-a"], "label_id": 4}}
    with pytest.raises(ValueError, match="label mismatch"):
        apply_metadata_reference(paired, bad_label)

    bad_count = {"flow-a": {**reference["flow-a"], "packet_metas": [packet_meta()]}}
    with pytest.raises(ValueError, match="packet count mismatch"):
        apply_metadata_reference(paired, bad_count)


def test_unseen_payload_packet_uses_label_free_informative_fallback():
    reliability, support, source = packet_identifiability(
        packet_meta(hex_prefix="", payload_len=32, flags="PA"),
        {"levels": {}},
    )

    assert reliability == 1.0
    assert support == 0
    assert source == "payload_fallback"


def test_tower2_samples_append_identifiability_before_meta_features(tmp_path):
    embedding_path = tmp_path / "embedding.npy"
    np.save(embedding_path, np.ones((2, 4), dtype=np.float32))
    row = {
        "flow_id": "flow-0",
        "label_id": 0,
        "embedding_path": str(embedding_path),
        "packet_metas": [packet_meta(), packet_meta()],
    }
    profile = build_identifiability_profile([row])

    seq_samples, graph_samples = build_samples_from_flow(
        row, window_size=2, stride=1, identifiability_profile=profile
    )

    assert len(seq_samples) == len(graph_samples) == 1
    sample = seq_samples[0]
    assert sample["x"].shape[1] == 4 + 2 + 14
    assert sample["packet_identifiability"].tolist() == [1.0, 1.0]
    assert sample["packet_identifiability_support"].tolist() == pytest.approx(
        [np.log1p(2), np.log1p(2)]
    )


def test_content_groups_are_attached_to_seq_and_graph_windows(tmp_path):
    embedding_a = tmp_path / "a.npy"
    embedding_b = tmp_path / "b.npy"
    np.save(embedding_a, np.ones((2, 4), dtype=np.float32))
    np.save(embedding_b, np.ones((2, 4), dtype=np.float32) * 2)
    shared_pcap = tmp_path / "shared.pcap"
    shared_pcap.write_bytes(b"same-flow-content")
    index_path = tmp_path / "flow_embedding_index.jsonl"
    index_path.write_text(
        "\n".join(
            [
                (
                    '{"flow_id":"flow-a","label_id":0,'
                    f'"pcap_path":"{shared_pcap}"'
                    "}"
                ),
                (
                    '{"flow_id":"flow-b","label_id":0,'
                    f'"pcap_path":"{shared_pcap}"'
                    "}"
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    rows = [
        {
            "flow_id": "flow-a",
            "label_id": 0,
            "embedding_path": str(embedding_a),
            "packet_metas": [packet_meta(), packet_meta(payload_len=1)],
        },
        {
            "flow_id": "flow-b",
            "label_id": 0,
            "embedding_path": str(embedding_b),
            "packet_metas": [packet_meta(), packet_meta(payload_len=1)],
        },
    ]

    rows, manifest = attach_content_groups(rows, str(index_path))
    seq_a, graph_a = build_samples_from_flow(rows[0], window_size=2, stride=1)
    seq_b, graph_b = build_samples_from_flow(rows[1], window_size=2, stride=1)

    assert manifest["num_flows"] == 2
    assert manifest["num_content_groups"] == 1
    assert manifest["duplicate_content_groups"] == 1
    assert rows[0]["content_group_id"] == rows[1]["content_group_id"]
    assert seq_a[0]["content_group_id"] == seq_b[0]["content_group_id"]
    assert graph_a[0]["content_hash"] == graph_b[0]["content_hash"]


def test_content_group_index_requires_matching_flow_and_label(tmp_path):
    embedding = tmp_path / "a.npy"
    np.save(embedding, np.ones((1, 4), dtype=np.float32))
    pcap = tmp_path / "a.pcap"
    pcap.write_bytes(b"content")
    index_path = tmp_path / "flow_embedding_index.jsonl"
    index_path.write_text(
        (
            '{"flow_id":"flow-a","label_id":1,'
            f'"pcap_path":"{pcap}"'
            "}\n"
        ),
        encoding="utf-8",
    )
    rows = [
        {
            "flow_id": "flow-a",
            "label_id": 0,
            "embedding_path": str(embedding),
            "packet_metas": [packet_meta()],
        }
    ]

    with pytest.raises(ValueError, match="label_mismatches"):
        attach_content_groups(rows, str(index_path))

    rows[0]["flow_id"] = "missing"
    with pytest.raises(ValueError, match="missing"):
        attach_content_groups(rows, str(index_path))


def test_content_group_metrics_count_duplicate_content_once():
    metrics = content_group_metrics_from_lists(
        y_true=[0, 0, 1, 1],
        y_pred=[0, 1, 1, 1],
        content_group_ids=[10, 10, 11, 12],
        num_classes=2,
    )

    assert metrics["content_group_count"] == 3
    assert metrics["content_group_rows"] == 4
    assert metrics["content_group_accuracy"] == pytest.approx(1.0)
    assert metrics["content_group_macro_f1"] == pytest.approx(1.0)
    assert selected_metric_value(metrics, "content_group_macro_f1") == pytest.approx(1.0)


def test_content_group_metric_selection_requires_group_metadata():
    with pytest.raises(ValueError, match="content_group_macro_f1"):
        selected_metric_value({"accuracy": 1.0, "macro_f1": 1.0}, "content_group_macro_f1")

    with pytest.raises(ValueError, match="conflicting labels"):
        content_group_metrics_from_lists(
            y_true=[0, 1],
            y_pred=[0, 1],
            content_group_ids=["same", "same"],
            num_classes=2,
        )


def test_content_group_mean_loss_downweights_duplicate_content():
    per_sample = torch.tensor([2.0, 4.0, 8.0])
    valid = torch.tensor([True, True, True])

    loss = content_group_mean_loss(per_sample, valid, ["dup", "dup", "unique"])

    assert loss.item() == pytest.approx(((2.0 + 4.0) / 2.0 + 8.0) / 2.0)
    assert loss.item() != pytest.approx(per_sample.mean().item())


def test_content_group_mean_loss_requires_complete_metadata():
    per_sample = torch.tensor([2.0, 4.0])
    valid = torch.tensor([True, True])

    with pytest.raises(ValueError, match="requires content_group_id"):
        content_group_mean_loss(per_sample, valid, None)
    with pytest.raises(ValueError, match="every valid sample"):
        content_group_mean_loss(per_sample, valid, ["known", None])


def test_flow_stat_aux_loss_uses_content_group_metadata_for_group_mean():
    class Args:
        flow_stat_aux_weight = 1.0
        label_smoothing = 0.0
        focal_gamma = 0.0
        content_group_loss_reduction = "group_mean"
        confidence_penalty_weight = 0.0

    logits = [
        torch.tensor([2.0, 0.0], requires_grad=True),
        torch.tensor([0.5, 1.0], requires_grad=True),
        torch.tensor([0.0, 2.0], requires_grad=True),
    ]
    labels = torch.tensor([0, 0, 1])

    loss = flow_stat_aux_loss(logits, labels, None, Args(), ["dup", "dup", "unique"])

    assert loss is not None
    assert loss.requires_grad
    loss.backward()
    assert logits[0].grad is not None

    with pytest.raises(ValueError, match="requires content_group_id"):
        flow_stat_aux_loss(logits, labels, None, Args())


def test_flow_stat_features_use_trailing_explicit_fields_and_distinct_counts():
    head = FlowAggregationHead(
        hidden_dim=4,
        num_classes=2,
        flow_stat_meta_dim=2,
        flow_stat_expert_weight=0.25,
    )
    windows = [
        torch.tensor([[1000.0, 1.0, 2.0], [2000.0, 3.0, 4.0]]),
        torch.tensor([[3000.0, 5.0, 6.0]]),
    ]

    features = head._flow_stat_features(windows, torch.zeros(4))

    assert features is not None
    assert features[:2].tolist() == pytest.approx([3.0, 4.0])
    assert features[2:4].tolist() == pytest.approx(
        torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]).std(
            dim=0, unbiased=False
        ).tolist()
    )
    assert features[4:6].tolist() == pytest.approx([1.0, 2.0])
    assert features[6:8].tolist() == pytest.approx([5.0, 6.0])
    assert features[8:].tolist() == pytest.approx(
        [torch.log1p(torch.tensor(3.0)).item(), torch.log1p(torch.tensor(2.0)).item()]
    )


def test_flow_stat_features_remove_padding_and_overlapping_packets():
    head = FlowAggregationHead(
        hidden_dim=4,
        num_classes=2,
        flow_stat_meta_dim=1,
        flow_stat_expert_weight=0.25,
    )
    windows = torch.tensor(
        [
            [[10.0, 1.0], [20.0, 2.0], [30.0, 3.0]],
            [[30.0, 3.0], [40.0, 4.0], [999.0, 999.0]],
        ]
    )
    mask = torch.tensor([[True, True, True], [True, True, False]])

    features = head._flow_stat_features(
        windows,
        torch.zeros(4),
        window_mask=mask,
        window_ranges=[(0, 3), (2, 4)],
    )

    assert features is not None
    assert features[:4].tolist() == pytest.approx(
        [
            2.5,
            torch.tensor([1.0, 2.0, 3.0, 4.0]).std(unbiased=False).item(),
            1.0,
            4.0,
        ]
    )
    assert features[4:].tolist() == pytest.approx(
        [torch.log1p(torch.tensor(4.0)).item(), torch.log1p(torch.tensor(2.0)).item()]
    )


def test_flow_stat_checkpoint_dimension_prefers_explicit_contract():
    assert checkpoint_flow_stat_meta_dim(
        {"meta_feature_dim": 141, "flow_stat_meta_dim": 13}
    ) == 13
    assert checkpoint_flow_stat_meta_dim({"meta_feature_dim": 141}) == 141


def test_flow_attention_prior_favors_identifiable_windows_and_can_be_disabled():
    h = torch.eye(2)
    reliability = torch.tensor([1.0, 0.1])
    plain = FlowAggregationHead(2, 2, identifiability_attention_prior=False)
    informed = FlowAggregationHead(
        2,
        2,
        identifiability_attention_prior=True,
        identifiability_prior_init=1.0,
    )
    for head in (plain, informed):
        for parameter in head.score.parameters():
            parameter.data.zero_()

    assert plain.pool(h, reliability).tolist() == pytest.approx([0.5, 0.5])
    assert informed.pool(h, reliability).tolist() == pytest.approx(
        [1.0 / 1.1, 0.1 / 1.1], rel=1e-5
    )


def test_packet_attention_pooling_uses_the_same_reliability_rule():
    h = torch.eye(2).unsqueeze(0)
    reliability = torch.tensor([[1.0, 0.1]])
    pool = AttentionPooling(2, reliability_prior=True, reliability_prior_init=1.0)
    for parameter in pool.score.parameters():
        parameter.data.zero_()

    output = pool(h, reliability=reliability)

    assert output.squeeze(0).tolist() == pytest.approx(
        [1.0 / 1.1, 0.1 / 1.1], rel=1e-5
    )


def test_dual_packet_pooling_preserves_high_and_low_identifiability_views():
    h = torch.eye(2).unsqueeze(0)
    reliability = torch.tensor([[1.0, 0.1]])
    pool = AttentionPooling(
        2, reliability_dual=True, reliability_prior_init=1.0
    )
    for parameter in pool.score.parameters():
        parameter.data.zero_()
    for parameter in pool.dual_gate.parameters():
        parameter.data.zero_()

    output = pool(h, reliability=reliability).squeeze(0)
    high = torch.tensor([1.0 / 1.1, 0.1 / 1.1])
    low = torch.tensor([0.05 / 0.95, 0.9 / 0.95])
    expected = (torch.tensor([0.5, 0.5]) + high + low) / 3.0

    assert output.tolist() == pytest.approx(expected.tolist(), rel=1e-5)


def test_residual_dual_pooling_keeps_a_minimum_base_weight():
    h = torch.eye(2).unsqueeze(0)
    reliability = torch.tensor([[1.0, 0.1]])
    pool = AttentionPooling(
        2,
        reliability_dual=True,
        reliability_prior_init=1.0,
        reliability_residual_max_weight=0.25,
        reliability_residual_init=1.0 - 1e-4,
    )
    for parameter in pool.score.parameters():
        parameter.data.zero_()
    for parameter in pool.dual_gate.parameters():
        parameter.data.zero_()

    pool(h, reliability=reliability)
    gate = pool.last_reliability_gate.squeeze(0)

    assert gate.sum().item() == pytest.approx(1.0)
    assert gate[0].item() >= 0.75
    assert gate[1:].sum().item() <= 0.25


def test_identifiability_adapter_only_freezes_the_backbone_and_flow_head():
    model = FlowTransformerClassifier(
        20,
        3,
        hidden_dim=16,
        num_layers=1,
        num_heads=4,
        identifiability_feature_index=4,
        identifiability_dual_pooling=True,
        identifiability_residual_max_weight=0.25,
    )
    flow_head = FlowAggregationHead(16, 3)

    trainable = configure_identifiability_adapter_only(model, flow_head)

    assert trainable
    assert all(name.startswith("pool.") for name in trainable)
    assert all(not parameter.requires_grad for parameter in flow_head.parameters())
    assert all(
        parameter.requires_grad == (name in trainable)
        for name, parameter in model.named_parameters()
    )


def test_zero_profile_feature_mode_is_dimension_matched_and_pooling_free():
    torch.manual_seed(7)
    model = FlowTransformerClassifier(
        8,
        3,
        hidden_dim=8,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        identifiability_feature_index=2,
        identifiability_pooling=False,
        identifiability_feature_mode="zero",
    ).eval()
    x_a = torch.randn(2, 4, 8)
    x_b = x_a.clone()
    x_b[..., 2:4] = torch.randn_like(x_b[..., 2:4]) * 100.0
    mask = torch.ones(2, 4, dtype=torch.bool)

    out_a = model(x_a, mask)
    out_b = model(x_b, mask)

    assert torch.equal(out_a["logits"], out_b["logits"])
    assert not hasattr(model.pool, "reliability_prior_raw_scale")


def test_evidence_adapter_starts_at_identity_with_a_live_output_gradient():
    torch.manual_seed(11)
    h = torch.randn(2, 5, 8, requires_grad=True)
    reliability = torch.rand(2, 5)
    pool = AttentionPooling(
        8,
        reliability_evidence_adapter=True,
        reliability_prior_init=0.25,
        reliability_adapter_max_delta=0.25,
    )
    base_pool = AttentionPooling(8)
    base_pool.score.load_state_dict(pool.score.state_dict())

    output = pool(h, reliability=reliability)
    expected = base_pool(h)

    assert torch.equal(output, expected)
    output.sum().backward()
    assert pool.evidence_adapter[-1].bias.grad is not None
    assert pool.evidence_adapter[-1].bias.grad.abs().sum().item() > 0
