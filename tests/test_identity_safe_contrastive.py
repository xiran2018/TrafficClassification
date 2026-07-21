import pytest
import torch

from train_tower1_multitask import PacketAuxCollator
from models.qwen_packet_multitask import flow_aware_contrastive_loss

from models.identity_safe_contrastive import (
    first_packet_identity_mask,
    identity_safe_flow_aware_contrastive_loss,
)


def loss(z, labels, flows, packets):
    return identity_safe_flow_aware_contrastive_loss(
        z,
        torch.tensor(labels),
        torch.tensor(flows),
        torch.tensor(packets),
        temperature=0.2,
        same_flow_weight=1.0,
        same_label_weight=1.0,
    )


def test_first_packet_identity_mask_keeps_one_occurrence_per_identity():
    mask = first_packet_identity_mask(torch.tensor([7, 7, 9, 7, 11, 9]))
    assert mask.tolist() == [True, False, True, False, True, False]


def test_alias_embedding_cannot_change_identity_safe_loss_or_receive_gradient():
    unique = torch.tensor(
        [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]], requires_grad=True
    )
    unique_loss = loss(unique, [0, 0, 1], [10, 11, 12], [100, 200, 300])

    with_alias = torch.tensor(
        [[1.0, 0.0], [-99.0, 42.0], [0.8, 0.2], [0.0, 1.0]],
        requires_grad=True,
    )
    alias_loss = loss(
        with_alias,
        [0, 0, 0, 1],
        [10, 10, 11, 12],
        [100, 100, 200, 300],
    )

    assert alias_loss.detach().item() == pytest.approx(unique_loss.detach().item())
    alias_loss.backward()
    assert with_alias.grad[1].abs().sum().item() == 0.0
    assert with_alias.grad[[0, 2]].abs().sum().item() > 0.0


def test_unique_packet_batches_are_exactly_equivalent_to_legacy_flow_supcon():
    torch.manual_seed(9)
    embeddings = torch.randn(16, 7)
    labels = torch.randint(0, 4, (16,))
    flow_ids = torch.arange(16) // 2
    packet_ids = torch.arange(16)
    legacy = flow_aware_contrastive_loss(
        embeddings,
        labels,
        flow_ids,
        temperature=0.07,
        same_flow_weight=1.0,
        same_label_weight=1.0,
    )
    identity_safe = identity_safe_flow_aware_contrastive_loss(
        embeddings,
        labels,
        flow_ids,
        packet_ids,
        temperature=0.07,
        same_flow_weight=1.0,
        same_label_weight=1.0,
    )
    assert identity_safe.item() == legacy.item()


def test_no_real_positive_returns_differentiable_zero():
    z = torch.randn(3, 4, requires_grad=True)
    value = loss(z, [0, 1, 2], [10, 11, 12], [100, 200, 300])

    assert value.item() == 0.0
    value.backward()
    assert z.grad.abs().sum().item() == 0.0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"temperature": 0.0}, "temperature"),
        ({"same_flow_weight": -1.0}, "weights"),
        ({"same_label_weight": -1.0}, "weights"),
    ],
)
def test_invalid_objective_configuration_is_rejected(kwargs, message):
    z = torch.randn(2, 4)
    with pytest.raises(ValueError, match=message):
        identity_safe_flow_aware_contrastive_loss(
            z,
            torch.tensor([0, 0]),
            torch.tensor([1, 1]),
            torch.tensor([10, 11]),
            **kwargs,
        )


def test_batch_alignment_is_required():
    with pytest.raises(ValueError, match="share batch size"):
        identity_safe_flow_aware_contrastive_loss(
            torch.randn(2, 4),
            torch.tensor([0]),
            torch.tensor([1, 1]),
            torch.tensor([10, 11]),
        )


class _FakeTokenizer:
    def __call__(self, texts, **_kwargs):
        width = max(len(text) for text in texts)
        ids = torch.zeros((len(texts), width), dtype=torch.long)
        mask = torch.zeros_like(ids)
        for index, value in enumerate(texts):
            ids[index, : len(value)] = 1
            mask[index, : len(value)] = 1
        return {"input_ids": ids, "attention_mask": mask}


def test_collator_propagates_stable_packet_identity_for_alias_rows():
    collator = PacketAuxCollator(
        _FakeTokenizer(), max_length=16, require_packet_identity=True
    )
    rows = [
        {"prompt": "a", "label_id": 0, "flow_id": "f", "packet_uid": "f_0"},
        {"prompt": "a", "label_id": 0, "flow_id": "f", "packet_uid": "f_0"},
        {"prompt": "b", "label_id": 0, "flow_id": "f", "packet_uid": "f_1"},
    ]
    batch = collator(rows)
    assert batch["packet_ids"][0] == batch["packet_ids"][1]
    assert batch["packet_ids"][0] != batch["packet_ids"][2]


def test_collator_rejects_missing_identity_when_d1_is_enabled():
    collator = PacketAuxCollator(
        _FakeTokenizer(), max_length=16, require_packet_identity=True
    )
    with pytest.raises(ValueError, match="packet_uid"):
        collator([{"prompt": "a", "label_id": 0, "flow_id": "f"}])
