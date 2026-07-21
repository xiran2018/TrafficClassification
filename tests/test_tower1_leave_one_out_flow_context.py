from pathlib import Path

import pytest
import torch

from models.qwen_packet_multitask import packet_to_flow_prototype_loss
from train_tower1_multitask import (
    FlowBalancedPacketBatchSampler,
    RestartableDataIterator,
)


def test_leave_one_out_removes_self_inclusion_advantage():
    embeddings = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]],
        requires_grad=True,
    )
    labels = torch.tensor([0, 0, 1, 1])
    flow_ids = torch.tensor([10, 10, 20, 20])

    inclusive = packet_to_flow_prototype_loss(
        embeddings,
        labels,
        flow_ids,
        positive_mode="own_flow",
        context_mode="inclusive",
    )
    leave_one_out = packet_to_flow_prototype_loss(
        embeddings,
        labels,
        flow_ids,
        positive_mode="own_flow",
        context_mode="leave_one_out",
    )

    assert torch.isfinite(leave_one_out)
    assert leave_one_out > inclusive
    leave_one_out.backward()
    assert torch.isfinite(embeddings.grad).all()


def test_leave_one_out_own_flow_ignores_single_packet_flows():
    embeddings = torch.eye(3, requires_grad=True)
    labels = torch.tensor([0, 0, 1])
    flow_ids = torch.tensor([10, 20, 30])

    loss = packet_to_flow_prototype_loss(
        embeddings,
        labels,
        flow_ids,
        positive_mode="own_flow",
        context_mode="leave_one_out",
    )

    assert loss.item() == pytest.approx(0.0)
    loss.backward()
    assert embeddings.grad is not None


def test_distinct_flow_sampler_does_not_repeat_singleton_packet():
    rows = [
        {"flow_id": "singleton"},
        {"flow_id": "multi"},
        {"flow_id": "multi"},
    ]
    sampler = FlowBalancedPacketBatchSampler(
        rows,
        batch_size=4,
        packets_per_flow=2,
        seed=42,
        allow_packet_replacement=False,
    )

    batch = next(iter(sampler))
    assert batch.count(0) == 1
    assert sorted(batch) == [0, 1, 2]


def test_restartable_iterator_reenters_sampler_instead_of_caching_first_pass():
    rows = [
        {"flow_id": f"flow-{index}", "packet_id": index}
        for index in range(12)
    ]
    sampler = FlowBalancedPacketBatchSampler(
        rows,
        batch_size=4,
        packets_per_flow=1,
        seed=7,
    )
    iterator = RestartableDataIterator(sampler)

    first_pass = [next(iterator) for _ in range(len(sampler))]
    second_pass = [next(iterator) for _ in range(len(sampler))]

    assert iterator.completed_passes == 1
    assert sampler.epoch == 2
    assert first_pass != second_pass
    assert sorted(index for batch in first_pass for index in batch) == list(range(12))
    assert sorted(index for batch in second_pass for index in batch) == list(range(12))


def test_unknown_flow_context_mode_is_rejected():
    with pytest.raises(ValueError, match="context_mode"):
        packet_to_flow_prototype_loss(
            torch.eye(2),
            torch.tensor([0, 1]),
            torch.tensor([10, 20]),
            context_mode="unknown",
        )


def test_packet_and_flow_runners_expose_the_same_context_option():
    root = Path(__file__).resolve().parents[1]
    for runner in ("run_packet_level_pipeline.py", "run_stage8_flowaware_pipeline.py"):
        source = (root / runner).read_text(encoding="utf-8")
        assert '"--flow_proto_context"' in source
        assert 'choices=["inclusive", "leave_one_out"]' in source
