from train_tower1_multitask import FlowBalancedPacketBatchSampler


def _epoch_indices(sampler):
    return [index for batch in sampler for index in batch]


def test_coverage_cycle_visits_every_packet_before_repeating():
    rows = [{"flow_id": "flow-a", "packet_id": index} for index in range(5)]
    sampler = FlowBalancedPacketBatchSampler(
        rows,
        batch_size=2,
        packets_per_flow=2,
        seed=42,
        scheduler="coverage_cycle_dataloader_v1",
    )

    draws = _epoch_indices(sampler) + _epoch_indices(sampler) + _epoch_indices(sampler)

    assert len(set(draws[:5])) == 5
    assert draws[5] in set(draws[:5])


def test_coverage_cycle_is_reproducible_across_sampler_instances():
    rows = [
        {"flow_id": flow_id, "packet_id": packet_id}
        for flow_id in ("flow-a", "flow-b")
        for packet_id in range(7)
    ]
    samplers = [
        FlowBalancedPacketBatchSampler(
            rows,
            batch_size=4,
            packets_per_flow=2,
            seed=7,
            scheduler="coverage_cycle_dataloader_v1",
        )
        for _ in range(2)
    ]

    histories = [[_epoch_indices(sampler) for _ in range(5)] for sampler in samplers]

    assert histories[0] == histories[1]


def test_coverage_cycle_respects_no_replacement_for_short_flows():
    rows = [{"flow_id": "singleton", "packet_id": 0}]
    sampler = FlowBalancedPacketBatchSampler(
        rows,
        batch_size=2,
        packets_per_flow=2,
        allow_packet_replacement=False,
        scheduler="coverage_cycle_dataloader_v1",
    )

    assert _epoch_indices(sampler) == [0]
