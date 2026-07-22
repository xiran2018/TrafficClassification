from audit_tower1_epoch_sampling import epoch_sampling_audit


def test_epoch_sampling_audit_proves_deterministic_resampling_and_coverage():
    rows = []
    for flow in range(6):
        for packet in range(5):
            rows.append({"flow_id": f"flow-{flow}", "packet_id": packet})

    first = epoch_sampling_audit(
        rows, batch_size=6, packets_per_flow=2, seed=42, epochs=4
    )
    second = epoch_sampling_audit(
        rows, batch_size=6, packets_per_flow=2, seed=42, epochs=4
    )

    assert first == second
    assert first["all_epoch_batch_hashes_unique"] is True
    assert first["all_adjacent_epochs_change_flow_packet_selection"] is True
    assert first["final_cumulative_packet_coverage"] > 0.4
    assert [row["sampler_seed"] for row in first["epochs_detail"]] == [42, 43, 44, 45]


def test_singleton_flows_change_order_even_when_packet_identity_cannot_change():
    rows = [{"flow_id": f"flow-{index}"} for index in range(12)]
    report = epoch_sampling_audit(
        rows, batch_size=4, packets_per_flow=1, seed=7, epochs=3
    )

    assert report["all_epoch_batch_hashes_unique"] is True
    assert report["all_adjacent_epochs_change_flow_packet_selection"] is False
    assert report["final_cumulative_packet_coverage"] == 1.0


def test_coverage_cycle_audit_proves_no_repeat_before_full_flow_coverage():
    rows = [
        {"flow_id": f"flow-{flow}", "packet_id": packet}
        for flow in range(4)
        for packet in range(7)
    ]

    report = epoch_sampling_audit(
        rows,
        batch_size=4,
        packets_per_flow=2,
        seed=42,
        epochs=3,
        scheduler="coverage_cycle_dataloader_v1",
    )

    assert report["scheduler"] == "coverage_cycle_dataloader_v1"
    assert report["coverage_cycle_no_early_repeat_verified"] is True
    assert report["final_cumulative_packet_coverage"] == 24 / 28
