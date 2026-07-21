#!/usr/bin/env python3
"""Audit deterministic per-epoch packet resampling for Tower1 training."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from train_tower1_multitask import FlowBalancedPacketBatchSampler, file_sha256, load_jsonl


SCHEDULER = "epoch_resampled_dataloader_v1"


def epoch_sampling_audit(
    rows: list[dict[str, Any]],
    *,
    batch_size: int,
    packets_per_flow: int,
    seed: int,
    epochs: int,
) -> dict[str, Any]:
    if epochs < 2:
        raise ValueError("sampling audit requires at least two epochs")
    sampler = FlowBalancedPacketBatchSampler(
        rows,
        batch_size=batch_size,
        packets_per_flow=packets_per_flow,
        seed=seed,
    )
    epoch_rows: list[dict[str, Any]] = []
    cumulative_packets: set[int] = set()
    previous_packets: set[int] | None = None
    previous_by_flow: dict[str, tuple[int, ...]] | None = None

    for epoch in range(epochs):
        digest = hashlib.sha256()
        selected_packets: list[int] = []
        selected_by_flow: dict[str, list[int]] = {}
        batches = list(iter(sampler))
        for batch in batches:
            digest.update(json.dumps(batch, separators=(",", ":")).encode("ascii"))
            digest.update(b"\n")
            selected_packets.extend(batch)
            for index in batch:
                flow_id = str(rows[index].get("flow_id", index))
                selected_by_flow.setdefault(flow_id, []).append(index)
        packet_set = set(selected_packets)
        cumulative_packets.update(packet_set)
        normalized_by_flow = {
            flow_id: tuple(sorted(indices))
            for flow_id, indices in selected_by_flow.items()
        }
        if previous_packets is None:
            jaccard = None
            changed_flow_rate = None
        else:
            union = packet_set | previous_packets
            jaccard = len(packet_set & previous_packets) / max(1, len(union))
            common_flows = set(normalized_by_flow) & set(previous_by_flow or {})
            changed = sum(
                normalized_by_flow[flow_id] != previous_by_flow[flow_id]
                for flow_id in common_flows
            )
            changed_flow_rate = changed / max(1, len(common_flows))
        epoch_rows.append(
            {
                "epoch": epoch + 1,
                "sampler_seed": seed + epoch,
                "batch_sha256": digest.hexdigest(),
                "batches": len(batches),
                "drawn_packet_rows": len(selected_packets),
                "unique_packet_rows": len(packet_set),
                "cumulative_unique_packet_rows": len(cumulative_packets),
                "packet_jaccard_vs_previous": jaccard,
                "changed_flow_selection_rate_vs_previous": changed_flow_rate,
            }
        )
        previous_packets = packet_set
        previous_by_flow = normalized_by_flow

    hashes = [row["batch_sha256"] for row in epoch_rows]
    adjacent_change_rates = [
        row["changed_flow_selection_rate_vs_previous"]
        for row in epoch_rows[1:]
    ]
    return {
        "schema": "tower1_epoch_sampling_audit_v1",
        "scheduler": SCHEDULER,
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "packets_per_flow": packets_per_flow,
        "input_packet_rows": len(rows),
        "input_flows": len(sampler.flows),
        "all_epoch_batch_hashes_unique": len(set(hashes)) == len(hashes),
        "all_adjacent_epochs_change_flow_packet_selection": all(
            value is not None and value > 0.0 for value in adjacent_change_rates
        ),
        "final_cumulative_packet_coverage": len(cumulative_packets) / max(1, len(rows)),
        "epochs_detail": epoch_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet_aux_jsonl", required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--packets_per_flow", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    input_path = Path(args.packet_aux_jsonl)
    rows = load_jsonl(input_path, show_progress=True)
    report = epoch_sampling_audit(
        rows,
        batch_size=args.batch_size,
        packets_per_flow=args.packets_per_flow,
        seed=args.seed,
        epochs=args.epochs,
    )
    report["input"] = {
        "path": str(input_path.resolve()),
        "sha256": file_sha256(input_path),
    }
    trainer = Path(__file__).resolve().parent / "train_tower1_multitask.py"
    report["trainer_source"] = {
        "path": str(trainer),
        "sha256": file_sha256(trainer),
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
