#!/usr/bin/env python3
"""Audit effective cross-scale supervision under the exact Tower1 sampler."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from audit_tower1_contrastive_exposure import (
    file_sha256,
    load_rows,
    sampled_batches,
)


def load_paired_identities(path: Path) -> set[str]:
    identities: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            packet_uid = str(row.get("packet_uid") or "")
            if not packet_uid:
                raise ValueError(f"{path}:{line_number}: packet_uid is required")
            if packet_uid in identities:
                raise ValueError(f"duplicate paired packet_uid={packet_uid}")
            identities.add(packet_uid)
    return identities


def batch_cross_scale_exposure(
    rows: list[dict[str, Any]],
    batch: list[int],
    paired_identities: set[str],
) -> dict[str, int]:
    identities: dict[str, dict[str, Any]] = {}
    sampled_identity_counts: dict[str, int] = defaultdict(int)
    for row_index in batch:
        row = rows[row_index]
        uid = row["packet_uid"]
        sampled_identity_counts[uid] += 1
        previous = identities.setdefault(uid, row)
        if (
            previous["flow_id"] != row["flow_id"]
            or previous["label_id"] != row["label_id"]
        ):
            raise ValueError("one packet identity has conflicting metadata")

    by_flow: dict[str, list[str]] = defaultdict(list)
    flow_labels: dict[str, int] = {}
    for uid, row in identities.items():
        flow_id = row["flow_id"]
        previous_label = flow_labels.setdefault(flow_id, row["label_id"])
        if previous_label != row["label_id"]:
            raise ValueError("one flow has conflicting labels")
        by_flow[flow_id].append(uid)

    factual_context_flows = {
        flow_id for flow_id, uids in by_flow.items() if len(uids) > 0
    }
    intervened_context_flows = {
        flow_id
        for flow_id, uids in by_flow.items()
        if any(uid in paired_identities for uid in uids)
    }
    factual_to_intervened = 0
    intervened_to_factual = 0
    bidirectional = 0
    distinct_own_context = 0
    alias_only_false_context = 0
    for uid, row in identities.items():
        flow_id = row["flow_id"]
        label = row["label_id"]
        other_uids = [candidate for candidate in by_flow[flow_id] if candidate != uid]
        has_factual_own_context = bool(other_uids)
        has_intervened_own_context = any(
            candidate in paired_identities for candidate in other_uids
        )
        has_factual_negative = any(
            flow_labels[candidate_flow] != label
            for candidate_flow in factual_context_flows
            if candidate_flow != flow_id
        )
        has_intervened_negative = any(
            flow_labels[candidate_flow] != label
            for candidate_flow in intervened_context_flows
            if candidate_flow != flow_id
        )
        f2i = has_intervened_own_context and has_intervened_negative
        i2f = (
            uid in paired_identities
            and has_factual_own_context
            and has_factual_negative
        )
        factual_to_intervened += int(f2i)
        intervened_to_factual += int(i2f)
        bidirectional += int(f2i and i2f)
        distinct_own_context += int(has_factual_own_context)
        alias_only_false_context += int(
            sampled_identity_counts[uid] > 1 and not has_factual_own_context
        )

    unique_count = len(identities)
    return {
        "sampled_rows": len(batch),
        "unique_packet_identities": unique_count,
        "duplicate_rows": len(batch) - unique_count,
        "paired_packet_identities": sum(uid in paired_identities for uid in identities),
        "distinct_own_context_anchors": distinct_own_context,
        "factual_to_intervened_valid_anchors": factual_to_intervened,
        "intervened_to_factual_valid_anchors": intervened_to_factual,
        "bidirectional_valid_anchors": bidirectional,
        "alias_only_false_context_anchors": alias_only_false_context,
    }


def aggregate(rows: list[dict[str, int]]) -> dict[str, Any]:
    totals = {
        key: sum(row[key] for row in rows)
        for key in rows[0]
    } if rows else {}
    unique = max(int(totals.get("unique_packet_identities", 0)), 1)
    sampled = max(int(totals.get("sampled_rows", 0)), 1)
    totals.update(
        {
            "num_batches": len(rows),
            "duplicate_row_rate": totals.get("duplicate_rows", 0) / sampled,
            "paired_identity_rate": totals.get("paired_packet_identities", 0) / unique,
            "distinct_own_context_anchor_rate": totals.get(
                "distinct_own_context_anchors", 0
            )
            / unique,
            "factual_to_intervened_valid_anchor_rate": totals.get(
                "factual_to_intervened_valid_anchors", 0
            )
            / unique,
            "intervened_to_factual_valid_anchor_rate": totals.get(
                "intervened_to_factual_valid_anchors", 0
            )
            / unique,
            "bidirectional_valid_anchor_rate": totals.get(
                "bidirectional_valid_anchors", 0
            )
            / unique,
            "alias_only_false_context_anchor_rate": totals.get(
                "alias_only_false_context_anchors", 0
            )
            / unique,
        }
    )
    return totals


def audit(
    rows: list[dict[str, Any]],
    paired_identities: set[str],
    *,
    batch_size: int,
    packets_per_flow: int,
    epochs: int,
    seed: int,
) -> dict[str, Any]:
    epoch_reports = []
    all_batches = []
    for epoch in range(epochs):
        batches = [
            batch_cross_scale_exposure(rows, batch, paired_identities)
            for batch in sampled_batches(
                rows,
                batch_size,
                packets_per_flow,
                seed,
                epoch,
                flow_pairing="random",
            )
        ]
        all_batches.extend(batches)
        epoch_reports.append({"epoch": epoch + 1, **aggregate(batches)})
    return {
        "source_packets": len(rows),
        "source_flows": len({row["flow_id"] for row in rows}),
        "batch_size": batch_size,
        "packets_per_flow": packets_per_flow,
        "epochs": epochs,
        "seed": seed,
        "aggregate": aggregate(all_batches),
        "epochs_detail": epoch_reports,
    }


def parse_input(value: str) -> tuple[str, Path, Path]:
    parts = value.split("=", 1)
    if len(parts) != 2 or not parts[0] or "," not in parts[1]:
        raise argparse.ArgumentTypeError("--input must be NAME=FACTUAL_JSONL,PAIRED_JSONL")
    factual, paired = parts[1].split(",", 1)
    return parts[0], Path(factual), Path(paired)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", type=parse_input, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--packets_per_flow", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()
    if len({name for name, _, _ in args.input}) != len(args.input):
        parser.error("input names must be unique")

    reports = {}
    for name, factual, paired in args.input:
        report = audit(
            load_rows(factual),
            load_paired_identities(paired),
            batch_size=args.batch_size,
            packets_per_flow=args.packets_per_flow,
            epochs=args.epochs,
            seed=args.seed,
        )
        report["factual_path"] = str(factual.resolve())
        report["factual_sha256"] = file_sha256(factual)
        report["paired_path"] = str(paired.resolve())
        report["paired_sha256"] = file_sha256(paired)
        reports[name] = report
    payload = {
        "schema": "cross_scale_sampler_exposure_audit_v1",
        "scope": "training_inputs_and_exact_sampler_only",
        "test_predictions_used": False,
        "identity_policy": "exact_packet_uid_compaction",
        "context_policy": "leave_one_distinct_packet_out",
        "reports": reports,
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
