import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows(path: Path) -> List[dict]:
    rows = []
    seen_uids = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            flow_id = str(row.get("flow_id", ""))
            packet_uid = str(row.get("packet_uid", ""))
            if not flow_id or not packet_uid:
                raise ValueError(
                    f"{path}:{line_number} requires explicit flow_id and packet_uid"
                )
            if packet_uid in seen_uids:
                raise ValueError(f"duplicate source packet_uid={packet_uid} in {path}")
            seen_uids.add(packet_uid)
            rows.append(
                {
                    "flow_id": flow_id,
                    "packet_uid": packet_uid,
                    "label_id": int(row["label_id"]),
                }
            )
    if not rows:
        raise ValueError(f"No rows loaded from {path}")
    return rows


def sampled_batches(
    rows: List[dict],
    batch_size: int,
    packets_per_flow: int,
    seed: int,
    epoch: int,
    flow_pairing: str = "random",
) -> Iterable[List[int]]:
    if batch_size <= 0 or packets_per_flow <= 0:
        raise ValueError("batch_size and packets_per_flow must be positive")
    flow_to_indices: Dict[str, List[int]] = {}
    for index, row in enumerate(rows):
        flow_to_indices.setdefault(row["flow_id"], []).append(index)
    if flow_pairing not in {"random", "same_class"}:
        raise ValueError("flow_pairing must be random or same_class")
    flow_labels = {}
    for row in rows:
        previous = flow_labels.setdefault(row["flow_id"], row["label_id"])
        if previous != row["label_id"]:
            raise ValueError(f"conflicting labels for flow_id={row['flow_id']}")

    flows_per_batch = max(1, batch_size // packets_per_flow)
    if flow_pairing == "same_class" and flows_per_batch % 2:
        raise ValueError(
            "same_class flow pairing requires an even number of flows per batch"
        )
    rng = random.Random(seed + epoch)
    if flow_pairing == "random":
        flows = list(flow_to_indices)
        rng.shuffle(flows)
        flow_batches = [
            flows[start : start + flows_per_batch]
            for start in range(0, len(flows), flows_per_batch)
        ]
    else:
        class_flows = defaultdict(list)
        for flow_id in flow_to_indices:
            class_flows[flow_labels[flow_id]].append(flow_id)
        pair_units = []
        singleton_units = []
        for label in sorted(class_flows):
            label_flows = class_flows[label]
            rng.shuffle(label_flows)
            for start in range(0, len(label_flows), 2):
                unit = label_flows[start : start + 2]
                (pair_units if len(unit) == 2 else singleton_units).append(unit)
        rng.shuffle(pair_units)
        rng.shuffle(singleton_units)
        flow_batches = []
        while pair_units or singleton_units:
            current = []
            while len(current) < flows_per_batch:
                remaining = flows_per_batch - len(current)
                if pair_units and remaining >= 2:
                    current.extend(pair_units.pop())
                elif singleton_units:
                    current.extend(singleton_units.pop())
                else:
                    break
            if current:
                flow_batches.append(current)

    for selected_flows in flow_batches:
        batch = []
        for flow_id in selected_flows:
            indices = flow_to_indices[flow_id]
            if len(indices) >= packets_per_flow:
                batch.extend(rng.sample(indices, packets_per_flow))
            else:
                batch.extend(rng.choice(indices) for _ in range(packets_per_flow))
        if batch:
            yield batch[:batch_size]


def batch_exposure(
    rows: List[dict],
    batch: List[int],
    same_flow_weight: float,
    same_label_weight: float,
) -> dict:
    first_identity_positions = []
    seen_uids = set()
    for position, row_index in enumerate(batch):
        packet_uid = rows[row_index]["packet_uid"]
        if packet_uid not in seen_uids:
            seen_uids.add(packet_uid)
            first_identity_positions.append(position)

    def positive_mass(positions: List[int]) -> tuple[float, int]:
        mass = 0.0
        valid_anchors = 0
        for left in positions:
            left_row = rows[batch[left]]
            anchor_mass = 0.0
            for right in positions:
                if left == right:
                    continue
                right_row = rows[batch[right]]
                if left_row["label_id"] == right_row["label_id"]:
                    anchor_mass += same_label_weight
                if left_row["flow_id"] == right_row["flow_id"]:
                    anchor_mass += same_flow_weight
            mass += anchor_mass
            valid_anchors += int(anchor_mass > 0.0)
        return mass, valid_anchors

    all_positions = list(range(len(batch)))
    naive_mass, naive_valid_anchors = positive_mass(all_positions)
    safe_mass, safe_valid_anchors = positive_mass(first_identity_positions)
    alias_mass = 0.0
    alias_pairs = 0
    for left in all_positions:
        left_row = rows[batch[left]]
        for right in all_positions:
            if left == right:
                continue
            right_row = rows[batch[right]]
            if left_row["packet_uid"] != right_row["packet_uid"]:
                continue
            alias_pairs += 1
            if left_row["label_id"] == right_row["label_id"]:
                alias_mass += same_label_weight
            if left_row["flow_id"] == right_row["flow_id"]:
                alias_mass += same_flow_weight

    num_rows = len(batch)
    num_unique = len(first_identity_positions)
    return {
        "sampled_rows": num_rows,
        "unique_packet_identities": num_unique,
        "duplicate_rows": num_rows - num_unique,
        "naive_denominator_pairs": num_rows * max(num_rows - 1, 0),
            "identity_safe_denominator_pairs": num_unique * max(num_unique - 1, 0),
        "naive_positive_weight_mass": naive_mass,
        "identity_safe_positive_weight_mass": safe_mass,
        "alias_positive_pairs": alias_pairs,
        "alias_positive_weight_mass": alias_mass,
        "naive_valid_anchors": naive_valid_anchors,
        "identity_safe_valid_anchors": safe_valid_anchors,
    }


def aggregate_exposure(stats: List[dict]) -> dict:
    totals = defaultdict(float)
    integer_fields = {
        "sampled_rows",
        "unique_packet_identities",
        "duplicate_rows",
        "naive_denominator_pairs",
        "identity_safe_denominator_pairs",
        "alias_positive_pairs",
        "naive_valid_anchors",
        "identity_safe_valid_anchors",
    }
    for row in stats:
        for key, value in row.items():
            totals[key] += value
    for key in integer_fields:
        totals[key] = int(totals[key])

    naive_mass = totals["naive_positive_weight_mass"]
    naive_denominator = totals["naive_denominator_pairs"]
    sampled_rows = totals["sampled_rows"]
    totals.update(
        {
            "num_batches": len(stats),
            "duplicate_row_rate": totals["duplicate_rows"] / max(sampled_rows, 1),
            "alias_share_of_naive_positive_weight": (
                totals["alias_positive_weight_mass"] / max(naive_mass, 1e-12)
            ),
            "positive_weight_removed_by_identity_dedup_rate": (
                (naive_mass - totals["identity_safe_positive_weight_mass"])
                / max(naive_mass, 1e-12)
            ),
            "denominator_pairs_removed_by_identity_dedup_rate": (
                (naive_denominator - totals["identity_safe_denominator_pairs"])
                / max(naive_denominator, 1)
            ),
            "naive_valid_anchor_rate": (
                totals["naive_valid_anchors"] / max(sampled_rows, 1)
            ),
            "identity_safe_valid_anchor_rate": (
                totals["identity_safe_valid_anchors"]
                / max(totals["unique_packet_identities"], 1)
            ),
        }
    )
    return dict(totals)


def audit_rows(
    rows: List[dict],
    batch_size: int,
    packets_per_flow: int,
    epochs: int,
    seed: int,
    same_flow_weight: float,
    same_label_weight: float,
    flow_pairing: str = "random",
) -> dict:
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    epoch_reports = []
    all_batch_stats = []
    for epoch in range(epochs):
        batch_stats = [
            batch_exposure(rows, batch, same_flow_weight, same_label_weight)
            for batch in sampled_batches(
                rows,
                batch_size,
                packets_per_flow,
                seed,
                epoch,
                flow_pairing=flow_pairing,
            )
        ]
        all_batch_stats.extend(batch_stats)
        epoch_reports.append({"epoch": epoch + 1, **aggregate_exposure(batch_stats)})
    return {
        "num_source_packets": len(rows),
        "num_source_flows": len({row["flow_id"] for row in rows}),
        "expected_batches_per_epoch": math.ceil(
            len({row["flow_id"] for row in rows})
            / max(1, batch_size // packets_per_flow)
        ),
        "batch_size": batch_size,
        "packets_per_flow": packets_per_flow,
        "epochs": epochs,
        "seed": seed,
        "same_flow_positive_weight": same_flow_weight,
        "same_label_positive_weight": same_label_weight,
        "flow_pairing": flow_pairing,
        "aggregate": aggregate_exposure(all_batch_stats),
        "epochs_detail": epoch_reports,
        "interpretation": {
            "scope": "training_input_and_sampler_only_no_validation_or_test_predictions",
            "naive": "every sampled row participates as anchor and denominator candidate",
            "identity_safe": (
                "only the first occurrence of each packet_uid in a batch participates "
                "in contrastive roles; classification exposure is unchanged"
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--packets_per_flow", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--same_flow_weight", type=float, default=1.0)
    parser.add_argument("--same_label_weight", type=float, default=1.0)
    parser.add_argument(
        "--flow_pairing", choices=["random", "same_class"], default="random"
    )
    parser.add_argument("--output_json", default="")
    args = parser.parse_args()

    reports = []
    for value in args.paths:
        path = Path(value)
        report = audit_rows(
            load_rows(path),
            batch_size=args.batch_size,
            packets_per_flow=args.packets_per_flow,
            epochs=args.epochs,
            seed=args.seed,
            same_flow_weight=args.same_flow_weight,
            same_label_weight=args.same_label_weight,
            flow_pairing=args.flow_pairing,
        )
        report["path"] = str(path)
        report["input_sha256"] = file_sha256(path)
        reports.append(report)

    payload = {
        "schema": "tower1_contrastive_exposure_audit_v1",
        "reports": reports,
    }
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    for report in reports:
        aggregate = report["aggregate"]
        print(
            f"{report['path']}: flows={report['num_source_flows']} "
            f"flow_pairing={report['flow_pairing']} "
            f"duplicate_rows={aggregate['duplicate_row_rate']:.2%} "
            f"alias_positive_mass={aggregate['alias_share_of_naive_positive_weight']:.2%} "
            f"dedup_positive_mass_removed="
            f"{aggregate['positive_weight_removed_by_identity_dedup_rate']:.2%} "
            f"dedup_denominator_pairs_removed="
            f"{aggregate['denominator_pairs_removed_by_identity_dedup_rate']:.2%}"
        )


if __name__ == "__main__":
    main()
