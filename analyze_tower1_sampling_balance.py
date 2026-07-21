import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict


def normalized_class_weights(counts: Dict[int, int], method: str, beta: float, strength: float) -> Dict[int, float]:
    if method == "inverse":
        weights = {label: 1.0 / max(count, 1) for label, count in counts.items()}
    elif method == "effective":
        weights = {
            label: (1.0 - beta) / max(1.0 - beta ** count, 1e-12)
            for label, count in counts.items()
        }
    else:
        weights = {label: 1.0 for label in counts}
    mean_weight = sum(weights.values()) / max(len(weights), 1)
    weights = {label: (weight / max(mean_weight, 1e-12)) ** strength for label, weight in weights.items()}
    mean_weight = sum(weights.values()) / max(len(weights), 1)
    return {label: weight / max(mean_weight, 1e-12) for label, weight in weights.items()}


def flow_balanced_objective_exposure(
    flow_counts: Dict[int, int], class_weights: Dict[int, float]
) -> dict:
    """Expected weighted CE mass when each flow is visited once per epoch."""
    exposure = {
        label: float(count) * float(class_weights.get(label, 1.0))
        for label, count in flow_counts.items()
    }
    positive = [value for value in exposure.values() if value > 0.0]
    ratio = max(positive) / min(positive) if positive else 0.0
    total = sum(exposure.values())
    normalized = {
        label: value / total if total > 0.0 else 0.0
        for label, value in exposure.items()
    }
    return {
        "class_weighted_mass": dict(sorted(exposure.items())),
        "normalized_mass": dict(sorted(normalized.items())),
        "imbalance_ratio": ratio,
    }


def audit(
    path: Path,
    method: str,
    beta: float,
    strengths: list[float],
    packets_per_flow: int = 2,
) -> dict:
    if packets_per_flow <= 0:
        raise ValueError("packets_per_flow must be positive")
    packet_counts: Counter[int] = Counter()
    flow_labels: Dict[str, int] = {}
    flow_packet_counts: Counter[str] = Counter()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            label = int(row["label_id"])
            flow_id = str(row.get("flow_id", ""))
            if not flow_id:
                raise ValueError(f"Missing flow_id in {path}")
            previous = flow_labels.setdefault(flow_id, label)
            if previous != label:
                raise ValueError(f"Conflicting labels for flow_id={flow_id} in {path}")
            packet_counts[label] += 1
            flow_packet_counts[flow_id] += 1
    flow_counts = Counter(flow_labels.values())
    short_flows = {
        flow_id: count
        for flow_id, count in flow_packet_counts.items()
        if count < packets_per_flow
    }
    replacement_slots = sum(packets_per_flow - count for count in short_flows.values())
    sampled_slots = len(flow_packet_counts) * packets_per_flow
    class_short_counts = Counter(flow_labels[flow_id] for flow_id in short_flows)
    length_histogram = Counter(min(count, 10) for count in flow_packet_counts.values())
    flow_count_weights = {
        str(strength): normalized_class_weights(flow_counts, method, beta, strength)
        for strength in strengths
    }
    unit_weights = {label: 1.0 for label in flow_counts}
    return {
        "path": str(path),
        "num_classes": len(packet_counts),
        "num_packets": sum(packet_counts.values()),
        "num_flows": len(flow_labels),
        "packet_counts": dict(sorted(packet_counts.items())),
        "flow_counts": dict(sorted(flow_counts.items())),
        "flow_length_audit": {
            "packets_per_flow": packets_per_flow,
            "singleton_flows": sum(count == 1 for count in flow_packet_counts.values()),
            "singleton_flow_rate": (
                sum(count == 1 for count in flow_packet_counts.values())
                / max(len(flow_packet_counts), 1)
            ),
            "flows_requiring_replacement": len(short_flows),
            "flow_replacement_rate": len(short_flows) / max(len(flow_packet_counts), 1),
            "replacement_slots_per_epoch": replacement_slots,
            "sampled_slots_per_epoch": sampled_slots,
            "replacement_slot_rate": replacement_slots / max(sampled_slots, 1),
            "class_flows_requiring_replacement": dict(sorted(class_short_counts.items())),
            "length_histogram_capped_at_10": dict(sorted(length_histogram.items())),
            "supcon_interpretation": (
                "replacement rows preserve flow-balanced CE exposure but create "
                "duplicate-packet same-flow pairs unless packet identity is masked"
            ),
        },
        "packet_imbalance_ratio": max(packet_counts.values()) / max(min(packet_counts.values()), 1),
        "flow_imbalance_ratio": max(flow_counts.values()) / max(min(flow_counts.values()), 1),
        "flow_count_weights": flow_count_weights,
        "flow_balanced_objective_exposure": {
            "unweighted": flow_balanced_objective_exposure(flow_counts, unit_weights),
            **{
                str(strength): flow_balanced_objective_exposure(flow_counts, weights)
                for strength, weights in (
                    (strength, flow_count_weights[str(strength)])
                    for strength in strengths
                )
            },
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--method", choices=["none", "inverse", "effective"], default="effective")
    ap.add_argument("--beta", type=float, default=0.9999)
    ap.add_argument("--strengths", default="0.5,1.0")
    ap.add_argument("--packets_per_flow", type=int, default=2)
    ap.add_argument("--output_json", default="")
    args = ap.parse_args()
    strengths = [float(value) for value in args.strengths.split(",") if value.strip()]
    if not strengths or any(not 0.0 <= value <= 1.0 for value in strengths):
        raise ValueError("--strengths must contain comma-separated values in [0,1]")
    reports = [
        audit(
            Path(path),
            args.method,
            args.beta,
            strengths,
            packets_per_flow=args.packets_per_flow,
        )
        for path in args.paths
    ]
    payload = {
        "method": args.method,
        "beta": args.beta,
        "packets_per_flow": args.packets_per_flow,
        "reports": reports,
    }
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
    for report in reports:
        print(
            f"{report['path']}: classes={report['num_classes']} packets={report['num_packets']} "
            f"flows={report['num_flows']} packet_ratio={report['packet_imbalance_ratio']:.2f} "
            f"flow_ratio={report['flow_imbalance_ratio']:.2f}"
        )
        flow_audit = report["flow_length_audit"]
        print(
            "  replacement exposure: "
            f"flows={flow_audit['flows_requiring_replacement']} "
            f"({flow_audit['flow_replacement_rate']:.2%}), "
            f"slots={flow_audit['replacement_slots_per_epoch']} "
            f"({flow_audit['replacement_slot_rate']:.2%})"
        )
        for strength, weights in report["flow_count_weights"].items():
            exposure = report["flow_balanced_objective_exposure"][strength]
            print(
                f"  strength={strength} weight_range="
                f"[{min(weights.values()):.4f}, {max(weights.values()):.4f}] "
                f"objective_exposure_ratio={exposure['imbalance_ratio']:.2f}"
            )


if __name__ == "__main__":
    main()
