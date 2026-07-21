#!/usr/bin/env python3
"""Compare VPN/TLS results with protocol-matched SWEET paper references."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from unified_framework_spec import FLOW_LEVEL_RESULTS, PACKET_LEVEL_RESULTS


SWEET_PAPER = "The Sweet Danger of Sugar-- Debunking Representation Learning"
SWEET_REFERENCES = {
    "packet-level": {
        "vpn-app": {
            "frozen_representation": {
                "model": "Pcap-Encoder",
                "table": 3,
                "accuracy": 0.835,
                "macro_f1": 0.710,
            },
            "end_to_end": {
                "model": "Pcap-Encoder (unfrozen)",
                "table": 4,
                "accuracy": 0.856,
                "macro_f1": 0.748,
            },
        },
        "tls-120": {
            "frozen_representation": {
                "model": "Pcap-Encoder",
                "table": 3,
                "accuracy": 0.710,
                "macro_f1": 0.637,
            },
            "end_to_end": {
                "model": "Pcap-Encoder (unfrozen)",
                "table": 4,
                "accuracy": 0.773,
                "macro_f1": 0.692,
            },
        },
    },
    "flow-level": {
        "vpn-app": {
            "frozen_representation": {
                "model": "Pcap-Encoder",
                "table": 9,
                "accuracy": 0.692,
                "macro_f1": 0.622,
            },
            "end_to_end": {
                "model": "YaTC (best paired AC/F1 among reported unfrozen models)",
                "table": 9,
                "accuracy": 0.600,
                "macro_f1": 0.548,
            },
        },
        "tls-120": {
            "frozen_representation": {
                "model": "Pcap-Encoder",
                "table": 9,
                "accuracy": 0.713,
                "macro_f1": 0.681,
            },
            "end_to_end": {
                "model": "netFound (unfrozen)",
                "table": 9,
                "accuracy": 0.908,
                "macro_f1": 0.897,
            },
        },
    },
}


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def extract_metrics(payload: dict[str, Any], task: str) -> tuple[float, float]:
    values = payload.get("metrics") or payload.get("test_metrics") or {}
    if task == "flow-level":
        values = values.get("flow_level") or values
    else:
        values = values.get("packet_level") or values
    accuracy = values.get("accuracy", values.get("flow_acc"))
    macro_f1 = values.get("macro_f1", values.get("flow_macro_f1"))
    if accuracy is None or macro_f1 is None:
        raise ValueError(f"missing {task} accuracy/macro_f1")
    return float(accuracy), float(macro_f1)


def compare_metrics(accuracy: float, macro_f1: float, reference: dict[str, Any]) -> dict[str, Any]:
    delta_accuracy = accuracy - float(reference["accuracy"])
    delta_macro_f1 = macro_f1 - float(reference["macro_f1"])
    return {
        **reference,
        "delta_accuracy": delta_accuracy,
        "delta_macro_f1": delta_macro_f1,
        "exceeds_accuracy": delta_accuracy > 0.0,
        "exceeds_macro_f1": delta_macro_f1 > 0.0,
        "exceeds_both": delta_accuracy > 0.0 and delta_macro_f1 > 0.0,
    }


def compare_result(task: str, dataset: str, path: str) -> dict[str, Any]:
    if task not in SWEET_REFERENCES or dataset not in SWEET_REFERENCES[task]:
        raise ValueError(f"no SWEET reference for {task}/{dataset}")
    payload = load_json(path)
    accuracy, macro_f1 = extract_metrics(payload, task)
    provenance = payload.get("publication_provenance") or {}
    references = SWEET_REFERENCES[task][dataset]
    spec = (PACKET_LEVEL_RESULTS if task == "packet-level" else FLOW_LEVEL_RESULTS)[dataset]
    target_accuracy = spec.target_accuracy
    target_macro_f1 = spec.target_macro_f1
    return {
        "task": task,
        "dataset": dataset,
        "path": path,
        "our_protocol": "downstream_adapted_lora",
        "publication_provenance": provenance,
        "strict_shared_core_v2_result": provenance.get("status") == "strict_shared_core_v2",
        "primary_comparator": "end_to_end",
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "predeclared_target": {
            "accuracy": target_accuracy,
            "macro_f1": target_macro_f1,
            "met": bool(
                target_accuracy is not None
                and target_macro_f1 is not None
                and accuracy >= target_accuracy
                and macro_f1 >= target_macro_f1
            ),
        },
        "sweet": {
            name: compare_metrics(accuracy, macro_f1, reference)
            for name, reference in references.items()
        },
        "headline_sweet_claim": (
            "exceeds_protocol_matched_end_to_end"
            if compare_metrics(accuracy, macro_f1, references["end_to_end"])["exceeds_both"]
            else "does_not_exceed_protocol_matched_end_to_end"
        ),
    }


def default_results() -> list[tuple[str, str, str]]:
    rows = []
    for dataset in ("vpn-app", "tls-120"):
        rows.append(("packet-level", dataset, PACKET_LEVEL_RESULTS[dataset].path))
        rows.append(("flow-level", dataset, FLOW_LEVEL_RESULTS[dataset].path))
    return rows


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Protocol-Matched SWEET Comparison",
        "",
        "Current models use downstream-supervised LoRA adaptation, so the primary comparator is the SWEET unfrozen/end-to-end column. Frozen Pcap-Encoder is reported separately as representation-reference context.",
        "",
        "| Task | Dataset | Ours AC/F1 | SWEET frozen AC/F1 | Gap | SWEET end-to-end AC/F1 | Gap | Headline |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in report["results"]:
        frozen = row["sweet"]["frozen_representation"]
        end_to_end = row["sweet"]["end_to_end"]
        lines.append(
            "| {task} | {dataset} | {acc:.4f}/{f1:.4f} | {f_acc:.4f}/{f_f1:.4f} | {f_da:+.4f}/{f_df:+.4f} | {e_acc:.4f}/{e_f1:.4f} | {e_da:+.4f}/{e_df:+.4f} | {claim} |".format(
                task=row["task"],
                dataset=row["dataset"],
                acc=row["accuracy"],
                f1=row["macro_f1"],
                f_acc=frozen["accuracy"],
                f_f1=frozen["macro_f1"],
                f_da=frozen["delta_accuracy"],
                f_df=frozen["delta_macro_f1"],
                e_acc=end_to_end["accuracy"],
                e_f1=end_to_end["macro_f1"],
                e_da=end_to_end["delta_accuracy"],
                e_df=end_to_end["delta_macro_f1"],
                claim=row["headline_sweet_claim"],
            )
        )
    lines.extend(
        [
            "",
            "A predeclared project target is an engineering promotion gate, not evidence that the method exceeds every SWEET baseline.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result",
        nargs=3,
        action="append",
        metavar=("TASK", "DATASET", "JSON"),
        default=[],
    )
    parser.add_argument("--output_json", default="reasoningDataset/sweet_protocol_comparison.json")
    parser.add_argument("--output_md", default="reasoningDataset/sweet_protocol_comparison.md")
    args = parser.parse_args()
    inputs = [tuple(row) for row in args.result] or default_results()
    report = {
        "source": {
            "paper": SWEET_PAPER,
            "split": "per-flow",
            "metrics": ["accuracy", "macro_f1"],
            "tables": [3, 4, 9],
        },
        "comparison_policy": {
            "current_method_protocol": "downstream_adapted_lora",
            "primary_comparator": "end_to_end",
            "frozen_comparator_is_secondary": True,
        },
        "results": [compare_result(*row) for row in inputs],
    }
    report["summary"] = {
        "all_results_strict_shared_core_v2": all(
            row["strict_shared_core_v2_result"] for row in report["results"]
        ),
        "all_results_exceed_end_to_end_accuracy_and_macro_f1": all(
            row["sweet"]["end_to_end"]["exceeds_both"]
            for row in report["results"]
        ),
        "strict_unified_method_exceeds_all_end_to_end": all(
            row["strict_shared_core_v2_result"]
            and row["sweet"]["end_to_end"]["exceeds_both"]
            for row in report["results"]
        ),
    }
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    output_md.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"output_json": str(output_json), "output_md": str(output_md)}, indent=2))


if __name__ == "__main__":
    main()
