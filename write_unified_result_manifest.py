#!/usr/bin/env python3
"""Bind an existing paper result JSON to the unified framework audit.

This is intentionally metadata-only: it does not train a model or change the
result. It records that a metric-bearing paper result is the artifact being
claimed, along with the result's source inputs and the shared framework modules
that future reproduction runs must preserve.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple

from unified_framework_spec import (
    FLOW_LEVEL_RESULTS,
    PACKET_LEVEL_RESULTS,
    build_framework_manifest,
    profile_shared_status,
)


def safe_name(value: str) -> str:
    return value.replace("/", "_").replace("-", "_").replace(".", "_")


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def result_spec(task: str, dataset: str):
    specs = FLOW_LEVEL_RESULTS if task == "flow-level" else PACKET_LEVEL_RESULTS
    if dataset not in specs:
        raise SystemExit(f"No configured {task} result spec for dataset={dataset}")
    return specs[dataset]


def extract_metrics(data: Dict[str, Any], task: str) -> Tuple[float | None, float | None, int | None]:
    if task == "flow-level":
        metrics = data.get("metrics")
        if isinstance(metrics, dict):
            flow = metrics.get("flow_level")
            if isinstance(flow, dict):
                return flow.get("accuracy"), flow.get("macro_f1"), flow.get("num_flows")
            return metrics.get("accuracy"), metrics.get("macro_f1"), metrics.get("num_flows") or metrics.get("num_samples")
        return None, None, None

    for key in ("metrics", "test_metrics", "packet_level"):
        metrics = data.get(key)
        if isinstance(metrics, dict):
            packet = metrics.get("packet_level")
            if isinstance(packet, dict):
                acc = packet.get("accuracy")
                f1 = packet.get("macro_f1")
                n = packet.get("num_samples") or packet.get("num_packets")
                if acc is not None or f1 is not None:
                    return acc, f1, n
            acc = metrics.get("accuracy")
            f1 = metrics.get("macro_f1")
            n = metrics.get("num_samples") or metrics.get("num_packets")
            if acc is not None or f1 is not None:
                return acc, f1, n
    return None, None, None


def default_output_path(task: str, dataset: str, result_json: str) -> Path:
    if task == "flow-level":
        stem = safe_name(Path(result_json).stem)
        return Path("reasoningDataset") / dataset / f"stage8_flowaware_manifest_result_bound_paper_unified_{stem}.json"
    return Path("reasoningDataset/packet-level") / dataset / "result_bound" / "packet_framework_manifest.json"


def main() -> None:
    ap = argparse.ArgumentParser(description="Write a result-bound unified framework manifest.")
    ap.add_argument("--task", choices=["flow-level", "packet-level"], required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--result_json", required=True)
    ap.add_argument("--framework_profile", choices=["paper_unified"], default="paper_unified")
    ap.add_argument("--output_manifest", default="")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    spec = result_spec(args.task, args.dataset)
    if spec.path != args.result_json:
        raise SystemExit(f"Result path does not match configured paper result: {args.result_json} != {spec.path}")
    result_path = Path(args.result_json)
    if not result_path.exists():
        raise SystemExit(f"Missing result JSON: {result_path}")
    data = load_json(args.result_json)
    provenance = data.get("publication_provenance") or {}
    if args.dataset in {"vpn-app", "tls-120"} and (
        provenance.get("status") != "strict_shared_core_v2"
        or not provenance.get("shared_core_config_sha256")
        or provenance.get("runtime_mechanism_evidence_required") is not True
        or provenance.get("flow_native_extraction_evidence_required") is not True
        or provenance.get("fixed_consensus") != "equal_log_mean_three_folds"
        or len(provenance.get("audit_paths") or []) != 3
    ):
        raise SystemExit(
            "VPN/TLS paper_unified binding requires strict-v2 publication provenance"
        )
    acc, f1, n = extract_metrics(data, args.task)
    if acc is None or f1 is None:
        raise SystemExit(f"Missing accuracy/macro_f1 in result JSON: {result_path}")
    if spec.target_accuracy is not None and float(acc) < spec.target_accuracy:
        raise SystemExit(f"Accuracy {acc} is below target {spec.target_accuracy}")
    if spec.target_macro_f1 is not None and float(f1) < spec.target_macro_f1:
        raise SystemExit(f"Macro-F1 {f1} is below target {spec.target_macro_f1}")

    shared_status = profile_shared_status(args.framework_profile)
    shared_status.update(
        {
            "semantic_tower1_channel": "present_or_available_in_source_inputs",
            "bounded_tri_channel_router": "semantic_anchor_residual_0_25",
            "content_group_empirical_risk": "group_mean",
            "fixed_cross_fold_consensus": "equal_log_mean_bound",
            "validation_only_selection": "enforced_by_source_results",
        }
    )
    source_inputs = data.get("inputs") if isinstance(data.get("inputs"), list) else []
    if args.task == "packet-level":
        source_paths = [
            str(item.get("path", "")) if isinstance(item, dict) else str(item)
            for item in source_inputs
        ]
        invalid_sources = [
            path for path in source_paths
            if not path.endswith("test_unified_packet_single_head.npz")
        ]
        if not source_paths or invalid_sources:
            raise SystemExit(
                "Packet paper result must be a consensus of only "
                "test_unified_packet_single_head.npz fold artifacts; invalid="
                + repr(invalid_sources or source_paths)
            )
    framework = build_framework_manifest(
        task=args.task,
        dataset=args.dataset,
        input_unit="flow_packet_sequence_crossfold_result" if args.task == "flow-level" else "one_current_packet_crossfold_result",
        stage="paper_result_binding",
        shared_module_status=shared_status,
        task_module_status=(
            {
                "packet_to_window_flow_aggregator": "source_result_bound",
                "flow_level_classifier": "crossfold_consensus",
            }
            if args.task == "flow-level"
            else {
                "strict_current_packet_protocol": "source_result_bound",
                "packet_level_classifier": "shared_representation_single_head_crossfold_consensus",
            }
        ),
        notes={
            "framework_profile": args.framework_profile,
            "completed": not args.dry_run,
            "dry_run": args.dry_run,
            "binding_only": True,
            "published_result_paths": [args.result_json],
            "result_metrics": {"accuracy": acc, "macro_f1": f1, "num_samples": n},
            "source_input_count": len(source_inputs),
            "source_inputs": source_inputs,
            "paper_main_experts": [],
            "semantic_fusion_level": "representation",
            "publication_provenance": provenance,
        },
    )
    payload = {
        "task": args.task,
        "dataset": args.dataset,
        "stage": "paper_result_binding",
        "framework_profile": args.framework_profile,
        "result_json": args.result_json,
        "framework": framework,
    }
    out_path = Path(args.output_manifest) if args.output_manifest else default_output_path(args.task, args.dataset, args.result_json)
    print(json.dumps({"output_manifest": str(out_path), "accuracy": acc, "macro_f1": f1, "dry_run": args.dry_run}, indent=2))
    if not args.dry_run:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
