#!/usr/bin/env python3
"""Create the next reproduction plan for missing paper_unified provenance.

The unified audit separates metric quality from publication provenance. A
result can have strong accuracy/F1 while still needing rerun evidence that it
was produced through the paper_unified profile. This helper turns those audit
gaps into concrete conda commands for the next automated loop.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List


PACKET_FOLDS = (0, 1, 2)
FLOW_FOLDS = (0, 1, 2)
FLOW_NUM_CLASSES = {"vpn-app": 16, "tls-120": 120}


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def shell_join(parts: List[str]) -> str:
    return " ".join(parts)


def canonical_sha256(payload: Dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def frozen_config_evidence(path: str) -> Dict[str, Any]:
    source = Path(path).resolve()
    if not source.is_file():
        raise ValueError(f"shared-core config does not exist: {source}")
    payload = load_json(str(source))
    if payload.get("schema") != "exact_shared_packet_core_v2":
        raise ValueError("shared-core config has the wrong schema")
    recorded = str(payload.get("config_sha256") or "")
    unsigned = dict(payload)
    unsigned.pop("config_sha256", None)
    recomputed = canonical_sha256(unsigned)
    if recorded != recomputed:
        raise ValueError("shared-core config canonical fingerprint mismatch")
    return {
        "path": str(source),
        "file_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "config_sha256": recorded,
        "status": payload.get("status"),
    }


def append_shared_core_config(argv: List[str], path: str | None) -> List[str]:
    if path:
        argv += ["--shared_core_config", str(Path(path).resolve())]
    return argv


def flow_embedding_suffix(fold: int) -> str:
    return f"rawproj_paper_unified_dualview_fold{fold}"


def flow_run_tag(run_tag: str, fold: int) -> str:
    return f"{run_tag}_fold{fold}"


def flow_argv(
    dataset: str,
    fold: int,
    run_tag: str,
    stage: str,
    shared_core_config: str | None = None,
) -> List[str]:
    argv = [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        "llm-factory",
        "python",
        "run_stage8_flowaware_pipeline.py",
        "--dataset",
        dataset,
        "--fold",
        str(fold),
        "--stage",
        stage,
        "--framework_profile",
        "paper_unified",
        "--embedding_header_policy",
        "full",
        "--run_tag",
        flow_run_tag(run_tag, fold),
        "--require_cuda",
        "--num_classes",
        str(FLOW_NUM_CLASSES[dataset]),
        "--source_suffix",
        "change_weight",
        "--output_suffix",
        f"paper_unified_dualview_fold{fold}",
        "--tower1_data_suffix",
        f"paper_unified_dualview_fold{fold}",
        "--embedding_suffix",
        flow_embedding_suffix(fold),
        "--tower1_output_dir",
        f"checkpoints/tower1_qwen_multitask_{dataset.replace('-', '_')}_paper_unified_fold{fold}",
        "--tower1_device",
        "cuda:5",
        "--gradient_checkpointing",
        "--sft_batch_size",
        "1",
        "--packet_batch_size",
        "16",
        "--embedding_num_shards",
        "8",
        "--embedding_cuda_devices",
        "0,1,2,3,4,5,6,7",
        "--paper_unified_stages",
        "model",
    ]
    return append_shared_core_config(argv, shared_core_config)


def flow_fold_result_path(dataset: str, fold: int, run_tag: str) -> str:
    suffix = flow_embedding_suffix(fold)
    native_suffix = f"shared_content_paper_unified_dualview_fold{fold}"
    tag = flow_run_tag(run_tag, fold).replace("-", "_")
    return (
        f"reasoningDataset/{dataset}/test_seq_metrics_flow_{suffix}_native_{native_suffix}_stage8_flowaware_"
        f"{tag}_probs.json"
    )


def flow_consensus_argv(dataset: str, run_tag: str, output_json: str) -> List[str]:
    argv = [
        "conda", "run", "--no-capture-output", "-n", "llm-factory",
        "python", "cross_fold_consensus.py",
    ]
    for fold in FLOW_FOLDS:
        argv += ["--input", f"fold{fold}", flow_fold_result_path(dataset, fold, run_tag)]
    argv += [
        "--mode", "log_mean",
        "--label_map", f"reasoningDataset/{dataset}/train_tower1_paper_unified_dualview_fold0/label_map.json",
        "--output_json", output_json,
        "--no_report",
    ]
    return argv


def flow_result_bind_argv(dataset: str, result_json: str) -> List[str]:
    return [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        "llm-factory",
        "python",
        "write_unified_result_manifest.py",
        "--task",
        "flow-level",
        "--dataset",
        dataset,
        "--result_json",
        result_json,
        "--framework_profile",
        "paper_unified",
    ]


def packet_result_bind_argv(dataset: str, result_json: str) -> List[str]:
    return [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        "llm-factory",
        "python",
        "write_unified_result_manifest.py",
        "--task",
        "packet-level",
        "--dataset",
        dataset,
        "--result_json",
        result_json,
        "--framework_profile",
        "paper_unified",
    ]


def packet_consensus_argv(dataset: str, output_json: str) -> List[str]:
    inputs = [
        f"reasoningDataset/packet-level/{dataset}/fold{fold}/test_unified_packet_single_head.npz"
        for fold in PACKET_FOLDS
    ]
    return [
        "conda", "run", "--no-capture-output", "-n", "llm-factory",
        "python", "fuse_packet_crossfold.py",
        "--inputs", *inputs,
        "--label_map", f"reasoningDataset/packet-level/{dataset}/fold0/train/label_map.json",
        "--method", "log_mean",
        "--output_json", output_json,
    ]


def packet_argv(
    dataset: str,
    fold: int,
    stage: str,
    shared_core_config: str | None = None,
) -> List[str]:
    argv = [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        "llm-factory",
        "python",
        "run_packet_level_pipeline.py",
        "--dataset",
        dataset,
        "--fold",
        str(fold),
        "--stage",
        stage,
        "--framework_profile",
        "paper_unified",
    ]
    return append_shared_core_config(argv, shared_core_config)


def completed_packet_folds(row: Dict[str, Any]) -> set[int]:
    provenance = row.get("framework_provenance") or {}
    folds: set[int] = set()
    for manifest in provenance.get("matching_manifests") or []:
        path = Path(str(manifest))
        for part in path.parts:
            if part.startswith("fold") and part[4:].isdigit():
                folds.add(int(part[4:]))
                break
    return folds


def completed_flow_folds(row: Dict[str, Any]) -> set[int]:
    provenance = row.get("framework_provenance") or {}
    return {int(fold) for fold in provenance.get("completed_folds") or []}


def missing_rows(audit: Dict[str, Any], section: str) -> List[Dict[str, Any]]:
    return [
        row
        for row in audit.get(section, [])
        if row.get("metric_status") == "pass" and row.get("publication_status") != "pass"
    ]


def can_bind_existing_result(provenance: Dict[str, Any]) -> bool:
    """Only bind when an executed paper_unified candidate already exists.

    Binding is metadata-only. If a default profile has changed and the audit has
    zero candidate manifests, a new run is needed; otherwise an old metric JSON
    could be promoted into the paper path without reproduction evidence.
    """
    return (
        provenance.get("status") == "missing_bound_result_manifest"
        and int(provenance.get("candidate_manifest_count") or 0) > 0
    )


def build_plan(
    audit: Dict[str, Any],
    flow_stage: str,
    packet_stage: str,
    run_tag: str,
    shared_core_config: str | None = None,
) -> Dict[str, Any]:
    config_evidence = (
        frozen_config_evidence(shared_core_config)
        if shared_core_config
        else None
    )
    flow_gaps = missing_rows(audit, "flow_level")
    packet_gaps = missing_rows(audit, "packet_level")
    actions: List[Dict[str, Any]] = []
    for row in flow_gaps:
        dataset = row["dataset"]
        provenance = row.get("framework_provenance") or {}
        completed_folds = completed_flow_folds(row)
        for fold in FLOW_FOLDS:
            if fold in completed_folds:
                continue
            argv = flow_argv(
                dataset,
                fold,
                run_tag,
                flow_stage,
                shared_core_config,
            )
            actions.append(
                {
                    "id": f"flow:{dataset}:fold{fold}",
                    "task": "flow-level",
                    "dataset": dataset,
                    "fold": fold,
                    "reason": row.get("publication_status"),
                    "metric_status": row.get("metric_status"),
                    "provenance_status": provenance.get("status", ""),
                    "argv": argv,
                    "command": shell_join(argv),
                    "expected_manifest_glob": row.get("framework_manifest_glob"),
                }
            )
        consensus_argv = flow_consensus_argv(dataset, run_tag, row["path"])
        actions.append(
            {
                "id": f"flow-consensus:{dataset}",
                "task": "flow-level",
                "dataset": dataset,
                "reason": "build_fixed_log_mean_crossfold_consensus",
                "metric_status": row.get("metric_status"),
                "argv": consensus_argv,
                "command": shell_join(consensus_argv),
                "expected_result": row["path"],
            }
        )
        bind_argv = flow_result_bind_argv(dataset, row["path"])
        actions.append(
            {
                "id": f"flow-result:{dataset}",
                "task": "flow-level",
                "dataset": dataset,
                "reason": "bind_reproduced_crossfold_result",
                "metric_status": row.get("metric_status"),
                "argv": bind_argv,
                "command": shell_join(bind_argv),
                "expected_manifest_glob": row.get("framework_manifest_glob"),
            }
        )
    for row in packet_gaps:
        dataset = row["dataset"]
        result_path = row.get("path") or f"reasoningDataset/packet-level/{dataset}/paper_default_result.json"
        provenance = row.get("framework_provenance") or {}
        completed_folds = completed_packet_folds(row)
        for fold in PACKET_FOLDS:
            if fold in completed_folds:
                continue
            argv = packet_argv(
                dataset,
                fold,
                packet_stage,
                shared_core_config,
            )
            actions.append(
                {
                    "id": f"packet:{dataset}:fold{fold}",
                    "task": "packet-level",
                    "dataset": dataset,
                    "fold": fold,
                    "reason": row.get("publication_status"),
                    "metric_status": row.get("metric_status"),
                    "argv": argv,
                    "command": shell_join(argv),
                    "expected_manifest_glob": row.get("framework_manifest_glob"),
                }
            )
        if (
            provenance.get("status") == "missing_bound_result_manifest"
            and set(PACKET_FOLDS).issubset(completed_folds)
        ):
            bind_argv = packet_result_bind_argv(dataset, result_path)
            actions.append(
                {
                    "id": f"packet-result:{dataset}",
                    "task": "packet-level",
                    "dataset": dataset,
                    "reason": "bind_reproduced_crossfold_result",
                    "metric_status": row.get("metric_status"),
                    "argv": bind_argv,
                    "command": shell_join(bind_argv),
                    "expected_manifest_glob": row.get("framework_manifest_glob"),
                }
            )
            continue
        consensus_argv = packet_consensus_argv(dataset, result_path)
        actions.append(
            {
                "id": f"packet-consensus:{dataset}",
                "task": "packet-level",
                "dataset": dataset,
                "reason": "build_fixed_log_mean_crossfold_consensus",
                "metric_status": row.get("metric_status"),
                "argv": consensus_argv,
                "command": shell_join(consensus_argv),
                "expected_result": result_path,
            }
        )
        bind_argv = packet_result_bind_argv(dataset, result_path)
        actions.append(
            {
                "id": f"packet-result:{dataset}",
                "task": "packet-level",
                "dataset": dataset,
                "reason": "bind_reproduced_crossfold_result",
                "metric_status": row.get("metric_status"),
                "argv": bind_argv,
                "command": shell_join(bind_argv),
                "expected_manifest_glob": row.get("framework_manifest_glob"),
            }
        )
    return {
        "status": "ready" if actions else "no_missing_paper_unified_provenance",
        "audit_status": audit.get("status"),
        "flow_gaps": len(flow_gaps),
        "packet_gaps": len(packet_gaps),
        "shared_core_config": config_evidence,
        "num_actions": len(actions),
        "actions": actions,
    }


def render_markdown(plan: Dict[str, Any]) -> str:
    lines = [
        "# Unified Reproduction Plan",
        "",
        f"Status: `{plan['status']}`",
        "",
        f"Audit status: `{plan.get('audit_status')}`",
        "",
        f"Flow gaps: `{plan['flow_gaps']}`",
        f"Packet gaps: `{plan['packet_gaps']}`",
        f"Commands: `{plan['num_actions']}`",
        "",
    ]
    if not plan["actions"]:
        return "\n".join(lines)
    lines += [
        "| Task | Dataset | Fold | Reason | Command |",
        "| --- | --- | ---: | --- | --- |",
    ]
    for row in plan["actions"]:
        lines.append(
            "| {task} | {dataset} | {fold} | {reason} | `{command}` |".format(
                task=row["task"],
                dataset=row["dataset"],
                fold=row.get("fold", "-"),
                reason=row.get("reason", "-"),
                command=row["command"],
            )
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Plan paper_unified reruns for audit provenance gaps.")
    ap.add_argument("--audit_json", default="reasoningDataset/unified_framework_audit.json")
    ap.add_argument(
        "--flow_stage",
        default="all",
        help=(
            "Flow runner stage for provenance reruns. Default all rebuilds masked "
            "packet prompts, retrains Tower-1, re-extracts policy-attested embeddings, "
            "and runs the unified Tower-2 model/fusion path for each fold."
        ),
    )
    ap.add_argument(
        "--packet_stage",
        default="paper_unified",
        help=(
            "Packet runner stage for provenance reruns. Default paper_unified "
            "includes preprocessing, split audit, structural/byte experts, and "
            "validation fusion under the shared header policy without forcing "
            "the heavyweight Qwen semantic ablation on every fold."
        ),
    )
    ap.add_argument("--run_tag", default="paper_unified_repro")
    ap.add_argument(
        "--shared_core_config",
        default="",
        help=(
            "Optional frozen exact_shared_packet_core_v2 config. When supplied, "
            "its file hash and canonical method fingerprint are verified and the "
            "absolute path is embedded in every Packet/Flow training action."
        ),
    )
    ap.add_argument("--output_json", default="reasoningDataset/unified_repro_plan.json")
    ap.add_argument("--output_md", default="reasoningDataset/unified_repro_plan.md")
    args = ap.parse_args()

    audit = load_json(args.audit_json)
    plan = build_plan(
        audit,
        args.flow_stage,
        args.packet_stage,
        args.run_tag,
        args.shared_core_config or None,
    )
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
    out_md = Path(args.output_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(render_markdown(plan), encoding="utf-8")
    print(json.dumps({"status": plan["status"], "num_actions": plan["num_actions"]}, indent=2))


if __name__ == "__main__":
    main()
