#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path
from typing import Any, Dict, List

from recommend_next_experiment import cuda_summary


def run(cmd: List[str], execute: bool) -> None:
    print("$ " + " ".join(shlex.quote(x) for x in cmd), flush=True)
    if execute:
        subprocess.run(cmd, check=True)


def safe_name(value: str) -> str:
    return value.replace("/", "_").replace("-", "_")


def result_suffix(embedding_suffix: str, run_tag: str = "") -> str:
    suffix = f"{embedding_suffix}_stage8_flowaware"
    if run_tag:
        suffix += f"_{safe_name(run_tag)}"
    return suffix


def paired_view_outputs_exist(args) -> bool:
    root = Path("reasoningDataset") / args.dataset
    suffix = args.paired_embedding_suffix
    required = [
        root / f"train_tower2_{suffix}" / "seq_dataset.pt",
        root / f"train_tower2_{suffix}" / "graph_dataset.pt",
        root / f"valid_tower2_{suffix}" / "seq_dataset.pt",
        root / f"valid_tower2_{suffix}" / "graph_dataset.pt",
        root / f"test_tower2_{suffix}" / "seq_dataset.pt",
        root / f"test_tower2_{suffix}" / "graph_dataset.pt",
    ]
    return all(path.exists() for path in required)


def final_outputs_exist(args) -> bool:
    root = Path("reasoningDataset") / args.dataset
    suffix = result_suffix(args.embedding_suffix, args.run_tag)
    model_names = "_".join(selected_model_types(args))
    return (root / f"test_fusion_{model_names}_{suffix}_safe_prior_residual.json").exists()


def selected_model_types(args) -> List[str]:
    out: List[str] = []
    for raw in args.model_types.split(","):
        model_type = raw.strip()
        if not model_type:
            continue
        if model_type not in {"graph", "seq"}:
            raise ValueError(f"Unknown model_type={model_type}; use graph,seq")
        if model_type not in out:
            out.append(model_type)
    return out or ["graph", "seq"]


def runner_base(args) -> List[str]:
    return [
        "python",
        "run_stage8_flowaware_pipeline.py",
        "--dataset",
        args.dataset,
        "--num_classes",
        str(args.num_classes),
        "--model_types",
        args.model_types,
        "--no_progress",
    ]


def recommendation_cmd(args) -> List[str]:
    cmd = [
        "python",
        "recommend_next_experiment.py",
        "--dataset",
        args.dataset,
        "--output_json",
        str(Path("reasoningDataset") / args.dataset / f"next_experiment_recommendation_{args.run_tag}.json"),
        "--output_md",
        str(Path("reasoningDataset") / args.dataset / f"next_experiment_recommendation_{args.run_tag}.md"),
    ]
    for extra_dataset in args.extra_recommendation_dataset:
        cmd += ["--dataset", extra_dataset]
    return cmd


def stage_commands(args) -> List[Dict[str, Any]]:
    base = runner_base(args)
    paired_common = [
        "--output_suffix",
        args.paired_output_suffix,
        "--embedding_suffix",
        args.paired_embedding_suffix,
    ]
    train_common = [
        "--embedding_suffix",
        args.embedding_suffix,
        "--run_tag",
        args.run_tag,
        "--paired_embedding_suffix",
        args.paired_embedding_suffix,
        "--paired_view_weight",
        str(args.paired_view_weight),
        "--paired_consistency_weight",
        str(args.paired_consistency_weight),
        "--consistency_weight",
        str(args.consistency_weight),
        "--meta_dropout_prob",
        str(args.meta_dropout_prob),
        "--embedding_dropout_prob",
        str(args.embedding_dropout_prob),
        "--window_dropout_prob",
        str(args.window_dropout_prob),
        "--edge_attr_dropout_prob",
        str(args.edge_attr_dropout_prob),
        "--tower2_epochs",
        str(args.tower2_epochs),
        "--tower2_early_stop_patience",
        str(args.tower2_early_stop_patience),
    ]
    return [
        {
            "name": "diagnose_before",
            "cmd": recommendation_cmd(args),
            "requires_cuda": False,
            "skip_if": False,
        },
        {
            "name": "paired_tower1_preprocess",
            "cmd": base
            + ["--stage", "tower1_preprocess", "--output_suffix", args.paired_output_suffix, "--embedding_header_policy", args.embedding_header_policy],
            "requires_cuda": False,
            "skip_if": args.skip_existing and paired_view_outputs_exist(args),
        },
        {
            "name": "paired_embeddings",
            "cmd": base + ["--stage", "embeddings"] + paired_common + ["--require_cuda"],
            "requires_cuda": True,
            "skip_if": args.skip_existing and paired_view_outputs_exist(args),
        },
        {
            "name": "paired_tower2_preprocess",
            "cmd": base + ["--stage", "tower2_preprocess", "--embedding_suffix", args.paired_embedding_suffix],
            "requires_cuda": False,
            "skip_if": args.skip_existing and paired_view_outputs_exist(args),
        },
        {
            "name": "paired_tower2_train",
            "cmd": base + ["--stage", "tower2_train"] + train_common,
            "requires_cuda": args.require_cuda_for_tower2,
            "skip_if": args.skip_existing and final_outputs_exist(args),
        },
        {
            "name": "paired_eval",
            "cmd": base + ["--stage", "eval", "--embedding_suffix", args.embedding_suffix, "--run_tag", args.run_tag],
            "requires_cuda": False,
            "skip_if": args.skip_existing and final_outputs_exist(args),
        },
        {
            "name": "paired_fusion",
            "cmd": base + ["--stage", "fusion", "--embedding_suffix", args.embedding_suffix, "--run_tag", args.run_tag],
            "requires_cuda": False,
            "skip_if": args.skip_existing and final_outputs_exist(args),
        },
        {
            "name": "paired_prior",
            "cmd": base + ["--stage", "prior", "--embedding_suffix", args.embedding_suffix, "--run_tag", args.run_tag],
            "requires_cuda": False,
            "skip_if": args.skip_existing and final_outputs_exist(args),
        },
        {
            "name": "diagnose_after",
            "cmd": recommendation_cmd(args),
            "requires_cuda": False,
            "skip_if": False,
        },
    ]


def write_plan(args, stages: List[Dict[str, Any]], cuda: Dict[str, Any], execute: bool) -> None:
    payload = {
        "dataset": args.dataset,
        "num_classes": args.num_classes,
        "execute": execute,
        "cuda": cuda,
        "embedding_suffix": args.embedding_suffix,
        "paired_embedding_suffix": args.paired_embedding_suffix,
        "run_tag": args.run_tag,
        "stages": [
            {
                "name": stage["name"],
                "requires_cuda": stage["requires_cuda"],
                "skip_if": stage["skip_if"],
                "cmd": stage["cmd"],
            }
            for stage in stages
        ],
    }
    out = Path(args.plan_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Autopilot for the next recommended unified Stage-8 experiment.")
    ap.add_argument("--dataset", default="vpn-app")
    ap.add_argument("--num_classes", type=int, default=16)
    ap.add_argument("--extra_recommendation_dataset", action="append", default=["tls-120", "ustc-app"])
    ap.add_argument("--embedding_suffix", default="rawproj_flowaware_change_weight")
    ap.add_argument("--paired_output_suffix", default="flowaware_ipport_rand_change_weight")
    ap.add_argument("--paired_embedding_suffix", default="rawproj_flowaware_ipport_rand_change_weight")
    ap.add_argument("--embedding_header_policy", choices=["randomize_ip_port", "mask_ip_port"], default="randomize_ip_port")
    ap.add_argument("--run_tag", default="paired_ipport")
    ap.add_argument("--model_types", default="graph,seq")
    ap.add_argument("--paired_view_weight", type=float, default=0.2)
    ap.add_argument("--paired_consistency_weight", type=float, default=0.1)
    ap.add_argument("--consistency_weight", type=float, default=0.05)
    ap.add_argument("--meta_dropout_prob", type=float, default=0.1)
    ap.add_argument("--embedding_dropout_prob", type=float, default=0.05)
    ap.add_argument("--window_dropout_prob", type=float, default=0.1)
    ap.add_argument("--edge_attr_dropout_prob", type=float, default=0.1)
    ap.add_argument("--tower2_epochs", type=int, default=30)
    ap.add_argument("--tower2_early_stop_patience", type=int, default=8)
    ap.add_argument("--require_cuda_for_tower2", action="store_true")
    ap.add_argument("--execute", action="store_true", help="Actually run the recommended commands. Default prints a dry-run plan.")
    ap.add_argument("--allow_no_cuda", action="store_true", help="Allow --execute even when CUDA is unavailable; useful only for tiny CPU probes.")
    ap.add_argument("--skip_existing", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--plan_json", default="reasoningDataset/recommended_experiment_plan.json")
    args = ap.parse_args()

    cuda = cuda_summary()
    stages = stage_commands(args)
    execute = bool(args.execute)
    if execute and not args.allow_no_cuda:
        cuda_stages = [stage["name"] for stage in stages if stage["requires_cuda"] and not stage["skip_if"]]
        if cuda_stages and not cuda.get("available"):
            print(
                "CUDA is unavailable; switching to dry-run for GPU stages: "
                + ", ".join(cuda_stages),
                flush=True,
            )
            execute = False
    write_plan(args, stages, cuda, execute)

    for stage in stages:
        if stage["skip_if"]:
            print(f"# skip {stage['name']} (existing outputs detected)", flush=True)
            continue
        if execute:
            run(stage["cmd"], execute=True)
        else:
            print(f"# stage {stage['name']} requires_cuda={stage['requires_cuda']}", flush=True)
            run(stage["cmd"], execute=False)


if __name__ == "__main__":
    main()
