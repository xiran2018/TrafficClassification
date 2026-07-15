#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path
from typing import Any, Dict, List

from recommend_next_experiment import cuda_summary


DEFAULT_UNIFIED_EXPERT_SLOTS = "base,graph,seq,prior_base,emb_lr,emb_et,proto_emb,paired,slot_stacker"


DATASET_PRESETS = {
    "vpn-app": {
        "num_classes": 16,
        "label_map": "reasoningDataset/vpn-app/train_tower1_change_weight/label_map.json",
        "base_selector_input": "reasoningDataset/vpn-app/test_selector_best_prior_embedding_experts_calib_shift000_valid_macro.json",
        "max_prediction_change_rate": 0.0,
        "final_selector_rank_metric": "bootstrap_gain_quantile",
        "final_selector_rank_select_metric": "accuracy",
        "final_selector_rank_candidate_limit": 256,
    },
    "tls-120": {
        "num_classes": 120,
        "label_map": "reasoningDataset/tls-120/train_tower1_change_weight/label_map.json",
        "base_selector_input": "reasoningDataset/tls-120/test_selector_unified_slot_stacker_tls120_valid_macro.json",
        "max_prediction_change_rate": 0.05,
        "final_selector_rank_metric": "bootstrap_gain_quantile",
        "final_selector_rank_select_metric": "macro_f1",
        "final_selector_rank_candidate_limit": 64,
    },
    "ustc-app": {
        "num_classes": 20,
        "label_map": "reasoningDataset/ustc-app/train_tower1_flowaware_change_weight/label_map.json",
        "base_selector_input": "reasoningDataset/ustc-app/test_selector_base_flowproto_full_s200_w002_step150_calib_shift005_valid_macro.json",
        "max_prediction_change_rate": 0.05,
    },
}


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


def paired_prior_output_path(args) -> str:
    root = Path("reasoningDataset") / args.dataset
    suffix = result_suffix(args.embedding_suffix, args.run_tag)
    model_names = "_".join(selected_model_types(args))
    return str(root / f"test_fusion_{model_names}_{suffix}_safe_prior_residual.json")


def default_base_selector_input(args) -> str:
    return DATASET_PRESETS.get(args.dataset, {}).get("base_selector_input", "")


def default_label_map(args) -> str:
    return DATASET_PRESETS.get(args.dataset, {}).get(
        "label_map",
        f"reasoningDataset/{args.dataset}/train_tower1_change_weight/label_map.json",
    )


def final_selector_output_path(args) -> str:
    if args.final_selector_output:
        return args.final_selector_output
    root = Path("reasoningDataset") / args.dataset
    suffix = result_suffix(args.embedding_suffix, args.run_tag)
    return str(root / f"test_selector_best_plus_{suffix}_valid_macro.json")


def final_selector_output_exists(args) -> bool:
    return Path(final_selector_output_path(args)).exists()


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
        "--seed",
        str(args.seed),
        "--flow_pooling",
        args.flow_pooling,
        "--multi_view_gate_entropy_weight",
        str(args.multi_view_gate_entropy_weight),
        "--no_progress",
    ]


def recommendation_cmd(args) -> List[str]:
    datasets = []
    for dataset in [args.dataset] + list(args.extra_recommendation_dataset):
        if dataset not in datasets:
            datasets.append(dataset)
    cmd = [
        "python",
        "recommend_next_experiment.py",
        "--output_json",
        str(Path("reasoningDataset") / args.dataset / f"next_experiment_recommendation_{args.run_tag}.json"),
        "--output_md",
        str(Path("reasoningDataset") / args.dataset / f"next_experiment_recommendation_{args.run_tag}.md"),
    ]
    for dataset in datasets:
        cmd += ["--dataset", dataset]
    return cmd


def final_selector_cmd(args) -> List[str]:
    base_input = args.base_selector_input or default_base_selector_input(args)
    if not base_input:
        raise ValueError("--base_selector_input is required for datasets without a built-in final selector default.")
    cmd = [
        "python",
        "validation_gated_selector.py",
        "--input",
        "base",
        base_input,
        "--input",
        "paired",
        paired_prior_output_path(args),
        "--label_map",
        args.label_map or default_label_map(args),
        "--select_metric",
        args.final_selector_metric,
        "--rank_select_metric",
        args.final_selector_rank_select_metric,
        "--rank_metric",
        args.final_selector_rank_metric,
        "--rank_bootstrap_samples",
        str(args.final_selector_rank_bootstrap_samples),
        "--rank_candidate_limit",
        str(args.final_selector_rank_candidate_limit),
        "--strategies",
        args.final_selector_strategies,
        "--alpha_grid",
        args.final_selector_alpha_grid,
        "--metric_margin_grid",
        args.final_selector_metric_margin_grid,
        "--expert_conf_grid",
        args.final_selector_expert_conf_grid,
        "--expert_margin_grid",
        args.final_selector_expert_margin_grid,
        "--base_conf_max_grid",
        args.final_selector_base_conf_max_grid,
        "--delta_conf_grid",
        args.final_selector_delta_conf_grid,
        "--delta_margin_grid",
        args.final_selector_delta_margin_grid,
        "--reliability_power_grid",
        args.final_selector_reliability_power_grid,
        "--confidence_power_grid",
        args.final_selector_confidence_power_grid,
        "--reliability_min_weight_grid",
        args.final_selector_reliability_min_weight_grid,
        "--reliability_temperature_grid",
        args.final_selector_reliability_temperature_grid,
        "--calibration_strength_grid",
        args.final_selector_calibration_strength_grid,
        "--calibration_temperature_grid",
        args.final_selector_calibration_temperature_grid,
        "--min_valid_gain_over_base",
        str(args.final_selector_min_valid_gain_over_base),
        "--bootstrap_samples",
        str(args.final_selector_bootstrap_samples),
        "--bootstrap_min_win_rate",
        str(args.final_selector_bootstrap_min_win_rate),
        "--bootstrap_min_gain_quantile",
        str(args.final_selector_bootstrap_min_gain_quantile),
        "--max_prediction_change_rate",
        str(args.final_selector_max_prediction_change_rate),
        "--output_json",
        final_selector_output_path(args),
    ]
    if args.final_selector_unified_expert_slots:
        cmd += ["--unified_expert_slots", args.final_selector_unified_expert_slots]
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
            "name": "final_selector",
            "cmd": final_selector_cmd(args),
            "requires_cuda": False,
            "skip_if": args.skip_existing and final_selector_output_exists(args),
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
        "experiment_config": {
            "paired_view_weight": args.paired_view_weight,
            "paired_consistency_weight": args.paired_consistency_weight,
            "consistency_weight": args.consistency_weight,
            "meta_dropout_prob": args.meta_dropout_prob,
            "embedding_dropout_prob": args.embedding_dropout_prob,
            "window_dropout_prob": args.window_dropout_prob,
            "edge_attr_dropout_prob": args.edge_attr_dropout_prob,
            "tower2_epochs": args.tower2_epochs,
            "tower2_early_stop_patience": args.tower2_early_stop_patience,
            "model_types": args.model_types,
            "seed": args.seed,
            "flow_pooling": args.flow_pooling,
            "multi_view_gate_entropy_weight": args.multi_view_gate_entropy_weight,
            "final_selector_unified_expert_slots": args.final_selector_unified_expert_slots,
        },
        "base_selector_input": args.base_selector_input or default_base_selector_input(args),
        "paired_prior_output": paired_prior_output_path(args),
        "final_selector_output": final_selector_output_path(args),
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
    ap.add_argument("--num_classes", type=int, default=0, help="Defaults from dataset presets when 0.")
    ap.add_argument("--extra_recommendation_dataset", action="append", default=["vpn-app", "tls-120", "ustc-app"])
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
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--flow_pooling", choices=["mean", "attention", "late_fusion", "transformer", "multi_view"], default="mean")
    ap.add_argument("--multi_view_gate_entropy_weight", type=float, default=0.0)
    ap.add_argument("--base_selector_input", default="", help="Current best probability JSON used as the first input to the final validation-gated selector.")
    ap.add_argument("--label_map", default="", help="Label map for final selector; defaults from dataset presets.")
    ap.add_argument("--final_selector_output", default="", help="Optional output path for the final validation-gated selector.")
    ap.add_argument("--final_selector_metric", choices=["accuracy", "macro_f1"], default="macro_f1")
    ap.add_argument(
        "--final_selector_rank_select_metric",
        choices=["accuracy", "macro_f1"],
        default="",
        help="Metric used inside bootstrap_* ranking for the final selector. Defaults from dataset presets, then final selector metric.",
    )
    ap.add_argument(
        "--final_selector_rank_metric",
        choices=[
            "select_metric",
            "accuracy",
            "macro_f1",
            "bootstrap_gain_quantile",
            "bootstrap_mean_gain",
            "bootstrap_win_rate",
        ],
        default="",
        help="Candidate ranking metric for the final selector. Defaults from dataset presets, then select_metric.",
    )
    ap.add_argument(
        "--final_selector_rank_bootstrap_samples",
        type=int,
        default=0,
        help="Bootstrap resamples used for final-selector robust ranking. Defaults to final selector bootstrap samples.",
    )
    ap.add_argument("--final_selector_rank_candidate_limit", type=int, default=-1)
    ap.add_argument("--final_selector_strategies", default="always,class_precision,reliability_fusion,threshold_switch,class_bias_calibration")
    ap.add_argument("--final_selector_alpha_grid", default="0.5,5")
    ap.add_argument("--final_selector_metric_margin_grid", default="0,0.05")
    ap.add_argument("--final_selector_expert_conf_grid", default="0.3,0.85")
    ap.add_argument("--final_selector_expert_margin_grid", default="0.05")
    ap.add_argument("--final_selector_base_conf_max_grid", default="1")
    ap.add_argument("--final_selector_delta_conf_grid", default="-1,0.05")
    ap.add_argument("--final_selector_delta_margin_grid", default="-1,0.1")
    ap.add_argument("--final_selector_reliability_power_grid", default="4")
    ap.add_argument("--final_selector_confidence_power_grid", default="1")
    ap.add_argument("--final_selector_reliability_min_weight_grid", default="0")
    ap.add_argument("--final_selector_reliability_temperature_grid", default="0.5")
    ap.add_argument("--final_selector_calibration_strength_grid", default="1.0")
    ap.add_argument("--final_selector_calibration_temperature_grid", default="1.25")
    ap.add_argument("--final_selector_min_valid_gain_over_base", type=float, default=0.0)
    ap.add_argument("--final_selector_bootstrap_samples", type=int, default=300)
    ap.add_argument("--final_selector_bootstrap_min_win_rate", type=float, default=0.6)
    ap.add_argument("--final_selector_bootstrap_min_gain_quantile", type=float, default=-0.001)
    ap.add_argument("--final_selector_max_prediction_change_rate", type=float, default=-1.0, help="Defaults from dataset presets when <0.")
    ap.add_argument(
        "--final_selector_unified_expert_slots",
        default=DEFAULT_UNIFIED_EXPERT_SLOTS,
        help=(
            "Comma-separated expert slots exposed by every dataset in the final selector. "
            "Missing slots are filled as identity experts from the base input; pass an empty string to disable."
        ),
    )
    ap.add_argument("--require_cuda_for_tower2", action="store_true")
    ap.add_argument("--execute", action="store_true", help="Actually run the recommended commands. Default prints a dry-run plan.")
    ap.add_argument("--allow_no_cuda", action="store_true", help="Allow --execute even when CUDA is unavailable; useful only for tiny CPU probes.")
    ap.add_argument("--skip_existing", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--plan_json", default="", help="Defaults to reasoningDataset/DATASET/recommended_experiment_plan_RUN_TAG.json.")
    args = ap.parse_args()

    preset = DATASET_PRESETS.get(args.dataset, {})
    if args.num_classes <= 0:
        if "num_classes" not in preset:
            raise SystemExit(f"--num_classes is required for dataset={args.dataset}")
        args.num_classes = int(preset["num_classes"])
    if args.final_selector_max_prediction_change_rate < 0:
        args.final_selector_max_prediction_change_rate = float(preset.get("max_prediction_change_rate", 0.05))
    if not args.final_selector_rank_metric:
        args.final_selector_rank_metric = str(preset.get("final_selector_rank_metric", "select_metric"))
    if not args.final_selector_rank_select_metric:
        args.final_selector_rank_select_metric = str(preset.get("final_selector_rank_select_metric", args.final_selector_metric))
    if args.final_selector_rank_candidate_limit < 0:
        args.final_selector_rank_candidate_limit = int(preset.get("final_selector_rank_candidate_limit", 256))
    if not args.plan_json:
        args.plan_json = str(Path("reasoningDataset") / args.dataset / f"recommended_experiment_plan_{args.run_tag}.json")

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
