#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from test_tower2 import (
    aggregate_by_flow,
    compute_metrics,
    load_model,
    predict_graph,
    predict_seq,
    softmax_np,
)
from train_tower2 import build_class_to_coarse


def predict_windows(model, model_type: str, dataset: str, device: str, batch_size: int):
    if model_type == "seq":
        return predict_seq(model, dataset, device, batch_size)
    return predict_graph(model, dataset, device)


def aggregate_prediction(raw, ckpt, flow_head, device: str, mode: str, topk: int):
    y_true, _, flow_ids, logits, embeddings, xs = raw
    class_to_coarse = None
    if ckpt.get("num_coarse_classes", 0) > 0:
        class_to_coarse, _ = build_class_to_coarse(
            ckpt.get("coarse_groups", "vpn_app"), ckpt["num_classes"], device
        )
    values = aggregate_by_flow(
        y_true,
        flow_ids,
        logits,
        embeddings,
        xs,
        flow_head,
        device,
        class_to_coarse,
        ckpt.get("hierarchical_logit_weight", 0.0),
        ckpt.get("hierarchical_mode", "logit"),
        mode,
        topk,
    )
    flow_true, flow_pred, out_ids, flow_logits, gate_summary, stat_gate_summary = values
    prob = softmax_np(np.stack(flow_logits, axis=0))
    return {
        "y_true": flow_true,
        "y_pred": flow_pred,
        "flow_ids": out_ids,
        "prob": prob,
        "metrics": compute_metrics(flow_true, flow_pred),
        "multi_view_gate": gate_summary,
        "flow_stat_gate": stat_gate_summary,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Select flow-window aggregation on validation and apply it once to test.")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--valid_dataset", required=True)
    ap.add_argument("--test_dataset", required=True)
    ap.add_argument("--output_json", required=True)
    ap.add_argument("--modes", default="checkpoint,mean_logits,mean_probs,max_conf,topk_logits,vote")
    ap.add_argument("--topk_grid", default="1,2,3,4,5")
    ap.add_argument("--select_metric", choices=["accuracy", "macro_f1"], default="macro_f1")
    ap.add_argument(
        "--min_valid_gain_over_checkpoint",
        type=float,
        default=0.005,
        help="Require this validation gain before replacing the checkpoint flow head.",
    )
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    model, ckpt, flow_head = load_model(args.checkpoint, args.device)
    valid_raw = predict_windows(model, ckpt["model_type"], args.valid_dataset, args.device, args.batch_size)
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    topk_grid = [int(value.strip()) for value in args.topk_grid.split(",") if value.strip()]
    checkpoint_result = aggregate_prediction(valid_raw, ckpt, flow_head, args.device, "checkpoint", 3)
    checkpoint_row = {
        "mode": "checkpoint",
        "topk": 3,
        "metrics": checkpoint_result["metrics"],
        "eligible": True,
        "valid_gain_over_checkpoint": 0.0,
    }
    candidates = [checkpoint_row]
    checkpoint_key = (
        checkpoint_result["metrics"][args.select_metric],
        checkpoint_result["metrics"]["macro_f1"],
        checkpoint_result["metrics"]["accuracy"],
    )
    best = (checkpoint_key, checkpoint_row, checkpoint_result)
    print("valid " + json.dumps(checkpoint_row, sort_keys=True), flush=True)
    for mode in modes:
        if mode == "checkpoint":
            continue
        ks = topk_grid if mode == "topk_logits" else [3]
        for topk in ks:
            result = aggregate_prediction(valid_raw, ckpt, flow_head, args.device, mode, topk)
            gain = result["metrics"][args.select_metric] - checkpoint_result["metrics"][args.select_metric]
            eligible = gain >= args.min_valid_gain_over_checkpoint
            row = {
                "mode": mode,
                "topk": topk,
                "metrics": result["metrics"],
                "eligible": eligible,
                "valid_gain_over_checkpoint": float(gain),
            }
            candidates.append(row)
            key = (
                result["metrics"][args.select_metric],
                result["metrics"]["macro_f1"],
                result["metrics"]["accuracy"],
            )
            if eligible and key > best[0]:
                best = (key, row, result)
            print("valid " + json.dumps(row, sort_keys=True), flush=True)

    if best is None:
        raise RuntimeError("No flow pooling candidates were evaluated.")
    selected = best[1]
    valid_result = best[2]
    test_raw = predict_windows(model, ckpt["model_type"], args.test_dataset, args.device, args.batch_size)
    test_result = aggregate_prediction(
        test_raw, ckpt, flow_head, args.device, selected["mode"], selected["topk"]
    )
    print("selected " + json.dumps(selected, sort_keys=True), flush=True)
    print("test " + json.dumps(test_result["metrics"], sort_keys=True), flush=True)

    payload = {
        "method": "validation_selected_flow_eval_pooling",
        "result_scope": "single_fold",
        "metrics": {"flow_level": test_result["metrics"]},
        "valid_metrics": valid_result["metrics"],
        "selected": selected,
        "candidate_reports": candidates,
        "flow_ids": test_result["flow_ids"],
        "flow_y_true": test_result["y_true"],
        "flow_y_pred": test_result["y_pred"],
        "flow_prob": test_result["prob"].tolist(),
        "valid_flow_ids": valid_result["flow_ids"],
        "valid_y_true": valid_result["y_true"],
        "valid_y_pred": valid_result["y_pred"],
        "valid_prob": valid_result["prob"].tolist(),
        "audit": {
            "selection_split": "validation",
            "test_labels_used_for_selection": False,
            "checkpoint": args.checkpoint,
            "valid_dataset": args.valid_dataset,
            "test_dataset": args.test_dataset,
            "select_metric": args.select_metric,
            "min_valid_gain_over_checkpoint": args.min_valid_gain_over_checkpoint,
        },
        "eval_config": {
            "multi_view_gate": test_result["multi_view_gate"],
            "flow_stat_gate": test_result["flow_stat_gate"],
        },
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


if __name__ == "__main__":
    main()
