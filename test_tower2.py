#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

from train_tower2 import (
    FlowAggregationHead,
    SeqDataset,
    GraphDataset,
    collate_seq,
    apply_hierarchical_logits,
    build_class_to_coarse,
    parse_label_groups,
)
from models.flow_transformer import FlowTransformerClassifier
from models.flow_graph_transformer import FlowGraphTransformerClassifier


def load_model(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device)
    if ckpt["model_type"] == "seq":
        model = FlowTransformerClassifier(ckpt["input_dim"], ckpt["num_classes"], ckpt["hidden_dim"], ckpt["num_layers"], ckpt["num_heads"], ckpt["dropout"])
    else:
        model = FlowGraphTransformerClassifier(
            ckpt["input_dim"],
            ckpt["num_classes"],
            ckpt["hidden_dim"],
            ckpt["num_layers"],
            ckpt["num_heads"],
            edge_attr_dim=ckpt.get("edge_attr_dim", 4),
            dropout=ckpt["dropout"],
        )
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    flow_head = None
    if "flow_head_state" in ckpt:
        hierarchical_mode = ckpt.get("hierarchical_mode", "logit")
        class_groups = ckpt.get("class_groups")
        if class_groups is None and hierarchical_mode == "expert":
            class_groups = parse_label_groups(ckpt.get("coarse_groups", "vpn_app"), ckpt["num_classes"])
        flow_head = FlowAggregationHead(
            ckpt["hidden_dim"],
            ckpt["num_classes"],
            pooling=ckpt.get("flow_pooling", "attention"),
            num_coarse_classes=ckpt.get("num_coarse_classes", 0),
            class_groups=class_groups if hierarchical_mode == "expert" else None,
            hierarchical_mode=hierarchical_mode,
            flow_transformer_layers=ckpt.get("flow_transformer_layers", 1),
            flow_transformer_heads=ckpt.get("flow_transformer_heads", 4),
            dropout=ckpt.get("dropout", 0.1),
        )
        flow_head.load_state_dict(ckpt["flow_head_state"])
        flow_head.to(device).eval()
    return model, ckpt, flow_head


@torch.no_grad()
def predict_seq(model, dataset_path: str, device: str, batch_size: int):
    ds = SeqDataset(dataset_path)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_seq)
    y_true, y_pred, flow_ids, logits_all, emb_all = [], [], [], [], []
    for batch in dl:
        out = model(batch["x"].to(device), batch["mask"].to(device))
        logits = out["logits"].cpu()
        emb = out["embedding"].cpu()
        labels = batch["label"]
        for i in range(logits.size(0)):
            if int(labels[i]) < 0:
                continue
            y_true.append(int(labels[i]))
            y_pred.append(int(logits[i].argmax()))
            flow_ids.append(batch["flow_id"][i])
            logits_all.append(logits[i].numpy())
            emb_all.append(emb[i].numpy())
    return y_true, y_pred, flow_ids, logits_all, emb_all


@torch.no_grad()
def predict_graph(model, dataset_path: str, device: str):
    ds = GraphDataset(dataset_path)
    y_true, y_pred, flow_ids, logits_all, emb_all = [], [], [], [], []
    for item in ds:
        label = int(item.get("label", -1))
        if label < 0:
            continue
        out = model(item["x"].to(device), item["edge_index"].to(device), item["edge_attr"].to(device))
        logits = out["logits"].squeeze(0).cpu().numpy()
        y_true.append(label)
        y_pred.append(int(logits.argmax()))
        flow_ids.append(str(item.get("flow_id", "")))
        logits_all.append(logits)
        emb_all.append(out["embedding"].cpu().numpy())
    return y_true, y_pred, flow_ids, logits_all, emb_all


def aggregate_by_flow(
    y_true,
    flow_ids,
    logits_all,
    emb_all=None,
    flow_head=None,
    device: str = "cpu",
    class_to_coarse=None,
    hierarchical_logit_weight: float = 0.0,
    hierarchical_mode: str = "logit",
    flow_eval_pooling: str = "checkpoint",
    flow_eval_topk: int = 3,
):
    buckets = defaultdict(list)
    emb_buckets = defaultdict(list)
    labels = {}
    for i, (y, fid, logit) in enumerate(zip(y_true, flow_ids, logits_all)):
        buckets[fid].append(logit)
        if emb_all is not None:
            emb_buckets[fid].append(emb_all[i])
        labels[fid] = y
    flow_true, flow_pred, out_flow_ids, flow_logits_all = [], [], [], []
    multi_view_gates = []
    for fid, arrs in buckets.items():
        if flow_head is None or flow_eval_pooling != "checkpoint":
            pooling = "mean_logits" if flow_eval_pooling == "checkpoint" else flow_eval_pooling
            logits = pool_window_logits(arrs, pooling, flow_eval_topk)
        else:
            emb = torch.tensor(np.stack(emb_buckets[fid], axis=0), dtype=torch.float32, device=device)
            win_logits = torch.tensor(np.stack(arrs, axis=0), dtype=torch.float32, device=device)
            pooled = flow_head(emb, window_logits=win_logits)
            if pooled.get("multi_view_gate") is not None:
                multi_view_gates.append(pooled["multi_view_gate"].detach().cpu().numpy())
            logits = pooled["logits"]
            if hierarchical_mode != "expert":
                logits = apply_hierarchical_logits(
                    logits,
                    pooled.get("coarse_logits"),
                    class_to_coarse,
                    hierarchical_logit_weight,
                )
            logits = logits.detach().cpu().numpy()
        flow_true.append(labels[fid])
        flow_pred.append(int(logits.argmax()))
        out_flow_ids.append(fid)
        flow_logits_all.append(np.asarray(logits, dtype=np.float32))
    gate_summary = None
    if multi_view_gates:
        gate_arr = np.stack(multi_view_gates, axis=0)
        gate_summary = {
            "branches": ["mean", "max", "std", "attention"],
            "mean": gate_arr.mean(axis=0).astype(float).tolist(),
            "std": gate_arr.std(axis=0).astype(float).tolist(),
            "num_flows": int(gate_arr.shape[0]),
        }
    return flow_true, flow_pred, out_flow_ids, flow_logits_all, gate_summary


def softmax_np(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=-1, keepdims=True).clip(min=1e-12)


def pool_window_logits(arrs, mode: str, topk: int) -> np.ndarray:
    logits = np.stack(arrs, axis=0)
    if mode == "mean_logits":
        return logits.mean(axis=0)

    probs = softmax_np(logits)
    if mode == "mean_probs":
        return probs.mean(axis=0)

    conf = probs.max(axis=1)
    if mode == "max_conf":
        return logits[int(conf.argmax())]
    if mode == "topk_logits":
        k = min(max(1, int(topk)), logits.shape[0])
        idx = np.argsort(conf)[-k:]
        return logits[idx].mean(axis=0)
    if mode == "vote":
        votes = logits.argmax(axis=1)
        counts = np.bincount(votes, minlength=logits.shape[1]).astype(np.float32)
        return counts + 1e-6 * probs.mean(axis=0)
    raise ValueError(f"Unknown --flow_eval_pooling: {mode}")


def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred) if y_true else 0.0
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    p_weight, r_weight, f_weight, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    return {
        "accuracy": acc,
        "macro_precision": p_macro,
        "macro_recall": r_macro,
        "macro_f1": f_macro,
        "weighted_precision": p_weight,
        "weighted_recall": r_weight,
        "weighted_f1": f_weight,
    }


def load_label_names(path: str):
    if not path:
        return None, None
    with open(path, "r", encoding="utf-8") as f:
        label_map = json.load(f)
    if not label_map:
        return None, label_map
    max_id = max(int(v) for v in label_map.values())
    label_names = [str(i) for i in range(max_id + 1)]
    for name, idx in label_map.items():
        label_names[int(idx)] = name
    return label_names, label_map


def print_report(title: str, y_true, y_pred, label_names):
    print(f"\n{title}:")
    if label_names:
        labels = list(range(len(label_names)))
        print(classification_report(y_true, y_pred, labels=labels, target_names=label_names, zero_division=0))
    else:
        print(classification_report(y_true, y_pred, zero_division=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--label_map", default="", help="Optional JSON mapping label name to label id, used for readable reports.")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--output_json", default="")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no_report", action="store_true", help="Only print the metrics JSON, without classification reports.")
    ap.add_argument(
        "--flow_eval_pooling",
        default="checkpoint",
        choices=["checkpoint", "mean_logits", "mean_probs", "max_conf", "topk_logits", "vote"],
        help="Flow-level aggregation at eval time. 'checkpoint' uses the trained flow head when available.",
    )
    ap.add_argument("--flow_eval_topk", type=int, default=3, help="Top-k windows for --flow_eval_pooling topk_logits.")
    args = ap.parse_args()
    model, ckpt, flow_head = load_model(args.checkpoint, args.device)
    label_names, label_map = load_label_names(args.label_map)
    class_to_coarse = None
    if ckpt.get("num_coarse_classes", 0) > 0:
        class_to_coarse, _ = build_class_to_coarse(ckpt.get("coarse_groups", "vpn_app"), ckpt["num_classes"], args.device)
    if ckpt["model_type"] == "seq":
        y_true, y_pred, flow_ids, logits_all, emb_all = predict_seq(model, args.dataset, args.device, args.batch_size)
    else:
        y_true, y_pred, flow_ids, logits_all, emb_all = predict_graph(model, args.dataset, args.device)

    window_metrics = compute_metrics(y_true, y_pred)
    flow_true, flow_pred, out_flow_ids, flow_logits_all, multi_view_gate_summary = aggregate_by_flow(
        y_true,
        flow_ids,
        logits_all,
        emb_all=emb_all,
        flow_head=flow_head,
        device=args.device,
        class_to_coarse=class_to_coarse,
        hierarchical_logit_weight=ckpt.get("hierarchical_logit_weight", 0.0),
        hierarchical_mode=ckpt.get("hierarchical_mode", "logit"),
        flow_eval_pooling=args.flow_eval_pooling,
        flow_eval_topk=args.flow_eval_topk,
    )
    flow_metrics = compute_metrics(flow_true, flow_pred)
    metrics = {
        "window_level": window_metrics,
        "flow_level": flow_metrics,
        "eval_config": {
            "flow_eval_pooling": args.flow_eval_pooling,
            "flow_eval_topk": args.flow_eval_topk,
            "multi_view_gate": multi_view_gate_summary,
        },
    }
    print(json.dumps(metrics, indent=2))
    if not args.no_report:
        print_report("Window-level report", y_true, y_pred, label_names)
        print_report("Flow-level report", flow_true, flow_pred, label_names)
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "metrics": metrics,
                    "label_map": label_map,
                    "window_y_true": y_true,
                    "window_y_pred": y_pred,
                    "window_prob": softmax_np(np.stack(logits_all, axis=0)).tolist() if logits_all else [],
                    "window_flow_ids": flow_ids,
                    "flow_y_true": flow_true,
                    "flow_y_pred": flow_pred,
                    "flow_prob": softmax_np(np.stack(flow_logits_all, axis=0)).tolist() if flow_logits_all else [],
                    "flow_ids": out_flow_ids,
                },
                f,
                indent=2,
            )


if __name__ == "__main__":
    main()
