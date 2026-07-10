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
):
    buckets = defaultdict(list)
    emb_buckets = defaultdict(list)
    labels = {}
    for i, (y, fid, logit) in enumerate(zip(y_true, flow_ids, logits_all)):
        buckets[fid].append(logit)
        if emb_all is not None:
            emb_buckets[fid].append(emb_all[i])
        labels[fid] = y
    flow_true, flow_pred = [], []
    for fid, arrs in buckets.items():
        if flow_head is None:
            logits = np.mean(np.stack(arrs, axis=0), axis=0)
        else:
            emb = torch.tensor(np.stack(emb_buckets[fid], axis=0), dtype=torch.float32, device=device)
            win_logits = torch.tensor(np.stack(arrs, axis=0), dtype=torch.float32, device=device)
            pooled = flow_head(emb, window_logits=win_logits)
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
    return flow_true, flow_pred


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
    flow_true, flow_pred = aggregate_by_flow(
        y_true,
        flow_ids,
        logits_all,
        emb_all=emb_all,
        flow_head=flow_head,
        device=args.device,
        class_to_coarse=class_to_coarse,
        hierarchical_logit_weight=ckpt.get("hierarchical_logit_weight", 0.0),
        hierarchical_mode=ckpt.get("hierarchical_mode", "logit"),
    )
    flow_metrics = compute_metrics(flow_true, flow_pred)
    metrics = {"window_level": window_metrics, "flow_level": flow_metrics}
    print(json.dumps(metrics, indent=2))
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
                    "flow_y_true": flow_true,
                    "flow_y_pred": flow_pred,
                },
                f,
                indent=2,
            )


if __name__ == "__main__":
    main()
