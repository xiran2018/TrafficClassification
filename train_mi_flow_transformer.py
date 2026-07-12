#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset


def load_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_label_names(path: str):
    if not path:
        return None, None
    with open(path, "r", encoding="utf-8") as f:
        label_map = json.load(f)
    max_id = max(int(v) for v in label_map.values())
    label_names = [str(i) for i in range(max_id + 1)]
    for name, idx in label_map.items():
        label_names[int(idx)] = name
    return label_names, label_map


def compute_metrics(y_true, y_pred):
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    p_weight, r_weight, f_weight, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    return {
        "accuracy": accuracy_score(y_true, y_pred) if len(y_true) else 0.0,
        "macro_precision": p_macro,
        "macro_recall": r_macro,
        "macro_f1": f_macro,
        "weighted_precision": p_weight,
        "weighted_recall": r_weight,
        "weighted_f1": f_weight,
    }


def meta_features(metas: Sequence[Dict[str, Any]], max_packets: int) -> np.ndarray:
    feats = []
    last_t = None
    for i in range(max_packets):
        if i >= len(metas):
            feats.append([0.0] * 14)
            continue
        m = metas[i]
        direction = 1.0 if m.get("direction") == "C2S" else -1.0
        packet_len = max(float(m.get("packet_len", 0) or 0), 0.0)
        payload_len = max(float(m.get("payload_len", 0) or 0), 0.0)
        captured = max(float(m.get("l3_captured_len", 0) or 0), 0.0)
        entropy = max(float(m.get("payload_entropy", 0) or 0), 0.0) / 8.0
        iat = max(float(m.get("iat", 0) or 0), 0.0)
        if last_t is None:
            rel_time = 0.0
        else:
            rel_time = max(float(m.get("time", 0) or 0) - last_t, 0.0)
        last_t = float(m.get("time", 0) or 0)
        l4 = str(m.get("l4", ""))
        flags = str(m.get("tcp_flags", ""))
        sport = int(m.get("sport", -1) or -1)
        dport = int(m.get("dport", -1) or -1)
        feats.append(
            [
                direction,
                math.log1p(packet_len) / math.log1p(1514.0),
                math.log1p(payload_len) / math.log1p(1460.0),
                math.log1p(captured) / math.log1p(1514.0),
                math.log1p(iat),
                math.log1p(rel_time),
                entropy,
                1.0 if l4 == "TCP" else 0.0,
                1.0 if l4 == "UDP" else 0.0,
                1.0 if "S" in flags else 0.0,
                1.0 if "A" in flags else 0.0,
                1.0 if "P" in flags else 0.0,
                math.log1p(max(sport, 0)) / math.log1p(65535.0),
                math.log1p(max(dport, 0)) / math.log1p(65535.0),
            ]
        )
    return np.asarray(feats, dtype=np.float32)


class FlowEmbeddingDataset(Dataset):
    def __init__(self, index_path: str, max_packets: int, embedding_dropout_prob: float = 0.0):
        self.rows = list(load_jsonl(index_path))
        self.max_packets = max_packets
        self.embedding_dropout_prob = embedding_dropout_prob

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        emb = np.load(row["embedding_path"]).astype(np.float32)
        emb = emb[: self.max_packets]
        n = emb.shape[0]
        dim = int(row.get("embedding_dim", emb.shape[1] if emb.ndim == 2 else 0))
        x = np.zeros((self.max_packets, dim), dtype=np.float32)
        if emb.ndim == 2 and n > 0:
            x[:n, : emb.shape[1]] = emb
        meta = meta_features(row.get("packet_metas", [])[: self.max_packets], self.max_packets)
        mask = np.zeros((self.max_packets,), dtype=np.bool_)
        mask[:n] = True
        return {
            "x": x,
            "meta": meta,
            "mask": mask,
            "label": int(row["label_id"]),
            "flow_id": str(row.get("flow_id", idx)),
        }


def collate(batch):
    return {
        "x": torch.tensor(np.stack([b["x"] for b in batch]), dtype=torch.float32),
        "meta": torch.tensor(np.stack([b["meta"] for b in batch]), dtype=torch.float32),
        "mask": torch.tensor(np.stack([b["mask"] for b in batch]), dtype=torch.bool),
        "label": torch.tensor([b["label"] for b in batch], dtype=torch.long),
        "flow_id": [b["flow_id"] for b in batch],
    }


class MIFlowTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        meta_dim: int,
        num_classes: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.2,
        pooling: str = "cls_attention",
    ):
        super().__init__()
        self.pooling = pooling
        self.input_norm = nn.LayerNorm(input_dim)
        self.embedding_proj = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout))
        self.meta_proj = nn.Sequential(nn.Linear(meta_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout))
        self.cls = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.pos = nn.Parameter(torch.zeros(1, 1 + 256, hidden_dim))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.attn = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
        self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes))
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x: torch.Tensor, meta: torch.Tensor, mask: torch.Tensor):
        h = self.embedding_proj(self.input_norm(x)) + self.meta_proj(meta)
        bsz = h.size(0)
        cls = self.cls.expand(bsz, -1, -1)
        h = torch.cat([cls, h], dim=1)
        h = h + self.pos[:, : h.size(1)]
        key_padding_mask = torch.cat([torch.zeros(bsz, 1, dtype=torch.bool, device=mask.device), ~mask], dim=1)
        h = self.encoder(h, src_key_padding_mask=key_padding_mask)
        if self.pooling == "cls":
            z = h[:, 0]
        else:
            token_h = h[:, 1:]
            score = self.attn(token_h).squeeze(-1).masked_fill(~mask, -1e9)
            weights = torch.softmax(score, dim=-1)
            attn_z = torch.sum(token_h * weights.unsqueeze(-1), dim=1)
            z = 0.5 * h[:, 0] + 0.5 * attn_z if self.pooling == "cls_attention" else attn_z
        return {"embedding": z, "logits": self.head(z)}


def class_weights(labels: Sequence[int], num_classes: int, beta: float, strength: float, device: str):
    counts = torch.zeros(num_classes, dtype=torch.float32)
    for label in labels:
        counts[int(label)] += 1.0
    present = counts > 0
    weights = torch.ones(num_classes, dtype=torch.float32)
    beta = min(max(float(beta), 0.0), 0.999999)
    eff = (1.0 - beta) / (1.0 - torch.pow(torch.tensor(beta), counts[present]))
    eff = eff / eff.mean().clamp(min=1e-12)
    weights[present] = 1.0 + min(max(strength, 0.0), 1.0) * (eff - 1.0)
    return weights.to(device)


def evaluate(model, loader, device: str, num_classes: int):
    model.eval()
    y_true, y_pred, probs, fids = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            out = model(batch["x"].to(device), batch["meta"].to(device), batch["mask"].to(device))
            prob = torch.softmax(out["logits"], dim=-1).cpu().numpy()
            pred = prob.argmax(axis=1)
            probs.extend(prob.tolist())
            y_pred.extend(pred.tolist())
            y_true.extend(batch["label"].tolist())
            fids.extend(batch["flow_id"])
    return compute_metrics(y_true, y_pred), y_true, y_pred, probs, fids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_index", required=True)
    ap.add_argument("--valid_index", required=True)
    ap.add_argument("--test_index", required=True)
    ap.add_argument("--label_map", default="")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--output_json", default="")
    ap.add_argument("--num_classes", type=int, required=True)
    ap.add_argument("--max_packets", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--num_heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.25)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.03)
    ap.add_argument("--label_smoothing", type=float, default=0.05)
    ap.add_argument("--class_weight_beta", type=float, default=0.9999)
    ap.add_argument("--class_weight_strength", type=float, default=0.5)
    ap.add_argument("--pooling", choices=["cls", "attention", "cls_attention"], default="cls_attention")
    ap.add_argument("--select_metric", choices=["accuracy", "macro_f1"], default="macro_f1")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_ds = FlowEmbeddingDataset(args.train_index, args.max_packets)
    valid_ds = FlowEmbeddingDataset(args.valid_index, args.max_packets)
    test_ds = FlowEmbeddingDataset(args.test_index, args.max_packets)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    sample = train_ds[0]
    model = MIFlowTransformer(
        input_dim=sample["x"].shape[1],
        meta_dim=sample["meta"].shape[1],
        num_classes=args.num_classes,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        pooling=args.pooling,
    ).to(args.device)
    weights = class_weights([r["label_id"] for r in train_ds.rows], args.num_classes, args.class_weight_beta, args.class_weight_strength, args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, ok, cnt = 0.0, 0, 0
        for batch in train_loader:
            x = batch["x"].to(args.device)
            meta = batch["meta"].to(args.device)
            mask = batch["mask"].to(args.device)
            y = batch["label"].to(args.device)
            out = model(x, meta, mask)
            loss = F.cross_entropy(out["logits"], y, weight=weights, label_smoothing=args.label_smoothing)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.item()) * y.numel()
            ok += int((out["logits"].argmax(dim=-1) == y).sum())
            cnt += int(y.numel())
        val_metrics, *_ = evaluate(model, valid_loader, args.device, args.num_classes)
        score = val_metrics[args.select_metric]
        print(
            f"epoch={epoch} train_loss={total/max(cnt,1):.4f} train_acc={ok/max(cnt,1):.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} val_macro_f1={val_metrics['macro_f1']:.4f} "
            f"select={args.select_metric}:{score:.4f}",
            flush=True,
        )
        if score > best:
            best = score
            torch.save({"model_state": model.state_dict(), "args": vars(args), "input_dim": sample["x"].shape[1], "meta_dim": sample["meta"].shape[1]}, out_dir / "best.pt")

    ckpt = torch.load(out_dir / "best.pt", map_location=args.device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    valid_metrics, valid_y, valid_pred, valid_prob, valid_fids = evaluate(model, valid_loader, args.device, args.num_classes)
    test_metrics, test_y, test_pred, test_prob, test_fids = evaluate(model, test_loader, args.device, args.num_classes)
    print("best_valid", json.dumps(valid_metrics, sort_keys=True))
    print("test", json.dumps(test_metrics, indent=2, sort_keys=True))
    label_names, label_map = load_label_names(args.label_map)
    if label_names:
        print(classification_report(test_y, test_pred, labels=list(range(len(label_names))), target_names=label_names, zero_division=0))
    else:
        print(classification_report(test_y, test_pred, zero_division=0))

    if args.output_json:
        payload = {
            "metrics": {"flow_level": test_metrics},
            "valid_metrics": valid_metrics,
            "label_map": label_map,
            "flow_ids": test_fids,
            "flow_y_true": test_y,
            "flow_y_pred": test_pred,
            "flow_prob": test_prob,
            "valid_flow_ids": valid_fids,
            "valid_y_true": valid_y,
            "valid_y_pred": valid_pred,
            "valid_prob": valid_prob,
            "config": vars(args),
        }
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()
