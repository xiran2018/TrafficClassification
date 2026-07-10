#!/usr/bin/env python3
"""Tower-2 preprocessing.

Consumes flow_embedding_index.jsonl from extract_packet_embeddings_qwen.py and builds:
  - seq_dataset.pt: sequence windows for non-graph Packet Interaction Transformer
  - graph_dataset.pt: flow-window graphs for edge-aware Graph Transformer

Compared with the first draft, this version adds optional coherence negative windows
for Same-flow Coherence Prediction and keeps labels consistent across splits.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from traffic_utils import meta_feature_vector

EDGE_TYPES = {
    "temporal_next": 0,
    "same_direction": 1,
    "opposite_direction": 2,
    "ack_candidate": 3,
    "seq_continuity": 4,
    "same_burst": 5,
    "retransmission_candidate": 6,
}


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def meta_vec_from_dict(m: Dict[str, Any]) -> List[float]:
    class Obj:
        pass
    o = Obj()
    for k, v in m.items():
        setattr(o, k, v)
    return meta_feature_vector(o)


def length_bin(x: int) -> int:
    if x < 128:
        return 0
    if x < 512:
        return 1
    if x < 1000:
        return 2
    return 3


def iat_bin(x: float) -> int:
    if x < 0.001:
        return 0
    if x < 0.01:
        return 1
    if x < 0.1:
        return 2
    return 3


def window_indices(n: int, window_size: int, stride: int):
    if n <= 0:
        return
    if n <= window_size:
        yield 0, n
    else:
        for start in range(0, n - window_size + 1, stride):
            yield start, start + window_size
        if (n - window_size) % stride != 0:
            yield n - window_size, n


def build_edges(metas: List[Dict[str, Any]], burst_iat_threshold: float = 0.01) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    srcs: List[int] = []
    dsts: List[int] = []
    attrs: List[List[float]] = []
    ack_labels: List[int] = []
    same_burst_labels: List[int] = []
    retrans_labels: List[int] = []

    def add(i: int, j: int, etype: str, label_ack: int = -1, label_burst: int = -1, label_retrans: int = -1):
        mi, mj = metas[i], metas[j]
        dt = max(0.0, float(mj.get("time", 0)) - float(mi.get("time", 0)))
        seq_delta = float(mj.get("seq", -1)) - float(mi.get("seq", -1)) if mi.get("seq", -1) >= 0 and mj.get("seq", -1) >= 0 else 0.0
        ack_delta = float(mj.get("ack", -1)) - float(mi.get("ack", -1)) if mi.get("ack", -1) >= 0 and mj.get("ack", -1) >= 0 else 0.0
        srcs.append(i)
        dsts.append(j)
        attrs.append([EDGE_TYPES[etype], np.log1p(dt), np.sign(seq_delta), np.sign(ack_delta)])
        ack_labels.append(label_ack)
        same_burst_labels.append(label_burst)
        retrans_labels.append(label_retrans)

    n = len(metas)
    for i in range(n - 1):
        add(i, i + 1, "temporal_next")
        add(i + 1, i, "temporal_next")
        same_dir = metas[i].get("direction") == metas[i + 1].get("direction")
        add(i, i + 1, "same_direction" if same_dir else "opposite_direction")
        add(i + 1, i, "same_direction" if same_dir else "opposite_direction")
        if same_dir and float(metas[i + 1].get("iat", 999)) <= burst_iat_threshold:
            add(i, i + 1, "same_burst", label_burst=1)
        else:
            add(i, i + 1, "same_burst", label_burst=0)

    # TCP relation candidate edges within a short horizon.
    for i in range(n):
        mi = metas[i]
        if mi.get("l4") != "TCP" or mi.get("seq", -1) < 0:
            continue
        expected_ack = int(mi.get("seq", 0)) + int(mi.get("payload_len", 0))
        # SYN and FIN also consume one sequence number.
        flags = mi.get("tcp_flags", "") or ""
        if "S" in flags or "F" in flags:
            expected_ack += 1
        for j in range(i + 1, min(n, i + 8)):
            mj = metas[j]
            if mj.get("l4") != "TCP":
                continue
            opposite = mi.get("direction") != mj.get("direction")
            ack_ok = opposite and int(mj.get("ack", -1)) >= expected_ack and expected_ack > 0
            add(i, j, "ack_candidate", label_ack=1 if ack_ok else 0)
            same_dir = mi.get("direction") == mj.get("direction")
            retrans = same_dir and int(mi.get("seq", -1)) == int(mj.get("seq", -2)) and int(mi.get("payload_len", 0)) == int(mj.get("payload_len", -1)) and int(mi.get("payload_len", 0)) > 0
            add(i, j, "retransmission_candidate", label_retrans=1 if retrans else 0)
            if same_dir:
                cont = int(mj.get("seq", -1)) >= int(mi.get("seq", 0)) + int(mi.get("payload_len", 0))
                add(i, j, "seq_continuity", label_ack=1 if cont else 0)

    if not srcs:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 4), dtype=torch.float32)
    else:
        edge_index = torch.tensor([srcs, dsts], dtype=torch.long)
        edge_attr = torch.tensor(attrs, dtype=torch.float32)
    targets = {
        "ack_labels": torch.tensor(ack_labels, dtype=torch.long),
        "same_burst_labels": torch.tensor(same_burst_labels, dtype=torch.long),
        "retrans_labels": torch.tensor(retrans_labels, dtype=torch.long),
    }
    return edge_index, edge_attr, targets


def build_samples_from_flow(row: Dict[str, Any], window_size: int, stride: int):
    emb = np.load(row["embedding_path"]).astype("float32")
    metas = row["packet_metas"]
    n = min(len(metas), emb.shape[0])
    if n == 0:
        return [], []
    emb, metas = emb[:n], metas[:n]
    meta_feats = np.asarray([meta_vec_from_dict(m) for m in metas], dtype="float32")
    x_all = np.concatenate([emb, meta_feats], axis=1)
    seq_samples = []
    graph_samples = []
    for s, e in window_indices(n, window_size, stride):
        x = torch.tensor(x_all[s:e], dtype=torch.float32)
        wm = metas[s:e]
        label = int(row["label_id"])
        next_targets = {"next_direction": -1, "next_length_bin": -1, "next_iat_bin": -1}
        if e < n:
            nxt = metas[e]
            next_targets = {
                "next_direction": 0 if nxt.get("direction") == "C2S" else 1,
                "next_length_bin": length_bin(int(nxt.get("packet_len", 0))),
                "next_iat_bin": iat_bin(float(nxt.get("iat", 0))),
            }
        common = {
            "x": x,
            "label": label,
            "coherence_label": torch.tensor(1, dtype=torch.long),
            "flow_id": row["flow_id"],
            "window": (s, e),
            **{k: torch.tensor(v, dtype=torch.long) for k, v in next_targets.items()},
        }
        seq_samples.append(common.copy())
        edge_index, edge_attr, edge_targets = build_edges(wm)
        graph_samples.append({
            **common,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            **edge_targets,
        })
    return seq_samples, graph_samples


def make_negative_coherence_samples(seq_samples: List[Dict[str, Any]], graph_samples: List[Dict[str, Any]], ratio: float, seed: int):
    """Create mixed-window samples for Same-flow Coherence Prediction.

    Negative samples have label=-1 so they are skipped by the main classification loss,
    but they are used by the coherence head.
    """
    if ratio <= 0 or len(seq_samples) < 2:
        return [], []
    rng = random.Random(seed)
    n_neg = int(len(seq_samples) * ratio)
    neg_seq = []
    neg_graph = []
    for _ in range(n_neg):
        a, b = rng.sample(range(len(seq_samples)), 2)
        sa, sb = seq_samples[a], seq_samples[b]
        if sa["x"].shape[0] < 2 or sb["x"].shape[0] < 2:
            continue
        n = min(sa["x"].shape[0], sb["x"].shape[0])
        cut = max(1, n // 2)
        x = torch.cat([sa["x"][:cut], sb["x"][cut:n]], dim=0)
        item = {
            "x": x,
            "label": -1,
            "coherence_label": torch.tensor(0, dtype=torch.long),
            "flow_id": f"mixed::{sa['flow_id']}::{sb['flow_id']}",
            "window": (0, n),
            "next_direction": torch.tensor(-1, dtype=torch.long),
            "next_length_bin": torch.tensor(-1, dtype=torch.long),
            "next_iat_bin": torch.tensor(-1, dtype=torch.long),
        }
        neg_seq.append(item)
        edge_index = torch.tensor([[i for i in range(n - 1)] + [i + 1 for i in range(n - 1)], [i + 1 for i in range(n - 1)] + [i for i in range(n - 1)]], dtype=torch.long) if n > 1 else torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.tensor([[EDGE_TYPES["temporal_next"], 0.0, 0.0, 0.0]] * edge_index.shape[1], dtype=torch.float32) if edge_index.numel() else torch.zeros((0, 4), dtype=torch.float32)
        neg_graph.append({
            **item,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "ack_labels": torch.full((edge_index.shape[1],), -1, dtype=torch.long),
            "same_burst_labels": torch.full((edge_index.shape[1],), -1, dtype=torch.long),
            "retrans_labels": torch.full((edge_index.shape[1],), -1, dtype=torch.long),
        })
    return neg_seq, neg_graph


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--flow_embedding_index", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--window_size", type=int, default=32)
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--negative_coherence_ratio", type=float, default=0.0, help="Use >0 for training only, e.g., 0.5.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    seq_samples: List[Dict[str, Any]] = []
    graph_samples: List[Dict[str, Any]] = []
    for row in load_jsonl(args.flow_embedding_index):
        ss, gs = build_samples_from_flow(row, args.window_size, args.stride)
        seq_samples.extend(ss)
        graph_samples.extend(gs)

    neg_seq, neg_graph = make_negative_coherence_samples(seq_samples, graph_samples, args.negative_coherence_ratio, args.seed)
    seq_samples.extend(neg_seq)
    graph_samples.extend(neg_graph)

    torch.save(seq_samples, Path(args.output_dir) / "seq_dataset.pt")
    torch.save(graph_samples, Path(args.output_dir) / "graph_dataset.pt")
    print(f"saved seq={len(seq_samples)}, graph={len(graph_samples)} to {args.output_dir}")
    print(f"coherence negatives: seq={len(neg_seq)}, graph={len(neg_graph)}")


if __name__ == "__main__":
    main()
