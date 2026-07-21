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
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from audit_packet_identifiability import packet_signature
from traffic_utils import current_packet_meta_feature_vector, meta_feature_vector

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


def sha256_file(path: str, cache: Dict[str, str]) -> str:
    key = str(Path(path).expanduser().resolve())
    cached = cache.get(key)
    if cached is not None:
        return cached
    h = hashlib.sha256()
    with open(key, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    digest = h.hexdigest()
    cache[key] = digest
    return digest


def attach_content_groups(
    rows: List[Dict[str, Any]],
    index_path: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any] | None]:
    """Attach exact-PCAP content group ids to flow rows.

    The group id is deterministic for one preprocessing run: all flows whose
    original PCAP bytes have the same SHA256 hash receive the same integer id.
    This is intentionally optional so existing Tower-2 datasets remain
    byte-for-byte behavior-compatible unless a caller asks for group metadata.
    """
    if not index_path:
        return rows, None

    references: Dict[str, Dict[str, Any]] = {}
    for row in load_jsonl(index_path):
        flow_id = str(row["flow_id"])
        if flow_id in references:
            raise ValueError(f"duplicate flow_id in content group index: {flow_id}")
        references[flow_id] = row

    cache: Dict[str, str] = {}
    flow_to_hash: Dict[str, str] = {}
    flow_to_path: Dict[str, str] = {}
    flow_to_label: Dict[str, int] = {}
    for flow_id, ref in references.items():
        pcap_path = str(ref.get("pcap_path") or "")
        if not pcap_path:
            raise ValueError(f"content group index row is missing pcap_path for flow_id={flow_id}")
        flow_to_hash[flow_id] = sha256_file(pcap_path, cache)
        flow_to_path[flow_id] = pcap_path
        flow_to_label[flow_id] = int(ref["label_id"])

    hash_to_group_id = {digest: idx for idx, digest in enumerate(sorted(set(flow_to_hash.values())))}
    output_rows: List[Dict[str, Any]] = []
    missing: List[str] = []
    label_mismatches: List[str] = []
    for row in rows:
        flow_id = str(row["flow_id"])
        digest = flow_to_hash.get(flow_id)
        if digest is None:
            missing.append(flow_id)
            continue
        if int(row["label_id"]) != flow_to_label[flow_id]:
            label_mismatches.append(flow_id)
            continue
        out = dict(row)
        out["content_hash"] = digest
        out["content_group_id"] = int(hash_to_group_id[digest])
        out["content_pcap_path"] = flow_to_path[flow_id]
        output_rows.append(out)

    if missing or label_mismatches:
        raise ValueError(
            "content group index does not align with flow rows: "
            f"missing={missing[:3]} label_mismatches={label_mismatches[:3]}"
        )

    group_sizes: Dict[int, int] = {}
    for row in output_rows:
        group_id = int(row["content_group_id"])
        group_sizes[group_id] = group_sizes.get(group_id, 0) + 1
    duplicate_groups = sum(1 for size in group_sizes.values() if size > 1)
    manifest = {
        "version": 1,
        "method": "exact_pcap_sha256_content_groups",
        "source_index": index_path,
        "num_flows": len(output_rows),
        "num_content_groups": len(group_sizes),
        "duplicate_content_groups": duplicate_groups,
        "max_group_size": max(group_sizes.values(), default=0),
    }
    return output_rows, manifest


def load_metadata_reference(path: str) -> Dict[str, Dict[str, Any]]:
    if not path:
        return {}
    references: Dict[str, Dict[str, Any]] = {}
    for row in load_jsonl(path):
        flow_id = str(row["flow_id"])
        if flow_id in references:
            raise ValueError(f"duplicate flow_id in metadata reference: {flow_id}")
        references[flow_id] = row
    return references


def load_structural_embedding_index(path: str) -> Dict[str, Dict[str, Any]]:
    if not path:
        return {}
    references: Dict[str, Dict[str, Any]] = {}
    for row in load_jsonl(path):
        flow_id = str(row["flow_id"])
        if flow_id in references:
            raise ValueError(f"duplicate flow_id in structural embedding index: {flow_id}")
        references[flow_id] = row
    return references


def apply_structural_embedding_reference(
    row: Dict[str, Any],
    references: Dict[str, Dict[str, Any]],
    required_scope: str = "",
) -> Dict[str, Any]:
    if not references:
        return row
    flow_id = str(row["flow_id"])
    reference = references.get(flow_id)
    if reference is None:
        raise ValueError(f"structural embedding index is missing flow_id={flow_id}")
    if int(row["label_id"]) != int(reference["label_id"]):
        raise ValueError(f"structural embedding label mismatch for flow_id={flow_id}")
    observed_scope = str(reference.get("representation_scope", "unknown"))
    if required_scope and observed_scope != required_scope:
        raise ValueError(
            f"structural representation scope mismatch for flow_id={flow_id}: "
            f"observed={observed_scope!r} required={required_scope!r}"
        )
    semantic = np.load(row["embedding_path"], mmap_mode="r")
    structural = np.load(reference["structural_embedding_path"], mmap_mode="r")
    metas = row.get("packet_metas", [])
    expected_count = min(int(semantic.shape[0]), len(metas))
    if int(structural.shape[0]) != expected_count:
        raise ValueError(
            f"structural packet count mismatch for flow_id={flow_id}: "
            f"semantic/meta={expected_count} structural={structural.shape[0]}"
        )
    expected_packet_ids = [int(meta.get("packet_id", index)) for index, meta in enumerate(metas[:expected_count])]
    observed_packet_ids = [int(value) for value in reference.get("packet_ids", expected_packet_ids)]
    if observed_packet_ids != expected_packet_ids:
        raise ValueError(f"structural packet_id order mismatch for flow_id={flow_id}")
    output = dict(row)
    output["structural_embedding_path"] = reference["structural_embedding_path"]
    output["structural_embedding_dim"] = int(structural.shape[1])
    output["structural_encoder_checkpoint"] = reference.get("encoder_checkpoint", "")
    output["structural_representation_scope"] = observed_scope
    output["structural_representation_name"] = reference.get("representation_name", "")
    return output


def apply_metadata_reference(
    row: Dict[str, Any], references: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    if not references:
        return row
    flow_id = str(row["flow_id"])
    reference = references.get(flow_id)
    if reference is None:
        raise ValueError(f"metadata reference is missing flow_id={flow_id}")
    if int(row["label_id"]) != int(reference["label_id"]):
        raise ValueError(f"metadata reference label mismatch for flow_id={flow_id}")
    embedding = np.load(row["embedding_path"], mmap_mode="r")
    metas = reference.get("packet_metas", [])
    if int(embedding.shape[0]) != len(metas):
        raise ValueError(
            f"metadata reference packet count mismatch for flow_id={flow_id}: "
            f"embedding={embedding.shape[0]} reference={len(metas)}"
        )
    output = dict(row)
    output["packet_metas"] = metas
    output["metadata_reference_flow_id"] = flow_id
    return output


def meta_vec_from_dict(m: Dict[str, Any], strict_current_packet: bool = False) -> List[float]:
    class Obj:
        pass
    o = Obj()
    for k, v in m.items():
        setattr(o, k, v)
    return current_packet_meta_feature_vector(o) if strict_current_packet else meta_feature_vector(o)


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


def _signature_reliability(label_counts: Dict[str, int]) -> float:
    counts = np.asarray(list(label_counts.values()), dtype=np.float64)
    probabilities = counts / counts.sum()
    if len(probabilities) <= 1:
        return 1.0
    entropy = float(-(probabilities * np.log(probabilities.clip(min=1e-12))).sum())
    return float(max(0.0, 1.0 - entropy / np.log(len(probabilities))))


def build_identifiability_profile(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    counts: Dict[str, Dict[str, Dict[str, int]]] = {
        "session": {},
        "semantic": {},
    }
    for row in rows:
        label = str(int(row["label_id"]))
        for meta in row.get("packet_metas", []):
            packet_row = {"meta": meta}
            for level in counts:
                signature = packet_signature(packet_row, level)
                label_counts = counts[level].setdefault(signature, {})
                label_counts[label] = label_counts.get(label, 0) + 1
    profile: Dict[str, Any] = {"version": 1, "levels": {}}
    for level, signatures in counts.items():
        profile["levels"][level] = {
            signature: {
                "support": int(sum(label_counts.values())),
                "reliability": _signature_reliability(label_counts),
                "label_counts": label_counts,
            }
            for signature, label_counts in signatures.items()
        }
    return profile


def packet_identifiability(
    meta: Dict[str, Any], profile: Dict[str, Any], min_support: int = 2
) -> Tuple[float, int, str]:
    packet_row = {"meta": meta}
    for level in ("session", "semantic"):
        signature = packet_signature(packet_row, level)
        entry = profile.get("levels", {}).get(level, {}).get(signature)
        if entry is not None and int(entry.get("support", 0)) >= min_support:
            return (
                float(entry.get("reliability", 0.0)),
                int(entry.get("support", 0)),
                level,
            )
    payload_len = int(meta.get("payload_len", 0))
    flags = str(meta.get("tcp_flags", "") or "")
    if payload_len > 0:
        return 1.0, 0, "payload_fallback"
    if meta.get("l4") == "TCP" and ("F" in flags or flags == "A"):
        return 0.25, 0, "control_fallback"
    return 0.5, 0, "unknown_fallback"


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


def build_samples_from_flow(
    row: Dict[str, Any],
    window_size: int,
    stride: int,
    identifiability_profile: Dict[str, Any] | None = None,
    identifiability_min_support: int = 2,
    strict_current_packet_features: bool = False,
):
    emb = np.load(row["embedding_path"]).astype("float32")
    metas = row["packet_metas"]
    n = min(len(metas), emb.shape[0])
    if n == 0:
        return [], []
    emb, metas = emb[:n], metas[:n]
    meta_feats = np.asarray(
        [meta_vec_from_dict(m, strict_current_packet_features) for m in metas],
        dtype="float32",
    )
    structural = None
    if row.get("structural_embedding_path"):
        structural = np.load(row["structural_embedding_path"]).astype("float32")[:n]
        if structural.shape[0] != n:
            raise ValueError(
                f"structural embedding packet count changed for flow_id={row['flow_id']}"
            )
    identifiability = None
    if identifiability_profile is not None:
        identifiability_rows = [
            packet_identifiability(meta, identifiability_profile, identifiability_min_support)
            for meta in metas
        ]
        identifiability = np.asarray(
            [
                [reliability, np.log1p(support)]
                for reliability, support, _ in identifiability_rows
            ],
            dtype="float32",
        )
        components = [emb]
        if structural is not None:
            components.append(structural)
        components.extend([identifiability, meta_feats])
        x_all = np.concatenate(components, axis=1)
    else:
        components = [emb]
        if structural is not None:
            components.append(structural)
        components.append(meta_feats)
        x_all = np.concatenate(components, axis=1)
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
        if "content_group_id" in row:
            common["content_group_id"] = int(row["content_group_id"])
            common["content_hash"] = str(row.get("content_hash", ""))
            common["content_pcap_path"] = str(row.get("content_pcap_path", row.get("pcap_path", "")))
        if identifiability is not None:
            common["packet_identifiability"] = torch.tensor(
                identifiability[s:e, 0], dtype=torch.float32
            )
            common["packet_identifiability_support"] = torch.tensor(
                identifiability[s:e, 1], dtype=torch.float32
            )
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
    ap.add_argument("--build_identifiability_profile", default="")
    ap.add_argument("--identifiability_profile", default="")
    ap.add_argument("--identifiability_min_support", type=int, default=2)
    ap.add_argument("--metadata_reference_index", default="", help="Optional clean-view flow embedding index whose fully parsed packet_metas replace a paired view's legacy/reduced metadata after strict flow/label/count checks.")
    ap.add_argument("--structural_embedding_index", default="", help="Optional flow_structural_embedding_index.jsonl from extract_native_flow_embeddings.py. Native packet embeddings are appended after semantic embeddings and before trailing metadata.")
    ap.add_argument("--require_structural_scope", default="", help="Reject structural embeddings that do not attest this packet-context scope.")
    ap.add_argument("--strict_current_packet_features", action="store_true", help="Exclude cross-packet IAT from the shared structural channel.")
    ap.add_argument("--content_group_index", default="", help="Optional flow_embedding_index.jsonl used to attach exact-PCAP SHA256 content_group_id metadata for group-aware validation/training.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    rows = list(load_jsonl(args.flow_embedding_index))
    rows, content_group_manifest = attach_content_groups(rows, args.content_group_index)
    if content_group_manifest is not None:
        manifest_path = Path(args.output_dir) / "content_group_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(content_group_manifest, handle, indent=2, ensure_ascii=False)
        print(
            "attached content groups "
            + json.dumps(content_group_manifest, sort_keys=True),
            flush=True,
        )
    metadata_references = load_metadata_reference(args.metadata_reference_index)
    if metadata_references:
        rows = [apply_metadata_reference(row, metadata_references) for row in rows]
        if len(rows) != len(metadata_references):
            missing = sorted(set(metadata_references) - {str(row["flow_id"]) for row in rows})
            raise ValueError(
                "metadata reference contains unmatched flows: "
                f"count={len(missing)} examples={missing[:3]}"
            )
        print(
            f"reused clean metadata for {len(rows)} paired flows from "
            f"{args.metadata_reference_index}"
        )
    structural_references = load_structural_embedding_index(args.structural_embedding_index)
    if structural_references:
        semantic_flow_ids = {str(row["flow_id"]) for row in rows}
        missing_from_semantic = sorted(structural_references.keys() - semantic_flow_ids)
        missing_from_structural = sorted(semantic_flow_ids - structural_references.keys())
        if missing_from_semantic or missing_from_structural:
            raise ValueError(
                "semantic/structural flow sets differ: "
                f"structural_only={missing_from_semantic[:3]} "
                f"semantic_only={missing_from_structural[:3]}"
            )
        rows = [
            apply_structural_embedding_reference(
                row, structural_references, args.require_structural_scope
            )
            for row in rows
        ]
        print(
            f"attached native structural embeddings for {len(rows)} flows from "
            f"{args.structural_embedding_index}"
        )
    if args.build_identifiability_profile and args.identifiability_profile:
        ap.error("use only one of --build_identifiability_profile or --identifiability_profile")
    profile = None
    if args.build_identifiability_profile:
        profile = build_identifiability_profile(rows)
        profile_path = Path(args.build_identifiability_profile)
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        with open(profile_path, "w", encoding="utf-8") as handle:
            json.dump(profile, handle)
        print(f"saved identifiability profile to {profile_path}")
    elif args.identifiability_profile:
        with open(args.identifiability_profile, "r", encoding="utf-8") as handle:
            profile = json.load(handle)

    seq_samples: List[Dict[str, Any]] = []
    graph_samples: List[Dict[str, Any]] = []
    for row in rows:
        ss, gs = build_samples_from_flow(
            row,
            args.window_size,
            args.stride,
            profile,
            args.identifiability_min_support,
            args.strict_current_packet_features,
        )
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
