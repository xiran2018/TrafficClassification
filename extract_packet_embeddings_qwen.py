#!/usr/bin/env python3
"""Extract packet embeddings with a Qwen-LoRA Packet Semantic Encoder.

Scheme B: last-token pooling for decoder-only LLM.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, TextIO

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from models.qwen_packet_multitask import MLPProjectionHead
try:
    from peft import PeftModel
except Exception:
    PeftModel = None


def load_jsonl(path: str | Path, show_progress: bool = True):
    path = Path(path)
    total_bytes = path.stat().st_size if show_progress else None
    with open(path, "rb") as f:
        pbar = tqdm(
            total=total_bytes,
            desc=f"load {path.name}",
            unit="B",
            unit_scale=True,
            disable=not show_progress,
        )
        for line in f:
            if show_progress:
                pbar.update(len(line))
            if line.strip():
                yield json.loads(line.decode("utf-8"))
        pbar.close()


@torch.no_grad()
def embed_batch(model, tokenizer, texts: List[str], max_length: int, embedding_mode: str, projection_head=None) -> np.ndarray:
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    backbone = getattr(model, "model", None)
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        backbone = getattr(model.base_model.model, "model", model.base_model.model)
    elif hasattr(model, "get_base_model"):
        base = model.get_base_model()
        backbone = getattr(base, "model", base)
    if backbone is not None and backbone is not model:
        out = backbone(**inputs, output_hidden_states=False, return_dict=True)
        h = out.last_hidden_state
    else:
        out = model(**inputs, output_hidden_states=True, return_dict=True)
        h = out.hidden_states[-1]
    last_idx = (inputs["attention_mask"].sum(dim=1) - 1).to(h.device)
    raw_hidden = h[torch.arange(h.size(0), device=h.device), last_idx].float()
    raw_emb = torch.nn.functional.normalize(raw_hidden, p=2, dim=-1)
    if embedding_mode == "raw":
        emb = raw_emb
    else:
        if projection_head is None:
            raise ValueError(f"--embedding_mode {embedding_mode} requires --tower1_heads")
        projection_head = projection_head.to(raw_hidden.device)
        projected = projection_head(raw_hidden)
        emb = projected if embedding_mode == "projected" else torch.cat([raw_emb, projected], dim=-1)
    return emb.cpu().numpy().astype("float32")


def write_flow_embeddings(
    outf: TextIO,
    emb_dir: Path,
    model,
    tokenizer,
    projection_head,
    flow_id: str,
    flow_rows: List[dict],
    batch_size: int,
    max_length: int,
    embedding_mode: str,
) -> None:
    flow_rows = sorted(flow_rows, key=lambda r: r["packet_id"])
    prompts = [r["prompt"] for r in flow_rows]
    all_emb = []
    for i in range(0, len(prompts), batch_size):
        all_emb.append(embed_batch(model, tokenizer, prompts[i:i + batch_size], max_length, embedding_mode, projection_head=projection_head))
    emb = np.concatenate(all_emb, axis=0)
    emb_path = emb_dir / f"{flow_id}.npy"
    np.save(emb_path, emb)
    out_row = {
        "flow_id": flow_id,
        "label": flow_rows[0]["label"],
        "label_id": flow_rows[0]["label_id"],
        "pcap_path": flow_rows[0]["pcap_path"],
        "embedding_path": str(emb_path),
        "embedding_mode": embedding_mode,
        "embedding_dim": int(emb.shape[1]),
        "packet_metas": [r["meta"] for r in flow_rows],
    }
    outf.write(json.dumps(out_row, ensure_ascii=False) + "\n")
    outf.flush()


def iter_flow_groups(rows: Iterable[dict]):
    current_flow_id = None
    current_rows: List[dict] = []
    for row in rows:
        flow_id = row["flow_id"]
        if current_flow_id is None:
            current_flow_id = flow_id
        if flow_id != current_flow_id:
            yield current_flow_id, current_rows
            current_flow_id = flow_id
            current_rows = []
        current_rows.append(row)
    if current_flow_id is not None:
        yield current_flow_id, current_rows


def flow_in_shard(flow_id: str, shard_index: int, num_shards: int) -> bool:
    if num_shards <= 1:
        return True
    digest = hashlib.sha1(flow_id.encode("utf-8", errors="ignore")).digest()
    value = int.from_bytes(digest[:8], "big")
    return value % num_shards == shard_index


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--packet_index", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--base_model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--lora_path", default="")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--torch_dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument(
        "--device",
        default="auto",
        help="Model placement: auto or an explicit device such as cuda:0/cuda:1/cpu.",
    )
    ap.add_argument("--tower1_heads", default="", help="Optional tower1_heads.pt saved by train_tower1_multitask.py")
    ap.add_argument("--embedding_mode", default="raw", choices=["raw", "projected", "concat"], help="raw: last-token hidden state; projected: contrastive projection head; concat: raw + projected.")
    ap.add_argument("--use_projected_embedding", action="store_true", help="Deprecated alias for --embedding_mode projected.")
    ap.add_argument("--no_progress", action="store_true", help="Disable JSONL loading and embedding progress bars.")
    ap.add_argument("--local_files_only", action="store_true", help="Load base model and LoRA only from local cache/paths.")
    ap.add_argument("--group_in_memory", action="store_true", help="Group all packets in memory; use only if packet_index is not sorted/grouped by flow_id.")
    ap.add_argument("--log_every", type=int, default=100, help="When --no_progress is set, print one embedding progress line every N flows.")
    ap.add_argument("--num_shards", type=int, default=1, help="Split flows into N deterministic shards by flow_id hash.")
    ap.add_argument("--shard_index", type=int, default=0, help="Shard id in [0, num_shards).")
    args = ap.parse_args()
    if args.num_shards <= 0:
        raise ValueError("--num_shards must be positive")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard_index must satisfy 0 <= shard_index < num_shards")
    if args.use_projected_embedding and args.embedding_mode == "raw":
        args.embedding_mode = "projected"
    show_progress = not args.no_progress
    os.makedirs(args.output_dir, exist_ok=True)
    emb_dir = Path(args.output_dir) / "packet_embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.torch_dtype]
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True, local_files_only=args.local_files_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    device_map = "auto" if args.device == "auto" else {"": args.device}
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    if args.lora_path:
        if PeftModel is None:
            raise RuntimeError("peft is not installed; pip install peft")
        model = PeftModel.from_pretrained(model, args.lora_path, local_files_only=args.local_files_only)
    projection_head = None
    if args.embedding_mode in {"projected", "concat"}:
        if not args.tower1_heads:
            raise ValueError(f"--embedding_mode {args.embedding_mode} requires --tower1_heads")
        ckpt = torch.load(args.tower1_heads, map_location="cpu")
        projection_dim = ckpt["projection_head"]["net.3.weight"].shape[0]
        cfg = getattr(model, "config", None)
        if cfg is None and hasattr(model, "get_base_model"):
            cfg = model.get_base_model().config
        hidden_size = int(cfg.hidden_size)
        device = next(model.parameters()).device
        projection_head = MLPProjectionHead(hidden_size, projection_dim=projection_dim).to(device)
        projection_head.load_state_dict(ckpt["projection_head"], strict=True)
        projection_head.eval()
    model.eval()

    with open(Path(args.output_dir) / "embedding_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "base_model": args.base_model,
                "lora_path": args.lora_path,
                "tower1_heads": args.tower1_heads,
                "embedding_mode": args.embedding_mode,
                "max_length": args.max_length,
                "device": args.device,
                "num_shards": args.num_shards,
                "shard_index": args.shard_index,
            },
            f,
            indent=2,
        )

    meta_out = Path(args.output_dir) / "flow_embedding_index.jsonl"
    with open(meta_out, "w", encoding="utf-8") as outf:
        if args.group_in_memory:
            rows = list(load_jsonl(args.packet_index, show_progress=show_progress))
            by_flow: Dict[str, List[dict]] = defaultdict(list)
            for row in tqdm(rows, desc="group packets", disable=not show_progress):
                by_flow[row["flow_id"]].append(row)
            shard_items = ((flow_id, flow_rows) for flow_id, flow_rows in by_flow.items() if flow_in_shard(flow_id, args.shard_index, args.num_shards))
            flow_iter = tqdm(shard_items, desc="embed flows", disable=not show_progress)
        else:
            shard_items = (
                (flow_id, flow_rows)
                for flow_id, flow_rows in iter_flow_groups(load_jsonl(args.packet_index, show_progress=show_progress))
                if flow_in_shard(flow_id, args.shard_index, args.num_shards)
            )
            flow_iter = tqdm(shard_items, desc="embed flows", disable=not show_progress)
        done = 0
        for flow_id, flow_rows in flow_iter:
            if show_progress:
                flow_iter.set_postfix(flow_id=flow_id, packets=len(flow_rows))
            write_flow_embeddings(
                outf,
                emb_dir,
                model,
                tokenizer,
                projection_head,
                flow_id,
                flow_rows,
                args.batch_size,
                args.max_length,
                args.embedding_mode,
            )
            done += 1
            if not show_progress and args.log_every > 0 and done % args.log_every == 0:
                print(f"embedded flows={done}, last_flow={flow_id}, packets={len(flow_rows)}", flush=True)
    print(f"saved {meta_out}")


if __name__ == "__main__":
    main()
