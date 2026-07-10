#!/usr/bin/env python3
"""Extract packet embeddings with a Qwen-LoRA Packet Semantic Encoder.

Scheme B: last-token pooling for decoder-only LLM.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

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
    out = model(**inputs, output_hidden_states=True, return_dict=True)
    h = out.hidden_states[-1]
    last_idx = inputs["attention_mask"].sum(dim=1) - 1
    raw_hidden = h[torch.arange(h.size(0), device=h.device), last_idx].float()
    raw_emb = torch.nn.functional.normalize(raw_hidden, p=2, dim=-1)
    if embedding_mode == "raw":
        emb = raw_emb
    else:
        if projection_head is None:
            raise ValueError(f"--embedding_mode {embedding_mode} requires --tower1_heads")
        projected = projection_head(raw_hidden)
        emb = projected if embedding_mode == "projected" else torch.cat([raw_emb, projected], dim=-1)
    return emb.cpu().numpy().astype("float32")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--packet_index", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--base_model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--lora_path", default="")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--torch_dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--tower1_heads", default="", help="Optional tower1_heads.pt saved by train_tower1_multitask.py")
    ap.add_argument("--embedding_mode", default="raw", choices=["raw", "projected", "concat"], help="raw: last-token hidden state; projected: contrastive projection head; concat: raw + projected.")
    ap.add_argument("--use_projected_embedding", action="store_true", help="Deprecated alias for --embedding_mode projected.")
    ap.add_argument("--no_progress", action="store_true", help="Disable JSONL loading and embedding progress bars.")
    args = ap.parse_args()
    if args.use_projected_embedding and args.embedding_mode == "raw":
        args.embedding_mode = "projected"
    show_progress = not args.no_progress
    os.makedirs(args.output_dir, exist_ok=True)
    emb_dir = Path(args.output_dir) / "packet_embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.torch_dtype]
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=dtype, device_map="auto", trust_remote_code=True)
    if args.lora_path:
        if PeftModel is None:
            raise RuntimeError("peft is not installed; pip install peft")
        model = PeftModel.from_pretrained(model, args.lora_path)
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
            },
            f,
            indent=2,
        )

    rows = list(load_jsonl(args.packet_index, show_progress=show_progress))
    by_flow: Dict[str, List[dict]] = defaultdict(list)
    for row in tqdm(rows, desc="group packets", disable=not show_progress):
        by_flow[row["flow_id"]].append(row)

    meta_out = Path(args.output_dir) / "flow_embedding_index.jsonl"
    with open(meta_out, "w", encoding="utf-8") as outf:
        flow_iter = tqdm(by_flow.items(), desc="embed flows", disable=not show_progress)
        for flow_id, flow_rows in flow_iter:
            flow_rows = sorted(flow_rows, key=lambda r: r["packet_id"])
            prompts = [r["prompt"] for r in flow_rows]
            flow_iter.set_postfix(flow_id=flow_id, packets=len(prompts))
            all_emb = []
            for i in range(0, len(prompts), args.batch_size):
                all_emb.append(embed_batch(model, tokenizer, prompts[i:i + args.batch_size], args.max_length, args.embedding_mode, projection_head=projection_head))
            emb = np.concatenate(all_emb, axis=0)
            emb_path = emb_dir / f"{flow_id}.npy"
            np.save(emb_path, emb)
            out_row = {
                "flow_id": flow_id,
                "label": flow_rows[0]["label"],
                "label_id": flow_rows[0]["label_id"],
                "pcap_path": flow_rows[0]["pcap_path"],
                "embedding_path": str(emb_path),
                "embedding_mode": args.embedding_mode,
                "embedding_dim": int(emb.shape[1]),
                "packet_metas": [r["meta"] for r in flow_rows],
            }
            outf.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            if not show_progress:
                print(f"embedded flow={flow_id}, packets={len(flow_rows)}")
    print(f"saved {meta_out}")


if __name__ == "__main__":
    main()
