#!/usr/bin/env python3
"""Extract packet embeddings with a Qwen-LoRA Packet Semantic Encoder.

Scheme B: last-token pooling for decoder-only LLM.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, TextIO

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from models.qwen_packet_multitask import MLPProjectionHead
try:
    from peft import PeftModel
except Exception:
    PeftModel = None


CROSS_FLOW_SCHEDULER = "cross_flow_length_bucketed_v1"
LEGACY_FLOW_SCHEDULER = "legacy_per_flow_v1"


def artifact_evidence(path: str | Path) -> dict:
    """Hash a checkpoint file or directory without depending on mtimes."""
    artifact = Path(path).resolve()
    if artifact.is_file():
        files = [artifact]
        root = artifact.parent
        kind = "file"
    elif artifact.is_dir():
        files = sorted(item for item in artifact.rglob("*") if item.is_file())
        root = artifact
        kind = "directory"
    else:
        raise FileNotFoundError(f"Missing model artifact: {artifact}")
    digest = hashlib.sha256()
    for item in files:
        relative = item.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return {
        "path": str(artifact),
        "kind": kind,
        "sha256": digest.hexdigest(),
        "file_count": len(files),
    }


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


def packet_index_policies(path: str | Path) -> tuple[str, str]:
    header_policies: set[str] = set()
    context_policies: set[str] = set()
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            header_policies.add(str(row.get("embedding_header_policy") or "unknown"))
            context_policies.add(str(row.get("packet_context_policy") or "unknown"))
    if not header_policies:
        raise ValueError(f"packet index is empty: {path}")
    if len(header_policies) != 1 or len(context_policies) != 1:
        raise ValueError(
            "packet index mixes embedding policies: "
            f"header={sorted(header_policies)} context={sorted(context_policies)}"
        )
    return next(iter(header_policies)), next(iter(context_policies))


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
    last_idx = last_nonpadding_indices(inputs["attention_mask"]).to(h.device)
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


def last_nonpadding_indices(attention_mask: torch.Tensor) -> torch.Tensor:
    """Locate the final attended token for either left- or right-padded inputs."""
    if attention_mask.ndim != 2 or attention_mask.size(1) == 0:
        raise ValueError("attention_mask must be a non-empty 2D tensor")
    attended = attention_mask.to(dtype=torch.bool)
    if not bool(attended.any(dim=1).all()):
        raise ValueError("every embedding input must contain at least one token")
    reverse_offset = attended.flip(dims=(1,)).to(dtype=torch.int64).argmax(dim=1)
    return attention_mask.size(1) - 1 - reverse_offset


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


def iter_flow_batches(
    flow_items: Iterable[tuple[str, List[dict]]],
    max_packets: int,
) -> Iterator[list[tuple[str, List[dict]]]]:
    """Buffer adjacent flows so small flows share packet-level model batches."""
    if max_packets <= 0:
        for item in flow_items:
            yield [item]
        return
    pending: list[tuple[str, List[dict]]] = []
    pending_packets = 0
    for flow_id, flow_rows in flow_items:
        flow_rows = sorted(flow_rows, key=lambda row: row["packet_id"])
        if pending and pending_packets + len(flow_rows) > max_packets:
            yield pending
            pending = []
            pending_packets = 0
        pending.append((flow_id, flow_rows))
        pending_packets += len(flow_rows)
        if pending_packets >= max_packets:
            yield pending
            pending = []
            pending_packets = 0
    if pending:
        yield pending


def write_flow_batch_embeddings(
    outf: TextIO,
    emb_dir: Path,
    model,
    tokenizer,
    projection_head,
    flow_batch: list[tuple[str, List[dict]]],
    batch_size: int,
    max_length: int,
    embedding_mode: str,
) -> None:
    """Embed packets across flows, then restore the original per-flow artifacts."""
    prompts = [
        row["prompt"]
        for _flow_id, flow_rows in flow_batch
        for row in flow_rows
    ]
    # Length bucketing limits decoder padding without changing flow or packet order
    # in the persisted artifacts.
    prompt_order = sorted(range(len(prompts)), key=lambda index: (len(prompts[index]), index))
    all_embeddings = None
    for start in range(0, len(prompt_order), batch_size):
        indices = prompt_order[start:start + batch_size]
        embedded = embed_batch(
            model,
            tokenizer,
            [prompts[index] for index in indices],
            max_length,
            embedding_mode,
            projection_head=projection_head,
        )
        if all_embeddings is None:
            all_embeddings = np.empty(
                (len(prompts), embedded.shape[1]),
                dtype=embedded.dtype,
            )
        elif embedded.shape[1] != all_embeddings.shape[1]:
            raise RuntimeError("inconsistent embedding dimensions across micro-batches")
        all_embeddings[indices] = embedded
    if all_embeddings is None:
        raise ValueError("flow_batch must contain at least one packet")
    offset = 0
    for flow_id, flow_rows in flow_batch:
        count = len(flow_rows)
        embedding = all_embeddings[offset:offset + count]
        offset += count
        emb_path = emb_dir / f"{flow_id}.npy"
        np.save(emb_path, embedding)
        out_row = {
            "flow_id": flow_id,
            "label": flow_rows[0]["label"],
            "label_id": flow_rows[0]["label_id"],
            "pcap_path": flow_rows[0]["pcap_path"],
            "embedding_path": str(emb_path),
            "embedding_mode": embedding_mode,
            "embedding_dim": int(embedding.shape[1]),
            "packet_metas": [row["meta"] for row in flow_rows],
        }
        outf.write(json.dumps(out_row, ensure_ascii=False) + "\n")
        outf.flush()
    if offset != len(all_embeddings):
        raise RuntimeError(
            "cross-flow embedding split mismatch: "
            f"consumed={offset} produced={len(all_embeddings)}"
        )


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


def completed_flow_ids(index_path: Path) -> set[str]:
    completed: set[str] = set()
    if not index_path.exists():
        return completed
    with open(index_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            flow_id = str(row.get("flow_id", ""))
            embedding_path = row.get("embedding_path")
            if flow_id and embedding_path and Path(embedding_path).exists():
                completed.add(flow_id)
    return completed


def logical_cuda_index(device: str) -> int | None:
    if device == "cuda":
        return 0
    if device.startswith("cuda:"):
        return int(device.split(":", 1)[1])
    if device == "auto" and torch.cuda.is_available():
        return 0
    return None


def physical_cuda_token(device: str, visible_devices: str | None = None) -> str | None:
    logical_index = logical_cuda_index(device)
    if logical_index is None:
        return None
    visible = visible_devices if visible_devices is not None else os.environ.get("CUDA_VISIBLE_DEVICES", "")
    devices = [item.strip() for item in visible.split(",") if item.strip()]
    token = devices[logical_index] if logical_index < len(devices) else str(logical_index)
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in token)


def select_cuda_device(device: str) -> int | None:
    """Make index-less CUDA allocations follow the requested logical device."""
    logical_index = logical_cuda_index(device)
    if logical_index is None:
        return None
    torch.cuda.set_device(logical_index)
    return logical_index


def peft_device_kwargs(device: str) -> dict[str, str]:
    """Bind PEFT's safetensors load when model placement is explicit."""
    if device == "auto":
        return {}
    return {"torch_device": device}


def acquire_cuda_capacity_lock(
    device: str,
    min_free_gb: float,
    poll_seconds: float,
    lock_dir: str | Path,
):
    """Serialize Qwen loads per physical GPU and wait for enough free memory."""
    logical_index = logical_cuda_index(device)
    token = physical_cuda_token(device)
    if logical_index is None or token is None or not torch.cuda.is_available():
        return None
    lock_path = Path(lock_dir) / f"qwen_embedding_gpu_{token}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(lock_path, "a+", encoding="utf-8")
    print(f"waiting for embedding GPU lock: device={device} physical={token}", flush=True)
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    required_bytes = int(max(min_free_gb, 0.0) * (1024 ** 3))
    while required_bytes:
        free_bytes, total_bytes = torch.cuda.mem_get_info(logical_index)
        if free_bytes >= required_bytes:
            break
        print(
            "waiting for embedding GPU capacity: "
            f"device={device} physical={token} "
            f"free_gb={free_bytes / (1024 ** 3):.2f} "
            f"required_gb={min_free_gb:.2f} total_gb={total_bytes / (1024 ** 3):.2f}",
            flush=True,
        )
        time.sleep(max(poll_seconds, 1.0))
    print(f"acquired embedding GPU capacity: device={device} physical={token}", flush=True)
    return handle


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--packet_index", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--base_model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--lora_path", default="")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument(
        "--flow_batch_packets",
        type=int,
        default=0,
        help=(
            "Buffer this many packets across adjacent flows before embedding; "
            "model micro-batches still use --batch_size. The unified runners pass 128; "
            "the bare CLI keeps 0 for legacy per-flow reproducibility."
        ),
    )
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
    ap.add_argument("--resume_existing", action="store_true", help="Append to an existing flow index and skip flows whose embedding .npy already exists.")
    ap.add_argument(
        "--min_cuda_free_gb",
        type=float,
        default=20.0,
        help="Wait for this much free memory before loading Qwen on an explicit CUDA device.",
    )
    ap.add_argument("--cuda_capacity_poll_seconds", type=float, default=30.0)
    ap.add_argument("--cuda_lock_dir", default="/tmp/two_tower_embedding_gpu_locks")
    ap.add_argument("--disable_cuda_capacity_lock", action="store_true")
    args = ap.parse_args()
    if args.num_shards <= 0:
        raise ValueError("--num_shards must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive")
    if args.flow_batch_packets < 0:
        raise ValueError("--flow_batch_packets must be non-negative")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard_index must satisfy 0 <= shard_index < num_shards")
    if args.use_projected_embedding and args.embedding_mode == "raw":
        args.embedding_mode = "projected"
    show_progress = not args.no_progress
    os.makedirs(args.output_dir, exist_ok=True)
    emb_dir = Path(args.output_dir) / "packet_embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)

    gpu_capacity_lock = None
    if not args.disable_cuda_capacity_lock:
        gpu_capacity_lock = acquire_cuda_capacity_lock(
            args.device,
            args.min_cuda_free_gb,
            args.cuda_capacity_poll_seconds,
            args.cuda_lock_dir,
        )

    # PEFT loads adapter tensors through safetensors with an index-less `cuda`
    # device. Select the shard GPU before either checkpoint is materialized.
    select_cuda_device(args.device)

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
        model = PeftModel.from_pretrained(
            model,
            args.lora_path,
            local_files_only=args.local_files_only,
            **peft_device_kwargs(args.device),
        )
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

    packet_index_header_policy, packet_index_context_policy = packet_index_policies(
        args.packet_index
    )
    model_provenance = {
        "lora_adapter": artifact_evidence(args.lora_path) if args.lora_path else None,
        "tower1_heads": (
            artifact_evidence(args.tower1_heads) if args.tower1_heads else None
        ),
    }

    with open(Path(args.output_dir) / "embedding_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "base_model": args.base_model,
                "lora_path": args.lora_path,
                "tower1_heads": args.tower1_heads,
                "embedding_mode": args.embedding_mode,
                "batch_size": args.batch_size,
                "flow_batch_packets": args.flow_batch_packets,
                "scheduler": (
                    CROSS_FLOW_SCHEDULER
                    if args.flow_batch_packets > 0
                    else LEGACY_FLOW_SCHEDULER
                ),
                "max_length": args.max_length,
                "device": args.device,
                "num_shards": args.num_shards,
                "shard_index": args.shard_index,
                "packet_index": args.packet_index,
                "packet_index_header_policy": packet_index_header_policy,
                "packet_index_context_policy": packet_index_context_policy,
                "model_provenance": model_provenance,
            },
            f,
            indent=2,
        )

    meta_out = Path(args.output_dir) / "flow_embedding_index.jsonl"
    completed = completed_flow_ids(meta_out) if args.resume_existing else set()
    if completed:
        print(f"resume_existing: skip completed flows={len(completed)} from {meta_out}", flush=True)
    write_mode = "a" if args.resume_existing and meta_out.exists() else "w"
    with open(meta_out, write_mode, encoding="utf-8") as outf:
        if args.group_in_memory:
            rows = list(load_jsonl(args.packet_index, show_progress=show_progress))
            by_flow: Dict[str, List[dict]] = defaultdict(list)
            for row in tqdm(rows, desc="group packets", disable=not show_progress):
                by_flow[row["flow_id"]].append(row)
            shard_items = (
                (flow_id, flow_rows)
                for flow_id, flow_rows in by_flow.items()
                if flow_in_shard(flow_id, args.shard_index, args.num_shards)
                and flow_id not in completed
            )
            flow_iter = tqdm(shard_items, desc="embed flows", disable=not show_progress)
        else:
            shard_items = (
                (flow_id, flow_rows)
                for flow_id, flow_rows in iter_flow_groups(load_jsonl(args.packet_index, show_progress=show_progress))
                if flow_in_shard(flow_id, args.shard_index, args.num_shards)
                and flow_id not in completed
            )
            flow_iter = tqdm(shard_items, desc="embed flows", disable=not show_progress)
        done = 0
        if args.flow_batch_packets == 0:
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
                    print(
                        f"embedded flows={done}, last_flow={flow_id}, packets={len(flow_rows)}",
                        flush=True,
                    )
        else:
            for flow_batch in iter_flow_batches(flow_iter, args.flow_batch_packets):
                if show_progress:
                    last_flow_id, last_flow_rows = flow_batch[-1]
                    flow_iter.set_postfix(
                        flow_id=last_flow_id,
                        packets=len(last_flow_rows),
                        buffered_flows=len(flow_batch),
                    )
                write_flow_batch_embeddings(
                    outf,
                    emb_dir,
                    model,
                    tokenizer,
                    projection_head,
                    flow_batch,
                    args.batch_size,
                    args.max_length,
                    args.embedding_mode,
                )
                done += len(flow_batch)
                if not show_progress and args.log_every > 0 and done % args.log_every == 0:
                    last_flow_id, last_flow_rows = flow_batch[-1]
                    print(
                        f"embedded flows={done}, last_flow={last_flow_id}, "
                        f"packets={len(last_flow_rows)}",
                        flush=True,
                    )
    print(f"saved {meta_out}")


if __name__ == "__main__":
    main()
