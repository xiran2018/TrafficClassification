#!/usr/bin/env python3
"""Train Tower-1 Qwen-LoRA with protocol QA + weak packet classification + SupCon.

This script is intentionally separate from LLaMA-Factory because LLaMA-Factory SFT
cannot directly optimize packet embedding classification and contrastive losses.

Training objective:
    L = L_QA + alpha * L_packet_cls + beta * L_supcon

The packet embedding follows scheme B for decoder-only LLMs: last-token hidden state.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from itertools import cycle
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch
from torch.utils.data import BatchSampler, DataLoader, Dataset
from tqdm import tqdm

def load_jsonl(path: str | Path, show_progress: bool = True) -> List[dict]:
    path = Path(path)
    rows: List[dict] = []
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
                rows.append(json.loads(line))
                if show_progress and len(rows) % 10000 == 0:
                    pbar.set_postfix(rows=len(rows))
        pbar.close()
    print(f"loaded {len(rows)} rows from {path}", flush=True)
    return rows


class PacketSFTDataset(Dataset):
    def __init__(self, paths: List[str], show_progress: bool = True):
        self.rows: List[dict] = []
        for p in paths:
            if p:
                self.rows.extend(load_jsonl(p, show_progress=show_progress))
        if not self.rows:
            raise ValueError("No SFT samples loaded. Provide --sft_jsonl or use --no_sft.")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        return self.rows[idx]


class PacketAuxDataset(Dataset):
    def __init__(self, path: str, show_progress: bool = True):
        self.rows = load_jsonl(path, show_progress=show_progress)
        if not self.rows:
            raise ValueError(f"No packet auxiliary samples loaded from {path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        return self.rows[idx]


class FlowBalancedPacketBatchSampler(BatchSampler):
    """Sample several packets from each selected flow so flow-aware SupCon has positives."""

    def __init__(self, rows: List[dict], batch_size: int, packets_per_flow: int, seed: int = 42):
        self.flow_to_indices: Dict[str, List[int]] = {}
        for idx, row in enumerate(rows):
            self.flow_to_indices.setdefault(str(row.get("flow_id", idx)), []).append(idx)
        self.flows = list(self.flow_to_indices.keys())
        self.batch_size = max(1, int(batch_size))
        self.packets_per_flow = max(1, int(packets_per_flow))
        self.flows_per_batch = max(1, self.batch_size // self.packets_per_flow)
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        flows = list(self.flows)
        rng.shuffle(flows)
        for start in range(0, len(flows), self.flows_per_batch):
            batch = []
            for flow_id in flows[start:start + self.flows_per_batch]:
                indices = self.flow_to_indices[flow_id]
                if len(indices) >= self.packets_per_flow:
                    batch.extend(rng.sample(indices, self.packets_per_flow))
                else:
                    batch.extend(rng.choice(indices) for _ in range(self.packets_per_flow))
            if batch:
                yield batch[: self.batch_size]

    def __len__(self) -> int:
        return max(1, math.ceil(len(self.flows) / self.flows_per_batch))


def stable_flow_id(value: str) -> int:
    digest = hashlib.blake2b(str(value).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False) & ((1 << 63) - 1)


def build_sft_text(row: dict) -> tuple[str, str]:
    instruction = row.get("instruction", "Answer the packet protocol question.")
    inp = row.get("input", "")
    out = str(row.get("output", ""))
    prompt = f"{instruction}\n\n{inp}\n\nAnswer:"
    answer = " " + out
    return prompt, answer


class SFTCollator:
    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, rows: List[dict]) -> Dict[str, torch.Tensor]:
        input_ids: List[torch.Tensor] = []
        labels: List[torch.Tensor] = []
        eos = self.tokenizer.eos_token or ""
        for row in rows:
            prompt, answer = build_sft_text(row)
            prompt_ids = self.tokenizer(prompt, add_special_tokens=False).input_ids
            answer_ids = self.tokenizer(answer + eos, add_special_tokens=False).input_ids
            if len(answer_ids) >= self.max_length:
                prompt_ids = []
                answer_ids = answer_ids[: self.max_length]
            else:
                prompt_ids = prompt_ids[-(self.max_length - len(answer_ids)) :]
            full_ids = prompt_ids + answer_ids
            lab = [-100] * len(prompt_ids) + answer_ids.copy()
            input_ids.append(torch.tensor(full_ids, dtype=torch.long))
            labels.append(torch.tensor(lab, dtype=torch.long))
        batch = pad_lm_batch(input_ids, labels, self.tokenizer.pad_token_id)
        batch["valid_label_tokens"] = (batch["labels"] != -100).sum()
        return batch


class PacketAuxCollator:
    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, rows: List[dict]) -> Dict[str, torch.Tensor]:
        texts = [r["prompt"] for r in rows]
        toks = self.tokenizer(texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
        labels = torch.tensor([int(r["label_id"]) for r in rows], dtype=torch.long)
        weights = torch.tensor([float(r.get("packet_weight", 1.0)) for r in rows], dtype=torch.float32)
        flow_ids = torch.tensor([stable_flow_id(str(r.get("flow_id", ""))) for r in rows], dtype=torch.long)
        return {
            "input_ids": toks["input_ids"],
            "attention_mask": toks["attention_mask"],
            "labels": labels,
            "weights": weights,
            "flow_ids": flow_ids,
        }


def pad_lm_batch(input_ids: List[torch.Tensor], labels: List[torch.Tensor], pad_id: int) -> Dict[str, torch.Tensor]:
    max_len = max(x.numel() for x in input_ids)
    ids = torch.full((len(input_ids), max_len), pad_id, dtype=torch.long)
    labs = torch.full((len(input_ids), max_len), -100, dtype=torch.long)
    mask = torch.zeros((len(input_ids), max_len), dtype=torch.long)
    for i, (x, y) in enumerate(zip(input_ids, labels)):
        ids[i, : x.numel()] = x
        labs[i, : y.numel()] = y
        mask[i, : x.numel()] = 1
    return {"input_ids": ids, "attention_mask": mask, "labels": labs}


def move_to_device(batch: Optional[Dict[str, torch.Tensor]], device: torch.device) -> Optional[Dict[str, torch.Tensor]]:
    if batch is None:
        return None
    return {k: v.to(device) for k, v in batch.items()}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def infer_num_classes(label_map_path: str) -> int:
    with open(label_map_path, "r", encoding="utf-8") as f:
        label_map = json.load(f)
    return len(label_map)


def percentile_value(values: List[int], percentile: float) -> int:
    if not values:
        return 0
    values = sorted(values)
    percentile = min(100.0, max(0.0, percentile))
    idx = round((len(values) - 1) * percentile / 100.0)
    return values[idx]


def round_up(value: int, multiple: int) -> int:
    if multiple <= 1:
        return value
    return int(math.ceil(value / multiple) * multiple)


def estimate_sft_max_length(
    paths: List[str],
    tokenizer,
    max_length: int,
    percentile: float,
    multiple: int,
    show_progress: bool = True,
) -> int:
    eos = tokenizer.eos_token or ""
    lengths: List[int] = []
    for path_str in paths:
        path = Path(path_str)
        total_bytes = path.stat().st_size if show_progress else None
        with open(path, "rb") as f:
            pbar = tqdm(
                total=total_bytes,
                desc=f"scan sft length {path.name}",
                unit="B",
                unit_scale=True,
                disable=not show_progress,
            )
            for line in f:
                if show_progress:
                    pbar.update(len(line))
                if not line.strip():
                    continue
                row = json.loads(line)
                prompt, answer = build_sft_text(row)
                prompt_len = len(tokenizer(prompt, add_special_tokens=False).input_ids)
                answer_len = len(tokenizer(answer + eos, add_special_tokens=False).input_ids)
                lengths.append(prompt_len + answer_len)
            pbar.close()

    if not lengths:
        return max_length

    target_len = percentile_value(lengths, percentile)
    recommended = max(max_length, round_up(target_len, multiple))
    over_current = sum(1 for x in lengths if x > max_length)
    over_recommended = sum(1 for x in lengths if x > recommended)
    print(
        "SFT length stats: "
        f"n={len(lengths)}, max={max(lengths)}, p{percentile:g}={target_len}, "
        f"current_max_sft_length={max_length}, current_truncated={over_current} ({over_current / len(lengths) * 100:.4f}%), "
        f"recommended={recommended}, recommended_truncated={over_recommended} ({over_recommended / len(lengths) * 100:.4f}%)",
        flush=True,
    )
    return recommended


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--label_map", required=True)
    ap.add_argument("--packet_aux_jsonl", required=True, help="packet_auxiliary.jsonl from preprocess_tower1.py")
    ap.add_argument("--sft_jsonl", nargs="*", default=[], help="packet_instruction.jsonl and packet_validity.jsonl")
    ap.add_argument("--no_sft", action="store_true", help="Train only packet cls + SupCon without generative QA loss.")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--init_checkpoint_dir", default="", help="Optional Tower-1 checkpoint dir containing adapter/ and tower1_heads.pt for continued training.")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--max_steps", type=int, default=0, help="Override epochs if >0.")
    ap.add_argument("--sft_batch_size", type=int, default=2)
    ap.add_argument("--packet_batch_size", type=int, default=16)
    ap.add_argument("--max_sft_length", type=int, default=1792)
    ap.add_argument("--max_packet_length", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--head_lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--cls_weight", type=float, default=0.1)
    ap.add_argument("--contrastive_weight", type=float, default=0.3)
    ap.add_argument("--temperature", type=float, default=0.07)
    ap.add_argument("--same_flow_positive_weight", type=float, default=0.0, help="Extra positive weight for packets from the same flow in Tower-1 SupCon. 0 keeps label-only SupCon.")
    ap.add_argument("--same_label_positive_weight", type=float, default=1.0, help="Positive weight for same-label packets in flow-aware SupCon.")
    ap.add_argument("--flow_proto_weight", type=float, default=0.0, help="Weight for packet-to-flow prototype contrastive loss in Tower-1.")
    ap.add_argument("--flow_proto_positive", choices=["own_flow", "same_class"], default="same_class", help="Positive flow prototypes for --flow_proto_weight.")
    ap.add_argument("--flow_balanced_packet_batches", action="store_true", help="Sample packet batches as multiple packets per flow for flow-aware SupCon.")
    ap.add_argument("--packets_per_flow", type=int, default=2, help="Packets sampled per flow when --flow_balanced_packet_batches is set.")
    ap.add_argument("--projection_dim", type=int, default=256)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--local_files_only", action="store_true", help="Load the base model/tokenizer only from the local Hugging Face cache.")
    ap.add_argument("--gradient_accumulation_steps", type=int, default=1)
    ap.add_argument("--gradient_checkpointing", action="store_true")
    ap.add_argument("--log_steps", type=int, default=20)
    ap.add_argument("--save_steps", type=int, default=0)
    ap.add_argument("--no_load_progress", action="store_true", help="Disable JSONL loading progress bars.")
    ap.add_argument("--stop_on_nonfinite_loss", action="store_true", help="Raise an error instead of skipping a NaN/Inf loss step.")
    ap.add_argument("--auto_max_sft_length", action="store_true", help="Scan SFT token lengths and raise --max_sft_length to the requested percentile.")
    ap.add_argument("--sft_length_percentile", type=float, default=100.0, help="Percentile used by --auto_max_sft_length.")
    ap.add_argument("--sft_length_multiple", type=int, default=256, help="Round auto max SFT length up to this multiple.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    print(f"device={device}, dtype={args.dtype}", flush=True)
    print(f"loading label map: {args.label_map}", flush=True)
    num_classes = infer_num_classes(args.label_map)
    print(f"num_classes={num_classes}", flush=True)

    print("importing model code", flush=True)
    from models.qwen_packet_multitask import QwenPacketMultiTaskModel

    print(f"loading base model: {args.base_model}", flush=True)
    init_lora_path = str(Path(args.init_checkpoint_dir) / "adapter") if args.init_checkpoint_dir else ""
    model = QwenPacketMultiTaskModel(
        base_model_name_or_path=args.base_model,
        num_classes=num_classes,
        torch_dtype=dtype,
        lora_path=init_lora_path,
        create_lora=not bool(init_lora_path),
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        projection_dim=args.projection_dim,
        local_files_only=args.local_files_only,
    )
    if args.init_checkpoint_dir:
        load_packet_heads(model, Path(args.init_checkpoint_dir) / "tower1_heads.pt")
    print("base model loaded", flush=True)
    if args.gradient_checkpointing:
        print("enabling gradient checkpointing", flush=True)
        model.backbone.gradient_checkpointing_enable()
    print(f"moving model to {device}", flush=True)
    model.to(device)
    model.train()

    tokenizer = model.tokenizer
    show_load_progress = not args.no_load_progress

    if args.auto_max_sft_length and not args.no_sft:
        if not args.sft_jsonl:
            raise ValueError("Provide --sft_jsonl paths or set --no_sft")
        args.max_sft_length = estimate_sft_max_length(
            args.sft_jsonl,
            tokenizer,
            max_length=args.max_sft_length,
            percentile=args.sft_length_percentile,
            multiple=args.sft_length_multiple,
            show_progress=show_load_progress,
        )
        print(f"using max_sft_length={args.max_sft_length}", flush=True)

    print(f"loading packet auxiliary dataset: {args.packet_aux_jsonl}", flush=True)
    packet_ds = PacketAuxDataset(args.packet_aux_jsonl, show_progress=show_load_progress)
    if args.flow_balanced_packet_batches:
        packet_sampler = FlowBalancedPacketBatchSampler(
            packet_ds.rows,
            batch_size=args.packet_batch_size,
            packets_per_flow=args.packets_per_flow,
            seed=args.seed,
        )
        packet_loader = DataLoader(
            packet_ds,
            batch_sampler=packet_sampler,
            collate_fn=PacketAuxCollator(tokenizer, args.max_packet_length),
        )
        print(
            f"flow-balanced packet sampler: flows={len(packet_sampler.flows)}, "
            f"flows_per_batch={packet_sampler.flows_per_batch}, packets_per_flow={packet_sampler.packets_per_flow}",
            flush=True,
        )
    else:
        packet_loader = DataLoader(
            packet_ds,
            batch_size=args.packet_batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=PacketAuxCollator(tokenizer, args.max_packet_length),
        )
    print(f"packet samples={len(packet_ds)}, packet batches/epoch={len(packet_loader)}", flush=True)
    packet_iter = cycle(packet_loader)

    sft_loader = None
    if not args.no_sft:
        if not args.sft_jsonl:
            raise ValueError("Provide --sft_jsonl paths or set --no_sft")
        print(f"loading SFT datasets: {', '.join(args.sft_jsonl)}", flush=True)
        sft_ds = PacketSFTDataset(args.sft_jsonl, show_progress=show_load_progress)
        sft_loader = DataLoader(
            sft_ds,
            batch_size=args.sft_batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=SFTCollator(tokenizer, args.max_sft_length),
        )
        print(f"SFT samples={len(sft_ds)}, SFT batches/epoch={len(sft_loader)}", flush=True)
        sft_iter = cycle(sft_loader)
    else:
        print("SFT disabled; training packet cls + SupCon only", flush=True)
        sft_iter = None

    lora_params = []
    head_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "packet_classifier" in name or "projection_head" in name:
            head_params.append(p)
        else:
            lora_params.append(p)
    opt = torch.optim.AdamW(
        [
            {"params": lora_params, "lr": args.lr},
            {"params": head_params, "lr": args.head_lr},
        ],
        weight_decay=args.weight_decay,
    )

    steps_per_epoch = len(packet_loader)
    total_steps = args.max_steps if args.max_steps > 0 else steps_per_epoch * args.epochs
    print(
        f"starting training: epochs={args.epochs}, steps_per_epoch={steps_per_epoch}, total_steps={total_steps}",
        flush=True,
    )
    pbar = tqdm(range(total_steps), desc="train tower1")
    opt.zero_grad(set_to_none=True)

    running = {"loss": 0.0, "lm": 0.0, "cls": 0.0, "con": 0.0, "proto": 0.0, "acc": 0.0, "lm_tokens": 0.0, "n": 0}
    skipped_nonfinite = 0
    for step in pbar:
        sft_batch = next(sft_iter) if sft_iter is not None else None
        packet_batch = next(packet_iter)
        sft_batch = move_to_device(sft_batch, device)
        packet_batch = move_to_device(packet_batch, device)

        out = model.forward_multitask(
            sft_batch=sft_batch,
            packet_batch=packet_batch,
            cls_weight=args.cls_weight,
            contrastive_weight=args.contrastive_weight,
            temperature=args.temperature,
            same_flow_positive_weight=args.same_flow_positive_weight,
            same_label_positive_weight=args.same_label_positive_weight,
            flow_proto_weight=args.flow_proto_weight,
            flow_proto_positive=args.flow_proto_positive,
        )
        if not torch.isfinite(out.loss):
            skipped_nonfinite += 1
            opt.zero_grad(set_to_none=True)
            msg = (
                f"non-finite loss at step={step + 1}: "
                f"loss={float(out.loss.detach().cpu())} "
                f"lm={float(out.lm_loss.detach().cpu())} "
                f"pkt_cls={float(out.pkt_cls_loss.detach().cpu())} "
                f"supcon={float(out.supcon_loss.detach().cpu())} "
                f"proto={float(out.flow_proto_loss.detach().cpu())}"
            )
            if args.stop_on_nonfinite_loss:
                raise FloatingPointError(msg)
            tqdm.write(f"WARNING: {msg}; skipped optimizer update")
            pbar.set_postfix(skipped_nonfinite=skipped_nonfinite)
            continue

        loss = out.loss / args.gradient_accumulation_steps
        loss.backward()

        if (step + 1) % args.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)

        with torch.no_grad():
            running["loss"] += float(out.loss.detach().cpu())
            running["lm"] += float(out.lm_loss.detach().cpu())
            running["cls"] += float(out.pkt_cls_loss.detach().cpu())
            running["con"] += float(out.supcon_loss.detach().cpu())
            running["proto"] += float(out.flow_proto_loss.detach().cpu())
            if sft_batch is not None:
                running["lm_tokens"] += float(sft_batch.get("valid_label_tokens", torch.zeros(())).detach().cpu())
            if out.packet_logits is not None:
                pred = out.packet_logits.argmax(dim=-1)
                acc = (pred == packet_batch["labels"]).float().mean().item()
                running["acc"] += acc
            running["n"] += 1

        if (step + 1) % args.log_steps == 0:
            n = max(1, running["n"])
            msg = {
                "loss": running["loss"] / n,
                "lm": running["lm"] / n,
                "pkt_cls": running["cls"] / n,
                "supcon": running["con"] / n,
                "proto": running["proto"] / n,
                "pkt_acc": running["acc"] / n,
                "lm_tokens": running["lm_tokens"] / n,
            }
            pbar.set_postfix({k: f"{v:.4f}" for k, v in msg.items()})
            tqdm.write(
                "step={step}/{total} loss={loss:.4f} lm={lm:.4f} pkt_cls={pkt_cls:.4f} "
                "supcon={supcon:.4f} proto={proto:.4f} pkt_acc={pkt_acc:.4f} lm_tokens/batch={lm_tokens:.1f} "
                "skipped_nonfinite={skipped}".format(
                    step=step + 1,
                    total=total_steps,
                    skipped=skipped_nonfinite,
                    **msg,
                )
            )
            running = {"loss": 0.0, "lm": 0.0, "cls": 0.0, "con": 0.0, "proto": 0.0, "acc": 0.0, "lm_tokens": 0.0, "n": 0}

        if args.save_steps and (step + 1) % args.save_steps == 0:
            save_model(model, args.output_dir, suffix=f"step_{step+1}")

    print(f"training finished; skipped_nonfinite={skipped_nonfinite}", flush=True)
    save_model(model, args.output_dir)


def save_model(model: QwenPacketMultiTaskModel, output_dir: str, suffix: str = "") -> None:
    out = Path(output_dir) if not suffix else Path(output_dir) / suffix
    out.mkdir(parents=True, exist_ok=True)
    adapter_dir = out / "adapter"
    model.backbone.save_pretrained(adapter_dir)
    model.tokenizer.save_pretrained(out)
    model.save_packet_heads(str(out))
    with open(out / "tower1_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "base_model": model.base_model_name_or_path,
                "num_classes": model.num_classes,
                "hidden_size": model.hidden_size,
                "embedding_pooling": "last_token",
                "loss": "L_QA + alpha*L_packet_cls + beta*L_supcon + gamma*L_flow_proto",
                "supports_flow_aware_supcon": True,
                "supports_flow_prototype_loss": True,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"saved Tower-1 adapter and heads to {out}")


def load_packet_heads(model: QwenPacketMultiTaskModel, heads_path: Path) -> None:
    if not heads_path.exists():
        raise FileNotFoundError(f"Missing Tower-1 heads file: {heads_path}")
    state = torch.load(heads_path, map_location="cpu")
    if int(state.get("num_classes", model.num_classes)) != model.num_classes:
        raise ValueError(f"Head num_classes mismatch: checkpoint={state.get('num_classes')} current={model.num_classes}")
    model.packet_classifier.load_state_dict(state["packet_classifier"])
    model.projection_head.load_state_dict(state["projection_head"])
    print(f"loaded Tower-1 packet heads from {heads_path}", flush=True)


if __name__ == "__main__":
    main()
