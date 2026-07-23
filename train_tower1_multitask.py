#!/usr/bin/env python3
"""Train Tower-1 Qwen-LoRA with protocol QA + weak packet classification + SupCon.

This script is intentionally separate from LLaMA-Factory because LLaMA-Factory SFT
cannot directly optimize packet embedding classification and contrastive losses.

Training objective:
    L = L_QA + alpha * L_packet_cls + beta * L_supcon
        + gamma * L_flow_proto + delta * L_paired_consistency

The packet embedding follows scheme B for decoder-only LLMs: last-token hidden state.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
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
    def __init__(self, path: str, show_progress: bool = True, paired_path: str = ""):
        self.rows = load_jsonl(path, show_progress=show_progress)
        if not self.rows:
            raise ValueError(f"No packet auxiliary samples loaded from {path}")
        self.paired_rows = 0
        if paired_path:
            paired = load_jsonl(paired_path, show_progress=show_progress)
            paired_by_uid: Dict[str, dict] = {}
            for paired_row in paired:
                packet_uid = str(paired_row.get("packet_uid", ""))
                if not packet_uid:
                    raise ValueError(f"paired packet auxiliary rows require packet_uid: {paired_path}")
                if packet_uid in paired_by_uid:
                    raise ValueError(f"duplicate paired packet_uid {packet_uid!r}: {paired_path}")
                paired_by_uid[packet_uid] = paired_row

            factual_uids: set[str] = set()
            missing: List[str] = []
            label_mismatches: List[str] = []
            empty_prompts: List[str] = []
            for row in self.rows:
                packet_uid = str(row.get("packet_uid", ""))
                if not packet_uid:
                    raise ValueError(f"factual packet auxiliary rows require packet_uid: {path}")
                if packet_uid in factual_uids:
                    raise ValueError(f"duplicate factual packet_uid {packet_uid!r}: {path}")
                factual_uids.add(packet_uid)
                paired_row = paired_by_uid.get(packet_uid)
                if paired_row is None:
                    missing.append(packet_uid)
                    continue
                if int(paired_row.get("label_id", -1)) != int(row.get("label_id", -2)):
                    label_mismatches.append(packet_uid)
                    continue
                paired_prompt = str(paired_row.get("prompt", ""))
                if not paired_prompt.strip():
                    empty_prompts.append(packet_uid)
                    continue
                row["paired_prompt"] = paired_prompt
                row["paired_embedding_header_policy"] = paired_row.get("embedding_header_policy", "")
                self.paired_rows += 1
            extra = sorted(set(paired_by_uid) - factual_uids)
            if missing or extra or label_mismatches or empty_prompts:
                raise ValueError(
                    "paired packet auxiliary views are not exactly aligned: "
                    f"factual={len(self.rows)} paired={len(paired)} matched={self.paired_rows} "
                    f"missing={missing[:5]} extra={extra[:5]} "
                    f"label_mismatches={label_mismatches[:5]} empty_prompts={empty_prompts[:5]}"
                )
            print(
                f"paired packet auxiliary: matched={self.paired_rows}/{len(self.rows)} "
                f"from {paired_path}",
                flush=True,
            )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        return self.rows[idx]


def flow_balanced_validation_rows(rows: List[dict], packets_per_flow: int, seed: int) -> List[dict]:
    """Select a deterministic, non-repeated packet subset with equal flow exposure."""
    if packets_per_flow <= 0:
        return list(rows)
    grouped: Dict[str, List[dict]] = {}
    for index, row in enumerate(rows):
        grouped.setdefault(str(row.get("flow_id", index)), []).append(row)

    selected: List[dict] = []
    for flow_id in sorted(grouped):
        candidates = grouped[flow_id]

        def stable_packet_key(row: dict) -> bytes:
            packet_id = row.get("packet_uid", row.get("packet_id", ""))
            value = f"{seed}:{flow_id}:{packet_id}"
            return hashlib.blake2b(value.encode("utf-8"), digest_size=16).digest()

        selected.extend(sorted(candidates, key=stable_packet_key)[:packets_per_flow])
    return selected


def validation_patience_exhausted(non_improving_evals: int, patience: int) -> bool:
    return int(patience) > 0 and int(non_improving_evals) >= int(patience)


def validate_aligned_validation_views(
    factual_rows: List[dict], intervened_rows: List[dict]
) -> None:
    """Require paired validation views to describe the same labeled packets."""
    def identities(rows: List[dict]) -> dict[str, int]:
        result: dict[str, int] = {}
        for row in rows:
            packet_uid = str(row.get("packet_uid", ""))
            if not packet_uid:
                raise ValueError("paired validation rows require packet_uid")
            if packet_uid in result:
                raise ValueError(f"duplicate paired validation packet_uid: {packet_uid}")
            result[packet_uid] = int(row.get("label_id", -1))
        return result

    factual = identities(factual_rows)
    intervened = identities(intervened_rows)
    if factual != intervened:
        missing = sorted(set(factual) - set(intervened))[:5]
        extra = sorted(set(intervened) - set(factual))[:5]
        mismatched = sorted(
            packet_uid
            for packet_uid in set(factual) & set(intervened)
            if factual[packet_uid] != intervened[packet_uid]
        )[:5]
        raise ValueError(
            "paired validation views are not aligned: "
            f"factual={len(factual)} intervened={len(intervened)} "
            f"missing={missing} extra={extra} label_mismatches={mismatched}"
        )


def validation_selection_key(
    factual_metrics: dict,
    intervened_metrics: Optional[dict],
    *,
    select_metric: str,
    paired_mode: str,
) -> tuple[tuple[float, ...], dict]:
    """Build a checkpoint key without hiding either intervention view."""
    if paired_mode == "disabled":
        value = float(factual_metrics[select_metric])
        return (
            value,
            float(factual_metrics["macro_f1"]),
            float(factual_metrics["accuracy"]),
        ), {
            "mode": "disabled",
            "score": value,
            "factual_macro_f1": float(factual_metrics["macro_f1"]),
            "factual_accuracy": float(factual_metrics["accuracy"]),
        }
    if intervened_metrics is None:
        raise ValueError(f"{paired_mode} requires intervened validation metrics")
    factual_f1 = float(factual_metrics["macro_f1"])
    intervened_f1 = float(intervened_metrics["macro_f1"])
    factual_acc = float(factual_metrics["accuracy"])
    intervened_acc = float(intervened_metrics["accuracy"])
    mean_f1 = 0.5 * (factual_f1 + intervened_f1)
    worst_f1 = min(factual_f1, intervened_f1)
    mean_acc = 0.5 * (factual_acc + intervened_acc)
    worst_acc = min(factual_acc, intervened_acc)
    if paired_mode == "worst_view_macro_f1":
        score = worst_f1
        key = (worst_f1, mean_f1, worst_acc, mean_acc, factual_f1, factual_acc)
    elif paired_mode == "mean_view_macro_f1":
        score = mean_f1
        key = (mean_f1, worst_f1, mean_acc, worst_acc, factual_f1, factual_acc)
    else:
        raise ValueError(f"unknown paired validation selection mode: {paired_mode}")
    return key, {
        "mode": paired_mode,
        "score": score,
        "factual_macro_f1": factual_f1,
        "intervened_macro_f1": intervened_f1,
        "mean_view_macro_f1": mean_f1,
        "worst_view_macro_f1": worst_f1,
        "factual_accuracy": factual_acc,
        "intervened_accuracy": intervened_acc,
        "mean_view_accuracy": mean_acc,
        "worst_view_accuracy": worst_acc,
    }


def tower1_training_config(args) -> dict:
    keys = (
        "base_model",
        "label_map",
        "packet_aux_jsonl",
        "paired_packet_aux_jsonl",
        "valid_packet_aux_jsonl",
        "valid_paired_packet_aux_jsonl",
        "sft_jsonl",
        "epochs",
        "max_steps",
        "packet_batch_size",
        "valid_batch_size",
        "valid_packets_per_flow",
        "max_packet_length",
        "lr",
        "head_lr",
        "weight_decay",
        "class_weighting",
        "class_weight_beta",
        "class_weight_basis",
        "class_weight_strength",
        "disable_packet_information_weights",
        "cls_weight",
        "contrastive_weight",
        "temperature",
        "same_flow_positive_weight",
        "same_label_positive_weight",
        "flow_proto_weight",
        "flow_proto_positive",
        "flow_proto_context",
        "paired_consistency_weight",
        "paired_cls_weight",
        "paired_logit_kl_weight",
        "paired_raw_consistency_weight",
        "flow_balanced_packet_batches",
        "packets_per_flow",
        "packet_batch_scheduler",
        "projection_dim",
        "lora_r",
        "lora_alpha",
        "lora_dropout",
        "gradient_accumulation_steps",
        "gradient_checkpointing",
        "dtype",
        "local_files_only",
        "init_checkpoint_dir",
        "init_adapter_only",
        "select_metric",
        "paired_validation_selection",
        "early_stop_patience",
        "no_sft",
        "seed",
    )
    config = {key: getattr(args, key, None) for key in keys}
    config["packet_batch_scheduler"] = getattr(
        args,
        "packet_batch_scheduler",
        "epoch_resampled_dataloader_v1",
    )
    return config


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_training_contract(
    output_dir: str | Path,
    args,
    *,
    status: str,
    completed_artifacts: Optional[dict] = None,
) -> Path:
    output = Path(output_dir) / "tower1_training_contract.json"
    source_path = Path(__file__).resolve()
    if status == "complete" and output.is_file():
        payload = json.loads(output.read_text(encoding="utf-8"))
        if payload.get("schema") != "tower1_training_contract_v1":
            raise ValueError(f"unexpected Tower-1 training contract schema: {output}")
        if payload.get("training_config") != tower1_training_config(args):
            raise ValueError("Tower-1 completion config differs from its launch contract")
        payload["status"] = "complete"
        payload["completed_artifacts"] = completed_artifacts or {}
        payload["completion_observed_trainer_source"] = {
            "path": str(source_path),
            "sha256": file_sha256(source_path),
        }
        output.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return output

    input_paths = {
        "label_map": args.label_map,
        "packet_aux_jsonl": args.packet_aux_jsonl,
        "valid_packet_aux_jsonl": args.valid_packet_aux_jsonl,
        "paired_packet_aux_jsonl": args.paired_packet_aux_jsonl,
        "valid_paired_packet_aux_jsonl": args.valid_paired_packet_aux_jsonl,
    }
    input_paths.update(
        {f"sft_jsonl_{index}": path for index, path in enumerate(args.sft_jsonl)}
    )
    input_evidence = {}
    for name, value in input_paths.items():
        if not value:
            continue
        path = Path(value).resolve()
        input_evidence[name] = {
            "path": str(path),
            "sha256": file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
    payload = {
        "schema": "tower1_training_contract_v1",
        "status": status,
        "argv": list(sys.argv),
        "training_config": tower1_training_config(args),
        "input_evidence": input_evidence,
        "trainer_source": {
            "path": str(source_path),
            "sha256": file_sha256(source_path),
        },
        "completed_artifacts": completed_artifacts or {},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return output


class FlowBalancedPacketBatchSampler(BatchSampler):
    """Sample several packets from each selected flow so flow-aware SupCon has positives."""

    def __init__(
        self,
        rows: List[dict],
        batch_size: int,
        packets_per_flow: int,
        seed: int = 42,
        allow_packet_replacement: bool = True,
        scheduler: str = "epoch_resampled_dataloader_v1",
    ):
        self.flow_to_indices: Dict[str, List[int]] = {}
        for idx, row in enumerate(rows):
            self.flow_to_indices.setdefault(str(row.get("flow_id", idx)), []).append(idx)
        self.flows = list(self.flow_to_indices.keys())
        self.batch_size = max(1, int(batch_size))
        self.packets_per_flow = max(1, int(packets_per_flow))
        self.flows_per_batch = max(1, self.batch_size // self.packets_per_flow)
        self.seed = seed
        self.allow_packet_replacement = bool(allow_packet_replacement)
        if scheduler not in {
            "epoch_resampled_dataloader_v1",
            "coverage_cycle_dataloader_v1",
        }:
            raise ValueError(f"unsupported packet batch scheduler: {scheduler}")
        self.scheduler = scheduler
        self.epoch = 0

    def _coverage_cycle_sample(self, flow_id: str, indices: List[int], epoch: int) -> List[int]:
        """Traverse a deterministic per-flow permutation before repeating packets."""
        selected: List[int] = []
        flow_seed = stable_flow_id(flow_id)
        start = epoch * self.packets_per_flow
        for virtual_position in range(start, start + self.packets_per_flow):
            cycle, position = divmod(virtual_position, len(indices))
            permutation = list(indices)
            cycle_seed = (self.seed + flow_seed + cycle * 1_000_003) & ((1 << 63) - 1)
            random.Random(cycle_seed).shuffle(permutation)
            selected.append(permutation[position])
        return selected

    def __iter__(self):
        epoch = self.epoch
        rng = random.Random(self.seed + epoch)
        self.epoch += 1
        flows = list(self.flows)
        rng.shuffle(flows)
        for start in range(0, len(flows), self.flows_per_batch):
            batch = []
            for flow_id in flows[start:start + self.flows_per_batch]:
                indices = self.flow_to_indices[flow_id]
                if self.scheduler == "coverage_cycle_dataloader_v1":
                    if not self.allow_packet_replacement and len(indices) < self.packets_per_flow:
                        batch.extend(indices)
                    else:
                        batch.extend(self._coverage_cycle_sample(flow_id, indices, epoch))
                elif len(indices) >= self.packets_per_flow:
                    batch.extend(rng.sample(indices, self.packets_per_flow))
                elif not self.allow_packet_replacement:
                    batch.extend(indices)
                else:
                    batch.extend(rng.choice(indices) for _ in range(self.packets_per_flow))
            if batch:
                yield batch[: self.batch_size]

    def __len__(self) -> int:
        return max(1, math.ceil(len(self.flows) / self.flows_per_batch))


class RestartableDataIterator:
    """Restart a loader at exhaustion without caching its first epoch."""

    def __init__(self, loader: Iterable):
        self.loader = loader
        self.iterator = iter(loader)
        self.completed_passes = 0

    def __next__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.completed_passes += 1
            self.iterator = iter(self.loader)
            try:
                return next(self.iterator)
            except StopIteration as exc:
                raise ValueError("cannot restart an empty training loader") from exc


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
        batch = {
            "input_ids": toks["input_ids"],
            "attention_mask": toks["attention_mask"],
            "labels": labels,
            "weights": weights,
            "flow_ids": flow_ids,
        }
        if any(r.get("paired_prompt") for r in rows):
            paired_texts = [r.get("paired_prompt") or r["prompt"] for r in rows]
            paired_toks = self.tokenizer(
                paired_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            batch["paired_input_ids"] = paired_toks["input_ids"]
            batch["paired_attention_mask"] = paired_toks["attention_mask"]
            batch["paired_mask"] = torch.tensor([bool(r.get("paired_prompt")) for r in rows], dtype=torch.bool)
        return batch


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


def load_label_names(label_map_path: str) -> List[str]:
    with open(label_map_path, "r", encoding="utf-8") as f:
        label_map = {str(k): int(v) for k, v in json.load(f).items()}
    names = [""] * len(label_map)
    for name, idx in label_map.items():
        names[idx] = name
    return names


def packet_class_counts(rows: List[dict], basis: str = "packet") -> Dict[int, int]:
    if basis == "packet":
        counts: Dict[int, int] = {}
        for row in rows:
            label = int(row["label_id"])
            counts[label] = counts.get(label, 0) + 1
        return counts
    if basis != "flow":
        raise ValueError(f"Unknown class-weight count basis: {basis}")

    flow_labels: Dict[str, int] = {}
    for row in rows:
        flow_id = str(row.get("flow_id", ""))
        if not flow_id:
            raise ValueError("Flow-based class weighting requires every packet row to have flow_id")
        label = int(row["label_id"])
        previous = flow_labels.setdefault(flow_id, label)
        if previous != label:
            raise ValueError(
                f"Conflicting labels for flow_id={flow_id}: {previous} versus {label}"
            )
    counts: Dict[int, int] = {}
    for label in flow_labels.values():
        counts[label] = counts.get(label, 0) + 1
    return counts


def configure_packet_weights(
    rows: List[dict],
    weighting: str,
    beta: float,
    disable_information_weights: bool,
    count_basis: str = "packet",
    strength: float = 1.0,
) -> Dict[int, float]:
    if not 0.0 <= strength <= 1.0:
        raise ValueError(f"class weight strength must be in [0, 1], got {strength}")
    counts = packet_class_counts(rows, basis=count_basis)
    if weighting == "inverse":
        class_weights = {label: 1.0 / max(count, 1) for label, count in counts.items()}
    elif weighting == "effective":
        class_weights = {
            label: (1.0 - beta) / max(1.0 - beta ** count, 1e-12)
            for label, count in counts.items()
        }
    else:
        class_weights = {label: 1.0 for label in counts}
    mean_weight = sum(class_weights.values()) / max(len(class_weights), 1)
    class_weights = {label: weight / max(mean_weight, 1e-12) for label, weight in class_weights.items()}
    class_weights = {label: weight ** strength for label, weight in class_weights.items()}
    mean_weight = sum(class_weights.values()) / max(len(class_weights), 1)
    class_weights = {label: weight / max(mean_weight, 1e-12) for label, weight in class_weights.items()}
    for row in rows:
        information_weight = 1.0 if disable_information_weights else float(row.get("packet_weight", 1.0))
        row["packet_weight"] = information_weight * class_weights[int(row["label_id"])]
    return class_weights


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
    ap.add_argument("--valid_packet_aux_jsonl", default="", help="Optional held-out packet validation JSONL used for best-checkpoint selection.")
    ap.add_argument("--sft_jsonl", nargs="*", default=[], help="packet_instruction.jsonl and packet_validity.jsonl")
    ap.add_argument("--no_sft", action="store_true", help="Train only packet cls + SupCon without generative QA loss.")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--init_checkpoint_dir", default="", help="Optional Tower-1 checkpoint dir containing adapter/ and tower1_heads.pt for continued training.")
    ap.add_argument(
        "--init_adapter_only",
        action="store_true",
        help="Warm-start the shared LoRA adapter but initialize dataset-specific packet/projection heads.",
    )
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--max_steps", type=int, default=0, help="Override epochs if >0.")
    ap.add_argument("--sft_batch_size", type=int, default=2)
    ap.add_argument("--packet_batch_size", type=int, default=16)
    ap.add_argument("--valid_batch_size", type=int, default=0, help="Validation batch size; 0 reuses --packet_batch_size.")
    ap.add_argument(
        "--valid_packets_per_flow",
        type=int,
        default=0,
        help="Deterministically evaluate at most this many packets per validation flow; 0 uses all packets.",
    )
    ap.add_argument("--max_sft_length", type=int, default=1792)
    ap.add_argument("--max_packet_length", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--head_lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--class_weighting", choices=["none", "inverse", "effective"], default="none")
    ap.add_argument("--class_weight_beta", type=float, default=0.9999)
    ap.add_argument(
        "--class_weight_basis",
        choices=["packet", "flow"],
        default="packet",
        help="Count raw packets or unique flows when deriving Tower-1 class-balanced CE weights.",
    )
    ap.add_argument(
        "--class_weight_strength",
        type=float,
        default=1.0,
        help="Exponent in [0,1] applied to normalized class weights; 0 disables reweighting and 1 uses it fully.",
    )
    ap.add_argument("--disable_packet_information_weights", action="store_true", help="Give ACK/control packets full CE weight; recommended when packet classification is the primary task.")
    ap.add_argument("--cls_weight", type=float, default=0.1)
    ap.add_argument("--contrastive_weight", type=float, default=0.3)
    ap.add_argument("--temperature", type=float, default=0.07)
    ap.add_argument("--same_flow_positive_weight", type=float, default=0.0, help="Extra positive weight for packets from the same flow in Tower-1 SupCon. 0 keeps label-only SupCon.")
    ap.add_argument("--same_label_positive_weight", type=float, default=1.0, help="Positive weight for same-label packets in flow-aware SupCon.")
    ap.add_argument("--flow_proto_weight", type=float, default=0.0, help="Weight for packet-to-flow prototype contrastive loss in Tower-1.")
    ap.add_argument("--flow_proto_positive", choices=["own_flow", "same_class"], default="same_class", help="Positive flow prototypes for --flow_proto_weight.")
    ap.add_argument(
        "--flow_proto_context",
        choices=["inclusive", "leave_one_out"],
        default="inclusive",
        help="Whether an anchor packet may contribute to its own flow prototype.",
    )
    ap.add_argument("--paired_packet_aux_jsonl", default="", help="Optional second-view packet_auxiliary.jsonl aligned by packet_uid, e.g. randomized IP/port prompts.")
    ap.add_argument(
        "--valid_paired_packet_aux_jsonl",
        default="",
        help="Aligned intervened validation view used for robust paired-view checkpoint selection.",
    )
    ap.add_argument("--paired_consistency_weight", type=float, default=0.0, help="Weight for Tower-1 full-header vs paired-view packet embedding/logit consistency.")
    ap.add_argument("--paired_cls_weight", type=float, default=0.0, help="Extra paired-view packet CE multiplier added inside the packet classification loss.")
    ap.add_argument("--paired_logit_kl_weight", type=float, default=0.5, help="Logit symmetric-KL weight inside Tower-1 paired consistency.")
    ap.add_argument(
        "--paired_raw_consistency_weight",
        type=float,
        default=1.0,
        help="Raw last-token cosine consistency inside Tower-1 paired loss; concat extraction exposes this representation downstream.",
    )
    ap.add_argument("--flow_balanced_packet_batches", action="store_true", help="Sample packet batches as multiple packets per flow for flow-aware SupCon.")
    ap.add_argument("--packets_per_flow", type=int, default=2, help="Packets sampled per flow when --flow_balanced_packet_batches is set.")
    ap.add_argument(
        "--packet_batch_scheduler",
        choices=["epoch_resampled_dataloader_v1", "coverage_cycle_dataloader_v1"],
        default="epoch_resampled_dataloader_v1",
        help="Per-flow packet exposure policy across training epochs.",
    )
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
    ap.add_argument("--eval_steps", type=int, default=0, help="Validation interval. 0 evaluates once per packet-loader epoch.")
    ap.add_argument("--select_metric", choices=["macro_f1", "accuracy"], default="macro_f1")
    ap.add_argument(
        "--paired_validation_selection",
        choices=["disabled", "worst_view_macro_f1", "mean_view_macro_f1"],
        default="disabled",
        help="Checkpoint rule over factual and intervened validation views.",
    )
    ap.add_argument(
        "--early_stop_patience",
        type=int,
        default=0,
        help="Stop after this many non-improving validation evaluations; 0 disables early stopping.",
    )
    ap.add_argument("--no_load_progress", action="store_true", help="Disable JSONL loading progress bars.")
    ap.add_argument("--stop_on_nonfinite_loss", action="store_true", help="Raise an error instead of skipping a NaN/Inf loss step.")
    ap.add_argument("--auto_max_sft_length", action="store_true", help="Scan SFT token lengths and raise --max_sft_length to the requested percentile.")
    ap.add_argument("--sft_length_percentile", type=float, default=100.0, help="Percentile used by --auto_max_sft_length.")
    ap.add_argument("--sft_length_multiple", type=int, default=256, help="Round auto max SFT length up to this multiple.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    if not 0.0 <= args.class_weight_strength <= 1.0:
        ap.error("--class_weight_strength must be in [0, 1]")
    if args.paired_raw_consistency_weight < 0:
        ap.error("--paired_raw_consistency_weight must be non-negative")
    if args.paired_validation_selection != "disabled":
        if not args.valid_packet_aux_jsonl or not args.valid_paired_packet_aux_jsonl:
            ap.error(
                "paired validation selection requires both --valid_packet_aux_jsonl "
                "and --valid_paired_packet_aux_jsonl"
            )
        if not args.paired_packet_aux_jsonl or args.paired_consistency_weight <= 0:
            ap.error(
                "paired validation selection requires paired-view training with "
                "--paired_packet_aux_jsonl and --paired_consistency_weight > 0"
            )

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    contract_path = write_training_contract(args.output_dir, args, status="launched")
    print(f"wrote Tower-1 training contract: {contract_path}", flush=True)
    device = torch.device(args.device)
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    print(f"device={device}, dtype={args.dtype}", flush=True)
    print(f"loading label map: {args.label_map}", flush=True)
    num_classes = infer_num_classes(args.label_map)
    label_names = load_label_names(args.label_map)
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
    if args.init_checkpoint_dir and not args.init_adapter_only:
        load_packet_heads(model, Path(args.init_checkpoint_dir) / "tower1_heads.pt")
    print("base model loaded", flush=True)
    if args.gradient_checkpointing:
        print("enabling gradient checkpointing", flush=True)
        model.backbone.gradient_checkpointing_enable()
        if hasattr(model.backbone, "enable_input_require_grads"):
            model.backbone.enable_input_require_grads()
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
    packet_ds = PacketAuxDataset(
        args.packet_aux_jsonl,
        show_progress=show_load_progress,
        paired_path=args.paired_packet_aux_jsonl,
    )
    class_weights = configure_packet_weights(
        packet_ds.rows,
        weighting=args.class_weighting,
        beta=args.class_weight_beta,
        disable_information_weights=args.disable_packet_information_weights,
        count_basis=args.class_weight_basis,
        strength=args.class_weight_strength,
    )
    class_counts = packet_class_counts(packet_ds.rows, basis=args.class_weight_basis)
    print(
        f"packet class weights (method={args.class_weighting}, basis={args.class_weight_basis}, "
        f"strength={args.class_weight_strength}) counts={class_counts} weights={class_weights}",
        flush=True,
    )
    if args.flow_balanced_packet_batches:
        distinct_flow_context = (
            args.flow_proto_weight > 0 and args.flow_proto_context == "leave_one_out"
        )
        packet_sampler = FlowBalancedPacketBatchSampler(
            packet_ds.rows,
            batch_size=args.packet_batch_size,
            packets_per_flow=args.packets_per_flow,
            seed=args.seed,
            allow_packet_replacement=not distinct_flow_context,
            scheduler=args.packet_batch_scheduler,
        )
        packet_loader = DataLoader(
            packet_ds,
            batch_sampler=packet_sampler,
            collate_fn=PacketAuxCollator(tokenizer, args.max_packet_length),
        )
        print(
            f"flow-balanced packet sampler: flows={len(packet_sampler.flows)}, "
            f"flows_per_batch={packet_sampler.flows_per_batch}, "
            f"packets_per_flow={packet_sampler.packets_per_flow}, "
            f"allow_packet_replacement={packet_sampler.allow_packet_replacement}, "
            f"scheduler={packet_sampler.scheduler}",
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
    packet_iter = RestartableDataIterator(packet_loader)

    valid_loader = None
    valid_paired_loader = None
    if args.valid_packet_aux_jsonl:
        valid_ds = PacketAuxDataset(args.valid_packet_aux_jsonl, show_progress=show_load_progress)
        full_valid_count = len(valid_ds.rows)
        valid_ds.rows = flow_balanced_validation_rows(
            valid_ds.rows,
            packets_per_flow=args.valid_packets_per_flow,
            seed=args.seed,
        )
        valid_loader = DataLoader(
            valid_ds,
            batch_size=args.valid_batch_size or args.packet_batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=PacketAuxCollator(tokenizer, args.max_packet_length),
        )
        print(
            f"validation packet samples={len(valid_ds)}/{full_valid_count}, "
            f"packets_per_flow={args.valid_packets_per_flow or 'all'}, batches={len(valid_loader)}",
            flush=True,
        )
        if args.valid_paired_packet_aux_jsonl:
            valid_paired_ds = PacketAuxDataset(
                args.valid_paired_packet_aux_jsonl,
                show_progress=show_load_progress,
            )
            full_paired_valid_count = len(valid_paired_ds.rows)
            valid_paired_ds.rows = flow_balanced_validation_rows(
                valid_paired_ds.rows,
                packets_per_flow=args.valid_packets_per_flow,
                seed=args.seed,
            )
            validate_aligned_validation_views(valid_ds.rows, valid_paired_ds.rows)
            valid_paired_loader = DataLoader(
                valid_paired_ds,
                batch_size=args.valid_batch_size or args.packet_batch_size,
                shuffle=False,
                drop_last=False,
                collate_fn=PacketAuxCollator(tokenizer, args.max_packet_length),
            )
            print(
                f"paired validation packet samples={len(valid_paired_ds)}/{full_paired_valid_count}, "
                f"packets_per_flow={args.valid_packets_per_flow or 'all'}, "
                f"batches={len(valid_paired_loader)}",
                flush=True,
            )

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
        sft_iter = RestartableDataIterator(sft_loader)
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

    running = {"loss": 0.0, "lm": 0.0, "cls": 0.0, "con": 0.0, "proto": 0.0, "pair": 0.0, "acc": 0.0, "lm_tokens": 0.0, "n": 0}
    skipped_nonfinite = 0
    best_key = None
    non_improving_evals = 0
    eval_interval = args.eval_steps if args.eval_steps > 0 else steps_per_epoch
    history_path = Path(args.output_dir) / "packet_validation_history.jsonl"
    training_config = tower1_training_config(args)
    history_path.unlink(missing_ok=True)
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
            flow_proto_context=args.flow_proto_context,
            paired_consistency_weight=args.paired_consistency_weight,
            paired_cls_weight=args.paired_cls_weight,
            paired_logit_kl_weight=args.paired_logit_kl_weight,
            paired_raw_consistency_weight=args.paired_raw_consistency_weight,
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
                f"proto={float(out.flow_proto_loss.detach().cpu())} "
                f"pair={float(out.paired_consistency_loss.detach().cpu())}"
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
            running["pair"] += float(out.paired_consistency_loss.detach().cpu())
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
                "pair": running["pair"] / n,
                "pkt_acc": running["acc"] / n,
                "lm_tokens": running["lm_tokens"] / n,
            }
            pbar.set_postfix({k: f"{v:.4f}" for k, v in msg.items()})
            tqdm.write(
                "step={step}/{total} loss={loss:.4f} lm={lm:.4f} pkt_cls={pkt_cls:.4f} "
                "supcon={supcon:.4f} proto={proto:.4f} pair={pair:.4f} "
                "pkt_acc={pkt_acc:.4f} lm_tokens/batch={lm_tokens:.1f} "
                "skipped_nonfinite={skipped}".format(
                    step=step + 1,
                    total=total_steps,
                    skipped=skipped_nonfinite,
                    **msg,
                )
            )
            running = {"loss": 0.0, "lm": 0.0, "cls": 0.0, "con": 0.0, "proto": 0.0, "pair": 0.0, "acc": 0.0, "lm_tokens": 0.0, "n": 0}

        if args.save_steps and (step + 1) % args.save_steps == 0:
            save_model(
                model,
                args.output_dir,
                suffix=f"step_{step+1}",
                training_config=training_config,
            )

        should_eval = valid_loader is not None and ((step + 1) % eval_interval == 0 or step + 1 == total_steps)
        if should_eval:
            from packet_eval_utils import evaluate_packet_model

            metrics = evaluate_packet_model(
                model,
                valid_loader,
                device=device,
                num_classes=num_classes,
                label_names=label_names,
                desc=f"valid step {step + 1}",
            )
            metrics.pop("y_true", None)
            metrics.pop("y_pred", None)
            paired_metrics = None
            if valid_paired_loader is not None:
                paired_metrics = evaluate_packet_model(
                    model,
                    valid_paired_loader,
                    device=device,
                    num_classes=num_classes,
                    label_names=label_names,
                    desc=f"valid paired step {step + 1}",
                )
                paired_metrics.pop("y_true", None)
                paired_metrics.pop("y_pred", None)
            key, checkpoint_selection = validation_selection_key(
                metrics,
                paired_metrics,
                select_metric=args.select_metric,
                paired_mode=args.paired_validation_selection,
            )
            record = {
                "step": step + 1,
                "select_metric": args.select_metric,
                "metrics": metrics,
                "paired_metrics": paired_metrics,
                "checkpoint_selection": checkpoint_selection,
            }
            with open(history_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            improved = best_key is None or key > best_key
            paired_message = ""
            if paired_metrics is not None:
                paired_message = (
                    f" paired_accuracy={paired_metrics['accuracy']:.4f} "
                    f"paired_macro_f1={paired_metrics['macro_f1']:.4f}"
                )
            print(
                f"validation step={step + 1} loss={metrics['loss']:.4f} "
                f"accuracy={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f} "
                f"select={checkpoint_selection['mode']}:{checkpoint_selection['score']:.4f}"
                f"{paired_message} improved={improved}",
                flush=True,
            )
            if improved:
                best_key = key
                non_improving_evals = 0
                save_model(model, args.output_dir, suffix="best", training_config=training_config)
                with open(Path(args.output_dir) / "best_packet_validation_metrics.json", "w", encoding="utf-8") as f:
                    json.dump(record, f, indent=2, ensure_ascii=False)
            else:
                non_improving_evals += 1
            if validation_patience_exhausted(
                non_improving_evals, args.early_stop_patience
            ):
                print(
                    "early stopping Tower1 after "
                    f"step={step + 1}; non_improving_evals={non_improving_evals} "
                    f"patience={args.early_stop_patience}",
                    flush=True,
                )
                break

    print(f"training finished; skipped_nonfinite={skipped_nonfinite}", flush=True)
    save_model(
        model,
        args.output_dir,
        suffix="final",
        training_config=training_config,
    )
    output_dir = Path(args.output_dir)
    completed_artifacts = {
        name: {
            "path": str(path.resolve()),
            "sha256": file_sha256(path),
        }
        for name, path in {
            "validation_history": output_dir / "packet_validation_history.jsonl",
            "final_heads": output_dir / "final" / "tower1_heads.pt",
            "final_config": output_dir / "final" / "tower1_config.json",
        }.items()
        if path.is_file()
    }
    write_training_contract(
        args.output_dir,
        args,
        status="complete",
        completed_artifacts=completed_artifacts,
    )


def save_model(
    model: QwenPacketMultiTaskModel,
    output_dir: str,
    suffix: str = "",
    training_config: Optional[dict] = None,
) -> None:
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
                "projection_dim": model.projection_head.net[-1].out_features,
                "embedding_pooling": "last_token",
                "loss": "L_QA + alpha*L_packet_cls + beta*L_supcon + gamma*L_flow_proto + delta*L_paired_consistency",
                "supports_flow_aware_supcon": True,
                "supports_flow_prototype_loss": True,
                "supports_paired_view_consistency": True,
                "paired_consistency_representation": "raw_and_projected",
                "training_config": training_config or {},
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
