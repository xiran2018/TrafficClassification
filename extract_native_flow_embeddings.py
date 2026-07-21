#!/usr/bin/env python3
"""Extract packet-aligned structural embeddings from a native flow encoder."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.native_flow_encoder import NativeFlowEncoder
from native_flow_data import PacketIndexFlowDataset, apply_session_invariant_mask


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--packet_index", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--max_flows", type=int, default=0)
    ap.add_argument(
        "--session_mask_probability",
        type=float,
        default=None,
        help="Session-field mask probability. Defaults to the checkpoint pretraining value.",
    )
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_config = checkpoint["model_config"]
    session_mask_probability = args.session_mask_probability
    if session_mask_probability is None:
        session_mask_probability = float(
            checkpoint.get("pretraining_config", {}).get(
                "session_mask_probability", 1.0
            )
        )
    if not 0.0 <= session_mask_probability <= 1.0:
        ap.error("--session_mask_probability must be in [0, 1]")
    model = NativeFlowEncoder(**model_config).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    dataset = PacketIndexFlowDataset(
        args.packet_index,
        max_packets=int(model_config["max_packets"]),
        max_bytes=int(model_config["max_bytes"]),
        max_flows=args.max_flows,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    output_dir = Path(args.output_dir)
    embedding_dir = output_dir / "packet_structural_embeddings"
    flow_embedding_dir = output_dir / "flow_structural_embeddings"
    projection_dir = output_dir / "flow_contrastive_projections"
    embedding_dir.mkdir(parents=True, exist_ok=True)
    flow_embedding_dir.mkdir(parents=True, exist_ok=True)
    projection_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "flow_structural_embedding_index.jsonl"
    flow_index_path = output_dir / "flow_structural_probe_index.jsonl"
    projection_index_path = output_dir / "flow_projection_probe_index.jsonl"

    with (
        open(index_path, "w", encoding="utf-8") as output,
        open(flow_index_path, "w", encoding="utf-8") as flow_output,
        open(projection_index_path, "w", encoding="utf-8") as projection_output,
    ):
        with torch.no_grad():
            for batch in tqdm(loader, desc="extract native structural embeddings"):
                byte_tokens = batch["byte_tokens"].to(device, non_blocking=True)
                field_ids = batch["field_ids"].to(device, non_blocking=True)
                byte_mask = batch["byte_mask"].to(device, non_blocking=True)
                packet_mask = batch["packet_mask"].to(device, non_blocking=True)
                directions = batch["directions"].to(device, non_blocking=True)
                packet_meta = batch["packet_meta"].to(device, non_blocking=True)
                byte_tokens, _ = apply_session_invariant_mask(
                    byte_tokens,
                    field_ids,
                    byte_mask,
                    session_mask_probability,
                )
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    encoded = model(
                        byte_tokens,
                        field_ids,
                        byte_mask,
                        packet_mask,
                        directions,
                        packet_meta,
                    )
                # The paper-facing shared content channel must be a strict
                # current-packet representation. Contextual packet_repr is
                # retained only for flow pretext heads and must not enter the
                # downstream shared packet encoder.
                packet_embeddings = encoded["packet_content"].float().cpu().numpy()
                flow_embeddings = encoded["flow_repr"].float().cpu().numpy()
                flow_projections = encoded["flow_projection"].float().cpu().numpy()
                packet_masks = batch["packet_mask"].numpy()
                packet_ids = batch["packet_ids"].numpy()
                packet_directions = batch["directions"].numpy()
                for index, flow_id in enumerate(batch["flow_id"]):
                    count = int(packet_masks[index].sum())
                    embedding = packet_embeddings[index, :count].astype(np.float32)
                    embedding_path = embedding_dir / f"{flow_id}.npy"
                    np.save(embedding_path, embedding)
                    row = {
                        "flow_id": flow_id,
                        "label": batch["label"][index],
                        "label_id": int(batch["label_id"][index]),
                        "pcap_path": batch["pcap_path"][index],
                        "structural_embedding_path": str(embedding_path),
                        "structural_embedding_dim": int(embedding.shape[1]),
                        "representation_scope": "strict_current_packet",
                        "representation_name": "native_packet_content",
                        # Generic aliases let frozen-representation probes use
                        # the same validation-only model-selection pipeline as
                        # semantic packet embeddings.
                        "embedding_path": str(embedding_path),
                        "embedding_dim": int(embedding.shape[1]),
                        "packet_count": count,
                        "packet_ids": packet_ids[index, :count].tolist(),
                        "packet_metas": [
                            {"direction": "C2S" if int(direction) == 1 else "S2C"}
                            for direction in packet_directions[index, :count]
                        ],
                        "encoder_checkpoint": str(Path(args.checkpoint)),
                        "representation": "native_protocol_current_packet_content",
                    }
                    output.write(json.dumps(row, ensure_ascii=False) + "\n")
                    flow_embedding_path = flow_embedding_dir / f"{flow_id}.npy"
                    projection_path = projection_dir / f"{flow_id}.npy"
                    np.save(
                        flow_embedding_path,
                        flow_embeddings[index : index + 1].astype(np.float32),
                    )
                    np.save(
                        projection_path,
                        flow_projections[index : index + 1].astype(np.float32),
                    )
                    probe_base = {
                        "flow_id": flow_id,
                        "label": batch["label"][index],
                        "label_id": int(batch["label_id"][index]),
                        "pcap_path": batch["pcap_path"][index],
                        "packet_count": count,
                        "packet_metas": [{"direction": "C2S"}],
                        "encoder_checkpoint": str(Path(args.checkpoint)),
                    }
                    flow_output.write(
                        json.dumps(
                            {
                                **probe_base,
                                "embedding_path": str(flow_embedding_path),
                                "embedding_dim": int(flow_embeddings.shape[1]),
                                "representation": "native_protocol_flow_repr",
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    projection_output.write(
                        json.dumps(
                            {
                                **probe_base,
                                "embedding_path": str(projection_path),
                                "embedding_dim": int(flow_projections.shape[1]),
                                "representation": "native_protocol_flow_projection",
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
    manifest = {
        "packet_index": str(Path(args.packet_index)),
        "checkpoint": str(Path(args.checkpoint)),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "flow_count": len(dataset),
        "structural_embedding_dim": int(model_config["hidden_dim"]),
        "packet_representation_scope": "strict_current_packet",
        "packet_representation_name": "native_protocol_current_packet_content",
        "contextual_flow_representations_exported_separately": True,
        "session_mask_probability": session_mask_probability,
        "flow_structural_embedding_index": str(index_path),
        "flow_structural_probe_index": str(flow_index_path),
        "flow_projection_probe_index": str(projection_index_path),
        "alignment": "flow_id_and_packet_id",
    }
    with open(output_dir / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
