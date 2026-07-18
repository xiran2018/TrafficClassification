#!/usr/bin/env python3
"""Tower-1 preprocessing for packet-level protocol instruction tuning.

This version fixes two important issues in the previous draft:
1) It uses raw packet-byte prompts for Q&A so checksum/field answers are not leaked
   through parsed `Derived` fields.
2) It supports a shared label map across train/valid/test to prevent label-id mismatch.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

from tqdm import tqdm
from traffic_utils import (
    corrupt_ipv4_checksum_only,
    corrupt_ipv4_total_len_keep_ip_checksum_valid,
    extract_packet_classification_flows,
    extract_flow_packets,
    iter_labeled_pcaps,
    make_label_map,
    stable_id,
    packet_information_weight,
)


def load_or_create_label_map(input_dir: str, label_map_in: str = "", pcaps: List[Tuple[str, Path]] | None = None) -> Dict[str, int]:
    if label_map_in:
        with open(label_map_in, "r", encoding="utf-8") as f:
            return {k: int(v) for k, v in json.load(f).items()}
    rows = pcaps if pcaps is not None else list(iter_labeled_pcaps(input_dir))
    labels = [lab for lab, _ in rows]
    return make_label_map(labels)


def qa_samples_for_packet(qa_prompt: str, meta: Dict, flow_id: str, label: str) -> List[Dict[str, str]]:
    samples: List[Dict[str, str]] = []
    base = f"{qa_prompt}\n\nQuestion: "

    # Retrieval questions. Since input is raw hex, these are not trivial copy tasks.
    retrieval = [
        ("What is the source IPv4 address?", meta["src_ip"]),
        ("What is the destination IPv4 address?", meta["dst_ip"]),
        ("What is the IPv4 identification field?", str(meta["ip_id"])),
        ("What is the IPv4 TTL?", str(meta["ip_ttl"])),
        ("What is the IPv4 total length?", str(meta["ip_total_len"])),
        ("What is the IPv4 header checksum?", f"0x{meta['ip_checksum']:04x}" if meta["ip_checksum"] >= 0 else "unknown"),
        ("What is the transport-layer protocol?", meta["l4"]),
        ("Which is the last byte offset of the network-layer header?", str(meta["ip_header_len"] - 1 if meta["ip_header_len"] > 0 else "unknown")),
        ("Which is the length of the payload in the network layer?", str(max(0, meta["ip_total_len"] - meta["ip_header_len"]) if meta["ip_total_len"] > 0 and meta["ip_header_len"] > 0 else "unknown")),
    ]
    if meta["l4"] == "TCP":
        retrieval.extend([
            ("What is the TCP source port?", str(meta["sport"])),
            ("What is the TCP destination port?", str(meta["dport"])),
            ("What is the TCP sequence number?", str(meta["seq"])),
            ("What is the TCP acknowledgment number?", str(meta["ack"])),
            ("What are the TCP flags?", meta["tcp_flags"]),
            ("What is the TCP checksum?", f"0x{meta['l4_checksum']:04x}" if meta["l4_checksum"] >= 0 else "unknown"),
            ("What is the TCP window size?", str(meta["tcp_window"])),
            ("Which is the length of the TCP payload?", str(meta["payload_len"])),
        ])
    elif meta["l4"] == "UDP":
        retrieval.extend([
            ("What is the UDP source port?", str(meta["sport"])),
            ("What is the UDP destination port?", str(meta["dport"])),
            ("What is the UDP length?", str(meta["udp_len"])),
            ("What is the UDP checksum?", f"0x{meta['l4_checksum']:04x}" if meta["l4_checksum"] >= 0 else "unknown"),
            ("Which is the length of the UDP payload?", str(max(0, meta["udp_len"] - 8) if meta["udp_len"] > 0 else "unknown")),
        ])

    for q, a in retrieval:
        samples.append({
            "instruction": "Answer the packet-level protocol question from the provided packet bytes.",
            "input": base + q,
            "output": str(a),
            "task": "packet_field_qa",
            "flow_id": flow_id,
            "label": label,
        })

    # Consistency and checksum questions.
    consistency = []
    if meta["ip_checksum_valid"] is not None:
        consistency.append(("Is the IPv4 header checksum correct?", "Yes" if meta["ip_checksum_valid"] else "No"))
    if meta["ip_total_len"] > 0 and meta["ip_header_len"] > 0:
        actual = meta.get("l3_captured_len", -1)
        if meta.get("full_l3_captured", False):
            ans = "Yes" if meta["ip_total_len"] <= actual and meta["ip_total_len"] >= meta["ip_header_len"] else "No"
        else:
            ans = "Cannot determine from the truncated capture."
        consistency.append(("Is the IPv4 total length consistent with the captured network-layer packet length?", ans))
    if meta["l4"] == "TCP":
        consistency.append(("Is the TCP data offset structurally valid?", "Yes" if meta["tcp_data_offset"] >= 20 else "No"))
        if meta.get("l4_checksum_valid") is not None:
            consistency.append(("Is the TCP checksum correct?", "Yes" if meta["l4_checksum_valid"] else "No"))
        else:
            consistency.append(("Can the TCP checksum be reliably verified from this input?", "No, it requires the complete TCP segment and pseudo-header context."))
    elif meta["l4"] == "UDP":
        consistency.append(("Is the UDP length structurally valid?", "Yes" if meta["udp_len"] >= 8 else "No"))
        if meta.get("l4_checksum_valid") is not None:
            consistency.append(("Is the UDP checksum correct?", "Yes" if meta["l4_checksum_valid"] else "No"))
        else:
            consistency.append(("Can the UDP checksum be reliably verified from this input?", "No, it requires the complete UDP datagram and pseudo-header context."))

    for q, a in consistency:
        samples.append({
            "instruction": "Answer the packet consistency question from the provided packet bytes.",
            "input": base + q,
            "output": a,
            "task": "packet_consistency_qa",
            "flow_id": flow_id,
            "label": label,
        })
    return samples


def validity_samples_for_packet(qa_prompt: str, meta: Dict, flow_id: str, label: str) -> List[Dict[str, str]]:
    samples: List[Dict[str, str]] = [{
        "instruction": "Determine whether the packet is structurally valid and explain the key evidence.",
        "input": qa_prompt,
        "output": "Valid: no structural inconsistency is intentionally introduced in this packet.",
        "task": "packet_validity",
        "flow_id": flow_id,
        "label": label,
    }]
    # Easy negative: checksum byte changed.
    neg1 = corrupt_ipv4_checksum_only(meta.get("l3_hex_prefix", ""))
    if neg1:
        samples.append({
            "instruction": "Determine whether the packet is structurally valid and explain the key evidence.",
            "input": f"[PacketBytes]\nL3PacketPrefixHex: {neg1}\nCapturedL3Length: {meta.get('l3_captured_len')}\nFullL3Captured: {meta.get('full_l3_captured')}\n[EndPacketBytes]",
            "output": "Invalid: the IPv4 header checksum is inconsistent with the header bytes.",
            "task": "packet_validity_negative_easy",
            "flow_id": flow_id,
            "label": label,
        })
    # Hard negative: total length corrupted but IP checksum recomputed, so checksum alone is insufficient.
    neg2 = corrupt_ipv4_total_len_keep_ip_checksum_valid(meta.get("l3_hex_prefix", ""))
    if neg2:
        samples.append({
            "instruction": "Determine whether the packet is structurally valid and explain the key evidence.",
            "input": f"[PacketBytes]\nL3PacketPrefixHex: {neg2}\nCapturedL3Length: {meta.get('l3_captured_len')}\nFullL3Captured: {meta.get('full_l3_captured')}\n[EndPacketBytes]",
            "output": "Invalid: the IPv4 total length field is inconsistent with the observed packet length even though the IPv4 checksum may be recomputed.",
            "task": "packet_validity_negative_hard",
            "flow_id": flow_id,
            "label": label,
        })
    return samples


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--input_layout", choices=["flow_pcaps", "class_packet_pcaps"], default="flow_pcaps", help="flow_pcaps: one PCAP is one flow; class_packet_pcaps: one SWEET packet-level PCAP contains many real flows of one class.")
    ap.add_argument("--max_packets_per_flow", type=int, default=64, help="Maximum packets retained per real flow. In class_packet_pcaps mode, 0 keeps every packet.")
    ap.add_argument("--payload_prefix_len", type=int, default=128)
    ap.add_argument("--l3_prefix_len", type=int, default=512)
    ap.add_argument("--max_flows", type=int, default=0)
    ap.add_argument("--label_map_in", default="", help="Use the train label_map.json for valid/test to keep label ids consistent.")
    ap.add_argument("--write_label_map", action="store_true", help="Write label_map.json to output_dir.")
    ap.add_argument(
        "--embedding_header_policy",
        choices=["full", "randomize_ip_port", "mask_ip_port", "mask_session_fields"],
        default="full",
        help="Header policy for packet_index prompts used during embedding extraction; QA/SFT prompts stay unchanged.",
    )
    ap.add_argument("--classification_only", action="store_true", help="Write packet_index and packet_auxiliary only; skip the much larger protocol QA/validity corpora.")
    ap.add_argument("--no_progress", action="store_true", help="Disable pcap preprocessing progress bar.")
    args = ap.parse_args()
    if args.input_layout == "flow_pcaps" and args.max_packets_per_flow <= 0:
        ap.error("--max_packets_per_flow must be positive for --input_layout flow_pcaps")
    os.makedirs(args.output_dir, exist_ok=True)

    show_progress = not args.no_progress
    pcaps = list(iter_labeled_pcaps(args.input_dir))
    if show_progress:
        print(f"found {len(pcaps)} pcap files under {args.input_dir}", flush=True)

    label_map = load_or_create_label_map(args.input_dir, args.label_map_in, pcaps=pcaps)
    if args.write_label_map or not args.label_map_in:
        with open(Path(args.output_dir) / "label_map.json", "w", encoding="utf-8") as f:
            json.dump(label_map, f, ensure_ascii=False, indent=2)

    packet_index_path = Path(args.output_dir) / "packet_index.jsonl"
    instruction_path = Path(args.output_dir) / "packet_instruction.jsonl"
    validity_path = Path(args.output_dir) / "packet_validity.jsonl"
    auxiliary_path = Path(args.output_dir) / "packet_auxiliary.jsonl"
    n_flows = n_packets = n_qa = n_validity = n_aux = 0
    with open(packet_index_path, "w", encoding="utf-8") as pidx, \
         open(instruction_path, "w", encoding="utf-8") as qaf, \
         open(validity_path, "w", encoding="utf-8") as vf, \
         open(auxiliary_path, "w", encoding="utf-8") as af:
        pbar = tqdm(pcaps, desc="preprocess tower1", unit="pcap", disable=not show_progress)
        stop = False
        for label, pcap in pbar:
            if label not in label_map:
                msg = f"skip label not in label_map: {label}"
                tqdm.write(msg) if show_progress else print(msg)
                continue
            if stop or (args.max_flows and n_flows >= args.max_flows):
                break
            try:
                if args.input_layout == "class_packet_pcaps":
                    flow_rows = extract_packet_classification_flows(
                        pcap,
                        max_packets_per_flow=args.max_packets_per_flow,
                        payload_prefix_len=args.payload_prefix_len,
                        l3_prefix_len=args.l3_prefix_len,
                        embedding_header_policy=args.embedding_header_policy,
                    )
                else:
                    flow_id = stable_id(str(pcap.resolve()))
                    metas, qa_prompts, embed_prompts = extract_flow_packets(
                        pcap,
                        max_packets=args.max_packets_per_flow,
                        payload_prefix_len=args.payload_prefix_len,
                        l3_prefix_len=args.l3_prefix_len,
                        embedding_header_policy=args.embedding_header_policy,
                        header_random_salt=flow_id,
                    )
                    flow_rows = [(flow_id, metas, qa_prompts, embed_prompts)]
            except Exception as exc:
                msg = f"skip {pcap}: {exc}"
                tqdm.write(msg) if show_progress else print(msg)
                continue
            try:
                for flow_id, metas, qa_prompts, embed_prompts in flow_rows:
                    if args.max_flows and n_flows >= args.max_flows:
                        stop = True
                        break
                    if not metas:
                        continue
                    for meta_obj, qa_prompt, embed_prompt in zip(metas, qa_prompts, embed_prompts):
                        meta = asdict(meta_obj)
                        packet_uid = f"{flow_id}_{meta['packet_id']}"
                        row = {
                            "flow_id": flow_id,
                            "pcap_path": str(pcap),
                            "label": label,
                            "label_id": label_map[label],
                            "packet_id": meta["packet_id"],
                            "packet_uid": packet_uid,
                            "prompt": embed_prompt,
                            "qa_prompt": qa_prompt,
                            "embedding_header_policy": args.embedding_header_policy,
                            "input_layout": args.input_layout,
                            "sample_unit": "packet" if args.input_layout == "class_packet_pcaps" else "flow_packet",
                            "packet_context_policy": "single_packet" if args.input_layout == "class_packet_pcaps" else "flow_context",
                            "meta": meta,
                        }
                        pidx.write(json.dumps(row, ensure_ascii=False) + "\n")
                        aux_row = {
                            "flow_id": flow_id,
                            "pcap_path": str(pcap),
                            "label": label,
                            "label_id": label_map[label],
                            "packet_id": meta["packet_id"],
                            "packet_uid": packet_uid,
                            "prompt": embed_prompt,
                            "embedding_header_policy": args.embedding_header_policy,
                            "input_layout": args.input_layout,
                            "sample_unit": "packet" if args.input_layout == "class_packet_pcaps" else "flow_packet",
                            "packet_context_policy": "single_packet" if args.input_layout == "class_packet_pcaps" else "flow_context",
                            "packet_weight": packet_information_weight(meta_obj),
                            "meta": {
                                "direction": meta.get("direction"),
                                "l4": meta.get("l4"),
                                "packet_len": meta.get("packet_len"),
                                "payload_len": meta.get("payload_len"),
                                "tcp_flags": meta.get("tcp_flags"),
                                "iat": meta.get("iat"),
                            },
                        }
                        af.write(json.dumps(aux_row, ensure_ascii=False) + "\n")
                        n_aux += 1
                        n_packets += 1
                        if not args.classification_only:
                            for sample in qa_samples_for_packet(qa_prompt, meta, flow_id, label):
                                qaf.write(json.dumps(sample, ensure_ascii=False) + "\n")
                                n_qa += 1
                            for sample in validity_samples_for_packet(qa_prompt, meta, flow_id, label):
                                vf.write(json.dumps(sample, ensure_ascii=False) + "\n")
                                n_validity += 1
                    n_flows += 1
            except Exception as exc:
                msg = f"skip remaining flows in {pcap}: {exc}"
                tqdm.write(msg) if show_progress else print(msg)
            if show_progress:
                pbar.set_postfix(
                    flows=n_flows,
                    packets=n_packets,
                    qa=n_qa,
                    validity=n_validity,
                    aux=n_aux,
                )
            elif n_flows % 100 == 0:
                print(f"flows={n_flows}, packets={n_packets}, qa={n_qa}, validity={n_validity}")
        pbar.close()
    print(f"saved {packet_index_path}")
    print(f"saved {instruction_path}")
    print(f"saved {validity_path}")
    print(f"saved {auxiliary_path}")


if __name__ == "__main__":
    main()
