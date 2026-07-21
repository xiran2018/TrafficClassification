#!/usr/bin/env python3
"""Audit whether packet and flow checkpoints support an exact shared-core claim."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from models.native_flow_encoder import NATIVE_PACKET_PRETRAINING_PROTOCOL


PACKET_PREFIX = "protocol_content_encoder."
NATIVE_PREFIX = "packet_content_encoder."
SHARED_REPRESENTATION_PREFIX = "shared_packet_encoder."
PRETRAINING_PROTOCOL = NATIVE_PACKET_PRETRAINING_PROTOCOL


def prefixed_schema(state: dict[str, torch.Tensor], prefix: str) -> dict[str, list[int]]:
    schema = {
        key[len(prefix) :]: list(value.shape)
        for key, value in state.items()
        if key.startswith(prefix)
    }
    if not schema:
        raise ValueError(f"Checkpoint has no parameters with prefix {prefix!r}")
    return dict(sorted(schema.items()))


def packet_architecture(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "max_bytes": int(config["max_bytes"]),
        "hidden_dim": int(config["hidden_dim"]),
        "num_layers": int(config["num_layers"]),
        "num_heads": int(config["num_heads"]),
        "dropout": float(config["dropout"]),
        "num_field_types": 9,
    }


def native_architecture(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "max_bytes": int(config["max_bytes"]),
        "hidden_dim": int(config["hidden_dim"]),
        "num_layers": int(config["byte_layers"]),
        "num_heads": int(config["num_heads"]),
        "dropout": float(config["dropout"]),
        "num_field_types": int(config.get("num_field_types", 9)),
    }


def load_group_reduction(path: str, task: str) -> str | None:
    if not path:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if task == "packet":
        config = payload.get("config") or payload.get("model_config") or {}
        return config.get("content_group_loss_reduction") or config.get(
            "byte_content_group_loss_reduction"
        )
    return payload.get("content_group_loss_reduction") or (
        payload.get("framework", {}).get("notes", {}).get("content_group_loss_reduction")
    )


def audit_shared_core(
    packet_checkpoint: dict[str, Any],
    native_checkpoint: dict[str, Any],
    flow_checkpoint: dict[str, Any] | None = None,
    packet_group_reduction: str | None = None,
    flow_group_reduction: str | None = None,
    native_checkpoint_sha256: str | None = None,
) -> dict[str, Any]:
    packet_config = packet_checkpoint["config"]
    native_config = native_checkpoint["model_config"]
    packet_arch = packet_architecture(packet_config)
    native_arch = native_architecture(native_config)
    architecture_differences = {
        key: {"packet": packet_arch[key], "flow": native_arch[key]}
        for key in packet_arch
        if packet_arch[key] != native_arch[key]
    }
    packet_schema = prefixed_schema(packet_checkpoint["state_dict"], PACKET_PREFIX)
    native_schema = prefixed_schema(native_checkpoint["state_dict"], NATIVE_PREFIX)
    schema_only_packet = sorted(set(packet_schema) - set(native_schema))
    schema_only_flow = sorted(set(native_schema) - set(packet_schema))
    schema_shape_mismatches = {
        key: {"packet": packet_schema[key], "flow": native_schema[key]}
        for key in sorted(set(packet_schema) & set(native_schema))
        if packet_schema[key] != native_schema[key]
    }
    initialization = packet_checkpoint.get("initialization") or {}
    initialization_protocol = initialization.get("protocol_content_pretraining")
    initialization_source = initialization.get("protocol_content_checkpoint", "")
    initialization_sha256 = initialization.get("protocol_content_checkpoint_sha256", "")
    native_protocol = native_checkpoint.get("pretraining_protocol")
    architecture_match = not architecture_differences
    schema_match = not schema_only_packet and not schema_only_flow and not schema_shape_mismatches
    initialization_match = (
        initialization_protocol == PRETRAINING_PROTOCOL
        and native_protocol == PRETRAINING_PROTOCOL
        and bool(initialization_source)
        and bool(native_checkpoint_sha256)
        and initialization_sha256 == native_checkpoint_sha256
    )
    risk_protocol_observed = (
        packet_group_reduction is not None and flow_group_reduction is not None
    )
    risk_protocol_match = (
        risk_protocol_observed and packet_group_reduction == flow_group_reduction
    )
    representation_observed = flow_checkpoint is not None
    representation_match = False
    representation_differences: dict[str, Any] = {
        "only_packet": [],
        "only_flow": [],
        "shape_mismatches": {},
        "configuration": {},
    }
    flow_only_boundary_observed = False
    if flow_checkpoint is not None:
        packet_representation_schema = prefixed_schema(
            packet_checkpoint["state_dict"], SHARED_REPRESENTATION_PREFIX
        )
        flow_representation_schema = prefixed_schema(
            flow_checkpoint["model_state"], SHARED_REPRESENTATION_PREFIX
        )
        representation_differences["only_packet"] = sorted(
            set(packet_representation_schema) - set(flow_representation_schema)
        )
        representation_differences["only_flow"] = sorted(
            set(flow_representation_schema) - set(packet_representation_schema)
        )
        representation_differences["shape_mismatches"] = {
            key: {
                "packet": packet_representation_schema[key],
                "flow": flow_representation_schema[key],
            }
            for key in sorted(
                set(packet_representation_schema) & set(flow_representation_schema)
            )
            if packet_representation_schema[key] != flow_representation_schema[key]
        }
        flow_config = flow_checkpoint
        expected_configuration = {
            "semantic_dim": int(packet_config.get("semantic_dim", -1)),
            "content_dim": int(packet_config.get("hidden_dim", -1)),
            "structural_dim": int(packet_config.get("meta_dim", -1)),
            "hidden_dim": int(packet_config.get("hidden_dim", -1)),
        }
        observed_configuration = {
            "semantic_dim": int(flow_config.get("input_dim", -1))
            - int(flow_config.get("meta_feature_dim", -1)),
            "content_dim": int(flow_config.get("native_structural_dim", -1)),
            "structural_dim": int(flow_config.get("meta_feature_dim", -1))
            - int(flow_config.get("native_structural_dim", -1)),
            "hidden_dim": int(flow_config.get("shared_packet_hidden_dim", -1)),
        }
        representation_differences["configuration"] = {
            key: {"packet": expected_configuration[key], "flow": observed_configuration[key]}
            for key in expected_configuration
            if expected_configuration[key] != observed_configuration[key]
        }
        packet_exact = packet_config.get("exact_shared_representation") is True
        flow_exact = flow_config.get("exact_shared_packet_encoder") is True
        session_masked = packet_config.get("mask_protocol_session_fields") is True
        flow_only_boundary_observed = any(
            key.startswith("packet_to_flow_proj.")
            for key in flow_checkpoint["model_state"]
        )
        representation_match = (
            packet_exact
            and flow_exact
            and session_masked
            and flow_only_boundary_observed
            and not representation_differences["only_packet"]
            and not representation_differences["only_flow"]
            and not representation_differences["shape_mismatches"]
            and not representation_differences["configuration"]
        )
    exact = (
        architecture_match
        and schema_match
        and initialization_match
        and risk_protocol_match
        and (representation_match if representation_observed else True)
    )
    return {
        "claim": "same_packet_core_only_learned_weights_may_differ",
        "status": "pass" if exact else "not_ready",
        "architecture_match": architecture_match,
        "parameter_schema_match": schema_match,
        "shared_pretraining_protocol_match": initialization_match,
        "content_risk_protocol_match": risk_protocol_match,
        "exact_shared_representation_observed": representation_observed,
        "exact_shared_representation_match": representation_match,
        "flow_only_projection_boundary_observed": flow_only_boundary_observed,
        "packet_architecture": packet_arch,
        "flow_architecture": native_arch,
        "architecture_differences": architecture_differences,
        "schema_differences": {
            "only_packet": schema_only_packet,
            "only_flow": schema_only_flow,
            "shape_mismatches": schema_shape_mismatches,
        },
        "initialization": {
            "required_protocol": PRETRAINING_PROTOCOL,
            "observed_protocol": initialization_protocol,
            "native_checkpoint_protocol": native_protocol,
            "source_checkpoint": initialization_source,
            "source_checkpoint_sha256": initialization_sha256,
            "audited_native_checkpoint_sha256": native_checkpoint_sha256,
        },
        "content_risk_protocol": {
            "packet": packet_group_reduction,
            "flow": flow_group_reduction,
            "observed_for_both_tasks": risk_protocol_observed,
        },
        "shared_representation_differences": representation_differences,
        "interpretation": (
            "A shared Python class is insufficient. Pass requires identical packet-core "
            "architecture and parameter schema, the same pretraining protocol, and the "
            "same content-group risk reduction. When a flow checkpoint is supplied, "
            "the semantic/content/structural packet module must also have an identical "
            "parameter schema and flow aggregation must begin after packet_to_flow_proj; "
            "trained parameter values may differ."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet_checkpoint", required=True)
    parser.add_argument("--native_checkpoint", required=True)
    parser.add_argument(
        "--flow_checkpoint",
        required=True,
        help="Tower2 seq checkpoint whose packet representation must match exactly.",
    )
    parser.add_argument("--packet_result_json", default="")
    parser.add_argument("--flow_manifest_json", default="")
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    packet = torch.load(args.packet_checkpoint, map_location="cpu", weights_only=False)
    native = torch.load(args.native_checkpoint, map_location="cpu", weights_only=False)
    flow = torch.load(args.flow_checkpoint, map_location="cpu", weights_only=False)
    native_hash = hashlib.sha256()
    with open(args.native_checkpoint, "rb") as native_file:
        for chunk in iter(lambda: native_file.read(1024 * 1024), b""):
            native_hash.update(chunk)
    report = audit_shared_core(
        packet,
        native,
        flow_checkpoint=flow,
        packet_group_reduction=load_group_reduction(args.packet_result_json, "packet"),
        flow_group_reduction=load_group_reduction(args.flow_manifest_json, "flow"),
        native_checkpoint_sha256=native_hash.hexdigest(),
    )
    report["inputs"] = {
        "packet_checkpoint": args.packet_checkpoint,
        "native_checkpoint": args.native_checkpoint,
        "flow_checkpoint": args.flow_checkpoint,
        "packet_result_json": args.packet_result_json,
        "flow_manifest_json": args.flow_manifest_json,
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "architecture_match", "parameter_schema_match", "shared_pretraining_protocol_match", "content_risk_protocol_match", "exact_shared_representation_match")}, sort_keys=True))


if __name__ == "__main__":
    main()
