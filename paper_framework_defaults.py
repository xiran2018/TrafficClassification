#!/usr/bin/env python3
from __future__ import annotations

from unified_framework_spec import (
    ABLATION_ONLY_MODULES,
    FLOW_LEVEL_RESULTS,
    FRAMEWORK_PROFILES,
    MODEL_SHARED_CORE_MODULES,
    PAPER_MAIN_MODULES,
    SHARED_CORE_MODULES,
    SHARED_PROTOCOL_GUARDS,
    UNIFIED_CANDIDATE_EXPERTS,
)


DEFAULT_FRAMEWORK_PROFILE = "paper_unified"


DEFAULT_TARGETS = {
    "vpn-app": (0.75, 0.65),
    "tls-120": (0.78, 0.70),
    "tls": (0.78, 0.70),
}

DEFAULT_FLOW_DATASETS = tuple(FLOW_LEVEL_RESULTS)

DEFAULT_PAPER_SAFE_RESULTS = {
    dataset: spec.path for dataset, spec in FLOW_LEVEL_RESULTS.items()
}

DEFAULT_SHARED_CORE_MODULES = SHARED_CORE_MODULES
DEFAULT_MODEL_SHARED_CORE_MODULES = MODEL_SHARED_CORE_MODULES
DEFAULT_SHARED_PROTOCOL_GUARDS = SHARED_PROTOCOL_GUARDS
DEFAULT_PAPER_MAIN_MODULES = PAPER_MAIN_MODULES
DEFAULT_UNIFIED_CANDIDATE_EXPERTS = UNIFIED_CANDIDATE_EXPERTS
DEFAULT_ABLATION_ONLY_MODULES = ABLATION_ONLY_MODULES
DEFAULT_FRAMEWORK_PROFILE_DESCRIPTION = FRAMEWORK_PROFILES[DEFAULT_FRAMEWORK_PROFILE]["description"]

DEFAULT_UNIFIED_EXPERT_SLOTS = [
    "base",
    "graph",
    "seq",
    "prior_base",
    "emb_lr",
    "emb_et",
    "proto_emb",
    "paired",
    "slot_stacker",
    "soft_gate",
]

DEFAULT_UNIFIED_EXPERT_SLOTS_CSV = ",".join(DEFAULT_UNIFIED_EXPERT_SLOTS)


def default_framework_results() -> list[tuple[str, str, float | None, float | None]]:
    return [
        (
            "vpn-app",
            DEFAULT_PAPER_SAFE_RESULTS["vpn-app"],
            DEFAULT_TARGETS["vpn-app"][0],
            DEFAULT_TARGETS["vpn-app"][1],
        ),
        (
            "tls-120",
            DEFAULT_PAPER_SAFE_RESULTS["tls-120"],
            DEFAULT_TARGETS["tls-120"][0],
            DEFAULT_TARGETS["tls-120"][1],
        ),
    ]
