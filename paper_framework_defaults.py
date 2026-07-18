#!/usr/bin/env python3
from __future__ import annotations


DEFAULT_TARGETS = {
    "vpn-app": (0.74, 0.65),
    "tls-120": (0.78, 0.70),
    "tls": (0.78, 0.70),
}

DEFAULT_FLOW_DATASETS = ("vpn-app", "tls-120")

DEFAULT_PAPER_SAFE_RESULTS = {
    "vpn-app": "reasoningDataset/vpn-app/test_selector_best_prior_embedding_experts_calib_shift000_valid_macro.json",
    "tls-120": "reasoningDataset/tls-120/test_selector_soft_gate_tls120_tol0015_calib_family_valid_macro.json",
}

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
