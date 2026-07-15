#!/usr/bin/env python3
from __future__ import annotations


DEFAULT_TARGETS = {
    "vpn-app": (0.74, 0.65),
    "tls-120": (0.78, 0.70),
    "tls": (0.78, 0.70),
}

DEFAULT_PAPER_SAFE_RESULTS = {
    "vpn-app": "reasoningDataset/vpn-app/test_selector_best_prior_embedding_experts_calib_shift000_valid_macro.json",
    "tls-120": "reasoningDataset/tls-120/test_selector_unified_slot_stacker_tls120_valid_macro.json",
    "ustc-app": "reasoningDataset/ustc-app/test_selector_base_flowproto_full_s200_w002_step150_calib_shift005_valid_macro.json",
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
        (
            "ustc-app",
            DEFAULT_PAPER_SAFE_RESULTS["ustc-app"],
            None,
            None,
        ),
    ]
