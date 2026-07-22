#!/usr/bin/env python3
"""Preregister the paired factual/intervention shared-core candidate."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from freeze_shared_core_v2_config import canonical_sha256
from shared_core_v2 import DEVELOPMENT_STATUS, load_frozen_shared_core


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_paired_candidate(
    base: dict[str, Any], *, source_path: Path, source_file_sha256: str
) -> dict[str, Any]:
    tower1 = base["tower1"]
    if (
        float(tower1.get("paired_consistency_weight", -1.0)) != 0.0
        or float(tower1.get("paired_cls_weight", -1.0)) != 0.0
        or tower1.get("paired_validation_selection", "disabled") != "disabled"
    ):
        raise ValueError("paired candidate source must be the disabled control")

    candidate = copy.deepcopy(base)
    candidate.pop("config_sha256", None)
    candidate["status"] = DEVELOPMENT_STATUS
    candidate_tower1 = candidate["tower1"]
    candidate_tower1.update(
        {
            "paired_consistency_weight": 0.05,
            "paired_cls_weight": 0.2,
            "paired_logit_kl_weight": 0.5,
            "paired_raw_consistency_weight": 1.0,
            "paired_validation_selection": "worst_view_macro_f1",
        }
    )
    candidate.setdefault("selection_evidence", {})["paired_invariance"] = {
        "status": "pre_registered_candidate_pending_validation",
        "source_config_path": str(source_path.resolve()),
        "source_config_file_sha256": source_file_sha256,
        "source_method_sha256": base["config_sha256"],
        "intervention": "mask_ip_port",
        "checkpoint_selection": "worst_view_macro_f1",
        "test_labels_used": False,
        "promotion_rule": (
            "the same paired candidate must improve heldout macro_f1 by at least "
            "0.005 on vpn-app and tls-120 without accuracy drop greater than 0.005"
        ),
    }
    candidate["method_selection"] = {
        "scope": "pre_registered_before_candidate_validation",
        "decision_status": "candidate_pending_vpn_tls_validation",
        "selected_method": "paired_factual_masked_robust_validation_candidate",
        "test_labels_used": False,
        "unbiased_final_claim_allowed": False,
    }
    candidate["config_sha256"] = canonical_sha256(candidate)
    return candidate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_config", required=True)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    source = Path(args.base_config)
    base = load_frozen_shared_core(source)
    candidate = build_paired_candidate(
        base,
        source_path=source,
        source_file_sha256=file_sha256(source),
    )
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(candidate, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"saved paired shared-core candidate to {output}")
    print(f"config_sha256={candidate['config_sha256']}")


if __name__ == "__main__":
    main()
