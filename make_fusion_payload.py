#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--valid_json", required=True)
    ap.add_argument("--test_json", required=True)
    ap.add_argument("--output_json", required=True)
    args = ap.parse_args()

    with open(args.valid_json, "r", encoding="utf-8") as f:
        valid = json.load(f)
    with open(args.test_json, "r", encoding="utf-8") as f:
        test = json.load(f)
    required = ["flow_ids", "flow_y_true", "flow_prob"]
    missing_valid = [k for k in required if k not in valid]
    missing_test = [k for k in required if k not in test]
    if missing_valid or missing_test:
        raise ValueError(f"Missing fields valid={missing_valid} test={missing_test}")
    payload = {
        "valid_flow_ids": valid["flow_ids"],
        "valid_y_true": valid["flow_y_true"],
        "valid_prob": valid["flow_prob"],
        "flow_ids": test["flow_ids"],
        "flow_y_true": test["flow_y_true"],
        "flow_prob": test["flow_prob"],
        "valid_json": args.valid_json,
        "test_json": args.test_json,
        "label_map": test.get("label_map") or valid.get("label_map"),
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {args.output_json}")


if __name__ == "__main__":
    main()
