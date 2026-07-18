#!/usr/bin/env python3
"""Concatenate deterministic packet-evaluation probability shards."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--output_npz", required=True)
    args = ap.parse_args()

    arrays = [np.load(path) for path in args.inputs]
    class_counts = {data["probabilities"].shape[1] for data in arrays}
    if len(class_counts) != 1:
        raise ValueError(f"class-count mismatch across shards: {sorted(class_counts)}")
    has_ranges = [all(key in data for key in ("sample_start", "sample_stop", "dataset_size")) for data in arrays]
    if any(has_ranges) and not all(has_ranges):
        raise ValueError("range metadata is present in only some shards")
    if all(has_ranges):
        dataset_sizes = {int(data["dataset_size"]) for data in arrays}
        if len(dataset_sizes) != 1:
            raise ValueError(f"dataset-size mismatch across shards: {sorted(dataset_sizes)}")
        expected_start = 0
        for path, data in zip(args.inputs, arrays):
            start = int(data["sample_start"])
            stop = int(data["sample_stop"])
            if start != expected_start or stop - start != len(data["y_true"]):
                raise ValueError(
                    f"non-contiguous shard {path}: range=[{start},{stop}), expected start={expected_start}, rows={len(data['y_true'])}"
                )
            expected_start = stop
        dataset_size = next(iter(dataset_sizes))
        if expected_start != dataset_size:
            raise ValueError(f"incomplete shard coverage: stopped at {expected_start}, dataset_size={dataset_size}")
    else:
        print("warning: legacy shards have no range metadata; preserving caller-provided order", flush=True)
    y_true = np.concatenate([data["y_true"].astype(np.int64) for data in arrays])
    probabilities = np.concatenate([data["probabilities"].astype(np.float32) for data in arrays])
    output = Path(args.output_npz)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, y_true=y_true, probabilities=probabilities)
    print(f"saved {len(y_true)} ordered packets to {output}")


if __name__ == "__main__":
    main()
