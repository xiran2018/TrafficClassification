#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard_dir", action="append", required=True, help="Shard output directory containing flow_embedding_index.jsonl.")
    ap.add_argument("--output_dir", required=True, help="Merged output directory. NPY files remain in shard directories.")
    args = ap.parse_args()

    rows = {}
    for raw_dir in args.shard_dir:
        index_path = Path(raw_dir) / "flow_embedding_index.jsonl"
        if not index_path.exists():
            raise FileNotFoundError(index_path)
        with open(index_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                flow_id = str(row["flow_id"])
                if flow_id in rows:
                    raise ValueError(f"Duplicate flow_id across shards: {flow_id}")
                rows[flow_id] = row

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "flow_embedding_index.jsonl", "w", encoding="utf-8") as f:
        for flow_id in sorted(rows):
            f.write(json.dumps(rows[flow_id], ensure_ascii=False) + "\n")
    with open(output_dir / "merge_manifest.json", "w", encoding="utf-8") as f:
        json.dump({"num_flows": len(rows), "shard_dirs": args.shard_dir}, f, indent=2)
    print(f"merged {len(rows)} flows to {output_dir / 'flow_embedding_index.jsonl'}")


if __name__ == "__main__":
    main()
