import json
import tempfile
import unittest
from pathlib import Path

from run_stage8_flowaware_pipeline import expected_embedding_shard_counts, jsonl_row_count


class EmbeddingShardResumeTest(unittest.TestCase):
    def test_expected_counts_count_grouped_flows_once(self):
        rows = [
            {"flow_id": "flow-a", "packet_id": 0},
            {"flow_id": "flow-b", "packet_id": 0},
            {"flow_id": "flow-a", "packet_id": 1},
            {"flow_id": "flow-c", "packet_id": 0},
            {"flow_id": "flow-c", "packet_id": 1},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "packet_index.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            counts = expected_embedding_shard_counts(path, 4)
            self.assertEqual(sum(counts), 3)

    def test_jsonl_row_count_ignores_blank_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rows.jsonl"
            path.write_text("{}\n\n{}\n", encoding="utf-8")
            self.assertEqual(jsonl_row_count(path), 2)


if __name__ == "__main__":
    unittest.main()
