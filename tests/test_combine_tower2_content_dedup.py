import tempfile
import unittest
from pathlib import Path

import torch

from combine_tower2_datasets import merge_items


class Tower2ContentDedupTest(unittest.TestCase):
    def test_content_duplicate_drops_second_flow_but_keeps_first_windows(self):
        items = [
            {"flow_id": "a", "window": (0, 32), "label": 1},
            {"flow_id": "a", "window": (16, 48), "label": 1},
            {"flow_id": "b", "window": (0, 32), "label": 1},
            {"flow_id": "b", "window": (16, 48), "label": 1},
            {"flow_id": "c", "window": (0, 32), "label": 2},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dataset.pt"
            torch.save(items, path)
            merged, stats = merge_items(
                [str(path)], set(), {"a": "same", "b": "same", "c": "other"}, set(), True
            )
        self.assertEqual([item["flow_id"] for item in merged], ["a", "a", "c"])
        self.assertEqual(stats["content_duplicate_flows"], 1)
        self.assertEqual(stats["content_duplicate_windows"], 2)

    def test_content_label_conflict_is_rejected(self):
        items = [
            {"flow_id": "a", "window": (0, 32), "label": 1},
            {"flow_id": "b", "window": (0, 32), "label": 2},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dataset.pt"
            torch.save(items, path)
            with self.assertRaises(ValueError):
                merge_items([str(path)], set(), {"a": "same", "b": "same"}, set(), True)


if __name__ == "__main__":
    unittest.main()
