import unittest
from pathlib import Path

from summarize_experiment_results import skip_result_file
from summarize_cross_split_results import result_scope


class ResultScopeTest(unittest.TestCase):
    def test_explicit_consensus_scope(self):
        self.assertEqual(
            result_scope({"result_scope": "cross_fold_consensus"}),
            "cross_fold_consensus",
        )

    def test_legacy_consensus_config(self):
        self.assertEqual(
            result_scope({"config": {"num_inputs": 3, "requested_mode": "auto_confidence"}}),
            "cross_fold_consensus",
        )

    def test_multi_expert_selector_is_single_fold(self):
        payload = {
            "inputs": [
                {"name": "base", "path": "base.json"},
                {"name": "expert", "path": "expert.json"},
            ],
            "selector": {"strategy": "class_precision"},
        }
        self.assertEqual(result_scope(payload), "single_fold")

    def test_raw_result_scanner_skips_only_fold_named_consensus_alias(self):
        payload = {"config": {"num_inputs": 3, "requested_mode": "auto_confidence"}}
        self.assertTrue(skip_result_file(Path("test_crossfold_consensus_fold2.json"), payload))
        self.assertFalse(skip_result_file(Path("test_crossfold_consensus.json"), payload))
        self.assertFalse(
            skip_result_file(
                Path("test_independent_fold2.json"),
                {"config": {"num_inputs": 1}},
            )
        )


if __name__ == "__main__":
    unittest.main()
