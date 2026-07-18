import unittest

from paper_framework_defaults import (
    DEFAULT_FLOW_DATASETS,
    DEFAULT_PAPER_SAFE_RESULTS,
    default_framework_results,
)


class PaperDatasetScopeTest(unittest.TestCase):
    def test_default_flow_scope_excludes_per_packet_split_datasets(self):
        self.assertEqual(DEFAULT_FLOW_DATASETS, ("vpn-app", "tls-120"))
        self.assertEqual([row[0] for row in default_framework_results()], list(DEFAULT_FLOW_DATASETS))
        self.assertNotIn("ustc-app", DEFAULT_PAPER_SAFE_RESULTS)


if __name__ == "__main__":
    unittest.main()
