import unittest

import numpy as np

from evaluate_content_unique_predictions import aggregate_unique_content


class ContentUniqueEvaluationTest(unittest.TestCase):
    def test_duplicate_probabilities_are_averaged_once(self):
        hashes, labels, prob, audit = aggregate_unique_content(
            ["a", "b", "c"],
            [0, 0, 1],
            np.asarray([[0.9, 0.1], [0.7, 0.3], [0.2, 0.8]]),
            {"a": "same", "b": "same", "c": "other"},
        )
        self.assertEqual(hashes, ["same", "other"])
        self.assertEqual(labels.tolist(), [0, 1])
        np.testing.assert_allclose(prob[0], [0.8, 0.2])
        self.assertEqual(audit["duplicate_rows_removed"], 1)

    def test_conflicting_duplicate_labels_are_rejected(self):
        with self.assertRaises(ValueError):
            aggregate_unique_content(
                ["a", "b"], [0, 1], np.asarray([[0.9, 0.1], [0.1, 0.9]]), {"a": "same", "b": "same"}
            )


if __name__ == "__main__":
    unittest.main()
