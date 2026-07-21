import unittest

import numpy as np

from cross_fold_consensus import (
    confusion_em_fusion,
    fuse_probs,
    validation_class_reliability,
    validation_confusion_likelihood,
    selective_anchor_vote,
    validation_selective_threshold,
)


class CrossFoldReliabilityTest(unittest.TestCase):
    def test_validation_reliability_prefers_well_classified_class(self):
        payload = {
            "valid_y_true": [0, 0, 1, 1],
            "valid_prob": [[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4]],
        }
        reliability = validation_class_reliability(payload, 2, smoothing=1.0)
        self.assertGreater(reliability[0], reliability[1])

    def test_reliability_vote_uses_validation_weight(self):
        probs = [
            np.asarray([[0.9, 0.1]], dtype=np.float64),
            np.asarray([[0.2, 0.8]], dtype=np.float64),
        ]
        reliability = [np.asarray([0.9, 0.2]), np.asarray([0.2, 0.8])]
        fused = fuse_probs(probs, "class_reliability_vote", reliability)
        self.assertEqual(int(fused.argmax(axis=1)[0]), 0)
        np.testing.assert_allclose(fused.sum(axis=1), 1.0)

    def test_confusion_em_corrects_a_systematic_label_swap(self):
        first = {
            "valid_y_true": [0, 0, 1, 1],
            "valid_prob": [[0.05, 0.95], [0.05, 0.95], [0.95, 0.05], [0.95, 0.05]],
        }
        second = {
            "valid_y_true": [0, 0, 1, 1],
            "valid_prob": [[0.95, 0.05], [0.95, 0.05], [0.05, 0.95], [0.05, 0.95]],
        }
        confusions = [
            validation_confusion_likelihood(first, 2, smoothing=0.01),
            validation_confusion_likelihood(second, 2, smoothing=0.01),
        ]
        target_probs = [
            np.asarray([[0.05, 0.95], [0.95, 0.05]]),
            np.asarray([[0.95, 0.05], [0.05, 0.95]]),
        ]
        posterior, prior, iterations = confusion_em_fusion(target_probs, confusions, prior_anchor_weight=0.1)
        self.assertEqual(posterior.argmax(axis=1).tolist(), [0, 1])
        np.testing.assert_allclose(posterior.sum(axis=1), 1.0)
        np.testing.assert_allclose(prior.sum(), 1.0)
        self.assertGreater(iterations, 0)

    def test_selective_anchor_threshold_uses_validation_only(self):
        payload = {
            "valid_y_true": [0, 1, 1, 0],
            "valid_prob": [
                [0.99, 0.01],
                [0.02, 0.98],
                [0.80, 0.20],
                [0.55, 0.45],
            ],
        }
        threshold, report = validation_selective_threshold(
            payload, min_precision=1.0, min_coverage=0.25
        )
        self.assertAlmostEqual(threshold, 0.98)
        self.assertEqual(report["selected_count"], 2)
        self.assertAlmostEqual(report["validation_precision"], 1.0)

    def test_selective_anchor_overrides_vote_only_above_threshold(self):
        anchor = np.asarray([[0.95, 0.05], [0.55, 0.45]])
        weak_a = np.asarray([[0.1, 0.9], [0.1, 0.9]])
        weak_b = np.asarray([[0.2, 0.8], [0.2, 0.8]])
        fused, count = selective_anchor_vote([anchor, weak_a, weak_b], 0.9)
        self.assertEqual(count, 1)
        self.assertEqual(fused.argmax(axis=1).tolist(), [0, 1])


if __name__ == "__main__":
    unittest.main()
