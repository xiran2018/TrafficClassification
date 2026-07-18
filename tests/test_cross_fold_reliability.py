import unittest

import numpy as np

from cross_fold_consensus import (
    confusion_em_fusion,
    fuse_probs,
    validation_class_reliability,
    validation_confusion_likelihood,
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


if __name__ == "__main__":
    unittest.main()
