import unittest

import numpy as np

from adapt_target_prototypes import (
    adapt_probabilities,
    align_rows,
    leave_one_out_prototype_prob,
    source_prototypes,
)


class TargetPrototypeAdaptationTest(unittest.TestCase):
    def test_align_rows_uses_flow_ids(self):
        values = align_rows(["b", "a"], ["a", "b"], [[0.9, 0.1], [0.2, 0.8]])
        np.testing.assert_allclose(values, [[0.2, 0.8], [0.9, 0.1]])

    def test_source_prototypes_are_normalized(self):
        features = np.asarray([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [0.2, 0.8]], dtype=np.float32)
        labels = np.asarray([0, 0, 1, 1])
        prototypes = source_prototypes(features, labels, 2)
        np.testing.assert_allclose(np.linalg.norm(prototypes, axis=1), 1.0, atol=1e-6)

    def test_leave_one_out_does_not_use_sample_as_its_own_prototype(self):
        features = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        pseudo_prob = np.asarray([[0.99, 0.01], [0.99, 0.01]], dtype=np.float32)
        source = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
        prob, diagnostics = leave_one_out_prototype_prob(
            features, pseudo_prob, source, threshold=0.9, topk=4, pseudo_power=1.0,
            target_weight=1.0, prototype_temperature=0.1
        )
        self.assertLess(prob[0, 0], prob[0, 1])
        self.assertEqual(diagnostics["selected_per_class"], [2, 0])

    def test_adaptation_returns_normalized_finite_probabilities(self):
        features = np.asarray([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [0.2, 0.8]], dtype=np.float32)
        base = np.asarray([[0.8, 0.2], [0.7, 0.3], [0.1, 0.9], [0.3, 0.7]], dtype=np.float32)
        source = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        config = {
            "base_temperature": 1.0,
            "confidence_threshold": 0.6,
            "topk_per_class": 4,
            "pseudo_power": 1.0,
            "target_prototype_weight": 0.5,
            "prototype_temperature": 0.1,
            "prototype_weight": 0.5,
            "uncertainty_power": 1.0,
            "iterations": 2,
        }
        prob, diagnostics = adapt_probabilities(features, base, source, config)
        self.assertTrue(np.isfinite(prob).all())
        np.testing.assert_allclose(prob.sum(axis=1), 1.0, atol=1e-6)
        self.assertIn("iteration_2", diagnostics)


if __name__ == "__main__":
    unittest.main()
