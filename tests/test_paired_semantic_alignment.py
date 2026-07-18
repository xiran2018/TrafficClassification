import unittest
from types import SimpleNamespace

import torch

from train_tower2 import paired_semantic_alignment_loss


def loss_args(**overrides):
    values = {
        "paired_alignment_weight": 0.1,
        "paired_crossview_contrastive_weight": 0.05,
        "paired_crossview_temperature": 0.07,
        "paired_variance_weight": 0.1,
        "paired_variance_target": 0.04,
        "paired_covariance_weight": 0.01,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class PairedSemanticAlignmentTest(unittest.TestCase):
    def test_loss_is_finite_and_differentiable(self):
        clean = torch.randn(4, 16, requires_grad=True)
        paired = (clean.detach() + 0.1 * torch.randn(4, 16)).requires_grad_(True)
        labels = torch.tensor([0, 0, 1, 1])
        loss = paired_semantic_alignment_loss(clean, paired, labels, loss_args())
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(clean.grad)
        self.assertIsNotNone(paired.grad)

    def test_missing_pair_is_differentiable_zero(self):
        clean = torch.randn(4, 8, requires_grad=True)
        labels = torch.tensor([0, 0, 1, 1])
        loss = paired_semantic_alignment_loss(clean, None, labels, loss_args())
        self.assertEqual(float(loss.detach()), 0.0)
        loss.backward()
        self.assertIsNotNone(clean.grad)

    def test_variance_term_penalizes_collapsed_embeddings(self):
        clean = torch.ones(4, 8, requires_grad=True)
        paired = torch.ones(4, 8, requires_grad=True)
        labels = torch.tensor([0, 0, 1, 1])
        args = loss_args(
            paired_alignment_weight=0.0,
            paired_crossview_contrastive_weight=0.0,
            paired_variance_weight=1.0,
            paired_covariance_weight=0.0,
        )
        loss = paired_semantic_alignment_loss(clean, paired, labels, args)
        self.assertGreater(float(loss.detach()), 0.0)


if __name__ == "__main__":
    unittest.main()
