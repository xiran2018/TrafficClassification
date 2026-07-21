import unittest

import torch

from train_tower2 import (
    class_conditional_environment_alignment_loss,
    environment_risk_variance_loss,
    iter_balanced_group_batches,
    split_flow_groups,
)


class EnvironmentGeneralizationTest(unittest.TestCase):
    def test_risk_variance_penalizes_environment_gap(self):
        logits = torch.tensor([[4.0, 0.0], [4.0, 0.0], [4.0, 0.0], [4.0, 0.0]], requires_grad=True)
        labels = torch.tensor([0, 0, 1, 1])
        environments = torch.tensor([0, 0, 1, 1])
        loss = environment_risk_variance_loss(logits, labels, environments, None, 1.0)
        self.assertGreater(float(loss.detach()), 0.0)
        loss.backward()
        self.assertIsNotNone(logits.grad)

    def test_class_alignment_is_zero_for_matching_means(self):
        embeddings = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
        labels = torch.tensor([0, 0, 1, 1])
        environments = torch.tensor([0, 1, 0, 1])
        loss = class_conditional_environment_alignment_loss(embeddings, labels, environments, 1.0)
        self.assertAlmostEqual(float(loss), 0.0, places=6)

    def test_balanced_sampler_draws_both_environments_per_class(self):
        groups = [
            {"flow_id": f"{label}-{env}", "label": label, "environment": env, "items": [{}]}
            for label in range(2) for env in range(2)
        ]
        batch = next(iter(iter_balanced_group_batches(groups, batch_size=4, classes_per_batch=2, samples_per_class=2)))
        for label in range(2):
            self.assertEqual({item["environment"] for item in batch if item["label"] == label}, {0, 1})

    def test_content_group_split_keeps_duplicate_content_on_one_side(self):
        groups = [
            {"flow_id": "a", "label": 0, "content_group_id": 10, "items": [{}]},
            {"flow_id": "b", "label": 0, "content_group_id": 10, "items": [{}]},
            {"flow_id": "c", "label": 1, "content_group_id": 20, "items": [{}]},
            {"flow_id": "d", "label": 1, "content_group_id": 30, "items": [{}]},
        ]

        train, valid = split_flow_groups(groups, valid_ratio=0.5, seed=3, group_key="content_group_id")

        train_groups = {item["content_group_id"] for item in train}
        valid_groups = {item["content_group_id"] for item in valid}
        self.assertFalse(train_groups & valid_groups)
        self.assertEqual(len(train) + len(valid), len(groups))

    def test_content_unique_balanced_sampler_avoids_duplicate_exact_content_when_possible(self):
        groups = [
            {"flow_id": "a0", "label": 0, "content_group_id": 10, "items": [{}]},
            {"flow_id": "a1", "label": 0, "content_group_id": 10, "items": [{}]},
            {"flow_id": "a2", "label": 0, "content_group_id": 11, "items": [{}]},
            {"flow_id": "b0", "label": 1, "content_group_id": 20, "items": [{}]},
            {"flow_id": "b1", "label": 1, "content_group_id": 21, "items": [{}]},
        ]

        batch = next(
            iter(
                iter_balanced_group_batches(
                    groups,
                    batch_size=4,
                    classes_per_batch=2,
                    samples_per_class=2,
                    content_group_unique=True,
                )
            )
        )

        content_groups = [item["content_group_id"] for item in batch]
        self.assertEqual(len(content_groups), len(set(content_groups)))


if __name__ == "__main__":
    unittest.main()
