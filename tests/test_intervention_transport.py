import pytest
import torch

from models.intervention_transport import (
    LowRankInterventionTransport,
    transport_alignment_loss,
)
from train_intervention_transport import align_packet_embeddings


def sample(flow_id, label, window, values):
    return {
        "flow_id": flow_id,
        "label": label,
        "window": window,
        "x": torch.tensor(values, dtype=torch.float32),
    }


def test_transport_is_identity_initialized_and_trainable():
    model = LowRankInterventionTransport(8, rank=3)
    x = torch.randn(5, 8)
    assert torch.equal(model(x), x)
    target = torch.randn(5, 8)
    loss, parts = transport_alignment_loss(model(x), target)
    loss.backward()
    assert parts["cosine_loss"].item() >= 0
    assert model.up.weight.grad is not None
    assert model.up.weight.grad.abs().sum().item() > 0


def test_packet_alignment_is_keyed_and_excludes_trailing_metadata():
    clean = [sample("f", 1, (0, 2), [[1, 2, 9], [3, 4, 9]])]
    paired = [sample("f", 1, (0, 2), [[5, 6, 8], [7, 8, 8]])]
    clean_x, paired_x, report = align_packet_embeddings(clean, paired, meta_dim=1)
    assert clean_x.tolist() == [[1, 2], [3, 4]]
    assert paired_x.tolist() == [[5, 6], [7, 8]]
    assert report == {
        "windows": 1,
        "flows": 1,
        "packets": 2,
        "embedding_dim": 2,
        "meta_dim": 1,
    }


def test_packet_alignment_rejects_shape_and_label_mismatch():
    clean = [sample("f", 1, (0, 2), [[1, 2, 9], [3, 4, 9]])]
    wrong_label = [sample("f", 2, (0, 2), [[1, 2, 9], [3, 4, 9]])]
    with pytest.raises(ValueError, match="label mismatch"):
        align_packet_embeddings(clean, wrong_label, meta_dim=1)
    wrong_shape = [sample("f", 1, (0, 2), [[1, 2, 9]])]
    with pytest.raises(ValueError, match="shape mismatch"):
        align_packet_embeddings(clean, wrong_shape, meta_dim=1)
