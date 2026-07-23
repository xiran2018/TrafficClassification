import numpy as np

from train_cross_environment_reliability_router import (
    aligned_test_consensus,
    aligned_split,
    normalize_prob,
    router_features,
    safe_prior_transport,
)


def test_router_features_are_class_aware_and_finite():
    semantic = np.asarray([[0.8, 0.2], [0.3, 0.7]], dtype=np.float32)
    structural = np.asarray([[0.6, 0.4], [0.9, 0.1]], dtype=np.float32)
    features = router_features(semantic, structural)
    assert features.shape == (2, 16)
    assert np.isfinite(features).all()
    assert features[0, 0] != features[1, 0]


def test_normalize_prob_handles_unnormalized_positive_scores():
    values = normalize_prob([[2.0, 1.0], [0.0, 3.0]])
    np.testing.assert_allclose(values.sum(axis=1), np.ones(2), atol=1e-6)
    assert (values > 0).all()


def test_aligned_split_accepts_standard_valid_payload_fields():
    payload = {
        "valid_flow_ids": ["b", "a"],
        "valid_y_true": [1, 0],
        "valid_prob": [[0.1, 0.9], [0.8, 0.2]],
    }
    labels, probability = aligned_split(payload, "valid", ["a", "b"])
    assert labels.tolist() == [0, 1]
    assert probability.argmax(axis=1).tolist() == [0, 1]


def test_aligned_test_consensus_accepts_prob_and_gate_tuple():
    outputs = [
        (["a"], np.asarray([0]), np.asarray([[0.8, 0.2]]), np.asarray([0.4])),
        (["a"], np.asarray([0]), np.asarray([[0.6, 0.4]]), np.asarray([0.7])),
    ]
    ids, labels, probability, gates = aligned_test_consensus(outputs)
    assert ids == ["a"]
    assert labels.tolist() == [0]
    assert probability.argmax(axis=1).tolist() == [0]
    assert gates.shape == (1, 2)


def test_safe_prior_transport_can_be_disabled():
    y = np.asarray([0, 1])
    valid = np.asarray([[0.8, 0.2], [0.2, 0.8]], dtype=np.float32)
    test = np.asarray([[0.7, 0.3]], dtype=np.float32)
    valid_out, test_out, report = safe_prior_transport(y, valid, test, 0.0, 0.05)
    np.testing.assert_allclose(valid_out, valid)
    np.testing.assert_allclose(test_out, test)
    assert report["enabled"] is False
