import numpy as np

from train_cross_environment_reliability_router import (
    aligned_test_consensus,
    aligned_split,
    input_provenance,
    load_packet_environment,
    normalize_prob,
    packet_split,
    router_features,
    safe_prior_transport,
    save_packet_probabilities,
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


def test_safe_prior_transport_rejects_macro_f1_regression():
    y = np.asarray([0, 0, 1, 1])
    valid = np.asarray(
        [[0.55, 0.45], [0.55, 0.45], [0.45, 0.55], [0.45, 0.55]],
        dtype=np.float32,
    )
    test = np.asarray([[0.9, 0.1], [0.85, 0.15], [0.8, 0.2]], dtype=np.float32)
    _, _, report = safe_prior_transport(y, valid, test, 0.3, 0.05)
    assert report["selected_strength"] == 0.0
    assert report["reports"][1]["macro_f1"] < report["reports"][0]["macro_f1"]


def test_packet_split_uses_packet_uids_when_available(tmp_path):
    path = tmp_path / "packet.npz"
    np.savez(
        path,
        y_true=np.asarray([0, 1]),
        probabilities=np.asarray([[0.8, 0.2], [0.1, 0.9]], dtype=np.float32),
        packet_uids=np.asarray(["flow-a_0", "flow-b_0"]),
    )
    ids, labels, probability = packet_split(str(path))
    assert ids == ["flow-a_0", "flow-b_0"]
    assert labels.tolist() == [0, 1]
    assert probability.argmax(axis=1).tolist() == [0, 1]


def test_input_provenance_records_paths_and_hashes(tmp_path):
    paths = []
    for index in range(4):
        path = tmp_path / f"artifact_{index}.npz"
        path.write_bytes(f"artifact-{index}".encode())
        paths.append(str(path))
    records = input_provenance([["fold0", *paths]], "packet_level")
    assert records[0]["environment"] == "fold0"
    assert set(records[0]["artifacts"]) == {
        "semantic_valid",
        "semantic_test",
        "structural_valid",
        "structural_test",
    }
    for role, path in zip(records[0]["artifacts"], paths, strict=True):
        artifact = records[0]["artifacts"][role]
        assert artifact["path"] == path
        assert len(artifact["sha256"]) == 64


def test_save_packet_probabilities_preserves_exact_identities(tmp_path):
    path = tmp_path / "routed.npz"
    save_packet_probabilities(
        path,
        np.asarray([0, 1]),
        np.asarray([[0.8, 0.2], [0.1, 0.9]], dtype=np.float32),
        np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
        ["flow-a_0", "flow-b_0"],
    )
    with np.load(path, allow_pickle=False) as payload:
        assert payload["packet_uids"].tolist() == ["flow-a_0", "flow-b_0"]


def test_load_packet_environment_rejects_expert_label_mismatch(tmp_path):
    semantic_valid = tmp_path / "semantic_valid.npz"
    semantic_test = tmp_path / "semantic_test.npz"
    structural_valid = tmp_path / "structural_valid.npz"
    structural_test = tmp_path / "structural_test.npz"
    probability = np.asarray([[0.8, 0.2], [0.1, 0.9]], dtype=np.float32)
    np.savez(semantic_valid, y_true=np.asarray([0, 1]), probabilities=probability)
    np.savez(semantic_test, y_true=np.asarray([0, 1]), probabilities=probability)
    np.savez(structural_valid, y_true=np.asarray([0, 0]), probabilities=probability)
    np.savez(structural_test, y_true=np.asarray([0, 1]), probabilities=probability)
    try:
        load_packet_environment(
            [
                "fold0",
                str(semantic_valid),
                str(semantic_test),
                str(structural_valid),
                str(structural_test),
            ]
        )
    except ValueError as error:
        assert "validation labels differ" in str(error)
    else:
        raise AssertionError("mismatched packet labels must be rejected")
