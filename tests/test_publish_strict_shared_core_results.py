import json
import subprocess
import sys
from pathlib import Path

from publish_strict_shared_core_results import (
    archive_frozen_method_evidence,
    canonical_sha256,
    validated_audits,
)


ROOT = Path(__file__).resolve().parents[1]


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_audits_share_method_identity_despite_distinct_effective_configs(tmp_path):
    root = tmp_path / "audits"
    method = "a" * 64
    for fold in range(3):
        write_json(
            root / "vpn-app" / f"fold{fold}" / "audit.json",
            {
                "status": "pass",
                "dataset": "vpn-app",
                "fold": fold,
                "shared_core_method_sha256": method,
                "shared_core_config_sha256": None,
                "packet_effective_shared_core_config_sha256": f"packet-{fold}",
                "flow_effective_shared_core_config_sha256": f"flow-{fold}",
                "runtime_mechanism_evidence_required": True,
                "flow_native_extraction_evidence_required": True,
                "algorithm_source_evidence_required": True,
                "algorithm_source_evidence_verified": True,
                "algorithm_source_fingerprint": "f" * 64,
                "runtime_mechanism_evidence": {"status": "pass"},
                "flow_native_extraction_evidence": {"status": "pass"},
            },
        )

    paths, fingerprint = validated_audits("vpn-app", root)

    assert len(paths) == 3
    assert fingerprint == method


def prepare_inputs(
    tmp_path,
    *,
    packet_accuracy=0.91,
    packet_f1=0.8,
    flow_accuracy=0.8,
    flow_f1=0.72,
):
    balance = write_json(tmp_path / "balance_selection.json", {"selected": "baseline"})
    paired = write_json(tmp_path / "paired_selection.json", {"selected": "candidate"})
    config_payload = {
        "schema": "exact_shared_packet_core_v2",
        "status": "frozen_from_cross_dataset_validation",
        "selection_evidence": {
            "balance": {
                "path": str(balance),
                "sha256": file_sha256(balance),
            },
            "paired_invariance": {
                "path": str(paired),
                "sha256": file_sha256(paired),
            },
        },
    }
    import hashlib

    encoded = json.dumps(
        config_payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    fingerprint = hashlib.sha256(encoded).hexdigest()
    config_payload["config_sha256"] = fingerprint
    shared_core_config = write_json(tmp_path / "frozen_config.json", config_payload)
    audit_root = tmp_path / "audits"
    manifest_root = tmp_path / "manifests"
    for fold in range(3):
        flow_manifest = write_json(
            tmp_path / "flow_manifests" / f"fold{fold}.json",
            {
                "framework": {
                    "notes": {
                        "completed": True,
                        "shared_core_config_sha256": fingerprint,
                    }
                }
            },
        )
        write_json(
            audit_root / "vpn-app" / f"fold{fold}" / "audit.json",
            {
                "status": "pass",
                "dataset": "vpn-app",
                "fold": fold,
                "shared_core_config_sha256": fingerprint,
                "runtime_mechanism_evidence_required": True,
                "flow_native_extraction_evidence_required": True,
                "algorithm_source_evidence_required": True,
                "algorithm_source_evidence_verified": True,
                "algorithm_source_fingerprint": "f" * 64,
                "runtime_mechanism_evidence": {"status": "pass"},
                "flow_native_extraction_evidence": {"status": "pass"},
                "inputs": {"flow_manifest": str(flow_manifest)},
            },
        )
        write_json(
            manifest_root / "vpn-app" / f"fold{fold}" / "packet_framework_manifest.json",
            {
                "framework": {
                    "notes": {
                        "completed": True,
                        "shared_core_config_sha256": fingerprint,
                    }
                }
            },
        )
    packet = write_json(
        tmp_path / "packet.json",
        {
            "method": "log_mean",
            "inputs": [f"fold{fold}/test_unified_packet_single_head.npz" for fold in range(3)],
            "metrics": {"accuracy": packet_accuracy, "macro_f1": packet_f1},
        },
    )
    flow = write_json(
        tmp_path / "flow.json",
        {
            "config": {"requested_mode": "log_mean", "selected_mode": "log_mean"},
            "inputs": [
                {"path": f"fold{fold}/test_seq_metrics_flow_strict_probs.json"}
                for fold in range(3)
            ],
            "metrics": {
                "flow_level": {"accuracy": flow_accuracy, "macro_f1": flow_f1}
            },
        },
    )
    packet_bootstrap = write_json(
        tmp_path / "packet_bootstrap.json",
        {
            "task": "packet",
            "method": "class_stratified_flow_cluster_bootstrap",
            "bootstrap_samples": 5000,
            "num_flow_clusters": 100,
            "num_classes": 16,
            "metrics": {
                "accuracy": {
                    "point_estimate": packet_accuracy,
                    "bootstrap_95_ci": [packet_accuracy - 0.03, packet_accuracy + 0.02],
                },
                "macro_f1": {
                    "point_estimate": packet_f1,
                    "bootstrap_95_ci": [packet_f1 - 0.04, packet_f1 + 0.03],
                },
            },
        },
    )
    flow_bootstrap = write_json(
        tmp_path / "flow_bootstrap.json",
        {
            "task": "flow",
            "method": "class_stratified_flow_cluster_bootstrap",
            "bootstrap_samples": 5000,
            "num_flow_clusters": 80,
            "num_classes": 16,
            "metrics": {
                "accuracy": {
                    "point_estimate": flow_accuracy,
                    "bootstrap_95_ci": [flow_accuracy - 0.05, flow_accuracy + 0.05],
                },
                "macro_f1": {
                    "point_estimate": flow_f1,
                    "bootstrap_95_ci": [flow_f1 - 0.05, flow_f1 + 0.05],
                },
            },
        },
    )
    def novelty(task, accuracy, macro_f1):
        inputs = {}
        train_rows = []
        for fold in range(3):
            path = write_json(tmp_path / f"{task}_train_{fold}.json", {"fold": fold})
            train_rows.append({"path": str(path), "sha256": file_sha256(path)})
        for name in ("test_packet_index", "predictions", "label_map"):
            path = write_json(tmp_path / f"{task}_{name}.json", {"name": name})
            inputs[name] = {"path": str(path), "sha256": file_sha256(path)}
        inputs["train_packet_indices"] = train_rows
        return {
            "schema": "session_novelty_evaluation_v1",
            "task": task,
            "selection_role": "reporting_only_no_training_or_model_selection",
            "group_definition_uses_test_labels": False,
            "training_reference": "union_of_all_supplied_training_signature_sets",
            "groups": {"all": {"metrics": {"accuracy": accuracy, "macro_f1": macro_f1}}},
            "seen_minus_novel_gaps": {"endpoint": {}, "port": {}, "session": {}},
            "inputs": inputs,
        }

    packet_novelty = write_json(
        tmp_path / "packet_novelty.json", novelty("packet", packet_accuracy, packet_f1)
    )
    flow_novelty = write_json(
        tmp_path / "flow_novelty.json", novelty("flow", flow_accuracy, flow_f1)
    )
    return (
        audit_root,
        manifest_root,
        packet,
        flow,
        packet_bootstrap,
        flow_bootstrap,
        packet_novelty,
        flow_novelty,
        shared_core_config,
    )


def file_sha256(path):
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_publish(
    tmp_path,
    monkeypatch,
    *,
    packet_accuracy=0.91,
    packet_f1=0.8,
    flow_accuracy=0.8,
    flow_f1=0.72,
):
    (
        audit_root,
        manifest_root,
        packet,
        flow,
        packet_bootstrap,
        flow_bootstrap,
        packet_novelty,
        flow_novelty,
        shared_core_config,
    ) = prepare_inputs(
        tmp_path,
        packet_accuracy=packet_accuracy,
        packet_f1=packet_f1,
        flow_accuracy=flow_accuracy,
        flow_f1=flow_f1,
    )
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "publication.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "publish_strict_shared_core_results.py"),
            "--dataset",
            "vpn-app",
            "--audit_root",
            str(audit_root),
            "--packet_manifest_root",
            str(manifest_root),
            "--shared_core_config",
            str(shared_core_config),
            "--method_archive_root",
            str(tmp_path / "reasoningDataset/shared-core-v2"),
            "--packet_candidate",
            str(packet),
            "--flow_candidate",
            str(flow),
            "--packet_bootstrap",
            str(packet_bootstrap),
            "--flow_bootstrap",
            str(flow_bootstrap),
            "--packet_session_novelty",
            str(packet_novelty),
            "--flow_session_novelty",
            str(flow_novelty),
            "--output_json",
            str(output),
        ],
        check=True,
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    return json.loads(output.read_text(encoding="utf-8"))


def test_publish_requires_audits_and_promotes_fixed_consensus(tmp_path, monkeypatch):
    report = run_publish(tmp_path, monkeypatch)
    assert report["status"] == "published"
    assert report["packet"]["published"] is True
    assert report["flow"]["published"] is True
    assert Path(report["packet"]["canonical"]).is_file()
    assert Path(report["flow"]["canonical"]).is_file()
    assert len(report["canonical_packet_manifests"]) == 3
    assert len(report["canonical_flow_manifests"]) == 3
    assert report["uncertainty_evidence"]["packet"]["method"] == (
        "class_stratified_flow_cluster_bootstrap"
    )
    assert report["uncertainty_evidence"]["packet"]["sha256"] == file_sha256(
        report["uncertainty_evidence"]["packet"]["path"]
    )
    assert report["session_novelty_evidence"]["packet"]["path"].endswith(
        "packet_novelty.json"
    )
    canonical = json.loads(Path(report["packet"]["canonical"]).read_text())
    assert canonical["publication_provenance"]["status"] == "strict_shared_core_v2"
    assert canonical["publication_provenance"]["session_novelty_sha256"]
    assert Path(canonical["publication_provenance"]["session_novelty"]).is_file()
    bootstrap_path = canonical["publication_provenance"]["bootstrap_evidence"]
    assert Path(bootstrap_path).is_file()
    assert canonical["publication_provenance"][
        "bootstrap_evidence_sha256"
    ] == file_sha256(bootstrap_path)
    method_manifest = canonical["publication_provenance"]["method_archive_manifest"]
    assert Path(method_manifest).is_file()
    assert canonical["publication_provenance"][
        "method_archive_manifest_sha256"
    ] == file_sha256(method_manifest)
    audit_evidence = canonical["publication_provenance"]["audit_evidence"]
    assert len(audit_evidence) == 3
    assert len({row["path"] for row in audit_evidence}) == 3
    assert all(file_sha256(row["path"]) == row["sha256"] for row in audit_evidence)
    assert report["packet"]["candidate_sha256"] == file_sha256(
        report["packet"]["candidate"]
    )
    assert report["packet"]["canonical_sha256"] == file_sha256(
        report["packet"]["canonical"]
    )
    assert len(report["audit_evidence"]) == 3
    assert all(
        file_sha256(row["path"]) == row["sha256"]
        for row in report["audit_evidence"]
    )
    archive = report["frozen_method_evidence"]
    assert archive["status"] == "verified_and_archived"
    assert archive["schema"] == "strict_shared_core_v2_method_archive_v1"
    assert Path(archive["shared_core_config"]["archived_path"]).is_file()
    assert Path(archive["selection_evidence"]["balance"]["archived_path"]).is_file()


def test_flow_target_gap_does_not_replace_canonical_flow_result(tmp_path, monkeypatch):
    canonical = tmp_path / "reasoningDataset/vpn-app/test_crossfold_consensus_auto_confidence.json"
    write_json(canonical, {"sentinel": "preserve"})
    report = run_publish(
        tmp_path, monkeypatch, flow_accuracy=0.70, flow_f1=0.60
    )
    assert report["status"] == "packet_published_flow_target_gap"
    assert report["flow"]["published"] is False
    assert json.loads(canonical.read_text(encoding="utf-8")) == {"sentinel": "preserve"}


def test_packet_target_gap_does_not_replace_canonical_packet_result(tmp_path, monkeypatch):
    canonical = tmp_path / "reasoningDataset/packet-level/vpn-app/paper_default_result.json"
    write_json(canonical, {"sentinel": "preserve"})
    report = run_publish(
        tmp_path,
        monkeypatch,
        packet_accuracy=0.89,
        packet_f1=0.75,
    )
    assert report["status"] == "flow_published_packet_target_gap"
    assert report["packet"]["published"] is False
    assert report["flow"]["published"] is True
    assert json.loads(canonical.read_text(encoding="utf-8")) == {"sentinel": "preserve"}


def test_publish_rejects_session_novelty_metric_mismatch(tmp_path, monkeypatch):
    inputs = list(prepare_inputs(tmp_path))
    novelty_path = inputs[-2]
    novelty = json.loads(novelty_path.read_text(encoding="utf-8"))
    novelty["groups"]["all"]["metrics"]["accuracy"] = 0.1
    write_json(novelty_path, novelty)
    monkeypatch.chdir(tmp_path)
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "publish_strict_shared_core_results.py"),
            "--dataset", "vpn-app",
            "--audit_root", str(inputs[0]),
            "--packet_manifest_root", str(inputs[1]),
            "--shared_core_config", str(inputs[8]),
            "--method_archive_root", str(tmp_path / "reasoningDataset/shared-core-v2"),
            "--packet_candidate", str(inputs[2]),
            "--flow_candidate", str(inputs[3]),
            "--packet_bootstrap", str(inputs[4]),
            "--flow_bootstrap", str(inputs[5]),
            "--packet_session_novelty", str(inputs[6]),
            "--flow_session_novelty", str(inputs[7]),
            "--output_json", str(tmp_path / "publication.json"),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "overall metrics do not match" in completed.stderr


def test_publish_rejects_mutated_frozen_selection_evidence(tmp_path, monkeypatch):
    inputs = list(prepare_inputs(tmp_path))
    config = json.loads(inputs[8].read_text(encoding="utf-8"))
    balance = Path(config["selection_evidence"]["balance"]["path"])
    balance.write_text('{"selected":"mutated"}', encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "publish_strict_shared_core_results.py"),
            "--dataset", "vpn-app",
            "--audit_root", str(inputs[0]),
            "--packet_manifest_root", str(inputs[1]),
            "--shared_core_config", str(inputs[8]),
            "--method_archive_root", str(tmp_path / "reasoningDataset/shared-core-v2"),
            "--packet_candidate", str(inputs[2]),
            "--flow_candidate", str(inputs[3]),
            "--packet_bootstrap", str(inputs[4]),
            "--flow_bootstrap", str(inputs[5]),
            "--packet_session_novelty", str(inputs[6]),
            "--flow_session_novelty", str(inputs[7]),
            "--output_json", str(tmp_path / "publication.json"),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "selection evidence hash mismatch" in completed.stderr


def test_archive_includes_final_d1_d2_and_flow_selection_evidence(tmp_path):
    inputs = prepare_inputs(tmp_path)
    config_path = inputs[8]
    config = json.loads(config_path.read_text(encoding="utf-8"))

    base = {
        "schema": "exact_shared_packet_core_v2",
        "status": "frozen_from_cross_dataset_validation",
    }
    base["config_sha256"] = canonical_sha256(base)
    base_path = write_json(tmp_path / "base_shared_core.json", base)
    reports = {}
    for name in (
        "d1",
        "d2_incremental",
        "d2_overall",
        "cross_scale_exposure",
        "flow_noninferiority",
    ):
        reports[name] = write_json(
            tmp_path / f"{name}.json",
            {"test_labels_used": False, "selected": "candidate"},
        )

    config["method_selection"] = {
        "decision_status": "final_after_preregistered_validation",
        "test_labels_used": False,
        "identity_safe_contrastive": True,
        "availability_aware_cross_scale": True,
        "packet_selected_identity_safe_contrastive": True,
        "packet_selected_availability_aware_cross_scale": True,
        "flow_noninferiority_passed": True,
        "selected_method": "availability_aware_cross_scale",
        "base_shared_core_config": {
            "path": str(base_path),
            "sha256": file_sha256(base_path),
            "config_sha256": base["config_sha256"],
        },
        **{
            name: {"path": str(path), "sha256": file_sha256(path)}
            for name, path in reports.items()
        },
    }
    config.pop("config_sha256")
    config["config_sha256"] = canonical_sha256(config)
    write_json(config_path, config)

    archive = archive_frozen_method_evidence(
        config_path,
        expected_fingerprint=config["config_sha256"],
        archive_root=tmp_path / "method_archive",
    )

    evidence = archive["method_selection_evidence"]
    assert archive["schema"] == "strict_shared_core_v2_method_archive_v2"
    assert set(evidence) == {
        "base_shared_core_config",
        "d1",
        "d2_incremental",
        "d2_overall",
        "cross_scale_exposure",
        "flow_noninferiority",
    }
    assert all(Path(row["archived_path"]).is_file() for row in evidence.values())


def test_archive_includes_hierarchy_selection_gate_and_preregistration(tmp_path):
    inputs = prepare_inputs(tmp_path)
    config_path = inputs[8]
    config = json.loads(config_path.read_text(encoding="utf-8"))
    balance = config["selection_evidence"]["balance"]
    gate_path = write_json(
        tmp_path / "hierarchy_gate.json",
        {
            "schema": "hierarchy_adaptive_class_weight_gate_v1",
            "test_labels_used": False,
        },
    )
    preregistration_path = write_json(
        tmp_path / "hierarchy_preregistration.json",
        {
            "schema": "hierarchy_adaptive_class_weight_preregistration_v1",
            "test_labels_used": False,
        },
    )
    hierarchy_path = write_json(
        tmp_path / "hierarchy_selection.json",
        {
            "schema": "hierarchy_adaptive_class_weight_selection_v1",
            "selection_scope": "heldout_validation_only",
            "test_labels_used": False,
            "shared_algorithm": "normalized_effective_flow_class_risk_power_eta",
            "inputs": {
                "class_weight_selection": balance,
                "gate": {
                    "path": str(gate_path),
                    "sha256": file_sha256(gate_path),
                },
                "preregistration": {
                    "path": str(preregistration_path),
                    "sha256": file_sha256(preregistration_path),
                },
            },
        },
    )
    config["selection_evidence"]["hierarchy_class_weight"] = {
        "path": str(hierarchy_path),
        "sha256": file_sha256(hierarchy_path),
    }
    config.pop("config_sha256")
    config["config_sha256"] = canonical_sha256(config)
    write_json(config_path, config)

    archive = archive_frozen_method_evidence(
        config_path,
        expected_fingerprint=config["config_sha256"],
        archive_root=tmp_path / "method_archive",
    )

    evidence = archive["selection_evidence"]
    assert {
        "hierarchy_class_weight",
        "hierarchy_gate",
        "hierarchy_preregistration",
    } <= set(evidence)
    assert all(
        Path(evidence[name]["archived_path"]).is_file()
        for name in (
            "hierarchy_class_weight",
            "hierarchy_gate",
            "hierarchy_preregistration",
        )
    )


def test_archive_rejects_mutated_hierarchy_gate(tmp_path):
    inputs = prepare_inputs(tmp_path)
    config_path = inputs[8]
    config = json.loads(config_path.read_text(encoding="utf-8"))
    balance = config["selection_evidence"]["balance"]
    gate_path = write_json(
        tmp_path / "hierarchy_gate.json",
        {"schema": "hierarchy_adaptive_class_weight_gate_v1", "test_labels_used": False},
    )
    preregistration_path = write_json(
        tmp_path / "hierarchy_preregistration.json",
        {
            "schema": "hierarchy_adaptive_class_weight_preregistration_v1",
            "test_labels_used": False,
        },
    )
    hierarchy_path = write_json(
        tmp_path / "hierarchy_selection.json",
        {
            "schema": "hierarchy_adaptive_class_weight_selection_v1",
            "selection_scope": "heldout_validation_only",
            "test_labels_used": False,
            "shared_algorithm": "normalized_effective_flow_class_risk_power_eta",
            "inputs": {
                "class_weight_selection": balance,
                "gate": {"path": str(gate_path), "sha256": file_sha256(gate_path)},
                "preregistration": {
                    "path": str(preregistration_path),
                    "sha256": file_sha256(preregistration_path),
                },
            },
        },
    )
    config["selection_evidence"]["hierarchy_class_weight"] = {
        "path": str(hierarchy_path),
        "sha256": file_sha256(hierarchy_path),
    }
    config.pop("config_sha256")
    config["config_sha256"] = canonical_sha256(config)
    write_json(config_path, config)
    gate_path.write_text('{"test_labels_used":true}', encoding="utf-8")

    import pytest

    with pytest.raises(ValueError, match="hierarchy gate evidence hash mismatch"):
        archive_frozen_method_evidence(
            config_path,
            expected_fingerprint=config["config_sha256"],
            archive_root=tmp_path / "method_archive",
        )


def test_archive_includes_bounded_hierarchy_upstream_evidence(tmp_path):
    inputs = prepare_inputs(tmp_path)
    config_path = inputs[8]
    config = json.loads(config_path.read_text(encoding="utf-8"))
    balance = config["selection_evidence"]["balance"]
    gate_path = write_json(
        tmp_path / "bounded_gate.json",
        {
            "schema": "hierarchy_adaptive_class_weight_gate_v1",
            "selection_scope": "heldout_validation_only",
            "test_labels_used": False,
        },
    )
    derivation_path = write_json(
        tmp_path / "bounded_derivation.json",
        {
            "schema": "bounded_hierarchy_risk_protocol_v1",
            "status": "derived_from_training_counts_only",
            "test_labels_used": False,
        },
    )
    preregistration_path = write_json(
        tmp_path / "bounded_preregistration.json",
        {
            "schema": "hierarchy_adaptive_class_weight_preregistration_v1",
            "test_labels_used": False,
            "bounded_risk_geometry_amendment": {
                "numeric_derivation": {
                    "path": str(derivation_path),
                    "sha256": file_sha256(derivation_path),
                }
            },
        },
    )
    shared_inputs = {
        "class_weight_selection": balance,
        "gate": {"path": str(gate_path), "sha256": file_sha256(gate_path)},
        "preregistration": {
            "path": str(preregistration_path),
            "sha256": file_sha256(preregistration_path),
        },
    }
    reference_path = write_json(
        tmp_path / "hierarchy_grid_reference.json",
        {
            "schema": "hierarchy_adaptive_class_weight_selection_v1",
            "selection_scope": "heldout_validation_only",
            "test_labels_used": False,
            "shared_algorithm": "normalized_effective_flow_class_risk_power_eta",
            "inputs": shared_inputs,
        },
    )
    hierarchy_inputs = {
        **shared_inputs,
        "reference_hierarchy_selection": {
            "path": str(reference_path),
            "sha256": file_sha256(reference_path),
        },
        "bounded_risk_derivation": {
            "path": str(derivation_path),
            "sha256": file_sha256(derivation_path),
        },
    }
    hierarchy_path = write_json(
        tmp_path / "bounded_hierarchy_selection.json",
        {
            "schema": "hierarchy_adaptive_class_weight_selection_v1",
            "selection_scope": "heldout_validation_only",
            "test_labels_used": False,
            "shared_algorithm": "bounded_effective_flow_class_risk_power_eta",
            "numeric_protocol_selection": {
                "schema": "bounded_hierarchy_risk_selection_v1",
                "selected": "candidate",
                "candidate_promoted_for_all_datasets": True,
            },
            "inputs": hierarchy_inputs,
        },
    )
    config["selection_evidence"]["hierarchy_class_weight"] = {
        "path": str(hierarchy_path),
        "sha256": file_sha256(hierarchy_path),
    }
    flow_datasets = {}
    for dataset in ("vpn-app", "tls-120"):
        source_path = write_json(
            tmp_path / f"flow_hierarchy_source_{dataset}.json",
            {
                "schema": "class_sampling_hierarchy_analysis_v1",
                "selection_role": "train_only_reporting_not_model_selection",
                "test_labels_used": False,
            },
        )
        flow_datasets[dataset] = {
            "input": {
                "path": str(source_path),
                "sha256": file_sha256(source_path),
            }
        }
    flow_derivation_path = write_json(
        tmp_path / "flow_task_hierarchy_derivation.json",
        {
            "schema": "bounded_hierarchy_risk_protocol_v1",
            "status": "derived_from_training_counts_only",
            "test_labels_used": False,
            "shared_algorithm": (
                "largest_flow_risk_power_subject_to_max_min_ratio"
            ),
            "datasets": flow_datasets,
        },
    )
    config["selection_evidence"]["flow_task_hierarchy_derivation"] = {
        "path": str(flow_derivation_path),
        "sha256": file_sha256(flow_derivation_path),
    }
    packet_dataset_names = {
        "vpn-app",
        "vpn-binary",
        "vpn-service",
        "tls-120",
        "ustc-app",
        "ustc-binary",
    }
    packet_datasets = {}
    for dataset in packet_dataset_names:
        source_path = write_json(
            tmp_path / f"packet_hierarchy_source_{dataset}.json",
            {
                "schema": "class_sampling_hierarchy_analysis_v1",
                "selection_role": "train_only_reporting_not_model_selection",
                "test_labels_used": False,
            },
        )
        packet_datasets[dataset] = {
            "input": {
                "path": str(source_path),
                "sha256": file_sha256(source_path),
            }
        }
    packet_derivation_path = write_json(
        tmp_path / "packet_task_hierarchy_derivation.json",
        {
            "schema": "bounded_hierarchy_risk_protocol_v1",
            "status": "derived_from_training_counts_only",
            "test_labels_used": False,
            "shared_algorithm": (
                "largest_flow_risk_power_subject_to_max_min_ratio"
            ),
            "datasets": packet_datasets,
        },
    )
    config["selection_evidence"]["packet_task_hierarchy_derivation"] = {
        "path": str(packet_derivation_path),
        "sha256": file_sha256(packet_derivation_path),
    }
    config.pop("config_sha256")
    config["config_sha256"] = canonical_sha256(config)
    write_json(config_path, config)

    archive = archive_frozen_method_evidence(
        config_path,
        expected_fingerprint=config["config_sha256"],
        archive_root=tmp_path / "method_archive",
    )

    evidence = archive["selection_evidence"]
    expected = {
        "hierarchy_reference_hierarchy_selection",
        "hierarchy_bounded_risk_derivation",
        "flow_task_hierarchy_derivation",
        "flow_task_hierarchy_source_vpn-app",
        "flow_task_hierarchy_source_tls-120",
        "packet_task_hierarchy_derivation",
        *{
            f"packet_task_hierarchy_source_{dataset}"
            for dataset in packet_dataset_names
        },
    }
    assert expected <= set(evidence)
    assert all(Path(evidence[name]["archived_path"]).is_file() for name in expected)


def test_archive_rejects_mutated_final_d1_evidence(tmp_path):
    inputs = prepare_inputs(tmp_path)
    config_path = inputs[8]
    config = json.loads(config_path.read_text(encoding="utf-8"))
    base = {"schema": "exact_shared_packet_core_v2"}
    base["config_sha256"] = canonical_sha256(base)
    base_path = write_json(tmp_path / "base.json", base)
    d1_path = write_json(
        tmp_path / "d1.json", {"test_labels_used": False, "selected": "baseline"}
    )
    config["method_selection"] = {
        "decision_status": "final_after_preregistered_validation",
        "test_labels_used": False,
        "identity_safe_contrastive": False,
        "availability_aware_cross_scale": False,
        "packet_selected_identity_safe_contrastive": False,
        "packet_selected_availability_aware_cross_scale": False,
        "flow_noninferiority_passed": None,
        "selected_method": "shared_core_v2_control",
        "base_shared_core_config": {
            "path": str(base_path),
            "sha256": file_sha256(base_path),
            "config_sha256": base["config_sha256"],
        },
        "d1": {"path": str(d1_path), "sha256": file_sha256(d1_path)},
    }
    config.pop("config_sha256")
    config["config_sha256"] = canonical_sha256(config)
    write_json(config_path, config)
    d1_path.write_text('{"selected":"mutated"}', encoding="utf-8")

    import pytest

    with pytest.raises(ValueError, match="method selection d1 evidence hash mismatch"):
        archive_frozen_method_evidence(
            config_path,
            expected_fingerprint=config["config_sha256"],
            archive_root=tmp_path / "method_archive",
        )
