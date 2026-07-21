import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def prepare_inputs(
    tmp_path,
    *,
    packet_accuracy=0.91,
    packet_f1=0.8,
    flow_accuracy=0.8,
    flow_f1=0.72,
):
    audit_root = tmp_path / "audits"
    manifest_root = tmp_path / "manifests"
    for fold in range(3):
        flow_manifest = write_json(
            tmp_path / "flow_manifests" / f"fold{fold}.json",
            {
                "framework": {
                    "notes": {
                        "completed": True,
                        "shared_core_config_sha256": "shared-v2",
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
                "shared_core_config_sha256": "shared-v2",
                "runtime_mechanism_evidence_required": True,
                "flow_native_extraction_evidence_required": True,
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
                        "shared_core_config_sha256": "shared-v2",
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
    assert report["session_novelty_evidence"]["packet"]["path"].endswith(
        "packet_novelty.json"
    )
    canonical = json.loads(Path(report["packet"]["canonical"]).read_text())
    assert canonical["publication_provenance"]["status"] == "strict_shared_core_v2"
    assert canonical["publication_provenance"]["session_novelty_sha256"]
    assert Path(canonical["publication_provenance"]["session_novelty"]).is_file()


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
