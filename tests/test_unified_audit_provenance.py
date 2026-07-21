import json

from audit_unified_framework import (
    audit_manifest,
    audit_cross_task_checkpoint_reports,
    audit_result_provenance,
    audit_shared_core_v2_fingerprints,
    canonical_result_path_status,
)
from make_unified_repro_plan import build_plan
from unified_framework_spec import ResultSpec, build_framework_manifest, profile_shared_status
from write_unified_result_manifest import extract_metrics


def write_manifest(path, *, dataset="demo", stage="paper_unified", result_path=""):
    notes = {
        "framework_profile": "paper_unified",
        "completed": True,
        "dry_run": False,
        "paper_main_experts": [],
        "semantic_fusion_level": "representation",
    }
    if result_path:
        notes["published_result_paths"] = [result_path]
    classifier = (
        "shared_representation_single_head_crossfold_consensus"
        if stage == "paper_result_binding"
        else "shared_representation_single_head"
    )
    framework = build_framework_manifest(
        task="packet-level",
        dataset=dataset,
        input_unit="one_current_packet",
        stage=stage,
        shared_module_status=profile_shared_status("paper_unified"),
        task_module_status={
            "strict_current_packet_protocol": "enforced",
            "packet_level_classifier": classifier,
        },
        notes=notes,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "task": "packet-level",
                "dataset": dataset,
                "stage": stage,
                "framework": framework,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def packet_spec(result_path: str) -> ResultSpec:
    return ResultSpec(
        dataset="demo",
        task="packet-level",
        path=result_path,
        framework_manifest_glob="reasoningDataset/packet-level/demo/*/packet_framework_manifest.json",
    )


def test_canonical_result_path_status_rejects_tmp_packet_defaults():
    canonical = ResultSpec(
        dataset="demo",
        task="packet-level",
        path="reasoningDataset/packet-level/demo/paper_default_result.json",
        framework_manifest_glob="reasoningDataset/packet-level/demo/*/packet_framework_manifest.json",
    )
    tmp_result = ResultSpec(
        dataset="demo",
        task="packet-level",
        path="/tmp/two_tower_runs/packet_level/demo/packet_feature_crossfold_mean.json",
        framework_manifest_glob="reasoningDataset/packet-level/demo/*/packet_framework_manifest.json",
    )

    assert canonical_result_path_status(canonical)["ok"] is True
    assert canonical_result_path_status(tmp_result)["ok"] is False


def test_packet_provenance_requires_result_bound_manifest(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result_path = "reasoningDataset/packet-level/demo/packet_crossfold_mean.json"
    for fold in range(3):
        write_manifest(
            tmp_path
            / "reasoningDataset"
            / "packet-level"
            / "demo"
            / f"fold{fold}"
            / "packet_framework_manifest.json"
        )

    provenance = audit_result_provenance(packet_spec(result_path))

    assert provenance["status"] == "missing_bound_result_manifest"
    assert provenance["completed_folds"] == [0, 1, 2]
    assert provenance["result_bound_manifest_count"] == 0

    write_manifest(
        tmp_path
        / "reasoningDataset"
        / "packet-level"
        / "demo"
        / "result_bound"
        / "packet_framework_manifest.json",
        stage="paper_result_binding",
        result_path=result_path,
    )

    provenance = audit_result_provenance(packet_spec(result_path))

    assert provenance["status"] == "pass"
    assert provenance["completed_folds"] == [0, 1, 2]
    assert provenance["result_bound_manifest_count"] == 1


def test_paper_unified_manifest_requires_current_profile_fingerprint(tmp_path):
    manifest_path = tmp_path / "packet_framework_manifest.json"
    write_manifest(manifest_path)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data["framework"].pop("framework_profile_fingerprint", None)
    manifest_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    row = audit_manifest(manifest_path, "packet-level")

    assert row["status"] == "profile_fingerprint_missing"
    assert row["matches_framework_profile_fingerprint"] is False


def test_repro_plan_adds_packet_result_binding_action():
    audit = {
        "status": "review",
        "flow_level": [],
        "packet_level": [
            {
                "dataset": "demo",
                "path": "reasoningDataset/packet-level/demo/packet_crossfold_mean.json",
                "metric_status": "pass",
                "publication_status": "needs_paper_unified_repro",
                "framework_manifest_glob": "reasoningDataset/packet-level/demo/*/packet_framework_manifest.json",
                "framework_provenance": {
                    "status": "missing_bound_result_manifest",
                    "matching_manifests": [
                        "reasoningDataset/packet-level/demo/fold0/packet_framework_manifest.json",
                        "reasoningDataset/packet-level/demo/fold1/packet_framework_manifest.json",
                        "reasoningDataset/packet-level/demo/fold2/packet_framework_manifest.json",
                    ],
                },
            }
        ],
    }

    plan = build_plan(audit, flow_stage="paper_unified", packet_stage="paper_unified", run_tag="paper_unified_repro")

    assert [action["id"] for action in plan["actions"]] == ["packet-result:demo"]
    argv = plan["actions"][0]["argv"]
    assert "--task" in argv
    assert argv[argv.index("--task") + 1] == "packet-level"
    assert argv[argv.index("--result_json") + 1] == "reasoningDataset/packet-level/demo/packet_crossfold_mean.json"


def test_result_binding_metric_extraction_accepts_packet_history_shapes():
    assert extract_metrics({"test_metrics": {"accuracy": 0.8, "macro_f1": 0.7, "num_samples": 12}}, "packet-level") == (
        0.8,
        0.7,
        12,
    )
    assert extract_metrics(
        {"metrics": {"packet_level": {"accuracy": 0.9, "macro_f1": 0.85, "num_packets": 34}}},
        "packet-level",
    ) == (0.9, 0.85, 34)


def exact_row(
    task, dataset, fold, fingerprint="shared-v2", method_fingerprint=""
):
    return {
        "path": f"reasoningDataset/{dataset}/fold{fold}/manifest.json",
        "task": task,
        "dataset": dataset,
        "fold": fold,
        "status": "pass",
        "execution_status": "candidate_executed",
        "framework_profile": "paper_unified",
        "shared_core_config_sha256": fingerprint,
        "shared_core_method_sha256": method_fingerprint or fingerprint,
    }


def test_exact_v2_requires_one_fingerprint_across_tasks_datasets_and_folds():
    flow = [
        exact_row("flow-level", dataset, fold)
        for dataset in ("vpn-app", "tls-120")
        for fold in range(3)
    ]
    packet = [
        exact_row("packet-level", dataset, fold)
        for dataset in ("vpn-app", "tls-120")
        for fold in range(3)
    ]
    assert audit_shared_core_v2_fingerprints(flow, packet)["status"] == "pass"

    packet[-1]["shared_core_config_sha256"] = "different"
    report = audit_shared_core_v2_fingerprints(flow, packet)
    assert report["status"] == "not_ready"
    assert report["cross_task_cross_dataset_fingerprint_match"] is False
    assert report["unified_method_status"] == "pass"
    assert report["cross_task_cross_dataset_method_match"] is True


def test_unified_v2_allows_independent_effective_hyperparameter_hashes():
    flow = [
        exact_row(
            "flow-level",
            dataset,
            fold,
            fingerprint=f"flow-{dataset}-{fold}",
            method_fingerprint="shared-method-v2",
        )
        for dataset in ("vpn-app", "tls-120")
        for fold in range(3)
    ]
    packet = [
        exact_row(
            "packet-level",
            dataset,
            fold,
            fingerprint=f"packet-{dataset}-{fold}",
            method_fingerprint="shared-method-v2",
        )
        for dataset in ("vpn-app", "tls-120")
        for fold in range(3)
    ]
    report = audit_shared_core_v2_fingerprints(flow, packet)
    assert report["status"] == "not_ready"
    assert report["unified_method_status"] == "pass"
    assert report["observed_method_fingerprints"] == ["shared-method-v2"]


def test_exact_v2_checkpoint_reports_require_every_dataset_and_fold(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "reasoningDataset" / "shared-core-audits"
    for dataset in ("vpn-app", "tls-120"):
        for fold in range(3):
            path = root / dataset / f"fold{fold}" / "audit.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "dataset": dataset,
                        "fold": fold,
                        "shared_core_config_sha256": "shared-v2",
                        "shared_core_method_sha256": "shared-method-v2",
                    }
                ),
                encoding="utf-8",
            )
    assert audit_cross_task_checkpoint_reports()["status"] == "pass"

    tuned = root / "tls-120" / "fold1" / "audit.json"
    payload = json.loads(tuned.read_text(encoding="utf-8"))
    payload["shared_core_config_sha256"] = "tls-effective-config"
    tuned.write_text(json.dumps(payload), encoding="utf-8")
    tuned_report = audit_cross_task_checkpoint_reports()
    assert tuned_report["status"] == "not_ready"
    assert tuned_report["unified_method_status"] == "pass"

    (root / "tls-120" / "fold2" / "audit.json").unlink()
    report = audit_cross_task_checkpoint_reports()
    assert report["status"] == "not_ready"
    assert report["groups"]["tls-120"]["completed_folds"] == [0, 1]
