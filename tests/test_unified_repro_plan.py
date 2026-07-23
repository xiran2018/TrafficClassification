import hashlib
import json

import pytest

import run_unified_repro_plan as repro_executor
from make_unified_repro_plan import build_plan
from run_unified_repro_plan import (
    action_fingerprint,
    completed_ids,
    selected_actions,
    verify_plan_shared_core_config,
)


def fake_audit():
    return {
        "status": "review",
        "flow_level": [
            {
                "dataset": "vpn-app",
                "path": "reasoningDataset/vpn-app/test_crossfold_consensus_auto_confidence.json",
                "metric_status": "pass",
                "publication_status": "needs_paper_unified_repro",
                "framework_manifest_glob": "reasoningDataset/vpn-app/stage8_flowaware_manifest_*paper_unified*.json",
                "framework_provenance": {
                    "status": "insufficient_fold_manifests",
                    "candidate_manifest_count": 0,
                    "completed_folds": [],
                },
            }
        ],
        "packet_level": [
            {
                "dataset": "tls-120",
                "metric_status": "pass",
                "publication_status": "needs_paper_unified_repro",
                "framework_manifest_glob": "reasoningDataset/packet-level/tls-120/*/packet_framework_manifest.json",
            }
        ],
    }


def frozen_config(tmp_path):
    payload = {
        "schema": "exact_shared_packet_core_v2",
        "status": "frozen_from_cross_dataset_validation",
        "packet_core": {"hidden_dim": 128},
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    payload["config_sha256"] = hashlib.sha256(encoded).hexdigest()
    path = tmp_path / "frozen_config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_repro_plan_records_safe_argv_and_stable_ids():
    plan = build_plan(fake_audit(), flow_stage="all", packet_stage="paper_unified", run_tag="paper_unified_repro")
    assert plan["num_actions"] == 10
    first = plan["actions"][0]
    assert first["id"] == "flow:vpn-app:fold0"
    assert first["argv"][:5] == ["conda", "run", "--no-capture-output", "-n", "llm-factory"]
    assert first["command"] == " ".join(first["argv"])
    assert "--framework_profile" in first["argv"]
    assert "paper_unified" in first["argv"]
    assert first["argv"][first["argv"].index("--stage") + 1] == "all"
    assert first["argv"][first["argv"].index("--fold") + 1] == "0"
    assert first["argv"][first["argv"].index("--embedding_header_policy") + 1] == "full"
    assert first["argv"][first["argv"].index("--paper_unified_stages") + 1] == "model"
    assert "--result_json" not in first["argv"]
    assert plan["actions"][3]["id"] == "flow-consensus:vpn-app"
    assert plan["actions"][4]["id"] == "flow-result:vpn-app"
    packet = plan["actions"][5]
    assert packet["argv"][packet["argv"].index("--stage") + 1] == "paper_unified"
    packet_consensus = plan["actions"][8]
    assert all(
        "test_unified_packet_single_head.npz" in value
        for value in packet_consensus["argv"]
        if value.endswith(".npz")
    )


def test_repro_plan_only_reruns_missing_flow_folds():
    audit = fake_audit()
    audit["flow_level"][0]["framework_provenance"] = {
        "status": "insufficient_fold_manifests",
        "candidate_manifest_count": 1,
        "completed_folds": [0],
    }
    plan = build_plan(audit, flow_stage="all", packet_stage="paper_unified", run_tag="paper_unified_repro")
    first = plan["actions"][0]
    assert first["id"] == "flow:vpn-app:fold1"
    assert first["argv"][first["argv"].index("--stage") + 1] == "all"
    assert "--require_cuda" in first["argv"]
    assert "--result_json" not in first["argv"]


def test_repro_executor_filters_and_skips_completed_actions():
    plan = build_plan(fake_audit(), flow_stage="all", packet_stage="paper_unified", run_tag="paper_unified_repro")
    first_packet = next(row for row in plan["actions"] if row["id"] == "packet:tls-120:fold0")
    ledger = {"runs": [{
        "id": "packet:tls-120:fold0",
        "status": "success",
        "action_fingerprint": action_fingerprint(first_packet),
    }]}
    actions = selected_actions(
        plan,
        task="packet-level",
        dataset="tls-120",
        start_index=0,
        max_actions=-1,
        skip_ids=completed_ids(ledger),
    )
    assert [row["id"] for row in actions] == [
        "packet:tls-120:fold1",
        "packet:tls-120:fold2",
        "packet-consensus:tls-120",
        "packet-result:tls-120",
    ]


def test_repro_executor_does_not_skip_stale_id_only_or_changed_commands():
    plan = build_plan(fake_audit(), flow_stage="all", packet_stage="paper_unified", run_tag="paper_unified_repro")
    first = plan["actions"][0]
    stale = {
        "runs": [
            {"id": first["id"], "status": "success"},
            {
                "id": first["id"],
                "status": "success",
                "action_fingerprint": "0" * 64,
            },
        ]
    }
    actions = selected_actions(
        plan,
        task="flow-level",
        dataset="vpn-app",
        start_index=0,
        max_actions=1,
        skip_ids=completed_ids(stale),
    )
    assert actions[0]["id"] == "flow:vpn-app:fold0"


def test_repro_plan_skips_packet_folds_with_matching_manifests():
    audit = fake_audit()
    audit["packet_level"][0]["framework_provenance"] = {
        "status": "insufficient_fold_manifests",
        "matching_manifest_count": 1,
        "matching_manifests": [
            "reasoningDataset/packet-level/tls-120/fold0/packet_framework_manifest.json"
        ],
    }
    plan = build_plan(audit, flow_stage="all", packet_stage="paper_unified", run_tag="paper_unified_repro")
    packet_ids = [row["id"] for row in plan["actions"] if row["task"] == "packet-level"]
    assert packet_ids == [
        "packet:tls-120:fold1",
        "packet:tls-120:fold2",
        "packet-consensus:tls-120",
        "packet-result:tls-120",
    ]


def test_repro_plan_binds_one_verified_frozen_config_to_every_training_action(
    tmp_path,
):
    config = frozen_config(tmp_path)
    plan = build_plan(
        fake_audit(),
        flow_stage="all",
        packet_stage="paper_unified",
        run_tag="paper_unified_repro",
        shared_core_config=str(config),
    )

    evidence = plan["shared_core_config"]
    assert evidence["path"] == str(config.resolve())
    assert evidence["file_sha256"] == hashlib.sha256(config.read_bytes()).hexdigest()
    assert evidence["config_sha256"] == json.loads(config.read_text())["config_sha256"]

    training = [row for row in plan["actions"] if row.get("fold") is not None]
    assert training
    for action in training:
        argv = action["argv"]
        assert argv[argv.index("--shared_core_config") + 1] == str(config.resolve())
    non_training = [row for row in plan["actions"] if row.get("fold") is None]
    assert all("--shared_core_config" not in row["argv"] for row in non_training)


def test_repro_plan_rejects_mutated_frozen_config(tmp_path):
    config = frozen_config(tmp_path)
    payload = json.loads(config.read_text())
    payload["packet_core"]["hidden_dim"] = 256
    config.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="canonical fingerprint mismatch"):
        build_plan(
            fake_audit(),
            flow_stage="all",
            packet_stage="paper_unified",
            run_tag="paper_unified_repro",
            shared_core_config=str(config),
        )


def test_repro_executor_rechecks_frozen_config_after_plan_creation(tmp_path):
    config = frozen_config(tmp_path)
    plan = build_plan(
        fake_audit(),
        flow_stage="all",
        packet_stage="paper_unified",
        run_tag="paper_unified_repro",
        shared_core_config=str(config),
    )
    assert verify_plan_shared_core_config(plan) == plan["shared_core_config"]

    payload = json.loads(config.read_text())
    payload["packet_core"]["hidden_dim"] = 256
    config.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="file hash mismatch"):
        verify_plan_shared_core_config(plan)


def test_repro_executor_rejects_training_action_with_different_config(tmp_path):
    config = frozen_config(tmp_path)
    plan = build_plan(
        fake_audit(),
        flow_stage="all",
        packet_stage="paper_unified",
        run_tag="paper_unified_repro",
        shared_core_config=str(config),
    )
    action = next(row for row in plan["actions"] if row.get("fold") is not None)
    index = action["argv"].index("--shared_core_config") + 1
    action["argv"][index] = str(tmp_path / "different.json")

    with pytest.raises(ValueError, match="binds a different shared-core config"):
        verify_plan_shared_core_config(plan)


def test_repro_executor_rechecks_config_before_every_action(tmp_path, monkeypatch):
    config = frozen_config(tmp_path)
    plan = build_plan(
        fake_audit(),
        flow_stage="all",
        packet_stage="paper_unified",
        run_tag="paper_unified_repro",
        shared_core_config=str(config),
    )
    actions = [row for row in plan["actions"] if row.get("fold") is not None][:2]
    calls = 0

    def fake_run_action(action, log_dir, dry_run):
        nonlocal calls
        calls += 1
        if calls == 1:
            payload = json.loads(config.read_text())
            payload["packet_core"]["hidden_dim"] = 256
            config.write_text(json.dumps(payload), encoding="utf-8")
        return {
            "id": action["id"],
            "status": "dry_run",
            "action_fingerprint": action_fingerprint(action),
        }

    monkeypatch.setattr(repro_executor, "run_action", fake_run_action)
    ledger_path = tmp_path / "ledger.json"
    with pytest.raises(ValueError, match="file hash mismatch"):
        repro_executor.run_actions(
            actions,
            ledger_path=ledger_path,
            log_dir=tmp_path / "logs",
            dry_run=True,
            continue_on_error=False,
            plan=plan,
        )

    ledger = json.loads(ledger_path.read_text())
    assert calls == 1
    assert [row["id"] for row in ledger["runs"]] == [actions[0]["id"]]
