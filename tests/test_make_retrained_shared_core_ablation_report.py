import json

from make_retrained_shared_core_ablation_report import DIAGNOSTICS, build_report


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def metric_payload(task, accuracy, macro_f1):
    metrics = {"accuracy": accuracy, "macro_f1": macro_f1}
    return {"metrics": {"flow_level": metrics} if task == "flow" else metrics}


def manifest(result_paths, completed=True):
    return {
        "framework": {
            "notes": {"result_paths": result_paths, "completed": completed}
        }
    }


def test_retrained_report_uses_validation_for_decisions_and_keeps_test_descriptive(tmp_path):
    summaries = []
    fingerprint = "a" * 64
    for dataset in ("vpn-app", "tls-120"):
        for diagnostic in DIAGNOSTICS:
            reference_paths = {}
            for task in ("packet", "flow"):
                paths = []
                for split in ("valid", "test"):
                    name = (
                        f"{split}_packet.json"
                        if task == "packet"
                        else f"{split}_seq_metrics.json"
                    )
                    path = tmp_path / dataset / diagnostic / "reference" / task / name
                    write_json(path, metric_payload(task, 0.80, 0.75))
                    paths.append(str(path))
                reference_paths[task] = write_json(
                    tmp_path / dataset / diagnostic / f"{task}_reference_manifest.json",
                    manifest(paths),
                )

            packet_ablation_paths = []
            flow_outputs = {}
            for split in ("valid", "test"):
                delta = -0.01 if split == "valid" and diagnostic == "no_content" else 0.0
                packet_path = tmp_path / dataset / diagnostic / f"{split}_packet_ablation.json"
                flow_path = tmp_path / dataset / diagnostic / f"{split}_flow_ablation.json"
                write_json(packet_path, metric_payload("packet", 0.80 + delta, 0.75 + delta))
                write_json(flow_path, metric_payload("flow", 0.80 + delta, 0.75 + delta))
                packet_ablation_paths.append(str(packet_path))
                flow_outputs[split] = str(flow_path)
            packet_manifest = write_json(
                tmp_path / dataset / diagnostic / "packet_ablation_manifest.json",
                manifest(packet_ablation_paths),
            )
            summary = {
                "scope": "retrained_ablation",
                "dataset": dataset,
                "fold": 0,
                "diagnostic": diagnostic,
                "shared_core_config_sha256": fingerprint,
                "packet_reference_manifest": reference_paths["packet"],
                "flow_reference_manifest": reference_paths["flow"],
                "test_labels_used_for_model_selection": False,
                "dry_run": False,
                "outputs": {
                    "packet": {"manifest": packet_manifest},
                    "flow": {"results": flow_outputs},
                },
            }
            summaries.append(
                write_json(
                    tmp_path / dataset / f"{diagnostic}_summary.json", summary
                )
            )

    report = build_report(summaries)
    decisions = {row["diagnostic"]: row for row in report["decisions"]}
    assert decisions["no_content"]["decision"] == "retain_module_and_expand_to_three_folds"
    assert decisions["fixed_fusion"]["decision"] == (
        "candidate_simplification_requires_three_fold_noninferiority"
    )
    assert report["test_rows_are_descriptive_only"] is True
    assert report["test_labels_used_for_model_selection"] is False
