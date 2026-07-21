import json
from pathlib import Path

from make_shared_core_sensitivity_report import build_report


def write(path: Path, payload: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return str(path)


def packet_result(acc=0.8, f1=0.7):
    return {"metrics": {"accuracy": acc, "macro_f1": f1}}


def flow_result(acc=0.8, f1=0.7):
    return {"metrics": {"flow_level": {"accuracy": acc, "macro_f1": f1}}}


def test_sensitivity_report_requires_and_aggregates_full_matrix(tmp_path):
    summaries = []
    for dataset in ("vpn-app", "tls-120"):
        for fold in range(3):
            root = tmp_path / dataset / str(fold)
            packet_base = write(root / "test_unified_packet_single_head.json", packet_result())
            flow_base = write(root / "test_seq_metrics_probs.json", flow_result())
            packet_manifest = write(
                root / "packet_manifest.json",
                {"framework": {"notes": {"result_paths": [packet_base]}}},
            )
            flow_manifest = write(
                root / "flow_manifest.json",
                {"framework": {"notes": {"result_paths": [flow_base]}}},
            )
            packet_ablation = write(root / "packet_no_content.json", packet_result(0.79, 0.68))
            flow_ablation = write(root / "flow_no_content.json", flow_result(0.78, 0.67))
            summaries.append(
                write(
                    root / "summary.json",
                    {
                        "scope": "inference_only_not_retrained_ablation",
                        "dataset": dataset,
                        "fold": fold,
                        "split": "test",
                        "shared_core_config_sha256": "a" * 64,
                        "packet_manifest": packet_manifest,
                        "flow_manifest": flow_manifest,
                        "diagnostics": {
                            "no_content": {
                                "packet": packet_ablation,
                                "flow": flow_ablation,
                            }
                        },
                    },
                )
            )

    report = build_report(summaries)

    assert report["coverage"]["num_rows"] == 12
    rows = {(row["diagnostic"], row["task"]): row for row in report["aggregate"]}
    assert abs(rows[("no_content", "packet")]["mean_delta_macro_f1"] + 0.02) < 1e-9
    assert abs(rows[("no_content", "flow")]["mean_delta_macro_f1"] + 0.03) < 1e-9
    assert report["module_decisions"][0]["automatic_module_removal_allowed"] is False
