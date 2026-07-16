#!/usr/bin/env python3
"""Summarize 3-fold traffic-classification results and emit rerun commands.

The SWEET flow-level datasets provide train_val_split_0/1/2 folders, each with
its own train/val split and a shared test folder. This script treats those
folders as the cross-split stability unit: it scans existing result JSON files,
reports the best result per fold, and writes commands for missing/weak folds.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


DATASET_CONFIG = {
    "vpn-app": {
        "data_root": "/home/jing/download/sweet/flow-level-classification/vpn-app",
        "num_classes": 16,
        "target_acc": 0.74,
        "target_f1": 0.65,
        "coarse_groups": "vpn_app",
        "confusion_groups": "vpn_app",
    },
    "tls-120": {
        "data_root": "/home/jing/download/sweet/flow-level-classification/tls",
        "num_classes": 120,
        "target_acc": 0.78,
        "target_f1": 0.70,
        "coarse_groups": "none",
        "confusion_groups": "none",
    },
}


def metric_from_payload(data: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    metrics = data.get("metrics")
    if isinstance(metrics, dict):
        flow = metrics.get("flow_level")
        if isinstance(flow, dict):
            return maybe_float(flow.get("accuracy")), maybe_float(flow.get("macro_f1"))
        return maybe_float(metrics.get("accuracy")), maybe_float(metrics.get("macro_f1"))
    flow = data.get("flow_level")
    if isinstance(flow, dict):
        return maybe_float(flow.get("accuracy")), maybe_float(flow.get("macro_f1"))
    return maybe_float(data.get("accuracy")), maybe_float(data.get("macro_f1"))


def maybe_float(value: Any) -> Optional[float]:
    try:
        return None if value is None else float(value)
    except Exception:
        return None


def infer_fold(path: Path) -> Optional[int]:
    name = path.name
    parts = path.parts
    text = " ".join(parts[-3:])
    for idx in range(3):
        tokens = [f"split{idx}", f"fold{idx}", f"train_val_split_{idx}"]
        if any(token in text for token in tokens):
            return idx
    # Historical split0/baseline files usually have no split token.
    if "split1" not in name and "split2" not in name and "multisplit" not in name and "content_clean" not in name:
        return 0
    return None


def score_key(acc: Optional[float], f1: Optional[float], rank_metric: str) -> Tuple[float, float, float]:
    a = acc if acc is not None else -1.0
    f = f1 if f1 is not None else -1.0
    if rank_metric == "accuracy":
        return (a, f, min(a, f))
    if rank_metric == "macro_f1":
        return (f, a, min(a, f))
    return (min(a, f), a, f)


def iter_result_files(dataset_dir: Path, patterns: Iterable[str]) -> Iterable[Path]:
    seen = set()
    for pattern in patterns:
        for path in dataset_dir.glob(pattern):
            if path.is_file() and path not in seen:
                seen.add(path)
                yield path


def scan_dataset(dataset: str, patterns: List[str], rank_metric: str) -> Dict[str, Any]:
    root = Path("reasoningDataset") / dataset
    fold_rows: Dict[int, List[Dict[str, Any]]] = {0: [], 1: [], 2: []}
    ignored = []
    for path in iter_result_files(root, patterns):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            ignored.append({"path": str(path), "reason": f"json_error:{exc}"})
            continue
        acc, f1 = metric_from_payload(data)
        if acc is None or f1 is None:
            ignored.append({"path": str(path), "reason": "missing_metrics"})
            continue
        fold = infer_fold(path)
        if fold is None:
            ignored.append({"path": str(path), "reason": "unassigned_fold"})
            continue
        fold_rows[fold].append(
            {
                "path": str(path),
                "accuracy": acc,
                "macro_f1": f1,
                "score": score_key(acc, f1, rank_metric),
            }
        )
    best = {}
    for fold, rows in fold_rows.items():
        rows = sorted(rows, key=lambda r: r["score"], reverse=True)
        best[fold] = rows[0] if rows else None
    return {"dataset": dataset, "best_by_fold": best, "all_by_fold": fold_rows, "ignored": ignored}


def stats(values: List[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"mean": None, "std": None, "min": None, "max": None}
    return {
        "mean": float(statistics.mean(values)),
        "std": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
        "min": float(min(values)),
        "max": float(max(values)),
    }


def fold_command(dataset: str, fold: int, config: Dict[str, Any], args: argparse.Namespace) -> str:
    data_root = Path(config["data_root"])
    suffix = f"flowaware_change_weight_fold{fold}"
    embedding_suffix = f"rawproj_flowaware_change_weight_fold{fold}"
    run_tag = f"fold{fold}_stage8_cv"
    cmd = [
        "conda run --no-capture-output -n llm-factory",
        "python run_stage8_flowaware_pipeline.py",
        f"--dataset {dataset}",
        f"--num_classes {config['num_classes']}",
        "--stage all",
        f"--train_dir {data_root / f'train_val_split_{fold}' / 'train'}",
        f"--valid_dir {data_root / f'train_val_split_{fold}' / 'val'}",
        f"--test_dir {data_root / 'test'}",
        f"--output_suffix {suffix}",
        f"--tower1_data_suffix {suffix}",
        f"--embedding_suffix {embedding_suffix}",
        f"--run_tag {run_tag}",
        f"--tower1_output_dir checkpoints/tower1_qwen_multitask_{dataset.replace('-', '_')}_{suffix}",
        "--gradient_checkpointing",
        "--model_types graph,seq",
        "--flow_pooling multi_view",
        "--multi_view_gate_entropy_weight 0.01",
        "--confidence_penalty_weight 0.02",
        "--dropout 0.2",
        "--weight_decay 0.05",
        f"--coarse_groups {config['coarse_groups']}",
        f"--confusion_groups {config['confusion_groups']}",
        "--require_cuda",
        "--no_progress",
    ]
    if args.dry_run_commands:
        cmd.append("--dry_run")
    return " \\\n  ".join(cmd)


def build_report(dataset: str, scan: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    config = DATASET_CONFIG[dataset]
    target_acc = args.target_acc if args.target_acc is not None else config["target_acc"]
    target_f1 = args.target_f1 if args.target_f1 is not None else config["target_f1"]
    folds = []
    acc_values: List[float] = []
    f1_values: List[float] = []
    for fold in range(3):
        row = scan["best_by_fold"][fold]
        if row:
            acc_values.append(row["accuracy"])
            f1_values.append(row["macro_f1"])
            status = "pass" if row["accuracy"] >= target_acc and row["macro_f1"] >= target_f1 else "weak"
        else:
            status = "missing"
        folds.append(
            {
                "fold": fold,
                "status": status,
                "best": row,
                "recommended_command": None
                if status == "pass" and not args.emit_commands_for_pass
                else fold_command(dataset, fold, config, args),
            }
        )
    return {
        "dataset": dataset,
        "targets": {"accuracy": target_acc, "macro_f1": target_f1},
        "folds": folds,
        "summary": {
            "accuracy": stats(acc_values),
            "macro_f1": stats(f1_values),
            "all_folds_present": all(f["best"] is not None for f in folds),
            "all_folds_pass": all(f["status"] == "pass" for f in folds),
        },
        "ignored_count": len(scan["ignored"]),
    }


def write_markdown(reports: List[Dict[str, Any]], path: Path) -> None:
    lines = ["# Cross-split Result Summary", ""]
    for report in reports:
        lines.append(f"## {report['dataset']}")
        t = report["targets"]
        s = report["summary"]
        lines.append(f"Targets: accuracy >= {t['accuracy']:.4f}, macro-F1 >= {t['macro_f1']:.4f}")
        lines.append("")
        lines.append("| fold | status | accuracy | macro-F1 | result |")
        lines.append("|---:|---|---:|---:|---|")
        for fold in report["folds"]:
            best = fold["best"] or {}
            acc = best.get("accuracy")
            f1 = best.get("macro_f1")
            lines.append(
                "| {fold} | {status} | {acc} | {f1} | `{path}` |".format(
                    fold=fold["fold"],
                    status=fold["status"],
                    acc="" if acc is None else f"{acc:.4f}",
                    f1="" if f1 is None else f"{f1:.4f}",
                    path=best.get("path", ""),
                )
            )
        lines.append("")
        lines.append(
            "Summary: acc mean={amean}, min={amin}; F1 mean={fmean}, min={fmin}; all_folds_pass={passed}".format(
                amean=format_optional(s["accuracy"]["mean"]),
                amin=format_optional(s["accuracy"]["min"]),
                fmean=format_optional(s["macro_f1"]["mean"]),
                fmin=format_optional(s["macro_f1"]["min"]),
                passed=s["all_folds_pass"],
            )
        )
        cmds = [fold["recommended_command"] for fold in report["folds"] if fold["recommended_command"]]
        if cmds:
            lines.append("")
            lines.append("Recommended commands:")
            for cmd in cmds:
                lines.append("")
                lines.append("```bash")
                lines.append(cmd)
                lines.append("```")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def format_optional(value: Optional[float]) -> str:
    return "" if value is None else f"{value:.4f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", action="append", choices=sorted(DATASET_CONFIG), default=[])
    ap.add_argument("--pattern", action="append", default=["test*.json"], help="Glob under reasoningDataset/DATASET. Can be repeated.")
    ap.add_argument("--rank_metric", choices=["accuracy", "macro_f1", "target_margin"], default="target_margin")
    ap.add_argument("--target_acc", type=float, default=None, help="Override dataset target accuracy.")
    ap.add_argument("--target_f1", type=float, default=None, help="Override dataset target macro-F1.")
    ap.add_argument("--emit_commands_for_pass", action="store_true")
    ap.add_argument("--dry_run_commands", action="store_true", help="Append --dry_run to recommended runner commands.")
    ap.add_argument("--output_json", default="reasoningDataset/cross_split_summary.json")
    ap.add_argument("--output_md", default="reasoningDataset/cross_split_summary.md")
    args = ap.parse_args()

    datasets = args.dataset or ["vpn-app", "tls-120"]
    reports = []
    scans = {}
    for dataset in datasets:
        scan = scan_dataset(dataset, args.pattern, args.rank_metric)
        scans[dataset] = scan
        reports.append(build_report(dataset, scan, args))
    payload = {"reports": reports, "scans": scans}
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(reports, Path(args.output_md))
    for report in reports:
        print(f"{report['dataset']}: all_folds_pass={report['summary']['all_folds_pass']}")
        for fold in report["folds"]:
            best = fold["best"]
            if best:
                print(
                    f"  fold{fold['fold']} {fold['status']}: "
                    f"acc={best['accuracy']:.4f} f1={best['macro_f1']:.4f} {best['path']}"
                )
            else:
                print(f"  fold{fold['fold']} missing")
    print(f"wrote {out_json}")
    print(f"wrote {args.output_md}")


if __name__ == "__main__":
    main()
