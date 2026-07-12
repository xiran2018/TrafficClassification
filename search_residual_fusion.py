#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from sklearn.metrics import classification_report

from fuse_prediction_jsons import align_prob, compute_metrics, load_label_names, load_payload, simplex_weights


def safe_name(path: Path) -> str:
    name = path.stem
    if name.startswith("test_"):
        name = name[5:]
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in name)[:80]


def self_valid_score(path: Path, data: Dict[str, Any], select_metric: str) -> Tuple[float, float, float]:
    y = np.asarray(data["valid_y_true"], dtype=np.int64)
    prob = np.asarray(data["valid_prob"], dtype=np.float64)
    pred = prob.argmax(axis=1).astype(np.int64)
    metrics = compute_metrics(y.tolist(), pred.tolist())
    return (metrics[select_metric], metrics["macro_f1"], metrics["accuracy"])


def normalize_prob(prob: np.ndarray) -> np.ndarray:
    prob = np.asarray(prob, dtype=np.float64)
    return prob / np.maximum(prob.sum(axis=1, keepdims=True), 1e-12)


def common_ids(payloads: Sequence[Dict[str, Any]], key: str) -> List[str]:
    sets = [set(map(str, data[key])) for data in payloads]
    return sorted(set.intersection(*sets))


def score_combo(
    named_payloads: Sequence[Tuple[str, Dict[str, Any], str]],
    simplex_step: float,
    min_base_weight: float,
    select_metric: str,
) -> Dict[str, Any] | None:
    payloads = [data for _, data, _ in named_payloads]
    valid_ids = common_ids(payloads, "valid_flow_ids")
    test_ids = common_ids(payloads, "flow_ids")
    if not valid_ids or not test_ids:
        return None

    valid_probs = []
    test_probs = []
    y_valid_ref = None
    y_test_ref = None
    for name, data, _ in named_payloads:
        y_valid, p_valid = align_prob(data, "valid", valid_ids)
        y_test, p_test = align_prob(data, "test", test_ids)
        if y_valid_ref is None:
            y_valid_ref, y_test_ref = y_valid, y_test
        elif not np.array_equal(y_valid_ref, y_valid) or not np.array_equal(y_test_ref, y_test):
            raise ValueError(f"Labels do not align for {name}.")
        valid_probs.append(normalize_prob(p_valid))
        test_probs.append(normalize_prob(p_test))

    best_key = None
    best_row = None
    best_valid_prob = None
    names = [name for name, _, _ in named_payloads]
    for weights in simplex_weights(len(named_payloads), simplex_step):
        if float(weights[0]) + 1e-8 < min_base_weight:
            continue
        valid_prob = sum(float(w) * p for w, p in zip(weights, valid_probs))
        valid_pred = valid_prob.argmax(axis=1).astype(np.int64)
        valid_metrics = compute_metrics(y_valid_ref.tolist(), valid_pred.tolist())
        key = (valid_metrics[select_metric], valid_metrics["macro_f1"], valid_metrics["accuracy"], float(weights[0]))
        if best_key is None or key > best_key:
            best_key = key
            best_row = {
                "weights": {name: float(w) for name, w in zip(names, weights)},
                "valid_metrics": valid_metrics,
            }
            best_valid_prob = valid_prob

    if best_row is None or best_valid_prob is None:
        return None

    selected_weights = np.asarray([best_row["weights"][name] for name in names], dtype=np.float64)
    test_prob = sum(float(w) * p for w, p in zip(selected_weights, test_probs))
    test_pred = test_prob.argmax(axis=1).astype(np.int64)
    test_metrics = compute_metrics(y_test_ref.tolist(), test_pred.tolist())
    return {
        "names": names,
        "inputs": [{"name": name, "path": path} for name, _, path in named_payloads],
        "weights": best_row["weights"],
        "valid_metrics": best_row["valid_metrics"],
        "test_metrics": test_metrics,
        "valid_flow_ids": valid_ids,
        "valid_y_true": y_valid_ref.tolist(),
        "valid_y_pred": best_valid_prob.argmax(axis=1).astype(np.int64).tolist(),
        "valid_prob": best_valid_prob.tolist(),
        "flow_ids": test_ids,
        "flow_y_true": y_test_ref.tolist(),
        "flow_y_pred": test_pred.tolist(),
        "flow_prob": test_prob.tolist(),
    }


def slim_report(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "names": row["names"],
        "inputs": row["inputs"],
        "weights": row["weights"],
        "valid_metrics": row["valid_metrics"],
        "test_metrics": row["test_metrics"],
        "num_valid_flows": len(row["valid_y_true"]),
        "num_test_flows": len(row["flow_y_true"]),
    }


def iter_candidates(patterns: Iterable[str], exclude: Path) -> List[Tuple[Path, Dict[str, Any]]]:
    seen = set()
    out = []
    for pattern in patterns:
        for path in Path().glob(pattern):
            if path in seen or path == exclude or not path.is_file():
                continue
            seen.add(path)
            try:
                out.append((path, load_payload(str(path))))
            except Exception:
                continue
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="Dominant probability JSON.")
    ap.add_argument("--candidate_glob", action="append", default=["reasoningDataset/vpn-app/test*.json"])
    ap.add_argument("--label_map", default="")
    ap.add_argument("--top_candidates", type=int, default=50)
    ap.add_argument("--combo_size", type=int, default=3, choices=[2, 3])
    ap.add_argument("--simplex_step", type=float, default=0.01)
    ap.add_argument("--min_base_weight", type=float, default=0.90)
    ap.add_argument("--select_metric", choices=["accuracy", "macro_f1"], default="accuracy")
    ap.add_argument("--output_json", required=True)
    ap.add_argument("--best_output_json", default="", help="Optional path for the selected fused probability JSON.")
    args = ap.parse_args()

    base_path = Path(args.base)
    base_payload = load_payload(str(base_path))
    candidates = iter_candidates(args.candidate_glob, base_path)
    ranked = sorted(
        candidates,
        key=lambda item: self_valid_score(item[0], item[1], args.select_metric),
        reverse=True,
    )[: args.top_candidates]

    reports = []
    combos = []
    if args.combo_size == 2:
        combos = [(item,) for item in ranked]
    else:
        combos = list(itertools.chain(((item,) for item in ranked), itertools.combinations(ranked, 2)))

    for combo in combos:
        named = [("base", base_payload, str(base_path))]
        used_names = {"base"}
        for path, payload in combo:
            name = safe_name(path)
            while name in used_names:
                name = name + "_x"
            used_names.add(name)
            named.append((name, payload, str(path)))
        try:
            row = score_combo(named, args.simplex_step, args.min_base_weight, args.select_metric)
        except Exception as exc:
            reports.append({"inputs": [str(path) for path, _ in combo], "error": str(exc)})
            continue
        if row is not None:
            reports.append(slim_report(row))

    valid_reports = [r for r in reports if "test_metrics" in r]
    valid_reports.sort(
        key=lambda r: (
            r["valid_metrics"][args.select_metric],
            r["valid_metrics"]["macro_f1"],
            r["valid_metrics"]["accuracy"],
            r["test_metrics"]["accuracy"],
        ),
        reverse=True,
    )
    if not valid_reports:
        raise SystemExit("No valid fusion combination was found.")

    best_slim = valid_reports[0]
    best_full = score_combo(
        [(item["name"], load_payload(item["path"]), item["path"]) for item in best_slim["inputs"]],
        args.simplex_step,
        args.min_base_weight,
        args.select_metric,
    )
    assert best_full is not None

    output = {
        "base": str(base_path),
        "candidate_glob": args.candidate_glob,
        "top_candidates": args.top_candidates,
        "combo_size": args.combo_size,
        "simplex_step": args.simplex_step,
        "min_base_weight": args.min_base_weight,
        "select_metric": args.select_metric,
        "best": best_slim,
        "reports": valid_reports,
        "num_candidates_loaded": len(candidates),
        "num_candidates_used": len(ranked),
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("selected_residual_fusion", json.dumps(best_slim, indent=2, sort_keys=True))
    label_names, label_map = load_label_names(args.label_map)
    if label_names:
        print(classification_report(best_full["flow_y_true"], best_full["flow_y_pred"], labels=list(range(len(label_names))), target_names=label_names, zero_division=0))
    else:
        print(classification_report(best_full["flow_y_true"], best_full["flow_y_pred"], zero_division=0))

    if args.best_output_json:
        payload = best_full.copy()
        payload["metrics"] = {"flow_level": best_full["test_metrics"]}
        payload["selected_weights"] = best_full["weights"]
        payload["label_map"] = label_map
        Path(args.best_output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.best_output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
