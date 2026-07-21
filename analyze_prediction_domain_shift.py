import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


def label_names(label_map: Dict[str, int]) -> list[str]:
    names = [""] * len(label_map)
    for name, index in label_map.items():
        names[int(index)] = str(name)
    if any(not name for name in names):
        raise ValueError("label_map indices must be contiguous from zero")
    return names


def prediction_arrays(payload: dict, level: str) -> tuple[np.ndarray, np.ndarray]:
    candidates = {
        "flow": [("flow_y_true", "flow_y_pred")],
        "window": [("window_y_true", "window_y_pred")],
        "packet": [("packet_y_true", "packet_y_pred"), ("y_true", "y_pred")],
    }[level]
    for true_key, pred_key in candidates:
        if true_key in payload and pred_key in payload:
            return np.asarray(payload[true_key], dtype=np.int64), np.asarray(payload[pred_key], dtype=np.int64)
    raise KeyError(f"Could not find {level}-level prediction arrays; tried {candidates}")


def normalized_histogram(values: np.ndarray, num_classes: int) -> np.ndarray:
    counts = np.bincount(values, minlength=num_classes).astype(np.float64)
    return counts / max(float(counts.sum()), 1.0)


def jensen_shannon(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    midpoint = 0.5 * (p + q)

    def kl(left: np.ndarray, right: np.ndarray) -> float:
        mask = left > 0
        return float(np.sum(left[mask] * np.log(left[mask] / np.maximum(right[mask], 1e-12))))

    return 0.5 * (kl(p, midpoint) + kl(q, midpoint))


def split_report(payload: dict, level: str, names: list[str]) -> dict:
    y_true, y_pred = prediction_arrays(payload, level)
    labels = np.arange(len(names))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "num_samples": int(y_true.size),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(np.mean(f1)),
        "true_prior": normalized_histogram(y_true, len(names)).tolist(),
        "predicted_prior": normalized_histogram(y_pred, len(names)).tolist(),
        "per_class": {
            name: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, name in enumerate(names)
        },
        "confusion_matrix": matrix.tolist(),
    }


def nested_get(mapping: dict, path: Iterable[str]) -> Any:
    value: Any = mapping
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def gate_means(payload: dict) -> dict:
    diagnostics = nested_get(payload, ("metrics", "eval_config", "learned_gate_diagnostics"))
    if not isinstance(diagnostics, dict):
        return {}
    result = {}
    for gate_name, values in diagnostics.items():
        if not isinstance(values, dict):
            continue
        result[gate_name] = {
            key: values[key]
            for key in (
                "mean",
                "effective_routing_mean",
                "effective_routing_p05",
                "effective_routing_p50",
                "effective_routing_p95",
                "effective_mean",
                "effective_p05",
                "effective_p50",
                "effective_p95",
                "base_mode",
                "channel_names",
                "view_names",
                "weight_semantics",
                "max_residual_weight",
                "theoretical_bounds",
                "bounds_satisfied",
            )
            if key in values
        }
    return result


def compare_payloads(valid_payload: dict, test_payload: dict, level: str, top_k: int) -> dict:
    if valid_payload.get("label_map") != test_payload.get("label_map"):
        raise ValueError("Validation and test label maps differ")
    names = label_names(valid_payload["label_map"])
    valid = split_report(valid_payload, level, names)
    test = split_report(test_payload, level, names)
    matrix = np.asarray(test["confusion_matrix"], dtype=np.int64)
    confusions = []
    for source in range(len(names)):
        support = max(int(matrix[source].sum()), 1)
        for target in range(len(names)):
            if source == target or matrix[source, target] == 0:
                continue
            confusions.append(
                {
                    "true": names[source],
                    "predicted": names[target],
                    "count": int(matrix[source, target]),
                    "rate_within_true_class": float(matrix[source, target] / support),
                }
            )
    confusions.sort(key=lambda row: (row["rate_within_true_class"], row["count"]), reverse=True)
    class_shift = []
    for name in names:
        valid_item = valid["per_class"][name]
        test_item = test["per_class"][name]
        class_shift.append(
            {
                "class": name,
                "valid_f1": valid_item["f1"],
                "test_f1": test_item["f1"],
                "delta_f1": test_item["f1"] - valid_item["f1"],
                "valid_support": valid_item["support"],
                "test_support": test_item["support"],
            }
        )
    class_shift.sort(key=lambda row: row["delta_f1"])
    valid_true = np.asarray(valid["true_prior"])
    test_true = np.asarray(test["true_prior"])
    valid_pred = np.asarray(valid["predicted_prior"])
    test_pred = np.asarray(test["predicted_prior"])
    return {
        "scope": f"validation_to_test_{level}_domain_shift",
        "label_names": names,
        "summary": {
            "valid_accuracy": valid["accuracy"],
            "test_accuracy": test["accuracy"],
            "delta_accuracy": test["accuracy"] - valid["accuracy"],
            "valid_macro_f1": valid["macro_f1"],
            "test_macro_f1": test["macro_f1"],
            "delta_macro_f1": test["macro_f1"] - valid["macro_f1"],
            "true_prior_total_variation": float(0.5 * np.abs(valid_true - test_true).sum()),
            "predicted_prior_total_variation": float(0.5 * np.abs(valid_pred - test_pred).sum()),
            "true_prior_jensen_shannon": jensen_shannon(valid_true, test_true),
            "predicted_prior_jensen_shannon": jensen_shannon(valid_pred, test_pred),
        },
        "per_class_shift": class_shift,
        "top_test_confusions": confusions[:top_k],
        "gate_diagnostics": {
            "valid": gate_means(valid_payload),
            "test": gate_means(test_payload),
        },
        "valid": valid,
        "test": test,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--valid", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--level", choices=["packet", "window", "flow"], default="flow")
    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--output_json", required=True)
    args = ap.parse_args()
    with open(args.valid, "r", encoding="utf-8") as f:
        valid_payload = json.load(f)
    with open(args.test, "r", encoding="utf-8") as f:
        test_payload = json.load(f)
    report = compare_payloads(valid_payload, test_payload, args.level, args.top_k)
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps({
        "summary": report["summary"],
        "largest_class_drops": report["per_class_shift"][:10],
        "top_test_confusions": report["top_test_confusions"][:10],
        "gate_diagnostics": report["gate_diagnostics"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
