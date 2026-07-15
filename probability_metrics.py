from __future__ import annotations

from typing import Any

import numpy as np


def calibration_metrics(y_true: Any, prob: Any, num_bins: int = 15) -> dict[str, Any]:
    if y_true is None or prob is None or len(y_true) == 0 or len(prob) == 0:
        return {
            "nll": None,
            "brier": None,
            "ece": None,
            "avg_confidence": None,
            "accuracy": None,
            "num_samples": 0,
            "num_bins": int(num_bins),
        }
    p = np.asarray(prob, dtype=np.float64)
    y = np.asarray(y_true, dtype=np.int64)
    if p.ndim != 2 or p.shape[0] != y.shape[0] or p.shape[1] == 0:
        return {
            "nll": None,
            "brier": None,
            "ece": None,
            "avg_confidence": None,
            "accuracy": None,
            "num_samples": int(len(y)),
            "num_bins": int(num_bins),
        }
    p = np.clip(p, 1e-12, None)
    p = p / p.sum(axis=1, keepdims=True).clip(min=1e-12)
    safe_y = y.clip(min=0, max=p.shape[1] - 1)
    pred = p.argmax(axis=1)
    conf = p.max(axis=1)
    correct = pred == y
    true_prob = p[np.arange(len(y)), safe_y].clip(min=1e-12)
    onehot = np.zeros_like(p)
    onehot[np.arange(len(y)), safe_y] = 1.0

    ece = 0.0
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    for i in range(num_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == num_bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        if mask.any():
            ece += float(mask.mean()) * abs(float(correct[mask].mean()) - float(conf[mask].mean()))

    return {
        "nll": float(-np.log(true_prob).mean()),
        "brier": float(((p - onehot) ** 2).sum(axis=1).mean()),
        "ece": float(ece),
        "avg_confidence": float(conf.mean()),
        "accuracy": float(correct.mean()),
        "num_samples": int(len(y)),
        "num_bins": int(num_bins),
    }
