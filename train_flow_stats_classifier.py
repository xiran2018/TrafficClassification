#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

COMMON_PORTS = [20, 21, 22, 25, 53, 80, 110, 143, 443, 465, 587, 993, 995, 1194, 6881]
MESSAGE_PREFIX_LEN = 16


def load_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def stats(values: List[float]) -> List[float]:
    if not values:
        return [0.0] * 8
    a = np.asarray(values, dtype=np.float32)
    return [
        float(len(a)),
        float(a.mean()),
        float(a.std()),
        float(a.min()),
        float(np.percentile(a, 25)),
        float(np.percentile(a, 50)),
        float(np.percentile(a, 75)),
        float(a.max()),
    ]


def hist(values: List[float], bins: List[float]) -> List[float]:
    if not values:
        return [0.0] * (len(bins) + 1)
    counts, _ = np.histogram(np.asarray(values, dtype=np.float32), bins=np.asarray(bins, dtype=np.float32))
    below = sum(1 for v in values if v < bins[0])
    above = sum(1 for v in values if v >= bins[-1])
    full = np.concatenate([[below], counts[1:-1], [above]]).astype(np.float32)
    return (full / max(float(len(values)), 1.0)).tolist()


def bool_float(value: Any) -> float:
    return 1.0 if bool(value) else 0.0


def port_features(ports: List[int]) -> List[float]:
    valid = [int(p) for p in ports if int(p) >= 0]
    if not valid:
        return [0.0, 0.0, 0.0] + [0.0] * len(COMMON_PORTS)
    c = Counter(valid)
    top_port, top_count = c.most_common(1)[0]
    feats = [math.log1p(top_port), top_count / max(len(valid), 1), len(c) / max(len(valid), 1)]
    feats.extend([1.0 if p in c else 0.0 for p in COMMON_PORTS])
    return feats


def build_segments(
    dirs: List[float],
    lengths: List[float],
    payloads: List[float],
    iats: List[float],
    entropies: List[float],
    split_gap: float | None = None,
) -> List[Dict[str, float]]:
    if not dirs:
        return []
    segments: List[Dict[str, float]] = []
    start = 0
    for i in range(1, len(dirs)):
        gap_break = split_gap is not None and iats[i] > split_gap
        if dirs[i] != dirs[i - 1] or gap_break:
            segments.append(segment_summary(start, i, dirs, lengths, payloads, iats, entropies))
            start = i
    segments.append(segment_summary(start, len(dirs), dirs, lengths, payloads, iats, entropies))
    return segments


def segment_summary(
    start: int,
    end: int,
    dirs: List[float],
    lengths: List[float],
    payloads: List[float],
    iats: List[float],
    entropies: List[float],
) -> Dict[str, float]:
    idxs = list(range(start, end))
    byte_sum = sum(lengths[i] for i in idxs)
    payload_sum = sum(payloads[i] for i in idxs)
    duration = sum(iats[i] for i in idxs[1:])
    return {
        "start": float(start),
        "end": float(end),
        "dir": dirs[start],
        "count": float(end - start),
        "bytes": float(byte_sum),
        "payload": float(payload_sum),
        "duration": float(duration),
        "gap": float(iats[start] if start > 0 else 0.0),
        "entropy": float(np.mean([entropies[i] for i in idxs])) if idxs else 0.0,
    }


def segment_set_features(segments: List[Dict[str, float]], packet_count: int, total_bytes: float, total_payload: float) -> List[float]:
    feats: List[float] = [math.log1p(len(segments))]
    if not segments:
        return feats + [0.0] * (7 + 3 * 7 * 8 + 8 + 7 + MESSAGE_PREFIX_LEN * 4)

    c2s = [s for s in segments if s["dir"] > 0]
    s2c = [s for s in segments if s["dir"] < 0]
    changed = sum(1 for i in range(1, len(segments)) if segments[i]["dir"] != segments[i - 1]["dir"])
    same_adjacent = max(len(segments) - 1 - changed, 0)
    max_run = 1
    cur_run = 1
    for i in range(1, len(segments)):
        if segments[i]["dir"] == segments[i - 1]["dir"]:
            cur_run += 1
        else:
            max_run = max(max_run, cur_run)
            cur_run = 1
    max_run = max(max_run, cur_run)

    feats.extend([
        len(c2s) / len(segments),
        len(s2c) / len(segments),
        sum(s["bytes"] for s in c2s) / max(total_bytes, 1.0),
        sum(s["payload"] for s in c2s) / max(total_payload, 1.0),
        changed / max(len(segments) - 1, 1),
        same_adjacent / max(len(segments) - 1, 1),
        max_run / max(len(segments), 1),
    ])

    for subset in [segments, c2s, s2c]:
        feats.extend(stats([math.log1p(s["count"]) for s in subset]))
        feats.extend(stats([math.log1p(s["bytes"]) for s in subset]))
        feats.extend(stats([math.log1p(s["payload"]) for s in subset]))
        feats.extend(stats([math.log1p(s["duration"]) for s in subset]))
        feats.extend(stats([math.log1p(s["gap"]) for s in subset]))
        feats.extend(stats([s["payload"] / max(s["bytes"], 1.0) for s in subset]))
        feats.extend(stats([s["entropy"] / 8.0 for s in subset]))

    feats.extend(hist([s["count"] for s in segments], [1, 2, 3, 4, 8, 16, 32]))
    feats.extend(hist([math.log1p(s["bytes"]) for s in segments], [3, 4, 5, 6, 7, 8]))

    scale = math.log1p(max(total_bytes, 1514.0))
    for i in range(MESSAGE_PREFIX_LEN):
        if i < len(segments):
            s = segments[i]
            feats.append(s["dir"] * math.log1p(s["bytes"]) / scale)
            feats.append(math.log1p(s["payload"]) / scale)
            feats.append(math.log1p(s["count"]) / math.log1p(max(packet_count, 2)))
            feats.append(math.log1p(s["gap"]))
        else:
            feats.extend([0.0, 0.0, 0.0, 0.0])
    return feats


def message_features(
    dirs: List[float],
    lengths: List[float],
    payloads: List[float],
    iats: List[float],
    entropies: List[float],
) -> List[float]:
    if not dirs:
        return []
    total_bytes = sum(lengths)
    total_payload = sum(payloads)
    feats: List[float] = []

    dir_segments = build_segments(dirs, lengths, payloads, iats, entropies)
    feats.extend(segment_set_features(dir_segments, len(dirs), total_bytes, total_payload))
    for gap in [0.01, 0.1, 1.0]:
        feats.extend(segment_set_features(build_segments(dirs, lengths, payloads, iats, entropies, split_gap=gap), len(dirs), total_bytes, total_payload))

    response_ratios = []
    response_delays = []
    response_payload_ratios = []
    for prev, cur in zip(dir_segments, dir_segments[1:]):
        if cur["dir"] == prev["dir"]:
            continue
        response_ratios.append(math.log1p(cur["bytes"]) - math.log1p(prev["bytes"]))
        response_payload_ratios.append(math.log1p(cur["payload"]) - math.log1p(prev["payload"]))
        response_delays.append(math.log1p(cur["gap"]))
    feats.extend(stats(response_ratios))
    feats.extend(stats(response_payload_ratios))
    feats.extend(stats(response_delays))

    same_dir_gaps = []
    same_dir_byte_deltas = []
    same_dir_count_deltas = []
    last_by_dir: Dict[float, Dict[str, float]] = {}
    for seg in dir_segments:
        prev = last_by_dir.get(seg["dir"])
        if prev is not None:
            same_dir_gaps.append(math.log1p(max(seg["gap"], 0.0)))
            same_dir_byte_deltas.append(math.log1p(seg["bytes"]) - math.log1p(prev["bytes"]))
            same_dir_count_deltas.append(math.log1p(seg["count"]) - math.log1p(prev["count"]))
        last_by_dir[seg["dir"]] = seg
    feats.extend(stats(same_dir_gaps))
    feats.extend(stats(same_dir_byte_deltas))
    feats.extend(stats(same_dir_count_deltas))

    if len(dirs) > 1:
        feats.extend([
            sum(1 for i in range(1, len(dirs)) if dirs[i] != dirs[i - 1]) / (len(dirs) - 1),
            sum(1 for i in range(1, len(dirs)) if dirs[i] == dirs[i - 1]) / (len(dirs) - 1),
        ])
    else:
        feats.extend([0.0, 0.0])
    return feats


def parse_hex_prefix(value: Any) -> List[int]:
    out: List[int] = []
    for part in str(value or "").replace(",", " ").split():
        try:
            out.append(int(part, 16))
        except ValueError:
            continue
    return out


def header_features(metas: List[Dict[str, Any]], dirs: List[float], prefix_len: int, full_byte_sketch: bool = False) -> List[float]:
    feats: List[float] = []
    if not metas:
        return feats
    c2s_idx = [i for i, m in enumerate(metas) if m.get("direction") == "C2S"]
    s2c_idx = [i for i, m in enumerate(metas) if m.get("direction") == "S2C"]
    numeric_fields = [
        ("ip_total_len", 1514.0),
        ("ip_header_len", 60.0),
        ("l3_captured_len", 1514.0),
        ("payload_prefix_len", 1400.0),
        ("tcp_window", 65535.0),
        ("tcp_data_offset", 60.0),
        ("udp_len", 1500.0),
    ]
    for idxs in [list(range(len(metas))), c2s_idx, s2c_idx]:
        for field, scale in numeric_fields:
            values = []
            for i in idxs:
                value = float(metas[i].get(field, 0) or 0)
                if value < 0:
                    value = 0.0
                values.append(math.log1p(value) / math.log1p(scale))
            feats.extend(stats(values))
        for field in ["ip_checksum_valid", "l4_checksum_valid", "full_l3_captured"]:
            feats.extend(stats([bool_float(metas[i].get(field, False)) for i in idxs]))

    tcp_windows = [float(m.get("tcp_window", 0) or 0) for m in metas]
    feats.extend(hist(tcp_windows, [1, 1024, 4096, 8192, 16384, 32768, 65535]))
    data_offsets = [float(m.get("tcp_data_offset", 0) or 0) for m in metas]
    feats.extend(hist(data_offsets, [20, 24, 28, 32, 40, 60]))

    # Structural L3/L4 byte sketch. The default avoids IPv4 src/dst bytes
    # (12..19) and checksums; full_byte_sketch is for shortcut audits.
    byte_positions = list(range(40)) if full_byte_sketch else [0, 1, 2, 3, 6, 7, 8, 9, 20, 21, 22, 23, 32, 33, 34, 35]
    for i in range(prefix_len):
        if i < len(metas):
            m = metas[i]
            tcp_window = max(float(m.get("tcp_window", 0) or 0), 0.0)
            ip_total_len = max(float(m.get("ip_total_len", 0) or 0), 0.0)
            l3_captured = max(float(m.get("l3_captured_len", 0) or 0), 0.0)
            payload_prefix = max(float(m.get("payload_prefix_len", 0) or 0), 0.0)
            feats.extend([
                dirs[i] * math.log1p(ip_total_len) / math.log1p(1514.0),
                math.log1p(l3_captured) / math.log1p(1514.0),
                math.log1p(payload_prefix) / math.log1p(1400.0),
                math.log1p(tcp_window) / math.log1p(65535.0),
                max(float(m.get("tcp_data_offset", 0) or 0), 0.0) / 60.0,
                bool_float(m.get("ip_checksum_valid", False)),
                bool_float(m.get("l4_checksum_valid", False)),
            ])
            raw_bytes = parse_hex_prefix(m.get("l3_hex_prefix", ""))
            for pos in byte_positions:
                feats.append((raw_bytes[pos] / 255.0) if pos < len(raw_bytes) else 0.0)
        else:
            feats.extend([0.0] * (7 + len(byte_positions)))
    return feats


def stable_bucket(value: str, buckets: int) -> int:
    digest = hashlib.md5(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False) % buckets


def ip24(value: Any) -> str:
    text = str(value or "")
    parts = text.split(".")
    if len(parts) == 4:
        return ".".join(parts[:3]) + ".0/24"
    return text


def hashed_counter_features(values: List[str], buckets: int) -> List[float]:
    feats = [0.0] * buckets
    if not values:
        return feats + [0.0, 0.0]
    for value in values:
        feats[stable_bucket(value, buckets)] += 1.0
    total = max(float(len(values)), 1.0)
    feats = [x / total for x in feats]
    return feats + [max(feats), sum(1 for x in feats if x > 0) / buckets]


def endpoint_features(metas: List[Dict[str, Any]], buckets: int = 256) -> List[float]:
    server_ips = []
    client_ips = []
    server_subnets = []
    client_subnets = []
    endpoint_pairs = []
    for m in metas:
        src = str(m.get("src_ip", ""))
        dst = str(m.get("dst_ip", ""))
        if m.get("direction") == "C2S":
            client, server = src, dst
        else:
            client, server = dst, src
        if server:
            server_ips.append(server)
            server_subnets.append(ip24(server))
        if client:
            client_ips.append(client)
            client_subnets.append(ip24(client))
        if client or server:
            endpoint_pairs.append(f"{ip24(client)}->{ip24(server)}")
    feats: List[float] = []
    for values in [server_ips, server_subnets, client_subnets, endpoint_pairs]:
        feats.extend(hashed_counter_features(values, buckets))
    return feats


def protocol_payload_prefix(meta: Dict[str, Any], limit: int = 8) -> List[int]:
    """Extract payload bytes while excluding IP and transport header fields."""
    raw = parse_hex_prefix(meta.get("l3_hex_prefix", ""))
    ip_header_len = max(int(meta.get("ip_header_len", 0) or 0), 0)
    if not raw or ip_header_len <= 0:
        return []
    l4 = str(meta.get("l4", "")).upper()
    if l4 == "TCP":
        transport_header_len = max(int(meta.get("tcp_data_offset", 0) or 0), 0)
        if transport_header_len <= 0:
            return []
    elif l4 == "UDP":
        transport_header_len = 8
    else:
        transport_header_len = 0
    start = min(ip_header_len + transport_header_len, len(raw))
    declared = max(int(meta.get("payload_prefix_len", 0) or 0), 0)
    available = min(declared, max(len(raw) - start, 0), limit)
    return raw[start : start + available]


def protocol_closed_payload_features(
    metas: List[Dict[str, Any]], prefix_len: int, bytes_per_packet: int = 8
) -> List[float]:
    feats: List[float] = []
    histogram = np.zeros(16, dtype=np.float64)
    total_bytes = 0
    for index in range(prefix_len):
        payload = (
            protocol_payload_prefix(metas[index], bytes_per_packet)
            if index < len(metas)
            else []
        )
        feats.extend([value / 255.0 for value in payload])
        feats.extend([0.0] * (bytes_per_packet - len(payload)))
        feats.append(len(payload) / max(bytes_per_packet, 1))
        for value in payload:
            histogram[min(value // 16, 15)] += 1.0
            total_bytes += 1
    if total_bytes:
        histogram /= total_bytes
    feats.extend(histogram.tolist())
    return feats


def flow_features(row: Dict[str, Any], max_packets: int, prefix_len: int, use_ports: bool, feature_version: str = "basic") -> List[float]:
    protocol_closed = feature_version in {"protocol_closed", "protocol_closed_payload"}
    if protocol_closed and use_ports:
        raise ValueError("protocol_closed flow features cannot include ports")
    metas = list(row.get("packet_metas", []))[:max_packets]
    feats: List[float] = [math.log1p(len(metas))]
    if not metas:
        return feats

    dirs = [1.0 if m.get("direction") == "C2S" else -1.0 for m in metas]
    lengths = [float(m.get("packet_len", 0) or 0) for m in metas]
    payloads = [float(m.get("payload_len", 0) or 0) for m in metas]
    iats = [float(m.get("iat", 0.0) or 0.0) for m in metas]
    entropies = [float(m.get("payload_entropy", 0.0) or 0.0) for m in metas]
    ttl = [float(m.get("ip_ttl", 0) or 0) for m in metas]
    tcp = [1.0 if m.get("l4") == "TCP" else 0.0 for m in metas]
    udp = [1.0 if m.get("l4") == "UDP" else 0.0 for m in metas]
    c2s_idx = [i for i, m in enumerate(metas) if m.get("direction") == "C2S"]
    s2c_idx = [i for i, m in enumerate(metas) if m.get("direction") == "S2C"]

    feats.extend([
        sum(1 for d in dirs if d > 0) / len(dirs),
        sum(1 for d in dirs if d < 0) / len(dirs),
        sum(lengths[i] for i in c2s_idx) / max(sum(lengths), 1.0),
        sum(payloads[i] for i in c2s_idx) / max(sum(payloads), 1.0),
        float(np.mean(tcp)),
        float(np.mean(udp)),
        float(np.mean([bool_float(m.get("full_l3_captured", False)) for m in metas])),
    ])

    for idxs in [list(range(len(metas))), c2s_idx, s2c_idx]:
        feats.extend(stats([math.log1p(lengths[i]) for i in idxs]))
        feats.extend(stats([math.log1p(payloads[i]) for i in idxs]))
        feats.extend(stats([math.log1p(iats[i]) for i in idxs]))
        feats.extend(stats([entropies[i] / 8.0 for i in idxs]))

    feats.extend(hist(lengths, [64, 128, 256, 512, 768, 1024, 1280, 1514]))
    feats.extend(hist(payloads, [1, 16, 64, 128, 256, 512, 1024, 1400]))
    feats.extend(hist([math.log1p(x) for x in iats], [0.0001, 0.001, 0.01, 0.1, 1.0, 5.0]))
    if not protocol_closed:
        feats.extend(stats(ttl))

    flag_counts = Counter()
    for m in metas:
        for flag in str(m.get("tcp_flags", "") or ""):
            flag_counts[flag] += 1
    for flag in ["S", "A", "P", "F", "R"]:
        feats.append(flag_counts[flag] / len(metas))

    for i in range(prefix_len):
        if i < len(metas):
            feats.append(dirs[i] * math.log1p(lengths[i]) / math.log1p(1514))
            feats.append(math.log1p(payloads[i]) / math.log1p(1514))
            feats.append(math.log1p(iats[i]))
            feats.append(entropies[i] / 8.0)
        else:
            feats.extend([0.0, 0.0, 0.0, 0.0])

    if feature_version in {"protocol_closed", "protocol_closed_payload", "message", "message_header", "message_header_endpoint", "message_header_fullbytes"}:
        feats.extend(message_features(dirs, lengths, payloads, iats, entropies))
    if feature_version == "protocol_closed_payload":
        feats.extend(protocol_closed_payload_features(metas, prefix_len))
    if feature_version in {"message_header", "message_header_endpoint", "message_header_fullbytes"}:
        feats.extend(header_features(metas, dirs, prefix_len, full_byte_sketch=feature_version == "message_header_fullbytes"))
    if feature_version == "message_header_endpoint":
        feats.extend(endpoint_features(metas))
    elif feature_version not in {"basic", "protocol_closed", "protocol_closed_payload", "message", "message_header", "message_header_fullbytes"}:
        raise ValueError(f"Unknown feature_version: {feature_version}")

    if use_ports:
        server_ports = []
        client_ports = []
        for m in metas:
            sport, dport = int(m.get("sport", -1)), int(m.get("dport", -1))
            if m.get("direction") == "C2S":
                server_ports.append(dport)
                client_ports.append(sport)
            else:
                server_ports.append(sport)
                client_ports.append(dport)
        feats.extend(port_features(server_ports))
        feats.extend(port_features(client_ports))
    return feats


def load_split(path: str, max_packets: int, prefix_len: int, use_ports: bool, feature_version: str = "basic") -> Tuple[np.ndarray, np.ndarray, List[str]]:
    xs, ys, fids = [], [], []
    for row in load_jsonl(path):
        xs.append(flow_features(row, max_packets, prefix_len, use_ports, feature_version))
        ys.append(int(row["label_id"]))
        fids.append(str(row.get("flow_id", "")))
    width = max(len(x) for x in xs)
    x_arr = np.zeros((len(xs), width), dtype=np.float32)
    for i, x in enumerate(xs):
        x_arr[i, :len(x)] = x
    return x_arr, np.asarray(ys, dtype=np.int64), fids


def compute_metrics(y_true, y_pred):
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    p_weight, r_weight, f_weight, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    return {
        "accuracy": accuracy_score(y_true, y_pred) if len(y_true) else 0.0,
        "macro_precision": p_macro,
        "macro_recall": r_macro,
        "macro_f1": f_macro,
        "weighted_precision": p_weight,
        "weighted_recall": r_weight,
        "weighted_f1": f_weight,
    }


def make_model(kind: str, n_estimators: int, max_depth: int | None, min_samples_leaf: int, class_weight: str | None, seed: int):
    if kind == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features="sqrt",
            class_weight=class_weight,
            random_state=seed,
            n_jobs=1,
        )
    if kind == "random_forest":
        return RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features="sqrt",
            class_weight=class_weight,
            random_state=seed,
            n_jobs=1,
        )
    if kind == "logistic":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs", class_weight=class_weight, random_state=seed),
        )
    if kind == "hist_gbdt":
        return HistGradientBoostingClassifier(
            max_iter=min(max(50, n_estimators), 300),
            learning_rate=0.05,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            l2_regularization=0.01,
            class_weight=class_weight,
            random_state=seed,
        )
    if kind == "gradient_boosting":
        return GradientBoostingClassifier(
            n_estimators=min(max(50, n_estimators), 300),
            learning_rate=0.05,
            max_depth=3 if max_depth is None else max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=seed,
        )
    if kind == "mlp":
        return make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(256, 128),
                activation="relu",
                alpha=1e-3,
                batch_size=64,
                learning_rate_init=1e-3,
                max_iter=500,
                early_stopping=True,
                n_iter_no_change=25,
                random_state=seed,
            ),
        )
    if kind == "knn":
        return make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(n_neighbors=max(3, min_samples_leaf * 3), weights="distance"),
        )
    if kind == "svc":
        return make_pipeline(
            StandardScaler(),
            SVC(C=3.0, gamma="scale", probability=True, class_weight=class_weight, random_state=seed),
        )
    raise ValueError(f"Unknown model kind: {kind}")


def aligned_predict_proba(model, x: np.ndarray, num_classes: int) -> np.ndarray:
    raw = model.predict_proba(x)
    classes = getattr(model, "classes_", None)
    if classes is None and hasattr(model, "named_steps"):
        classes = model.named_steps[list(model.named_steps.keys())[-1]].classes_
    out = np.zeros((x.shape[0], num_classes), dtype=np.float32)
    for col, cls in enumerate(classes):
        out[:, int(cls)] = raw[:, col]
    out = out / np.maximum(out.sum(axis=1, keepdims=True), 1e-12)
    return out


def iter_model_candidates(model_kinds: str):
    for kind in [x.strip() for x in model_kinds.split(",") if x.strip()]:
        if kind in {"extra_trees", "random_forest"}:
            for class_weight in [None, "balanced"]:
                for max_depth in [None, 8, 16]:
                    for min_samples_leaf in [1, 2, 4, 8]:
                        yield kind, max_depth, min_samples_leaf, class_weight
        elif kind == "hist_gbdt":
            for class_weight in [None, "balanced"]:
                for max_depth in [None, 4, 8]:
                    for min_samples_leaf in [5, 10, 20]:
                        yield kind, max_depth, min_samples_leaf, class_weight
        elif kind == "gradient_boosting":
            for max_depth in [2, 3, 5]:
                for min_samples_leaf in [1, 2, 4]:
                    yield kind, max_depth, min_samples_leaf, None
        elif kind == "logistic":
            for class_weight in [None, "balanced"]:
                yield kind, None, 1, class_weight
        elif kind == "mlp":
            yield kind, None, 1, None
        elif kind == "svc":
            for class_weight in [None, "balanced"]:
                yield kind, None, 1, class_weight
        elif kind == "knn":
            for min_samples_leaf in [1, 2, 4, 8]:
                yield kind, None, min_samples_leaf, None
        else:
            raise ValueError(f"Unknown model kind: {kind}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_index", required=True)
    ap.add_argument("--valid_index", required=True)
    ap.add_argument("--test_index", required=True)
    ap.add_argument("--label_map", default="")
    ap.add_argument("--model_kinds", default="extra_trees,random_forest")
    ap.add_argument("--max_packets", type=int, default=64)
    ap.add_argument("--prefix_len", type=int, default=32)
    ap.add_argument("--use_ports", action="store_true")
    ap.add_argument(
        "--feature_version",
        choices=[
            "basic",
            "protocol_closed",
            "protocol_closed_payload",
            "message",
            "message_header",
            "message_header_endpoint",
            "message_header_fullbytes",
        ],
        default="basic",
    )
    ap.add_argument("--select_metric", choices=["accuracy", "macro_f1"], default="macro_f1")
    ap.add_argument("--output_json", default="")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.feature_version in {"protocol_closed", "protocol_closed_payload"} and args.use_ports:
        ap.error("protocol-closed feature versions cannot be combined with --use_ports")

    x_train, y_train, _ = load_split(args.train_index, args.max_packets, args.prefix_len, args.use_ports, args.feature_version)
    x_valid, y_valid, valid_fids = load_split(args.valid_index, args.max_packets, args.prefix_len, args.use_ports, args.feature_version)
    x_test, y_test, test_fids = load_split(args.test_index, args.max_packets, args.prefix_len, args.use_ports, args.feature_version)
    print(f"features train={x_train.shape} valid={x_valid.shape} test={x_test.shape}")

    grids = list(iter_model_candidates(args.model_kinds))

    reports = []
    best = None
    for kind, max_depth, min_samples_leaf, class_weight in grids:
        model = make_model(kind, 500, max_depth, min_samples_leaf, class_weight, args.seed)
        model.fit(x_train, y_train)
        pred = model.predict(x_valid)
        metrics = compute_metrics(y_valid.tolist(), pred.tolist())
        row = {
            "kind": kind,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "class_weight": class_weight,
            "metrics": metrics,
        }
        reports.append(row)
        print("valid", json.dumps(row, sort_keys=True))
        key = (metrics[args.select_metric], metrics["accuracy"])
        if best is None or key > best[0]:
            best = (key, row)

    selected = best[1]
    print("selected", json.dumps(selected, sort_keys=True))
    selected_model = make_model(
        selected["kind"],
        800,
        selected["max_depth"],
        selected["min_samples_leaf"],
        selected["class_weight"],
        args.seed,
    )
    selected_model.fit(x_train, y_train)
    final_model = make_model(
        selected["kind"],
        800,
        selected["max_depth"],
        selected["min_samples_leaf"],
        selected["class_weight"],
        args.seed,
    )
    x_final = np.concatenate([x_train, x_valid], axis=0)
    y_final = np.concatenate([y_train, y_valid], axis=0)
    final_model.fit(x_final, y_final)
    y_pred = final_model.predict(x_test)
    num_classes = int(max(y_final.max(), y_test.max()) + 1)
    valid_prob = aligned_predict_proba(selected_model, x_valid, num_classes)
    test_prob = aligned_predict_proba(final_model, x_test, num_classes)
    test_metrics = compute_metrics(y_test.tolist(), y_pred.tolist())
    print("test", json.dumps(test_metrics, indent=2, sort_keys=True))

    label_names = None
    label_map = None
    if args.label_map:
        with open(args.label_map, "r", encoding="utf-8") as f:
            label_map = json.load(f)
        label_names = [str(i) for i in range(max(int(v) for v in label_map.values()) + 1)]
        for name, idx in label_map.items():
            label_names[int(idx)] = name
    if label_names:
        print(classification_report(y_test, y_pred, labels=list(range(len(label_names))), target_names=label_names, zero_division=0))
    else:
        print(classification_report(y_test, y_pred, zero_division=0))

    if args.output_json:
        payload = {
            "metrics": {"flow_level": test_metrics},
            "selected": selected,
            "valid_reports": reports,
            "label_map": label_map,
            "flow_ids": test_fids,
            "flow_y_true": y_test.tolist(),
            "flow_y_pred": y_pred.tolist(),
            "flow_prob": test_prob.tolist(),
            "valid_flow_ids": valid_fids,
            "valid_y_true": y_valid.tolist(),
            "valid_y_pred": valid_prob.argmax(axis=1).astype(np.int64).tolist(),
            "valid_prob": valid_prob.tolist(),
            "feature_config": {
                "max_packets": args.max_packets,
                "prefix_len": args.prefix_len,
                "use_ports": args.use_ports,
                "feature_version": args.feature_version,
            },
        }
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()
