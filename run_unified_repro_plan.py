#!/usr/bin/env python3
"""Execute a unified paper_unified reproduction plan with a resumable ledger."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_ledger(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"started_at": utc_now(), "runs": []}
    return json.loads(path.read_text(encoding="utf-8"))


def write_ledger(path: Path, ledger: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ledger["updated_at"] = utc_now()
    path.write_text(json.dumps(ledger, indent=2, ensure_ascii=False), encoding="utf-8")


def action_fingerprint(action: Dict[str, Any]) -> str:
    argv = action_argv(action)
    payload = json.dumps(argv, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def canonical_sha256(payload: Dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def verify_plan_shared_core_config(plan: Dict[str, Any]) -> Dict[str, Any] | None:
    evidence = plan.get("shared_core_config")
    training_actions = [
        action for action in plan.get("actions") or []
        if action.get("fold") is not None
    ]
    if evidence is None:
        if any("--shared_core_config" in action_argv(action) for action in training_actions):
            raise ValueError(
                "training actions bind a shared-core config but the plan has no hash evidence"
            )
        return None
    if not isinstance(evidence, dict):
        raise ValueError("plan shared_core_config evidence must be an object")

    source = Path(str(evidence.get("path") or "")).resolve()
    if not source.is_file():
        raise ValueError(f"plan shared-core config does not exist: {source}")
    file_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    if file_sha256 != evidence.get("file_sha256"):
        raise ValueError("plan shared-core config file hash mismatch")

    payload = load_json(str(source))
    if payload.get("schema") != "exact_shared_packet_core_v2":
        raise ValueError("plan shared-core config has the wrong schema")
    recorded = str(payload.get("config_sha256") or "")
    unsigned = dict(payload)
    unsigned.pop("config_sha256", None)
    if recorded != canonical_sha256(unsigned):
        raise ValueError("plan shared-core config canonical fingerprint mismatch")
    if recorded != evidence.get("config_sha256"):
        raise ValueError("plan shared-core config fingerprint differs from the plan")

    expected_path = str(source)
    for action in training_actions:
        argv = action_argv(action)
        positions = [
            index for index, value in enumerate(argv)
            if value == "--shared_core_config"
        ]
        if len(positions) != 1 or positions[0] + 1 >= len(argv):
            raise ValueError(
                f"training action {action.get('id')} does not bind exactly one shared-core config"
            )
        actual_path = str(Path(argv[positions[0] + 1]).resolve())
        if actual_path != expected_path:
            raise ValueError(
                f"training action {action.get('id')} binds a different shared-core config"
            )
    return evidence


def action_completion_key(action: Dict[str, Any]) -> str:
    return f"{action.get('id', '')}|{action_fingerprint(action)}"


def completed_ids(ledger: Dict[str, Any]) -> Set[str]:
    """Return completed action+command keys; legacy ID-only rows are unsafe."""
    return {
        f"{row.get('id', '')}|{row['action_fingerprint']}"
        for row in ledger.get("runs", [])
        if row.get("status") == "success" and row.get("action_fingerprint")
    }


def selected_actions(
    plan: Dict[str, Any],
    *,
    task: str,
    dataset: str,
    start_index: int,
    max_actions: int,
    skip_ids: Set[str],
) -> List[Dict[str, Any]]:
    actions = list(plan.get("actions") or [])
    if task:
        actions = [row for row in actions if row.get("task") == task]
    if dataset:
        wanted = {item.strip() for item in dataset.split(",") if item.strip()}
        actions = [row for row in actions if row.get("dataset") in wanted]
    actions = actions[start_index:]
    actions = [row for row in actions if action_completion_key(row) not in skip_ids]
    if max_actions >= 0:
        actions = actions[:max_actions]
    return actions


def action_argv(action: Dict[str, Any]) -> List[str]:
    argv = action.get("argv")
    if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv):
        raise ValueError(f"action {action.get('id')} is missing safe argv")
    return argv


def log_path(log_dir: Path, action: Dict[str, Any]) -> Path:
    safe_id = str(action.get("id", "action")).replace(":", "_").replace("/", "_")
    return log_dir / f"{safe_id}.log"


def run_action(action: Dict[str, Any], log_dir: Path, dry_run: bool) -> Dict[str, Any]:
    argv = action_argv(action)
    path = log_path(log_dir, action)
    started = time.time()
    record = {
        "id": action.get("id"),
        "task": action.get("task"),
        "dataset": action.get("dataset"),
        "fold": action.get("fold"),
        "command": action.get("command"),
        "argv": argv,
        "action_fingerprint": action_fingerprint(action),
        "log_path": str(path),
        "started_at": utc_now(),
        "dry_run": dry_run,
    }
    if dry_run:
        record.update({"status": "dry_run", "returncode": 0, "duration_sec": 0.0})
        return record

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, "w", encoding="utf-8") as log:
            log.write("+ " + " ".join(argv) + "\n")
            log.flush()
            proc = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT)
    except KeyboardInterrupt:
        record.update(
            {
                "status": "interrupted",
                "returncode": -2,
                "duration_sec": round(time.time() - started, 3),
                "finished_at": utc_now(),
            }
        )
        return record
    record.update(
        {
            "status": "success" if proc.returncode == 0 else "failed",
            "returncode": proc.returncode,
            "duration_sec": round(time.time() - started, 3),
            "finished_at": utc_now(),
        }
    )
    return record


def run_actions(
    actions: Iterable[Dict[str, Any]],
    *,
    ledger_path: Path,
    log_dir: Path,
    dry_run: bool,
    continue_on_error: bool,
    plan: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    ledger = load_ledger(ledger_path)
    new_runs = 0
    batch_statuses: List[str] = []
    for action in actions:
        if plan is not None:
            verify_plan_shared_core_config(plan)
        print(f"+ run {action.get('id')}: {action.get('command')}", flush=True)
        record = run_action(action, log_dir, dry_run)
        ledger.setdefault("runs", []).append(record)
        new_runs += 1
        batch_statuses.append(record["status"])
        write_ledger(ledger_path, ledger)
        if record["status"] == "failed" and not continue_on_error:
            break
    ledger["new_runs"] = new_runs
    ledger["last_batch_statuses"] = batch_statuses
    return ledger


def main() -> None:
    ap = argparse.ArgumentParser(description="Execute unified paper_unified repro actions.")
    ap.add_argument("--plan_json", default="reasoningDataset/unified_repro_plan.json")
    ap.add_argument("--ledger_json", default="reasoningDataset/unified_repro_ledger.json")
    ap.add_argument("--log_dir", default="logs/unified_repro")
    ap.add_argument("--task", choices=["", "flow-level", "packet-level"], default="")
    ap.add_argument("--dataset", default="", help="Optional comma-separated dataset filter.")
    ap.add_argument("--start_index", type=int, default=0)
    ap.add_argument(
        "--max_actions",
        type=int,
        default=1,
        help="Maximum actions to execute after filtering. Use -1 for all remaining actions.",
    )
    ap.add_argument("--no_skip_completed", action="store_true")
    ap.add_argument("--continue_on_error", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    plan = load_json(args.plan_json)
    verify_plan_shared_core_config(plan)
    ledger_path = Path(args.ledger_json)
    ledger = load_ledger(ledger_path)
    skip = set() if args.no_skip_completed else completed_ids(ledger)
    actions = selected_actions(
        plan,
        task=args.task,
        dataset=args.dataset,
        start_index=args.start_index,
        max_actions=args.max_actions,
        skip_ids=skip,
    )
    if not actions:
        print(json.dumps({"status": "no_actions", "ledger_json": str(ledger_path)}, indent=2))
        return
    ledger = run_actions(
        actions,
        ledger_path=ledger_path,
        log_dir=Path(args.log_dir),
        dry_run=args.dry_run,
        continue_on_error=args.continue_on_error,
        plan=plan,
    )
    batch_statuses = ledger.get("last_batch_statuses", [])
    historical_failures = sum(1 for row in ledger.get("runs", []) if row.get("status") == "failed")
    summary = {
        "status": "batch_complete" if all(status in {"success", "dry_run"} for status in batch_statuses) else "batch_has_failures",
        "new_runs": ledger.get("new_runs", 0),
        "total_runs": len(ledger.get("runs", [])),
        "historical_failures": historical_failures,
        "ledger_json": str(ledger_path),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
