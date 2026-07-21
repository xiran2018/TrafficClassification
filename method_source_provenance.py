#!/usr/bin/env python3
"""Hash the executable Python source tree before and after strict runs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


EXCLUDED_PARTS = {".git", "__pycache__", "tests"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def executable_python_paths(root: str | Path) -> list[Path]:
    root = Path(root).resolve()
    return sorted(
        (
            path
            for path in root.rglob("*.py")
            if path.is_file()
            and not any(part in EXCLUDED_PARTS for part in path.relative_to(root).parts)
        ),
        key=lambda path: path.relative_to(root).as_posix(),
    )


def source_tree_snapshot(root: str | Path) -> dict[str, Any]:
    root = Path(root).resolve()
    files = [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": file_sha256(path),
            "size": path.stat().st_size,
        }
        for path in executable_python_paths(root)
    ]
    if not files:
        raise ValueError(f"no executable Python sources found under {root}")
    identity = {
        "schema": "executable_python_source_tree_v1",
        "scope": "all_non_test_python_sources",
        "files": files,
    }
    return {
        **identity,
        "root": str(root),
        "num_files": len(files),
        "fingerprint": canonical_sha256(identity),
    }


def source_stability_evidence(
    launch: dict[str, Any], completion: dict[str, Any]
) -> dict[str, Any]:
    launch_files = {
        str(row["path"]): str(row["sha256"])
        for row in launch.get("files") or []
    }
    completion_files = {
        str(row["path"]): str(row["sha256"])
        for row in completion.get("files") or []
    }
    changed = sorted(
        path
        for path in set(launch_files) | set(completion_files)
        if launch_files.get(path) != completion_files.get(path)
    )
    stable = bool(
        launch.get("schema") == "executable_python_source_tree_v1"
        and completion.get("schema") == "executable_python_source_tree_v1"
        and launch.get("scope") == "all_non_test_python_sources"
        and completion.get("scope") == "all_non_test_python_sources"
        and launch.get("fingerprint")
        and launch.get("fingerprint") == completion.get("fingerprint")
        and not changed
    )
    return {
        "schema": "algorithm_source_stability_evidence_v1",
        "status": "pass" if stable else "fail",
        "scope": "all_non_test_python_sources",
        "launch_fingerprint": launch.get("fingerprint"),
        "completion_fingerprint": completion.get("fingerprint"),
        "num_launch_files": int(launch.get("num_files", 0)),
        "num_completion_files": int(completion.get("num_files", 0)),
        "changed_paths": changed,
        "launch_snapshot": launch,
    }


def complete_source_stability(
    launch: dict[str, Any], root: str | Path
) -> dict[str, Any]:
    return source_stability_evidence(launch, source_tree_snapshot(root))
