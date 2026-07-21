#!/usr/bin/env python3
"""Hash the executable Python source tree before and after strict runs."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Any


EXCLUDED_PARTS = {".git", "__pycache__", "tests"}
UNIFIED_METHOD_ENTRYPOINTS = (
    "run_packet_level_pipeline.py",
    "run_stage8_flowaware_pipeline.py",
)


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


def _module_index(root: Path) -> dict[str, Path]:
    modules = {}
    for path in executable_python_paths(root):
        relative = path.relative_to(root)
        if relative.name == "__init__.py":
            module = ".".join(relative.parent.parts)
        else:
            module = ".".join(relative.with_suffix("").parts)
        if module:
            modules[module] = path
    return modules


def _imported_local_paths(
    root: Path, path: Path, modules: dict[str, Path]
) -> set[Path]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    relative = path.relative_to(root)
    package = list(relative.parent.parts)
    imported_modules = set()
    literal_paths = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                keep = max(0, len(package) - (node.level - 1))
                prefix_parts = package[:keep]
                if node.module:
                    prefix_parts.extend(node.module.split("."))
                prefix = ".".join(prefix_parts)
            else:
                prefix = node.module or ""
            if prefix:
                imported_modules.add(prefix)
            for alias in node.names:
                if alias.name != "*" and prefix:
                    imported_modules.add(f"{prefix}.{alias.name}")
        elif (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value.endswith(".py")
        ):
            for candidate in (root / node.value, path.parent / node.value):
                if candidate.is_file() and root in candidate.resolve().parents:
                    literal_paths.add(candidate.resolve())

    dependencies = set(literal_paths)
    for module in imported_modules:
        candidate = modules.get(module)
        if candidate is not None:
            dependencies.add(candidate.resolve())
        parts = module.split(".")
        for end in range(1, len(parts)):
            package_init = modules.get(".".join(parts[:end]))
            if package_init is not None and package_init.name == "__init__.py":
                dependencies.add(package_init.resolve())
    return dependencies


def entrypoint_dependency_paths(
    root: str | Path, entrypoints: list[str] | tuple[str, ...]
) -> list[Path]:
    root = Path(root).resolve()
    modules = _module_index(root)
    pending = []
    for entrypoint in entrypoints:
        path = (root / entrypoint).resolve()
        if not path.is_file() or root not in path.parents:
            raise ValueError(f"invalid local Python entrypoint: {entrypoint}")
        pending.append(path)
    resolved = set()
    while pending:
        path = pending.pop()
        if path in resolved:
            continue
        resolved.add(path)
        pending.extend(_imported_local_paths(root, path, modules) - resolved)
    return sorted(resolved, key=lambda path: path.relative_to(root).as_posix())


def source_tree_snapshot(
    root: str | Path,
    entrypoints: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    root = Path(root).resolve()
    paths = (
        entrypoint_dependency_paths(root, entrypoints)
        if entrypoints is not None
        else executable_python_paths(root)
    )
    files = [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": file_sha256(path),
            "size": path.stat().st_size,
        }
        for path in paths
    ]
    if not files:
        raise ValueError(f"no executable Python sources found under {root}")
    scope = (
        "entrypoint_dependency_closure_v1"
        if entrypoints is not None
        else "all_non_test_python_sources"
    )
    identity = {
        "schema": "executable_python_source_tree_v1",
        "scope": scope,
        "entrypoints": sorted(entrypoints or []),
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
        and launch.get("scope") in {
            "all_non_test_python_sources",
            "entrypoint_dependency_closure_v1",
        }
        and launch.get("scope") == completion.get("scope")
        and launch.get("entrypoints") == completion.get("entrypoints")
        and launch.get("fingerprint")
        and launch.get("fingerprint") == completion.get("fingerprint")
        and not changed
    )
    return {
        "schema": "algorithm_source_stability_evidence_v1",
        "status": "pass" if stable else "fail",
        "scope": launch.get("scope"),
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
    entrypoints = (
        launch.get("entrypoints")
        if launch.get("scope") == "entrypoint_dependency_closure_v1"
        else None
    )
    return source_stability_evidence(
        launch, source_tree_snapshot(root, entrypoints=entrypoints)
    )


def unified_method_source_snapshot(root: str | Path) -> dict[str, Any]:
    return source_tree_snapshot(root, entrypoints=UNIFIED_METHOD_ENTRYPOINTS)
