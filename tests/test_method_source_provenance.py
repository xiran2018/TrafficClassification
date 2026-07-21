from method_source_provenance import (
    complete_source_stability,
    entrypoint_dependency_paths,
    source_stability_evidence,
    source_tree_snapshot,
)


def test_source_snapshot_is_stable_and_excludes_tests(tmp_path):
    (tmp_path / "model.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_model.py").write_text(
        "assert True\n", encoding="utf-8"
    )

    launch = source_tree_snapshot(tmp_path)
    evidence = complete_source_stability(launch, tmp_path)

    assert launch["num_files"] == 1
    assert launch["files"][0]["path"] == "model.py"
    assert evidence["status"] == "pass"
    assert evidence["changed_paths"] == []


def test_source_stability_detects_content_change_and_added_file(tmp_path):
    source = tmp_path / "model.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    launch = source_tree_snapshot(tmp_path)
    source.write_text("VALUE = 2\n", encoding="utf-8")
    (tmp_path / "new_module.py").write_text("VALUE = 3\n", encoding="utf-8")
    completion = source_tree_snapshot(tmp_path)

    evidence = source_stability_evidence(launch, completion)

    assert evidence["status"] == "fail"
    assert evidence["changed_paths"] == ["model.py", "new_module.py"]
    assert evidence["launch_fingerprint"] != evidence["completion_fingerprint"]


def test_entrypoint_closure_tracks_imports_and_python_command_literals(tmp_path):
    (tmp_path / "main.py").write_text(
        "import helper\nWORKER = 'worker.py'\n", encoding="utf-8"
    )
    (tmp_path / "helper.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "worker.py").write_text(
        "from helper import VALUE\n", encoding="utf-8"
    )
    (tmp_path / "unrelated.py").write_text("VALUE = 2\n", encoding="utf-8")

    paths = entrypoint_dependency_paths(tmp_path, ["main.py"])
    assert [path.name for path in paths] == ["helper.py", "main.py", "worker.py"]

    launch = source_tree_snapshot(tmp_path, entrypoints=["main.py"])
    (tmp_path / "unrelated.py").write_text("VALUE = 3\n", encoding="utf-8")
    assert complete_source_stability(launch, tmp_path)["status"] == "pass"

    (tmp_path / "helper.py").write_text("VALUE = 4\n", encoding="utf-8")
    evidence = complete_source_stability(launch, tmp_path)
    assert evidence["status"] == "fail"
    assert evidence["changed_paths"] == ["helper.py"]
