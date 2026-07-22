import fcntl

import run_packet_level_pipeline as pipeline


def test_gpu_command_guard_reuses_embedding_lock(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5")
    monkeypatch.setenv("PACKET_GPU_LOCK_DIR", str(tmp_path))
    monkeypatch.setenv("PACKET_GPU_MIN_FREE_MB", "40000")
    monkeypatch.setattr(pipeline.subprocess, "check_output", lambda *args, **kwargs: "45000\n")
    monkeypatch.setattr(pipeline.fcntl, "flock", lambda _fd, operation: calls.append(operation))

    command = ["python", "train_tower1_multitask.py"]
    with pipeline.gpu_command_guard(command, dry_run=False):
        assert (tmp_path / "qwen_embedding_gpu_5.lock").exists()

    assert calls == [fcntl.LOCK_EX, fcntl.LOCK_UN]


def test_gpu_command_guard_skips_cpu_command(monkeypatch):
    monkeypatch.setattr(
        pipeline.fcntl,
        "flock",
        lambda *_args: (_ for _ in ()).throw(AssertionError("CPU command must not lock a GPU")),
    )

    with pipeline.gpu_command_guard(["python", "preprocess_tower1.py"], dry_run=False):
        pass


def test_gpu_command_guard_uses_matching_inherited_reservation(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5")
    monkeypatch.setenv("SHARED_GPU_RESERVATION_TOKEN", "5")
    monkeypatch.setattr(
        pipeline.fcntl,
        "flock",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("inherited reservation must not relock")
        ),
    )

    with pipeline.gpu_command_guard(
        ["python", "train_tower1_multitask.py"], dry_run=False
    ):
        pass
