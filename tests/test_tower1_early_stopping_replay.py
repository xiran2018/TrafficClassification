import json

import pytest

from analyze_tower1_early_stopping import load_history, replay_patience


def test_replay_records_late_recovery_regret():
    result = replay_patience([0.4, 0.7, 0.69, 0.68, 0.75], patience=2)

    assert result["stop_epoch"] == 4
    assert result["selected_epoch"] == 2
    assert result["oracle_epoch"] == 5
    assert result["epochs_saved"] == 1
    assert result["metric_regret"] == pytest.approx(0.05)


def test_replay_keeps_best_checkpoint_before_terminal_stop():
    result = replay_patience([0.4, 0.7, 0.69, 0.71, 0.70], patience=2)

    assert result["stop_epoch"] == 5
    assert result["selected_epoch"] == 4
    assert result["selected_metric"] == pytest.approx(0.71)
    assert result["metric_regret"] == pytest.approx(0.0)


@pytest.mark.parametrize("patience", [0, -1])
def test_replay_rejects_nonpositive_patience(patience):
    with pytest.raises(ValueError, match="patience must be positive"):
        replay_patience([0.5], patience=patience)


def test_load_history_binds_strict_step_order(tmp_path):
    path = tmp_path / "history.jsonl"
    rows = [
        {"step": 10, "metrics": {"macro_f1": 0.5}},
        {"step": 10, "metrics": {"macro_f1": 0.6}},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    with pytest.raises(ValueError, match="steps must increase strictly"):
        load_history(path)
