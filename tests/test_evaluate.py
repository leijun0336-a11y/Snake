from pathlib import Path

import pytest

from snake_ai.evaluate import find_latest_checkpoint


def test_find_latest_checkpoint_uses_latest_experiment_latest_pt(
    tmp_path: Path,
) -> None:
    older = tmp_path / "dqn_20260101_000000"
    newer = tmp_path / "dqn_20260102_000000"
    older.mkdir()
    newer.mkdir()
    (older / "latest.pt").touch()
    (newer / "latest.pt").touch()
    (newer / "best.pt").touch()

    assert find_latest_checkpoint(tmp_path) == newer / "latest.pt"


def test_find_latest_checkpoint_does_not_fallback_to_best_pt(tmp_path: Path) -> None:
    latest_run = tmp_path / "dqn_20260102_000000"
    latest_run.mkdir()
    (latest_run / "best.pt").touch()

    with pytest.raises(FileNotFoundError, match=r"No latest\.pt found"):
        find_latest_checkpoint(tmp_path)
