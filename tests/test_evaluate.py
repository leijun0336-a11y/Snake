import sys
from pathlib import Path

import pytest

from snake_ai.evaluate import build_configs, find_latest_checkpoint, parse_args


def test_build_configs_resolves_evaluation_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["evaluate.py"])

    train_config, env_config = build_configs(parse_args())

    assert train_config.seed == 42
    assert env_config.width == 20
    assert env_config.height == 20
    assert parse_args().network == "q_network"


def test_parse_args_selects_old_q_network(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["evaluate.py", "--network", "q_network_old"],
    )

    assert parse_args().network == "q_network_old"


def test_build_configs_matches_environment_minimum_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["evaluate.py", "--height", "4"])

    with pytest.raises(ValueError, match="at least 5"):
        build_configs(parse_args())


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
