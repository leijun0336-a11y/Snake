import sys

import pytest

from snake_ai.train import build_configs, parse_args


def test_build_configs_resolves_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["train.py"])

    train_config, env_config = build_configs(parse_args())

    assert train_config.max_steps_per_episode is None
    assert train_config.epsilon_linear_episodes == 7500
    assert env_config.width == 20
    assert env_config.height == 20


def test_build_configs_matches_environment_minimum_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["train.py", "--width", "4"])

    with pytest.raises(ValueError, match="at least 5"):
        build_configs(parse_args())
