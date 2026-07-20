import sys
from types import SimpleNamespace

import pytest

from snake_ai.game import SnakeEnv
from snake_ai.train import (
    build_configs,
    build_mask_planner,
    certified_action_mask,
    parse_args,
    terminal_action_mask,
)


def test_build_configs_resolves_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["train.py"])

    train_config, env_config = build_configs(parse_args())

    assert train_config.max_steps_per_episode is None
    assert train_config.epsilon_linear_episodes == 7500
    assert env_config.width == 20
    assert env_config.height == 20
    args = parse_args()
    assert args.validation_interval == 500
    assert args.validation_episodes == 100
    assert args.confirmation_episodes == 500
    assert args.validation_patience == 8
    assert args.validation_max_steps == 1000
    assert args.wandb is False
    assert args.mask is False
    assert build_mask_planner(args) is None


def test_parse_args_enables_wandb_only_when_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["train.py", "--wandb"])

    assert parse_args().wandb is True


def test_mask_flag_builds_strict_10x10_planner_and_action_mask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["train.py", "--mask", "--width", "10", "--height", "10"],
    )
    args = parse_args()

    build_configs(args)
    planner = build_mask_planner(args)
    env = SnakeEnv(
        width=10,
        height=10,
        seed=1,
        state_mode="hybrid",
        reward_profile="experiment8",
    )
    try:
        mask = certified_action_mask(planner, env)
    finally:
        env.close()

    assert args.mask is True
    assert args.validation_max_steps == 5000
    assert len(mask) == 3
    assert any(mask)
    assert terminal_action_mask(3) == (True, True, True)


@pytest.mark.parametrize(
    "arguments",
    [
        ["--width", "6", "--height", "6"],
        ["--width", "10", "--height", "10", "--state-mode", "grid"],
        ["--width", "10", "--height", "10", "--reward-profile", "reference"],
        ["--width", "10", "--height", "10", "--mask-max-astar-expansions", "0"],
    ],
)
def test_mask_flag_rejects_incompatible_training_config(
    monkeypatch: pytest.MonkeyPatch,
    arguments: list[str],
) -> None:
    monkeypatch.setattr(sys, "argv", ["train.py", "--mask", *arguments])

    with pytest.raises(ValueError, match="--mask requires|must be positive"):
        build_configs(parse_args())


def test_disabled_mask_returns_before_reading_planner_arguments() -> None:
    args = SimpleNamespace(mask=False)

    assert build_mask_planner(args) is None


def test_build_configs_matches_environment_minimum_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["train.py", "--width", "4"])

    with pytest.raises(ValueError, match="at least 5"):
        build_configs(parse_args())


def test_build_configs_rejects_invalid_validation_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["train.py", "--validation-episodes", "0"],
    )

    with pytest.raises(ValueError, match="validation episode counts"):
        build_configs(parse_args())
