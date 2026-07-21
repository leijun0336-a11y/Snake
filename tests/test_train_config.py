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
    args = parse_args()
    assert args.validation_interval == 500
    assert args.validation_episodes == 100
    assert args.confirmation_episodes == 500
    assert args.validation_patience == 8
    assert args.validation_max_steps == 1000
    assert args.wandb is False
    assert args.n_step == 1
    assert train_config.n_step == 1
    assert train_config.learning_rate == pytest.approx(1e-4)
    assert args.learning_rate == pytest.approx(1e-4)


def test_parse_args_enables_wandb_only_when_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["train.py", "--wandb"])

    assert parse_args().wandb is True


def test_build_configs_accepts_n_step(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["train.py", "--n-step", "3"])

    train_config, _ = build_configs(parse_args())

    assert train_config.n_step == 3


def test_build_configs_rejects_non_positive_n_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["train.py", "--n-step", "0"])

    with pytest.raises(ValueError, match="n_step must be at least 1"):
        build_configs(parse_args())


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
