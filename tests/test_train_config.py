import sys

import pytest

from snake_ai.train import (
    TRAINING_START_DELAY_SECONDS,
    build_configs,
    build_ppo_config,
    parse_args,
)


def test_build_configs_resolves_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["train.py"])

    train_config, env_config = build_configs(parse_args())

    assert train_config.max_steps_per_episode is None
    assert train_config.epsilon_linear_episodes == 7500
    assert env_config.width == 6
    assert env_config.height == 6
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
    assert args.algorithm == "dqn"
    assert TRAINING_START_DELAY_SECONDS == 5


def test_ppo_config_uses_aligned_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["train.py", "--algorithm", "ppo"])

    args = parse_args()
    train_config, _ = build_configs(args)
    ppo_config = build_ppo_config(args)

    assert train_config.learning_rate == pytest.approx(1e-4)
    assert train_config.batch_size == 128
    assert ppo_config.rollout_steps == 2048
    assert ppo_config.update_epochs == 4
    assert ppo_config.gae_lambda == pytest.approx(0.95)
    assert ppo_config.target_kl == pytest.approx(0.02)
    assert ppo_config.entropy_coefficient == pytest.approx(0.05)
    assert ppo_config.entropy_coefficient_end == pytest.approx(0.001)
    assert ppo_config.entropy_anneal_episodes == train_config.episodes == 15_000
    assert args.argmax_cycle_fallback is False


def test_ppo_rollout_must_be_divisible_by_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["train.py", "--algorithm", "ppo", "--ppo-rollout-steps", "1000"],
    )

    with pytest.raises(ValueError, match="divisible"):
        build_configs(parse_args())


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
