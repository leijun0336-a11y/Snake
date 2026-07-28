from argparse import Namespace

import pytest

from snake_ai.config import (
    EXPERIMENT8_REWARD_CONFIG,
    EXPERIMENT_PPO_REWARD_CONFIG,
    get_reward_config,
)
from snake_ai.game import Direction, Point, SnakeEnv
from snake_ai.train import resolve_max_steps_per_episode


def _training_args(
    *,
    reward_profile: str,
    potential_reward: bool = True,
    no_cost_rewards: bool = False,
    max_steps_per_episode: int | None = None,
    gamma: float = 0.99,
) -> Namespace:
    return Namespace(
        reward_profile=reward_profile,
        potential_reward=potential_reward,
        no_cost_rewards=no_cost_rewards,
        max_steps_per_episode=max_steps_per_episode,
        gamma=gamma,
    )


def test_experiment8_profile_uses_current_frozen_values() -> None:
    config = EXPERIMENT8_REWARD_CONFIG

    assert config.potential_reward is True
    assert config.cost_rewards is True
    assert config.progress_beta == 1.0
    assert config.food_reward == 10.0
    assert config.collision_penalty == -100.0
    assert config.starvation_penalty == -12.0
    assert config.win_reward == 20.0
    assert config.step_penalty == -0.005
    assert config.hunger_penalty_scale == 0.02
    assert config.step_cost_scope == "all_legal_moves"
    assert config.terminal_cost_mode == "accumulate"
    assert config.starvation_limit_mode == "board_area"
    assert config.starvation_comparison == "gt"
    assert config.progress_mode == "pbrs_food_distance"
    assert config.historical_source_revision == ("62ff05d5a8d7b65472a984e56647f2c20bceb915")


def test_experiment_ppo_uses_validated_food_collision_scale() -> None:
    config = EXPERIMENT_PPO_REWARD_CONFIG

    assert config.food_reward == pytest.approx(4.0)
    assert config.collision_penalty == pytest.approx(-4.0)


def test_unknown_reward_profile_does_not_fallback() -> None:
    with pytest.raises(ValueError, match="unknown reward profile"):
        get_reward_config("missing")


def test_experiment8_eating_step_matches_current_components() -> None:
    env = SnakeEnv(width=6, height=6, seed=1, reward_profile="experiment8")
    env.snake = [Point(2, 2), Point(1, 2), Point(0, 2)]
    env.direction = Direction.RIGHT
    env.food = Point(3, 2)
    old_phi = env._food_potential(env.snake[0], env.food)

    _, reward, done, info = env.step(0)
    new_phi = env._food_potential(env.snake[0], env.food)
    expected_progress = env.progress_beta * (env.reward_gamma * new_phi - old_phi)

    assert done is False
    assert info["reward_food"] == 10.0
    assert info["reward_progress"] == pytest.approx(expected_progress)
    assert info["reward_step"] == -0.005
    assert info["reward_hunger"] == 0.0
    assert info["reward_terminal"] == 0.0
    assert reward == pytest.approx(10.0 + expected_progress - 0.005)


def test_experiment8_collision_applies_terminal_potential_correction() -> None:
    env = SnakeEnv(width=6, height=6, seed=1, reward_profile="experiment8")
    env.snake = [Point(5, 3), Point(4, 3), Point(3, 3)]
    env.direction = Direction.RIGHT
    old_phi = env._food_potential(env.snake[0], env.food)

    _, reward, done, info = env.step(0)
    expected_progress = -env.progress_beta * old_phi

    assert done is True
    assert reward == pytest.approx(-100.0 + expected_progress)
    assert info["termination_reason"] == "collision_wall"
    assert info["reward_terminal"] == -100.0
    assert info["reward_progress"] == pytest.approx(expected_progress)
    assert all(
        float(info[f"reward_{component}"]) == 0.0 for component in ("food", "step", "hunger")
    )


def test_experiment8_starvation_occurs_after_limit_and_accumulates_costs() -> None:
    env = SnakeEnv(width=6, height=6, seed=1, reward_profile="experiment8")
    env.snake = [Point(2, 2), Point(1, 2), Point(0, 2)]
    env.direction = Direction.RIGHT
    env.food = Point(5, 5)
    env.steps_since_food = 35

    _, _, done_at_36, _ = env.step(0)
    old_phi = env._food_potential(env.snake[0], env.food)
    _, reward_at_37, done_at_37, info = env.step(0)
    expected_progress = -env.progress_beta * old_phi

    assert env.starvation_limit == 36
    assert done_at_36 is False
    assert done_at_37 is True
    assert info["steps_since_food"] == 37
    assert info["termination_reason"] == "starvation"
    assert info["reward_progress"] == pytest.approx(expected_progress)
    assert info["reward_step"] == -0.005
    assert info["reward_hunger"] == -0.02
    assert info["reward_terminal"] == -12.0
    assert reward_at_37 == pytest.approx(-12.025 + expected_progress)


def test_experiment8_starvation_limit_does_not_change_with_snake_length() -> None:
    env = SnakeEnv(width=6, height=6, seed=1, reward_profile="experiment8")

    assert env.starvation_limit == 36
    env.snake.append(Point(0, 0))
    assert env.starvation_limit == 36
    env.steps_since_food = 36
    assert env._is_too_long_without_food() is False
    env.steps_since_food = 37
    assert env._is_too_long_without_food() is True


def test_experiment8_board_completion_matches_current_total() -> None:
    env = SnakeEnv(width=6, height=6, seed=1, reward_profile="experiment8")
    food = Point(5, 5)
    head = Point(4, 5)
    env.snake = [head] + [
        Point(x, y) for y in range(6) for x in range(6) if Point(x, y) not in (head, food)
    ]
    env.direction = Direction.RIGHT
    env.food = food
    old_phi = env._food_potential(env.snake[0], env.food)

    _, reward, done, info = env.step(0)
    expected_progress = -env.progress_beta * old_phi

    assert done is True
    assert len(env.snake) == 36
    assert info["termination_reason"] == "board_completed"
    assert info["reward_food"] == 10.0
    assert info["reward_progress"] == pytest.approx(expected_progress)
    assert info["reward_step"] == -0.005
    assert info["reward_hunger"] == 0.0
    assert info["reward_terminal"] == 20.0
    assert reward == pytest.approx(29.995 + expected_progress)


def test_profile_step_limit_resolution_is_explicit() -> None:
    assert resolve_max_steps_per_episode(_training_args(reward_profile="reference")) == 500
    assert resolve_max_steps_per_episode(_training_args(reward_profile="experiment8")) is None
    assert (
        resolve_max_steps_per_episode(
            _training_args(reward_profile="experiment8", potential_reward=False)
        )
        is None
    )
    assert (
        resolve_max_steps_per_episode(
            _training_args(
                reward_profile="experiment8",
                no_cost_rewards=True,
                gamma=0.95,
            )
        )
        is None
    )
    assert (
        resolve_max_steps_per_episode(
            _training_args(reward_profile="reference", max_steps_per_episode=800)
        )
        == 800
    )
    assert (
        resolve_max_steps_per_episode(
            _training_args(reward_profile="experiment8", max_steps_per_episode=500)
        )
        == 500
    )


def test_experiment_ppo_environment_keeps_base_food_and_collision_rewards() -> None:
    eating = SnakeEnv(width=6, height=6, seed=1, reward_profile="experiment_ppo")
    eating.snake = [Point(2, 2), Point(1, 2), Point(0, 2)]
    eating.direction = Direction.RIGHT
    eating.food = Point(3, 2)

    _, _, done, info = eating.step(0)

    assert done is False
    assert info["reward_food"] == pytest.approx(4.0)
    eating.close()

    collision = SnakeEnv(width=6, height=6, seed=1, reward_profile="experiment_ppo")
    collision.snake = [Point(5, 3), Point(4, 3), Point(3, 3)]
    collision.direction = Direction.RIGHT
    old_phi = collision._food_potential(collision.snake[0], collision.food)

    _, reward, done, info = collision.step(0)
    expected_progress = -collision.progress_beta * old_phi

    assert done is True
    assert reward == pytest.approx(-4.0 + expected_progress)
    assert info["reward_progress"] == pytest.approx(expected_progress)
    assert info["reward_terminal"] == pytest.approx(-4.0)
    collision.close()
