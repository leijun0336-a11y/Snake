import numpy as np
import pytest

from snake_ai.game import Direction, Point, SnakeEnv


def test_reset_returns_expected_state_size() -> None:
    env = SnakeEnv(width=8, height=8, seed=1)
    state = env.reset()

    assert len(state) == env.state_size
    assert env.state_size == 20
    assert all(-1.0 <= value <= 1.0 for value in state)


def test_state_includes_normalized_distance_features() -> None:
    env = SnakeEnv(width=6, height=6, seed=1)
    env.snake = [Point(2, 2), Point(1, 2), Point(0, 2), Point(2, 3)]
    env.direction = Direction.RIGHT
    env.food = Point(5, 0)

    state = env.get_state()

    assert state[11] == 0.6
    assert state[12] == -0.4
    assert state[13] == 0.6
    assert state[14] == 0.6
    assert state[15] == 0.4
    assert state[16] == 1.0
    assert state[17] == 0.2
    assert state[18] == 1.0
    assert state[19] == 0.0


def test_grid_state_contains_map_channels() -> None:
    env = SnakeEnv(width=6, height=6, seed=1)
    env.snake = [Point(2, 2), Point(1, 2), Point(0, 2)]
    env.direction = Direction.RIGHT
    env.food = Point(4, 3)

    grid = env.get_grid_state()

    assert isinstance(grid, np.ndarray)
    assert grid.dtype == np.float32
    assert grid.shape == env.grid_state_shape
    assert len(grid) == env.grid_channels
    assert len(grid[0]) == env.height
    assert len(grid[0][0]) == env.width
    assert grid[0][0][0] == 1.0
    assert grid[3][2][2] == 1.0
    assert grid[1][2][1] == 1.0
    assert grid[6][2][0] == 1.0
    assert grid[7][3][4] == 1.0
    assert grid[8][2][2] == 1.0
    assert grid[8][2][0] == 1 / 3
    assert sum(float(grid[channel].sum()) for channel in range(2, 6)) == 1.0


def test_state_and_grid_include_normalized_hunger_progress() -> None:
    env = SnakeEnv(width=6, height=6, seed=1)
    env.steps_since_food = 18

    assert env.get_state()[19] == pytest.approx(18 / 39)
    # Hunger remains in the vector state but is intentionally absent from Grid.
    assert env.get_grid_state().shape[0] == 9


def test_grid_state_does_not_overwrite_down_head_channel() -> None:
    env = SnakeEnv(width=6, height=6, seed=1)
    env.snake = [Point(2, 2), Point(2, 1), Point(2, 0)]
    env.direction = Direction.DOWN
    env.food = Point(4, 3)
    env.steps_since_food = 18

    grid = env.get_grid_state()

    assert grid[5][2][2] == 1.0
    assert float(grid[5].sum()) == 1.0
    assert sum(float(grid[channel].sum()) for channel in range(2, 6)) == 1.0


def test_hybrid_state_contains_grid_and_vector_state() -> None:
    env = SnakeEnv(width=6, height=6, seed=1)

    grid, vector_state = env.get_hybrid_state()

    assert len(grid) == env.grid_channels
    assert len(vector_state) == env.state_size
    assert vector_state == env.get_state()


def test_reset_and_step_return_selected_grid_observation() -> None:
    env = SnakeEnv(width=6, height=6, seed=1, state_mode="grid")

    state = env.reset()
    next_state, _, _, _ = env.step(0)

    assert isinstance(state, np.ndarray)
    assert isinstance(next_state, np.ndarray)
    assert state.shape == env.grid_state_shape
    assert next_state.shape == env.grid_state_shape


def test_reset_returns_selected_hybrid_observation() -> None:
    env = SnakeEnv(width=6, height=6, seed=1, state_mode="hybrid")

    grid, vector_state = env.reset()

    assert isinstance(grid, np.ndarray)
    assert len(vector_state) == env.state_size


def test_step_moves_snake_and_returns_gym_like_tuple() -> None:
    env = SnakeEnv(width=8, height=8, seed=1)
    state = env.reset()
    head_before = env.snake[0]

    next_state, reward, done, info = env.step(0)

    assert env.snake[0] != head_before
    assert len(next_state) == len(state)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert "score" in info
    assert "steps" in info
    assert "snake_length" in info
    assert "steps_since_food" in info
    assert "reward_progress" in info
    assert "reward_hunger" in info
    assert info["reward_total"] == pytest.approx(reward)
    assert info["termination_reason"] == "none"


def test_wall_collision_ends_episode() -> None:
    env = SnakeEnv(width=6, height=6, seed=1)
    env.snake = [Point(5, 3), Point(4, 3), Point(3, 3)]
    env.direction = Direction.RIGHT

    _, reward, done, _ = env.step(0)

    assert done is True
    assert reward == -100.0

def test_can_move_into_current_tail_when_not_eating() -> None:
    env = SnakeEnv(
        width=6,
        height=6,
        seed=1,
        potential_reward=False,
        cost_rewards=False,
    )
    env.snake = [Point(2, 2), Point(2, 3), Point(1, 3), Point(1, 2)]
    env.direction = Direction.LEFT
    env.food = Point(5, 5)

    _, reward, done, _ = env.step(0)

    assert done is False
    assert reward == 0.0
    assert env.snake[0] == Point(1, 2)
    assert len(env.snake) == 4


def test_current_tail_is_collision_when_tail_will_not_move() -> None:
    env = SnakeEnv(width=6, height=6, seed=1)
    env.snake = [Point(2, 2), Point(2, 3), Point(1, 3), Point(1, 2)]
    env.food = Point(1, 2)

    assert env._is_collision_after_move(Point(1, 2)) is True


def test_potential_reward_is_positive_when_moving_closer_to_food() -> None:
    env = SnakeEnv(
        width=6,
        height=6,
        seed=1,
        potential_reward=True,
        cost_rewards=False,
    )
    env.snake = [Point(2, 2), Point(1, 2), Point(0, 2)]
    env.direction = Direction.RIGHT
    env.food = Point(5, 2)

    _, reward, done, info = env.step(0)

    assert done is False
    assert reward == pytest.approx(info["reward_progress"])
    assert float(info["reward_progress"]) > 0.0
    assert info["reward_step"] == 0.0
    assert info["reward_hunger"] == 0.0


def test_cost_rewards_apply_step_and_optional_quadratic_hunger_penalties() -> None:
    env = SnakeEnv(
        width=6,
        height=6,
        seed=1,
        potential_reward=False,
        hunger_penalty_scale=0.02,
    )
    env.snake = [Point(2, 2), Point(1, 2), Point(0, 2)]
    env.direction = Direction.RIGHT
    env.food = Point(5, 5)
    env.steps_since_food = 17

    _, reward, done, info = env.step(0)

    expected_hunger = -0.02 * (18 / 39) ** 2
    assert done is False
    assert info["reward_progress"] == 0.0
    assert info["reward_step"] == pytest.approx(-0.01)
    assert info["reward_hunger"] == pytest.approx(expected_hunger)
    assert reward == pytest.approx(-0.01 + expected_hunger)


def test_disabling_both_reward_groups_restores_event_only_baseline() -> None:
    env = SnakeEnv(
        width=6,
        height=6,
        seed=1,
        potential_reward=False,
        cost_rewards=False,
    )
    env.snake = [Point(2, 2), Point(1, 2), Point(0, 2)]
    env.direction = Direction.RIGHT
    env.food = Point(5, 5)

    _, reward, done, info = env.step(0)

    assert done is False
    assert reward == 0.0
    assert all(
        float(info[f"reward_{name}"]) == 0.0
        for name in ("food", "progress", "step", "hunger", "terminal")
    )


@pytest.mark.parametrize(
    ("cost_rewards", "expected_terminal"),
    [(True, -100.0), (False, -100.0)],
)
def test_starvation_penalty_respects_cost_reward_switch(
    cost_rewards: bool, expected_terminal: float
) -> None:
    env = SnakeEnv(
        width=6,
        height=6,
        seed=1,
        potential_reward=False,
        cost_rewards=cost_rewards,
    )
    env.snake = [Point(2, 2), Point(1, 2), Point(0, 2)]
    env.direction = Direction.RIGHT
    env.food = Point(5, 5)
    env.steps_since_food = env.starvation_limit

    _, reward, done, info = env.step(0)

    assert done is True
    assert info["termination_reason"] == "starvation"
    assert info["reward_terminal"] == expected_terminal
    assert reward == expected_terminal
    assert info["reward_step"] == 0.0
    assert info["reward_hunger"] == 0.0


def test_starvation_limit_includes_current_snake_length() -> None:
    env = SnakeEnv(width=6, height=6, seed=1)

    assert len(env.snake) == 3
    assert env.starvation_limit == 39

    env.snake.append(Point(0, 0))

    assert env.starvation_limit == 40


def test_starvation_can_be_disabled_for_reference_evaluation() -> None:
    env = SnakeEnv(
        width=6,
        height=6,
        seed=1,
        starvation_enabled=False,
        potential_reward=False,
    )
    env.snake = [Point(2, 2), Point(1, 2), Point(0, 2)]
    env.direction = Direction.RIGHT
    env.food = Point(5, 5)
    env.steps_since_food = env.starvation_limit

    _, _, done, info = env.step(0)

    assert done is False
    assert info["termination_reason"] == "none"
    assert env.hunger_ratio == 0.0


def test_eating_food_reports_components_and_resets_hunger() -> None:
    env = SnakeEnv(width=6, height=6, seed=1)
    env.snake = [Point(2, 2), Point(1, 2), Point(0, 2)]
    env.direction = Direction.RIGHT
    env.food = Point(3, 2)
    env.steps_since_food = 20

    _, reward, done, info = env.step(0)

    assert done is False
    assert env.steps_since_food == 0
    assert info["reward_food"] == 10.0
    assert info["reward_hunger"] == 0.0
    assert reward == pytest.approx(
        float(info["reward_food"])
        + float(info["reward_progress"])
    )
    assert info["reward_step"] == 0.0


def test_completing_board_has_total_reward_of_one_hundred() -> None:
    env = SnakeEnv(width=5, height=5, seed=1, potential_reward=False)
    food = Point(4, 4)
    head = Point(3, 4)
    env.snake = [head] + [
        Point(x, y)
        for y in range(5)
        for x in range(5)
        if Point(x, y) not in (head, food)
    ]
    env.direction = Direction.RIGHT
    env.food = food

    _, reward, done, info = env.step(0)

    assert done is True
    assert len(env.snake) == 25
    assert info["termination_reason"] == "board_completed"
    assert info["reward_food"] == 10.0
    assert info["reward_terminal"] == 90.0
    assert info["reward_step"] == 0.0
    assert reward == pytest.approx(100.0)

