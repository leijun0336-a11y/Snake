from snake_ai.game import Direction, Point, SnakeEnv


def test_reset_returns_expected_state_size() -> None:
    env = SnakeEnv(width=8, height=8, seed=1)
    state = env.reset()

    assert len(state) == env.state_size
    assert all(value in (0, 1) for value in state)


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


def test_wall_collision_ends_episode() -> None:
    env = SnakeEnv(width=6, height=6, seed=1)
    env.snake = [Point(5, 3), Point(4, 3), Point(3, 3)]
    env.direction = Direction.RIGHT

    _, reward, done, _ = env.step(0)

    assert done is True
    assert reward == -10.0
