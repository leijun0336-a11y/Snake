from snake_ai.game import Direction, Point, SnakeEnv


def test_reset_returns_expected_state_size() -> None:
    env = SnakeEnv(width=8, height=8, seed=1)
    state = env.reset()

    assert len(state) == env.state_size
    assert env.state_size == 19
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


def test_grid_state_contains_map_channels_and_direction() -> None:
    env = SnakeEnv(width=6, height=6, seed=1)
    env.snake = [Point(2, 2), Point(1, 2), Point(0, 2)]
    env.direction = Direction.RIGHT
    env.food = Point(4, 3)

    grid, direction = env.get_grid_state()

    assert len(grid) == env.grid_channels
    assert len(grid[0]) == env.height
    assert len(grid[0][0]) == env.width
    assert direction == [0.0, 1.0, 0.0, 0.0]
    assert grid[0][0][0] == 1.0
    assert grid[2][2][2] == 1.0
    assert grid[1][2][1] == 1.0
    assert grid[3][3][4] == 1.0
    assert grid[4][2][2] == 1.0
    assert grid[4][2][0] == 1 / 3


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


def test_wall_collision_ends_episode() -> None:
    env = SnakeEnv(width=6, height=6, seed=1)
    env.snake = [Point(5, 3), Point(4, 3), Point(3, 3)]
    env.direction = Direction.RIGHT

    _, reward, done, _ = env.step(0)

    assert done is True
    assert reward == -10.0

def test_can_move_into_current_tail_when_not_eating() -> None:
    env = SnakeEnv(width=6, height=6, seed=1)
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

