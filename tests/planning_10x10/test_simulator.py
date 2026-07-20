from __future__ import annotations

import pytest

from snake_ai.game import Direction, Point, SnakeEnv
from snake_ai.planning_10x10.hamiltonian import HamiltonianCycle10x10
from snake_ai.planning_10x10.simulator import simulate_action
from snake_ai.planning_10x10.types import PlanningState


@pytest.mark.parametrize("action", [0, 1, 2])
def test_simulator_matches_real_environment_for_legal_initial_moves(action: int) -> None:
    env = SnakeEnv(width=10, height=10, seed=1, state_mode="hybrid")
    env.food = Point(8, 5)
    state = PlanningState.from_env(env)

    simulated = simulate_action(state, action)
    _, _, done, info = env.step(action)

    assert simulated.collision is False
    assert done is False
    assert simulated.next_state is not None
    assert simulated.next_state.snake == tuple(env.snake)
    assert simulated.next_state.direction == env.direction
    assert simulated.next_state.steps_since_food == env.steps_since_food
    assert simulated.ate_food == (int(info["score"]) == 1)
    assert state.snake == (Point(5, 5), Point(4, 5), Point(3, 5))


def test_simulator_allows_entering_current_tail_when_it_moves() -> None:
    state = PlanningState(
        width=10,
        height=10,
        snake=(
            Point(2, 2),
            Point(3, 2),
            Point(3, 3),
            Point(2, 3),
            Point(1, 3),
            Point(1, 2),
        ),
        direction=Direction.LEFT,
        food=Point(9, 9),
    )

    transition = simulate_action(state, 0)

    assert transition.collision is False
    assert transition.next_state is not None
    assert transition.next_state.head == Point(1, 2)
    assert transition.next_state.tail == Point(1, 3)
    assert len(transition.next_state.snake) == len(state.snake)


def test_simulator_grows_and_clears_planned_food_after_eating() -> None:
    state = PlanningState(
        width=10,
        height=10,
        snake=(Point(5, 5), Point(4, 5), Point(3, 5)),
        direction=Direction.RIGHT,
        food=Point(6, 5),
        steps_since_food=12,
    )

    transition = simulate_action(state, 0)

    assert transition.ate_food is True
    assert transition.next_state is not None
    assert len(transition.next_state.snake) == 4
    assert transition.next_state.tail == Point(3, 5)
    assert transition.next_state.food is None
    assert transition.next_state.steps_since_food == 0


def test_simulator_reports_wall_and_body_collisions_without_mutating_state() -> None:
    wall_state = PlanningState(
        width=10,
        height=10,
        snake=(Point(9, 5), Point(8, 5), Point(7, 5)),
        direction=Direction.RIGHT,
        food=Point(0, 0),
    )
    body_state = PlanningState(
        width=10,
        height=10,
        snake=(Point(2, 2), Point(3, 2), Point(3, 3), Point(2, 3), Point(1, 3)),
        direction=Direction.LEFT,
        food=Point(9, 9),
    )

    wall = simulate_action(wall_state, 0)
    body = simulate_action(body_state, 2)

    assert wall.collision is True and wall.next_state is None
    assert body.collision is True and body.next_state is None
    assert wall_state.head == Point(9, 5)
    assert body_state.head == Point(2, 2)


def test_simulator_accepts_the_last_food_without_tail_reachability() -> None:
    cycle = HamiltonianCycle10x10()
    food = cycle.cells[0]
    snake = tuple(reversed(cycle.cells[1:]))
    state = PlanningState(
        width=10,
        height=10,
        snake=snake,
        direction=_direction_between(snake[1], snake[0]),
        food=food,
    )

    transition = simulate_action(state, cycle.successor_action(state))

    assert transition.collision is False
    assert transition.ate_food is True
    assert transition.board_completed is True
    assert transition.next_state is not None
    assert len(transition.next_state.snake) == 100


@pytest.mark.parametrize(
    ("snake", "direction", "food", "message"),
    [
        (
            (Point(5, 5), Point(4, 5), Point(4, 5)),
            Direction.RIGHT,
            Point(8, 5),
            "duplicate",
        ),
        (
            (Point(5, 5), Point(3, 5)),
            Direction.RIGHT,
            Point(8, 5),
            "adjacent",
        ),
        (
            (Point(5, 5), Point(4, 5), Point(3, 5)),
            Direction.LEFT,
            Point(8, 5),
            "direction",
        ),
        (
            (Point(5, 5), Point(4, 5), Point(3, 5)),
            Direction.RIGHT,
            Point(4, 5),
            "overlap",
        ),
    ],
)
def test_planning_state_rejects_inconsistent_snapshots(
    snake: tuple[Point, ...],
    direction: Direction,
    food: Point,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        PlanningState(
            width=10,
            height=10,
            snake=snake,
            direction=direction,
            food=food,
        )


def test_simulator_matches_environment_across_many_reachable_states() -> None:
    """逐动作差分测试，覆盖吃食物、转弯、尾格释放与碰撞分支。"""

    source = SnakeEnv(
        width=10,
        height=10,
        seed=19,
        state_mode="hybrid",
        reward_profile="experiment8",
        starvation_enabled=True,
    )
    cycle = HamiltonianCycle10x10()

    for _ in range(40):
        state = PlanningState.from_env(source)
        for action in (0, 1, 2):
            simulated = simulate_action(state, action)
            real = _environment_from_state(state)
            old_score = real.score
            _, _, done, info = real.step(action)
            collision = str(info["termination_reason"]).startswith("collision_")

            assert simulated.collision is collision
            if simulated.collision:
                assert done is True
                continue

            assert simulated.next_state is not None
            assert simulated.next_state.snake == tuple(real.snake)
            assert simulated.next_state.direction == real.direction
            assert simulated.next_state.steps_since_food == real.steps_since_food
            assert simulated.ate_food is (real.score == old_score + 1)
            assert simulated.board_completed is (info["termination_reason"] == "board_completed")

        _, _, done, _ = source.step(cycle.successor_action(state))
        assert done is False


def _environment_from_state(state: PlanningState) -> SnakeEnv:
    env = SnakeEnv(
        width=state.width,
        height=state.height,
        seed=123,
        state_mode="hybrid",
        reward_profile="experiment8",
        starvation_enabled=True,
    )
    env.snake = list(state.snake)
    env.direction = state.direction
    env.food = state.food
    env.steps_since_food = state.steps_since_food
    env.score = len(state.snake) - 3
    return env


def _direction_between(start: Point, end: Point) -> Direction:
    delta = end.x - start.x, end.y - start.y
    return {
        (1, 0): Direction.RIGHT,
        (-1, 0): Direction.LEFT,
        (0, 1): Direction.DOWN,
        (0, -1): Direction.UP,
    }[delta]
