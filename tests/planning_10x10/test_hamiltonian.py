from __future__ import annotations

from itertools import pairwise
from random import Random

import pytest

from snake_ai.game import Direction, Point, SnakeEnv
from snake_ai.planning_10x10.hamiltonian import HamiltonianCycle10x10
from snake_ai.planning_10x10.simulator import simulate_action, simulate_path
from snake_ai.planning_10x10.types import PlanningState


def test_cycle_covers_board_once_and_closes_with_adjacent_edges() -> None:
    cycle = HamiltonianCycle10x10()

    assert len(cycle.cells) == 100
    assert len(set(cycle.cells)) == 100
    wrapped = (*cycle.cells, cycle.cells[0])
    assert all(_distance(first, second) == 1 for first, second in pairwise(wrapped))


def test_reset_state_is_ordered_and_successor_is_straight() -> None:
    env = SnakeEnv(width=10, height=10, seed=1)
    cycle = HamiltonianCycle10x10()
    state = PlanningState.from_env(env)

    assert cycle.is_state_compatible(state)
    assert cycle.successor(state.head) == Point(6, 5)
    assert cycle.successor_action(state) == 0


def test_order_check_supports_shortcut_gaps_but_rejects_reverse_order() -> None:
    cycle = HamiltonianCycle10x10()
    contiguous = (Point(5, 5), Point(4, 5), Point(3, 5))
    with_gap = (cycle.cells[50], cycle.cells[45], cycle.cells[40])

    assert cycle.is_ordered(contiguous)
    assert cycle.is_ordered(with_gap)
    assert not cycle.is_ordered(tuple(reversed(contiguous)))


def test_state_compatibility_rejects_a_successor_that_requires_direct_reverse() -> None:
    cycle = HamiltonianCycle10x10()
    state = PlanningState(
        width=10,
        height=10,
        snake=(Point(0, 0), Point(0, 1)),
        direction=Direction.UP,
        food=Point(9, 9),
    )

    # 单看环编号仍有序，但环后继 (0,1) 正是颈部，项目的三个动作不能直接反向。
    assert cycle.is_ordered(state.snake)
    assert not cycle.is_state_compatible(state)


def test_every_cycle_safe_action_preserves_order_after_exact_simulation() -> None:
    env = SnakeEnv(width=10, height=10, seed=2)
    cycle = HamiltonianCycle10x10()
    state = PlanningState.from_env(env)
    actions = cycle.cycle_safe_actions(state)

    assert actions
    assert cycle.successor_action(state) in actions
    for action in actions:
        transition = simulate_action(state, action)
        assert transition.next_state is not None
        assert cycle.is_state_compatible(transition.next_state)


def test_viability_mask_keeps_all_order_preserving_actions_with_full_budget() -> None:
    env = SnakeEnv(width=10, height=10, seed=3)
    cycle = HamiltonianCycle10x10()
    state = PlanningState.from_env(env)

    assert cycle.viability_safe_actions(state, starvation_limit=100) == (
        cycle.cycle_safe_actions(state)
    )


def test_viability_mask_only_allows_immediate_food_at_last_safe_step() -> None:
    cycle = HamiltonianCycle10x10()
    state = PlanningState(
        width=10,
        height=10,
        snake=(Point(5, 5), Point(4, 5), Point(3, 5)),
        direction=Direction.RIGHT,
        food=Point(6, 5),
        steps_since_food=100,
    )

    assert cycle.viability_safe_actions(state, starvation_limit=100) == (0,)


@pytest.mark.parametrize("seed", range(10))
def test_each_viability_action_has_a_constructive_route_before_starvation(seed: int) -> None:
    env = SnakeEnv(width=10, height=10, seed=seed)
    cycle = HamiltonianCycle10x10()
    state = PlanningState.from_env(env)
    budget = 100 - state.steps_since_food + 1

    actions = cycle.viability_safe_actions(state, starvation_limit=100)
    assert actions
    for action in actions:
        first = simulate_action(state, action)
        assert first.next_state is not None
        if first.ate_food:
            route = (action,)
        else:
            suffix = cycle.path_to_food(first.next_state)
            assert suffix is not None
            route = (action, *suffix)

        transitions = simulate_path(state, route)
        assert len(route) <= budget
        assert len(transitions) == len(route)
        assert all(not transition.collision for transition in transitions)
        assert transitions[-1].ate_food is True


def test_hamiltonian_food_path_is_dynamically_legal_and_eats() -> None:
    cycle = HamiltonianCycle10x10()
    for seed in range(10):
        env = SnakeEnv(width=10, height=10, seed=seed)
        state = PlanningState.from_env(env)
        path = cycle.path_to_food(state)

        assert path is not None
        assert 1 <= len(path) <= 97
        transitions = simulate_path(state, path)
        assert len(transitions) == len(path)
        assert all(not transition.collision for transition in transitions)
        assert all(
            transition.next_state is not None and cycle.is_state_compatible(transition.next_state)
            for transition in transitions
        )
        assert transitions[-1].ate_food is True


@pytest.mark.parametrize("seed", [0, 7, 42])
def test_pure_hamiltonian_policy_completes_training_semantics_episode(seed: int) -> None:
    env = SnakeEnv(
        width=10,
        height=10,
        seed=seed,
        state_mode="hybrid",
        reward_profile="experiment8",
        starvation_enabled=True,
    )
    cycle = HamiltonianCycle10x10()
    env.reset(seed=seed)
    done = False

    while not done and env.frame_iteration < 5_000:
        state = PlanningState.from_env(env)
        _, _, done, info = env.step(cycle.successor_action(state))

    assert done is True
    assert info["termination_reason"] == "board_completed"
    assert info["score"] == 97
    assert env.frame_iteration <= 5_000


def test_random_viability_mask_policy_never_falls_back_and_completes_board() -> None:
    env = SnakeEnv(
        width=10,
        height=10,
        seed=1,
        reward_profile="experiment8",
        starvation_enabled=True,
    )
    cycle = HamiltonianCycle10x10()
    random = Random(123)
    done = False

    while not done and env.frame_iteration < 10_000:
        state = PlanningState.from_env(env)
        actions = cycle.viability_safe_actions(state, starvation_limit=env.starvation_limit)
        assert actions  # 不允许回退到三个原始动作。
        _, _, done, info = env.step(random.choice(actions))

    assert done is True
    assert info["termination_reason"] == "board_completed"
    assert info["score"] == 97


def _distance(first: Point, second: Point) -> int:
    return abs(first.x - second.x) + abs(first.y - second.y)
