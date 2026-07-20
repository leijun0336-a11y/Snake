from __future__ import annotations

from snake_ai.game import Direction, Point
from snake_ai.planning_10x10.astar import find_path
from snake_ai.planning_10x10.hamiltonian import HamiltonianCycle10x10
from snake_ai.planning_10x10.tail_safe import assess_food_action, reachable_cells
from snake_ai.planning_10x10.types import PlanningState


def make_straight_state(food: Point = Point(8, 5)) -> PlanningState:
    return PlanningState(
        width=10,
        height=10,
        snake=(Point(5, 5), Point(4, 5), Point(3, 5)),
        direction=Direction.RIGHT,
        food=food,
    )


def test_astar_returns_deterministic_relative_actions() -> None:
    state = make_straight_state()

    result = find_path(state, Point(8, 5), max_expansions=100)

    assert result is not None
    assert result.actions == (0, 0, 0)
    assert result.expansions > 0


def test_astar_respects_direction_and_never_returns_direct_reverse() -> None:
    state = make_straight_state(Point(2, 5))

    result = find_path(state, Point(2, 5), max_expansions=500)

    assert result is not None
    assert result.actions
    assert result.actions[0] in (0, 1, 2)
    assert result.actions[0] != 3


def test_tail_safe_accepts_short_straight_food_path() -> None:
    state = make_straight_state()
    cycle = HamiltonianCycle10x10()

    assessment = assess_food_action(
        state,
        0,
        max_expansions=500,
        max_actions_to_food=101,
        state_validator=cycle.is_state_compatible,
    )

    assert assessment.safe is True
    assert assessment.actions == (0, 0, 0)
    assert assessment.tail_reachable is True
    assert assessment.reachable_area > 0


def test_tail_safe_rejects_path_when_budget_or_invariant_fails() -> None:
    state = make_straight_state()
    cycle = HamiltonianCycle10x10()

    over_budget = assess_food_action(
        state,
        0,
        max_expansions=500,
        max_actions_to_food=2,
        state_validator=cycle.is_state_compatible,
    )
    invalid_invariant = assess_food_action(
        state,
        0,
        max_expansions=500,
        max_actions_to_food=101,
        state_validator=lambda _: False,
    )

    assert over_budget.safe is False
    assert invalid_invariant.safe is False


def test_reachable_cells_includes_tail_but_not_fixed_body() -> None:
    state = make_straight_state()

    reachable = reachable_cells(state)

    assert state.head in reachable
    assert state.tail in reachable
    assert state.snake[1] not in reachable
