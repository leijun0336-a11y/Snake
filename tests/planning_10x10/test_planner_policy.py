from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from snake_ai.game import Direction, Point, SnakeEnv
from snake_ai.planning_10x10.config import Planner10x10Config
from snake_ai.planning_10x10.dqn_policy import PlannedDQNPolicy10x10
from snake_ai.planning_10x10.planner import StrictSafePlanner10x10
from snake_ai.planning_10x10.simulator import simulate_action
from snake_ai.planning_10x10.types import (
    DecisionTier,
    HamiltonianInvariantError,
    NoSafeActionError,
    PlanCommitmentError,
    PlanningState,
    PlannerDecision,
)


class FixedQNetwork(nn.Module):
    def __init__(self, values: tuple[float, float, float]) -> None:
        super().__init__()
        self.register_buffer("values", torch.tensor(values, dtype=torch.float32))

    def forward(self, state):
        grid, _ = state
        return self.values.unsqueeze(0).expand(grid.shape[0], -1)


class StubPlanner:
    def __init__(self, actions: tuple[int, ...]) -> None:
        self.actions = actions

    def certify(
        self,
        state: PlanningState,
        *,
        committed_path: tuple[int, ...] | None = None,
    ) -> PlannerDecision:
        del state, committed_path
        if not self.actions:
            raise NoSafeActionError("no certified action")
        return PlannerDecision(
            admissible_actions=self.actions,
            certified_paths=tuple((action, 0) for action in self.actions),
            tier=DecisionTier.SAFE_FOOD,
            food_assessments=(),
            cycle_safe_actions=self.actions,
        )


def make_fake_agent(values: tuple[float, float, float]):
    return SimpleNamespace(
        state_mode="hybrid",
        state_size=(9, 10, 10),
        device=torch.device("cpu"),
        policy_net=FixedQNetwork(values),
    )


def make_observation():
    return np.zeros((9, 10, 10), dtype=np.float32), [0.0] * 20


def make_state() -> PlanningState:
    return PlanningState(
        width=10,
        height=10,
        snake=(Point(5, 5), Point(4, 5), Point(3, 5)),
        direction=Direction.RIGHT,
        food=Point(8, 5),
    )


def test_policy_chooses_highest_q_only_inside_certified_set() -> None:
    policy = PlannedDQNPolicy10x10(make_fake_agent((9.0, 2.0, 4.0)), StubPlanner((1, 2)))

    result = policy.choose_action(make_observation(), make_state())

    assert result.raw_dqn_action == 0
    assert result.action == 2
    assert result.overridden is True


def test_policy_q_ties_keep_lowest_action_index() -> None:
    policy = PlannedDQNPolicy10x10(make_fake_agent((-1.0, 3.0, 3.0)), StubPlanner((1, 2)))

    result = policy.choose_action(make_observation(), make_state())

    assert result.action == 1


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_policy_rejects_non_finite_q_values(bad_value: float) -> None:
    policy = PlannedDQNPolicy10x10(
        make_fake_agent((bad_value, 1.0, 0.0)),
        StubPlanner((1,)),
    )

    with pytest.raises(ValueError, match="invalid Q values"):
        policy.choose_action(make_observation(), make_state())


def test_policy_does_not_fallback_when_planner_has_no_action() -> None:
    policy = PlannedDQNPolicy10x10(make_fake_agent((9.0, 2.0, 4.0)), StubPlanner(()))

    with pytest.raises(NoSafeActionError):
        policy.choose_action(make_observation(), make_state())


def test_strict_planner_rejects_broken_cycle_invariant() -> None:
    state = PlanningState(
        width=10,
        height=10,
        snake=(Point(3, 5), Point(4, 5), Point(5, 5)),
        direction=Direction.LEFT,
        food=Point(8, 8),
    )

    with pytest.raises(HamiltonianInvariantError):
        StrictSafePlanner10x10().certify(state)


def test_real_planner_only_returns_actions_preserving_cycle_order() -> None:
    env = SnakeEnv(width=10, height=10, seed=3, state_mode="hybrid")
    state = PlanningState.from_env(env)
    planner = StrictSafePlanner10x10()

    decision = planner.certify(state)

    assert decision.admissible_actions
    for action in decision.admissible_actions:
        transition = simulate_action(state, action)
        assert transition.next_state is not None
        assert planner.cycle.is_state_compatible(transition.next_state)


def test_planner_uses_certified_hamiltonian_path_when_astar_budget_is_exhausted() -> None:
    env = SnakeEnv(width=10, height=10, seed=3, state_mode="hybrid")
    planner = StrictSafePlanner10x10(Planner10x10Config(max_astar_expansions=1))

    decision = planner.certify(PlanningState.from_env(env))

    assert decision.tier is DecisionTier.HAMILTONIAN_CYCLE
    assert decision.admissible_actions == (
        planner.cycle.successor_action(PlanningState.from_env(env)),
    )


def test_planner_uses_experiment8_starvation_boundary_without_off_by_one() -> None:
    planner = StrictSafePlanner10x10()
    can_eat_now = PlanningState(
        width=10,
        height=10,
        snake=(Point(5, 5), Point(4, 5), Point(3, 5)),
        direction=Direction.RIGHT,
        food=Point(6, 5),
        steps_since_food=100,
    )
    already_expired = PlanningState(
        width=10,
        height=10,
        snake=can_eat_now.snake,
        direction=can_eat_now.direction,
        food=can_eat_now.food,
        steps_since_food=101,
    )

    assert planner.certify(can_eat_now).admissible_actions == (0,)
    with pytest.raises(NoSafeActionError, match="starvation budget"):
        planner.certify(already_expired)


@pytest.mark.parametrize("limit", [99, 101, 200])
def test_strict_config_cannot_change_the_environment_starvation_limit(limit: int) -> None:
    with pytest.raises(ValueError, match="starvation_limit=100"):
        Planner10x10Config(starvation_limit=limit)


def test_committed_safe_path_survives_next_frame_static_astar_false_negative() -> None:
    """回归：不能丢掉上一帧已经完整动态验证过的路径后缀。"""

    state = _commitment_regression_state()
    planner = StrictSafePlanner10x10()
    first_decision = planner.certify(state)
    selected_path = first_decision.path_for_action(2)
    transition = simulate_action(state, 2)

    assert selected_path == (2, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 0)
    assert transition.next_state is not None
    # 完整状态 A* 能找到静态 A* 漏掉的安全等长路径。
    dynamic_decision = planner.certify(transition.next_state)
    assert dynamic_decision.admissible_actions == (selected_path[1],)

    # 即使把后续搜索预算压到 1，已认证后缀仍可独立维持递归可行性。
    limited_planner = StrictSafePlanner10x10(Planner10x10Config(max_astar_expansions=1))
    with pytest.raises(NoSafeActionError):
        limited_planner.certify(transition.next_state)

    continued = limited_planner.certify(
        transition.next_state,
        committed_path=selected_path[1:],
    )

    assert continued.admissible_actions == (selected_path[1],)
    assert continued.path_for_action(selected_path[1]) == selected_path[1:]


def test_policy_tracks_selected_certified_path_and_rejects_state_divergence() -> None:
    state = _commitment_regression_state()
    policy = PlannedDQNPolicy10x10(
        make_fake_agent((1.0, 0.0, 9.0)),
        StrictSafePlanner10x10(),
    )

    first = policy.choose_action(make_observation(), state)
    transition = simulate_action(state, first.action)

    assert first.action == 2
    assert transition.next_state is not None
    second = policy.choose_action(make_observation(), transition.next_state)
    assert second.action == 0

    with pytest.raises(PlanCommitmentError, match="live state differs"):
        policy.choose_action(make_observation(), state)

    policy.reset()
    assert policy.choose_action(make_observation(), state).action == 2


def _commitment_regression_state() -> PlanningState:
    coordinates = (
        (2, 9),
        (1, 9),
        (0, 9),
        (0, 8),
        (0, 7),
        (0, 6),
        (0, 5),
        (0, 4),
        (0, 3),
        (0, 2),
        (0, 1),
        (0, 0),
        (1, 0),
        (2, 0),
        (3, 0),
        (4, 0),
        (4, 1),
        (3, 1),
        (3, 2),
        (3, 3),
        (2, 3),
        (2, 4),
        (2, 5),
        (2, 6),
    )
    return PlanningState(
        width=10,
        height=10,
        snake=tuple(Point(x, y) for x, y in coordinates),
        direction=Direction.RIGHT,
        food=Point(6, 0),
        steps_since_food=50,
    )
