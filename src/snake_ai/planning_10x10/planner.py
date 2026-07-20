from __future__ import annotations

from snake_ai.planning_10x10.config import Planner10x10Config
from snake_ai.planning_10x10.hamiltonian import HamiltonianCycle10x10
from snake_ai.planning_10x10.simulator import simulate_path
from snake_ai.planning_10x10.tail_safe import assess_food_actions, reachable_cells
from snake_ai.planning_10x10.types import (
    DecisionTier,
    FoodPathAssessment,
    HamiltonianInvariantError,
    NoSafeActionError,
    PlanCommitmentError,
    PlanningState,
    PlannerDecision,
)


class StrictSafePlanner10x10:
    """tail-safe A* 与 Hamiltonian 不变量组成的严格推理规划器。"""

    def __init__(self, config: Planner10x10Config | None = None) -> None:
        self.config = config or Planner10x10Config()
        self.cycle = HamiltonianCycle10x10()

    def certify(
        self,
        state: PlanningState,
        *,
        committed_path: tuple[int, ...] | None = None,
    ) -> PlannerDecision:
        self._require_compatible_state(state)
        if state.food is None:
            raise NoSafeActionError("live planning state has no food")

        cycle_safe = self.cycle.cycle_safe_actions(state)
        if not cycle_safe:
            raise NoSafeActionError("Hamiltonian guard certified no legal action")

        max_actions_to_food = self.config.starvation_limit - state.steps_since_food + 1
        if max_actions_to_food < 1:
            raise NoSafeActionError("no remaining starvation budget to reach food")

        if committed_path is not None:
            if not committed_path:
                raise PlanCommitmentError("committed food path must not be empty")
            if len(committed_path) > max_actions_to_food:
                raise PlanCommitmentError("committed food path exceeds the starvation budget")
            if not self._path_is_strictly_safe(state, committed_path):
                raise PlanCommitmentError("committed food path no longer matches the live state")

        assessments = assess_food_actions(
            state,
            cycle_safe,
            max_expansions=self.config.max_astar_expansions,
            max_actions_to_food=max_actions_to_food,
            state_validator=self.cycle.is_state_compatible,
            allow_dynamic_retry=False,
        )
        path_by_action = self._safe_paths(assessments)
        if committed_path is not None:
            action = committed_path[0]
            existing = path_by_action.get(action)
            if existing is None or len(committed_path) < len(existing):
                path_by_action[action] = committed_path

        if not path_by_action:
            # 只有快速静态候选和已认证后缀都无解时，才支付完整状态搜索成本。
            # 这保留严格性，同时避免在每个普通帧为所有被拒动作做昂贵搜索。
            assessments = assess_food_actions(
                state,
                cycle_safe,
                max_expansions=self.config.max_astar_expansions,
                max_actions_to_food=max_actions_to_food,
                state_validator=self.cycle.is_state_compatible,
                allow_dynamic_retry=True,
            )
            path_by_action = self._safe_paths(assessments)

        if path_by_action:
            # 只暴露最短安全食物路径的首动作，防止逐帧重新规划时反复绕开食物。
            shortest = min(len(path) for path in path_by_action.values())
            selected = tuple(
                sorted(
                    (action, path)
                    for action, path in path_by_action.items()
                    if len(path) == shortest
                )
            )
            return PlannerDecision(
                admissible_actions=tuple(action for action, _ in selected),
                certified_paths=tuple(path for _, path in selected),
                tier=DecisionTier.SAFE_FOOD,
                food_assessments=assessments,
                cycle_safe_actions=cycle_safe,
            )

        # 有界完整状态 A* 仍可能因 expansion 上限而没有找到路径。此时只允许
        # 经过完整动态模拟验证的环后继路径；这是安全基线，不退回原始 DQN。
        cycle_path = self.cycle.path_to_food(state)
        if cycle_path is None or len(cycle_path) > max_actions_to_food:
            raise NoSafeActionError("no certified path can reach food before starvation")
        if not self._path_is_strictly_safe(state, cycle_path):
            raise NoSafeActionError("Hamiltonian food path failed strict safety validation")

        return PlannerDecision(
            admissible_actions=(cycle_path[0],),
            certified_paths=(cycle_path,),
            tier=DecisionTier.HAMILTONIAN_CYCLE,
            food_assessments=assessments,
            cycle_safe_actions=cycle_safe,
        )

    def _path_is_strictly_safe(
        self,
        state: PlanningState,
        actions: tuple[int, ...],
    ) -> bool:
        if not actions:
            return False
        transitions = simulate_path(state, actions)
        if len(transitions) != len(actions):
            return False
        if any(
            transition.collision
            or transition.next_state is None
            or not self.cycle.is_state_compatible(transition.next_state)
            for transition in transitions
        ):
            return False
        final = transitions[-1]
        if not final.ate_food or final.next_state is None:
            return False
        if final.board_completed:
            return True
        return final.next_state.tail in reachable_cells(final.next_state)

    @staticmethod
    def _safe_paths(
        assessments: tuple[FoodPathAssessment, ...],
    ) -> dict[int, tuple[int, ...]]:
        return {
            assessment.action: assessment.actions
            for assessment in assessments
            if assessment.safe and assessment.actions is not None
        }

    def _require_compatible_state(self, state: PlanningState) -> None:
        if (state.width, state.height) != (self.config.width, self.config.height):
            raise ValueError("strict planner requires an exact 10x10 planning state")
        if not self.cycle.is_state_compatible(state):
            raise HamiltonianInvariantError(
                "snake body no longer follows the configured Hamiltonian order"
            )
