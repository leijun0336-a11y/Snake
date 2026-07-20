from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from snake_ai.agents import DQNAgent
from snake_ai.game.snake_env import HybridState
from snake_ai.planning_10x10.planner import StrictSafePlanner10x10
from snake_ai.planning_10x10.simulator import simulate_action
from snake_ai.planning_10x10.types import PlanCommitmentError, PlanningState, PlannerDecision


@dataclass(frozen=True, slots=True)
class PlannedAction:
    action: int
    raw_dqn_action: int
    q_values: tuple[float, float, float]
    planner_decision: PlannerDecision

    @property
    def overridden(self) -> bool:
        return self.action != self.raw_dqn_action


class PlannedDQNPolicy10x10:
    """保持原 DQN 权重不变，只在严格认证动作中进行 Q 值排序。"""

    def __init__(self, agent: DQNAgent, planner: StrictSafePlanner10x10) -> None:
        if agent.state_mode != "hybrid":
            raise ValueError("planned 10x10 policy requires a hybrid DQN agent")
        if agent.state_size != (9, 10, 10):
            raise ValueError("planned 10x10 policy requires state_size=(9, 10, 10)")
        self.agent = agent
        self.planner = planner
        self.agent.policy_net.eval()
        self._committed_actions: tuple[int, ...] = ()
        self._expected_state: PlanningState | None = None

    def reset(self) -> None:
        """开始新一局前清除上一局尚未执行完的路径承诺。"""

        self._committed_actions = ()
        self._expected_state = None

    def choose_action(
        self,
        observation: HybridState,
        planning_state: PlanningState,
    ) -> PlannedAction:
        committed_path: tuple[int, ...] | None = None
        if self._committed_actions:
            if planning_state != self._expected_state:
                raise PlanCommitmentError(
                    "live state differs from the next state of the committed food path"
                )
            committed_path = self._committed_actions

        # 先认证安全集合；认证失败立即报错，绝不让网络输出绕过规划器。
        decision = self.planner.certify(planning_state, committed_path=committed_path)
        q_values = self._q_values(observation)
        raw_action = max(range(3), key=q_values.__getitem__)
        action = max(decision.admissible_actions, key=q_values.__getitem__)
        self._commit_selected_path(planning_state, decision.path_for_action(action))
        return PlannedAction(
            action=action,
            raw_dqn_action=raw_action,
            q_values=q_values,
            planner_decision=decision,
        )

    def _commit_selected_path(
        self,
        state: PlanningState,
        path: tuple[int, ...],
    ) -> None:
        transition = simulate_action(state, path[0])
        if transition.collision or transition.next_state is None:
            raise PlanCommitmentError("planner returned a path with an invalid first action")

        remainder = path[1:]
        if not remainder:
            if not transition.ate_food:
                raise PlanCommitmentError("certified one-step food path did not eat food")
            self.reset()
            return

        self._committed_actions = remainder
        self._expected_state = transition.next_state

    def _q_values(self, observation: HybridState) -> tuple[float, float, float]:
        grid, auxiliary = observation
        grid_array = np.asarray(grid, dtype=np.float32)
        auxiliary_array = np.asarray(auxiliary, dtype=np.float32)
        if grid_array.shape != (9, 10, 10) or auxiliary_array.shape != (20,):
            raise ValueError("planned policy received an invalid 10x10 hybrid observation")

        grid_tensor = torch.from_numpy(np.ascontiguousarray(grid_array[None])).to(self.agent.device)
        auxiliary_tensor = torch.from_numpy(np.ascontiguousarray(auxiliary_array[None])).to(
            self.agent.device
        )
        with torch.inference_mode():
            values = self.agent.policy_net((grid_tensor, auxiliary_tensor))[0]
        raw_values = values.detach().cpu().tolist()
        if len(raw_values) != 3:
            raise ValueError("DQN produced an invalid action dimension")
        result = tuple(float(value) for value in raw_values)
        if not all(np.isfinite(value) for value in result):
            raise ValueError("DQN produced invalid Q values")
        return result[0], result[1], result[2]
