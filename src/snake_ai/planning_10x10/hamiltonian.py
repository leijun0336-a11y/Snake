from __future__ import annotations

from itertools import pairwise

from snake_ai.game import Point
from snake_ai.planning_10x10.simulator import action_to_adjacent_point, simulate_action
from snake_ai.planning_10x10.types import PlanningState


class HamiltonianCycle10x10:
    """与项目 10×10 初始蛇方向对齐的固定 Hamiltonian 环。"""

    width = 10
    height = 10

    def __init__(self) -> None:
        self.cells = self._build_cycle()
        self.index_by_point = {point: index for index, point in enumerate(self.cells)}
        self._validate_cycle()

    @staticmethod
    def _build_cycle() -> tuple[Point, ...]:
        cells: list[Point] = [Point(0, y) for y in range(10)]

        # 保留第 0 列用于闭环，其余格按行蛇形。第 5 行向右，正好对齐
        # reset 后 (3,5) -> (4,5) -> (5,5) 的身体方向。
        for y in range(9, 0, -1):
            xs = range(1, 10) if (9 - y) % 2 == 0 else range(9, 0, -1)
            cells.extend(Point(x, y) for x in xs)
        cells.extend(Point(x, 0) for x in range(9, 0, -1))
        return tuple(cells)

    def _validate_cycle(self) -> None:
        expected_size = self.width * self.height
        if len(self.cells) != expected_size or len(set(self.cells)) != expected_size:
            raise RuntimeError("Hamiltonian cycle must contain every 10x10 cell exactly once")
        wrapped = (*self.cells, self.cells[0])
        if any(_manhattan(first, second) != 1 for first, second in pairwise(wrapped)):
            raise RuntimeError("Hamiltonian cycle contains non-adjacent cells")

    def is_ordered(self, snake: tuple[Point, ...]) -> bool:
        """允许 shortcut 留下间隙，但禁止身体沿环累计绕行一整圈。"""

        if not snake or len(set(snake)) != len(snake):
            return False
        if any(point not in self.index_by_point for point in snake):
            return False

        tail_to_head = [self.index_by_point[point] for point in reversed(snake)]
        travelled = sum(
            (next_index - current_index) % len(self.cells)
            for current_index, next_index in pairwise(tail_to_head)
        )
        return travelled < len(self.cells)

    def is_state_compatible(self, state: PlanningState) -> bool:
        if (state.width, state.height) != (self.width, self.height):
            return False
        if not self.is_ordered(state.snake):
            return False
        # 顺序不变量还必须能落实为项目的相对动作；这会排除“后继格恰好在颈部、
        # 需要直接反向”的人工异常状态。
        return action_to_adjacent_point(state, self.successor(state.head)) is not None

    def cycle_safe_actions(self, state: PlanningState) -> tuple[int, ...]:
        safe: list[int] = []
        for action in (0, 1, 2):
            transition = simulate_action(state, action)
            if transition.collision or transition.next_state is None:
                continue
            if self.is_state_compatible(transition.next_state):
                safe.append(action)
        return tuple(safe)

    def forward_distance(self, start: Point, target: Point) -> int:
        """返回沿固定环从 ``start`` 前进到 ``target`` 所需的步数。"""

        return (self.index_by_point[target] - self.index_by_point[start]) % len(self.cells)

    def viability_safe_actions(
        self,
        state: PlanningState,
        *,
        starvation_limit: int,
    ) -> tuple[int, ...]:
        """认证一步捷径后仍能在饿死前沿 Hamiltonian 环吃到当前食物。

        这里不运行 A*。每个候选动作只精确模拟一步；若它保持蛇身沿环有序，后续就有
        一条可构造的恢复路线：始终走环后继直到食物。恢复路线长度可以用环编号之差
        O(1) 算出，因此适合在 DQN 训练的每一步调用。
        """

        if starvation_limit < 1:
            raise ValueError("starvation_limit must be positive")
        if not self.is_state_compatible(state) or state.food is None:
            return ()

        # experiment8 在 steps_since_food > limit 时饿死，所以当前还能执行
        # limit - steps_since_food + 1 步；最后一步若吃到食物会先把计数清零。
        max_actions_to_food = starvation_limit - state.steps_since_food + 1
        if max_actions_to_food < 1:
            return ()

        safe: list[int] = []
        for action in (0, 1, 2):
            transition = simulate_action(state, action)
            if transition.collision or transition.next_state is None:
                continue
            if not self.is_state_compatible(transition.next_state):
                continue
            if transition.ate_food:
                safe.append(action)
                continue

            remaining = self.forward_distance(transition.next_state.head, state.food)
            if 1 + remaining <= max_actions_to_food:
                safe.append(action)
        return tuple(safe)

    def successor(self, point: Point) -> Point:
        index = self.index_by_point[point]
        return self.cells[(index + 1) % len(self.cells)]

    def successor_action(self, state: PlanningState) -> int:
        action = action_to_adjacent_point(state, self.successor(state.head))
        if action is None:
            raise RuntimeError("Hamiltonian successor would require a reverse or non-adjacent move")
        return action

    def path_to_food(self, state: PlanningState) -> tuple[int, ...] | None:
        """沿环后继生成一条确定路径，并用动态身体状态逐步前进。"""

        if state.food is None:
            return None
        current = state
        actions: list[int] = []
        for _ in range(len(self.cells)):
            action = self.successor_action(current)
            transition = simulate_action(current, action)
            if transition.collision or transition.next_state is None:
                return None
            actions.append(action)
            if transition.ate_food:
                return tuple(actions)
            current = transition.next_state
        return None


def _manhattan(first: Point, second: Point) -> int:
    return abs(first.x - second.x) + abs(first.y - second.y)
