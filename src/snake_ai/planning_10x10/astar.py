from __future__ import annotations

from collections.abc import Callable
import heapq
from itertools import count

from snake_ai.game import Direction, Point
from snake_ai.planning_10x10.simulator import (
    advance_point,
    in_bounds,
    simulate_action,
    turned_direction,
)
from snake_ai.planning_10x10.types import AStarResult, PlanningState


SearchNode = tuple[Point, Direction]
DynamicSearchKey = tuple[tuple[Point, ...], Direction, Point | None]
StateValidator = Callable[[PlanningState], bool]


def find_path(
    state: PlanningState,
    goal: Point,
    *,
    max_expansions: int,
) -> AStarResult | None:
    """在静态占用图上搜索候选路径。

    节点包含方向以遵守相对动作约束。该搜索故意只负责提出候选，最终合法性由
    无副作用模拟器逐步验证，避免把静态蛇身近似误当成真实环境转移。
    """

    if max_expansions < 1:
        raise ValueError("max_expansions must be positive")
    if not (0 <= goal.x < state.width and 0 <= goal.y < state.height):
        raise ValueError("A* goal is out of bounds")
    if state.head == goal:
        return AStarResult(actions=(), expansions=0)

    # 当前尾格会在普通移动后释放；其余身体在静态候选图中保守地视为障碍。
    blocked = set(state.snake[1:-1])
    start: SearchNode = (state.head, state.direction)
    best_g: dict[SearchNode, int] = {start: 0}
    came_from: dict[SearchNode, tuple[SearchNode, int]] = {}
    serial = count()
    start_h = _manhattan(state.head, goal)
    queue: list[tuple[int, int, int, int, SearchNode]] = [
        (start_h, start_h, 0, next(serial), start)
    ]
    expansions = 0

    while queue:
        _, _, current_g, _, current = heapq.heappop(queue)
        if current_g != best_g.get(current):
            continue
        point, direction = current
        if point == goal:
            return AStarResult(
                actions=_reconstruct_actions(came_from, current),
                expansions=expansions,
            )
        if expansions >= max_expansions:
            break

        expansions += 1
        for action in (0, 1, 2):
            next_direction = turned_direction(direction, action)
            next_point = advance_point(point, next_direction)
            if not in_bounds(state, next_point):
                continue
            if next_point in blocked and next_point != goal:
                continue

            next_node = (next_point, next_direction)
            next_g = current_g + 1
            if next_g >= best_g.get(next_node, 1 << 30):
                continue
            best_g[next_node] = next_g
            came_from[next_node] = current, action
            heuristic = _manhattan(next_point, goal)
            heapq.heappush(
                queue,
                (next_g + heuristic, heuristic, next_g, next(serial), next_node),
            )

    return None


def find_dynamic_path(
    state: PlanningState,
    goal: Point,
    *,
    max_expansions: int,
    state_validator: StateValidator,
) -> AStarResult | None:
    """用完整蛇身状态搜索，扩展时立即剪掉碰撞和不满足不变量的分支。

    普通静态 A* 速度快，但它只返回一条几何最短路；若该路破坏 Hamiltonian
    顺序，不能据此断言不存在另一条同长度安全路。本函数用于消除这种假阴性。
    """

    if max_expansions < 1:
        raise ValueError("max_expansions must be positive")
    if not (0 <= goal.x < state.width and 0 <= goal.y < state.height):
        raise ValueError("A* goal is out of bounds")
    if state.head == goal:
        return AStarResult(actions=(), expansions=0)

    start_key = _dynamic_key(state)
    best_g: dict[DynamicSearchKey, int] = {start_key: 0}
    came_from: dict[DynamicSearchKey, tuple[DynamicSearchKey, int]] = {}
    serial = count()
    start_h = _manhattan(state.head, goal)
    queue: list[tuple[int, int, int, int, DynamicSearchKey, PlanningState]] = [
        (start_h, start_h, 0, next(serial), start_key, state)
    ]
    expansions = 0

    while queue:
        _, _, current_g, _, current_key, current = heapq.heappop(queue)
        if current_g != best_g.get(current_key):
            continue
        if current.head == goal:
            return AStarResult(
                actions=_reconstruct_actions(came_from, current_key),
                expansions=expansions,
            )
        if expansions >= max_expansions:
            break

        expansions += 1
        for action in (0, 1, 2):
            transition = simulate_action(current, action)
            if transition.collision or transition.next_state is None:
                continue
            next_state = transition.next_state
            if not state_validator(next_state):
                continue

            next_key = _dynamic_key(next_state)
            next_g = current_g + 1
            if next_g >= best_g.get(next_key, 1 << 30):
                continue
            best_g[next_key] = next_g
            came_from[next_key] = current_key, action
            heuristic = _manhattan(next_state.head, goal)
            heapq.heappush(
                queue,
                (
                    next_g + heuristic,
                    heuristic,
                    next_g,
                    next(serial),
                    next_key,
                    next_state,
                ),
            )

    return None


def _reconstruct_actions(
    came_from: dict[SearchNode, tuple[SearchNode, int]]
    | dict[DynamicSearchKey, tuple[DynamicSearchKey, int]],
    goal: SearchNode | DynamicSearchKey,
) -> tuple[int, ...]:
    actions: list[int] = []
    current = goal
    while current in came_from:
        current, action = came_from[current]
        actions.append(action)
    actions.reverse()
    return tuple(actions)


def _dynamic_key(state: PlanningState) -> DynamicSearchKey:
    # 相同身体、方向和食物下，更短路径同时拥有更宽松的饥饿预算，因此 key
    # 不需要包含 steps_since_food。
    return state.snake, state.direction, state.food


def _manhattan(first: Point, second: Point) -> int:
    return abs(first.x - second.x) + abs(first.y - second.y)
