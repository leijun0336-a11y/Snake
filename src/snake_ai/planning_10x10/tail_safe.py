from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable

from snake_ai.game import Direction, Point
from snake_ai.planning_10x10.astar import find_dynamic_path, find_path
from snake_ai.planning_10x10.simulator import advance_point, simulate_action, simulate_path
from snake_ai.planning_10x10.types import FoodPathAssessment, PlanningState


StateValidator = Callable[[PlanningState], bool]


def assess_food_action(
    state: PlanningState,
    action: int,
    *,
    max_expansions: int,
    max_actions_to_food: int,
    state_validator: StateValidator,
    allow_dynamic_retry: bool = True,
) -> FoodPathAssessment:
    """验证以指定首动作开始的 A* 食物路径是否 tail-safe。"""

    first = simulate_action(state, action)
    if first.collision or first.next_state is None or not state_validator(first.next_state):
        return _unsafe_assessment(action)

    if first.ate_food:
        path = (action,)
        expansions = 0
    else:
        if state.food is None:
            return _unsafe_assessment(action)
        search = find_path(first.next_state, state.food, max_expansions=max_expansions)
        if search is not None:
            path = (action, *search.actions)
            assessment = _assess_candidate_path(
                state,
                action,
                path,
                expansions=search.expansions,
                max_actions_to_food=max_actions_to_food,
                state_validator=state_validator,
            )
            if assessment.safe:
                return assessment
            expansions = search.expansions
        else:
            expansions = 0

        if not allow_dynamic_retry:
            return _unsafe_assessment(action, expansions=expansions)

        # 静态 A* 只产出一条最短路。若它被动态验证拒绝，再用完整蛇身状态
        # 搜索安全等长路/绕行路，避免把“第一条候选失败”误判成“无解”。
        dynamic = find_dynamic_path(
            first.next_state,
            state.food,
            max_expansions=max_expansions,
            state_validator=state_validator,
        )
        if dynamic is None:
            return _unsafe_assessment(action, expansions=expansions)
        path = (action, *dynamic.actions)
        expansions += dynamic.expansions

    return _assess_candidate_path(
        state,
        action,
        path,
        expansions=expansions,
        max_actions_to_food=max_actions_to_food,
        state_validator=state_validator,
    )


def assess_food_actions(
    state: PlanningState,
    actions: Iterable[int],
    *,
    max_expansions: int,
    max_actions_to_food: int,
    state_validator: StateValidator,
    allow_dynamic_retry: bool = True,
) -> tuple[FoodPathAssessment, ...]:
    return tuple(
        assess_food_action(
            state,
            action,
            max_expansions=max_expansions,
            max_actions_to_food=max_actions_to_food,
            state_validator=state_validator,
            allow_dynamic_retry=allow_dynamic_retry,
        )
        for action in actions
    )


def reachable_cells(state: PlanningState) -> frozenset[Point]:
    """静态连通检查将蛇尾视为可到达目标，其余身体视为障碍。"""

    blocked = set(state.snake[1:-1])
    visited = {state.head}
    queue = deque([state.head])
    directions = (
        # 这里只检查拓扑连通性，不承担真实动作生成，因此允许四邻接扩展。
        Direction.RIGHT,
        Direction.DOWN,
        Direction.LEFT,
        Direction.UP,
    )
    while queue:
        current = queue.popleft()
        for direction in directions:
            point = advance_point(current, direction)
            if not (0 <= point.x < state.width and 0 <= point.y < state.height):
                continue
            if point in blocked or point in visited:
                continue
            visited.add(point)
            queue.append(point)
    return frozenset(visited)


def _assess_candidate_path(
    state: PlanningState,
    action: int,
    path: tuple[int, ...],
    *,
    expansions: int,
    max_actions_to_food: int,
    state_validator: StateValidator,
) -> FoodPathAssessment:
    if not path or path[0] != action or len(path) > max_actions_to_food:
        return _unsafe_assessment(action, expansions=expansions)

    transitions = simulate_path(state, path)
    if len(transitions) != len(path):
        return _unsafe_assessment(action, expansions=expansions)
    if any(transition.collision for transition in transitions):
        return _unsafe_assessment(action, expansions=expansions)
    if any(
        transition.next_state is None or not state_validator(transition.next_state)
        for transition in transitions
    ):
        return _unsafe_assessment(action, expansions=expansions)

    final = transitions[-1]
    if not final.ate_food or final.next_state is None:
        return _unsafe_assessment(action, expansions=expansions)
    if final.board_completed:
        return FoodPathAssessment(
            action=action,
            safe=True,
            actions=path,
            expansions=expansions,
            tail_reachable=True,
            reachable_area=1,
        )

    reachable = reachable_cells(final.next_state)
    tail_reachable = final.next_state.tail in reachable
    return FoodPathAssessment(
        action=action,
        safe=tail_reachable,
        actions=path if tail_reachable else None,
        expansions=expansions,
        tail_reachable=tail_reachable,
        reachable_area=len(reachable),
    )


def _unsafe_assessment(action: int, *, expansions: int = 0) -> FoodPathAssessment:
    return FoodPathAssessment(
        action=action,
        safe=False,
        actions=None,
        expansions=expansions,
        tail_reachable=False,
        reachable_area=0,
    )
