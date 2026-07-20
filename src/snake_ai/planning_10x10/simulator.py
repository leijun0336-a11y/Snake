from __future__ import annotations

from collections.abc import Iterable

from snake_ai.game import Direction, Point
from snake_ai.planning_10x10.types import PlanningState, SimulatedTransition


_DIRECTIONS = (
    Direction.RIGHT,
    Direction.DOWN,
    Direction.LEFT,
    Direction.UP,
)


def turned_direction(direction: Direction, action: int) -> Direction:
    """把项目的相对动作编码转换成新的绝对方向。"""

    if action not in (0, 1, 2):
        raise ValueError("action must be 0 (straight), 1 (right), or 2 (left)")
    turn = 1 if action == 1 else -1 if action == 2 else 0
    index = _DIRECTIONS.index(direction)
    return _DIRECTIONS[(index + turn) % len(_DIRECTIONS)]


def advance_point(point: Point, direction: Direction) -> Point:
    if direction == Direction.RIGHT:
        return Point(point.x + 1, point.y)
    if direction == Direction.LEFT:
        return Point(point.x - 1, point.y)
    if direction == Direction.DOWN:
        return Point(point.x, point.y + 1)
    return Point(point.x, point.y - 1)


def in_bounds(state: PlanningState, point: Point) -> bool:
    return 0 <= point.x < state.width and 0 <= point.y < state.height


def simulate_action(state: PlanningState, action: int) -> SimulatedTransition:
    """无副作用地执行一步，并严格复刻 ``SnakeEnv.step`` 的身体语义。"""

    next_direction = turned_direction(state.direction, action)
    new_head = advance_point(state.head, next_direction)
    ate_food = state.food is not None and new_head == state.food

    # 未进食时当前尾格会释放；进食时尾巴不动，所以仍属于碰撞集合。
    body_to_check = state.snake[1:] if ate_food else state.snake[1:-1]
    if not in_bounds(state, new_head) or new_head in body_to_check:
        return SimulatedTransition(
            next_state=None,
            collision=True,
            ate_food=False,
            board_completed=False,
        )

    if ate_food:
        next_snake = (new_head, *state.snake)
        next_steps_since_food = 0
        # 规划器只规划当前食物；真实环境随后生成的新食物由下一帧重新读取。
        next_food = None
    else:
        next_snake = (new_head, *state.snake[:-1])
        next_steps_since_food = state.steps_since_food + 1
        next_food = state.food

    board_completed = ate_food and len(next_snake) == state.width * state.height
    next_state = PlanningState(
        width=state.width,
        height=state.height,
        snake=next_snake,
        direction=next_direction,
        food=next_food,
        steps_since_food=next_steps_since_food,
    )
    return SimulatedTransition(
        next_state=next_state,
        collision=False,
        ate_food=ate_food,
        board_completed=board_completed,
    )


def simulate_path(
    state: PlanningState,
    actions: Iterable[int],
) -> tuple[SimulatedTransition, ...]:
    transitions: list[SimulatedTransition] = []
    current = state
    for action in actions:
        transition = simulate_action(current, action)
        transitions.append(transition)
        if transition.collision or transition.next_state is None:
            break
        current = transition.next_state
    return tuple(transitions)


def action_to_adjacent_point(state: PlanningState, target: Point) -> int | None:
    """返回通向相邻格的相对动作；目标不是可达相邻格时返回 None。"""

    for action in (0, 1, 2):
        direction = turned_direction(state.direction, action)
        if advance_point(state.head, direction) == target:
            return action
    return None
