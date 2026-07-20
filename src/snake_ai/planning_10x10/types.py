from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from snake_ai.game import Direction, Point

if TYPE_CHECKING:
    from snake_ai.game import SnakeEnv


class DecisionTier(StrEnum):
    """规划器最终用于约束 DQN 的安全认证层级。"""

    SAFE_FOOD = "safe_food"
    HAMILTONIAN_CYCLE = "hamiltonian_cycle"


class PlanningError(RuntimeError):
    """10×10 严格规划器的基础异常。"""


class HamiltonianInvariantError(PlanningError):
    """输入局面已经不满足严格 Hamiltonian 顺序。"""


class NoSafeActionError(PlanningError):
    """规划器无法认证任何动作时抛出；不会退回原始 DQN。"""


class PlanCommitmentError(PlanningError):
    """已认证路径的后续状态或动作与真实环境不一致。"""


@dataclass(frozen=True, slots=True)
class PlanningState:
    width: int
    height: int
    snake: tuple[Point, ...]
    direction: Direction
    food: Point | None
    steps_since_food: int = 0

    def __post_init__(self) -> None:
        if self.width < 1 or self.height < 1:
            raise ValueError("planning board dimensions must be positive")
        if not self.snake:
            raise ValueError("planning state requires a non-empty snake")
        if len(set(self.snake)) != len(self.snake):
            raise ValueError("planning snake contains duplicate cells")
        if self.steps_since_food < 0:
            raise ValueError("steps_since_food must be non-negative")
        if not isinstance(self.direction, Direction):
            raise ValueError("planning direction must be a Direction value")
        for point in self.snake:
            if not (0 <= point.x < self.width and 0 <= point.y < self.height):
                raise ValueError("planning snake contains an out-of-bounds cell")
        for first, second in zip(self.snake, self.snake[1:], strict=False):
            if abs(first.x - second.x) + abs(first.y - second.y) != 1:
                raise ValueError("planning snake body must use adjacent cells")
        if self.food is not None:
            if not (0 <= self.food.x < self.width and 0 <= self.food.y < self.height):
                raise ValueError("planning food is out of bounds")
            if self.food in self.snake:
                raise ValueError("planning food must not overlap the snake")
        if len(self.snake) > 1:
            head = self.snake[0]
            neck = self.snake[1]
            expected_direction = _direction_between(neck, head)
            if self.direction != expected_direction:
                raise ValueError("planning direction does not match neck-to-head geometry")

    @classmethod
    def from_env(cls, env: SnakeEnv) -> PlanningState:
        """从真实环境读取不可变快照，不反向解析神经网络 observation。"""

        return cls(
            width=env.width,
            height=env.height,
            snake=tuple(env.snake),
            direction=env.direction,
            food=env.food,
            steps_since_food=env.steps_since_food,
        )

    @property
    def head(self) -> Point:
        return self.snake[0]

    @property
    def tail(self) -> Point:
        return self.snake[-1]


@dataclass(frozen=True, slots=True)
class SimulatedTransition:
    next_state: PlanningState | None
    collision: bool
    ate_food: bool
    board_completed: bool


@dataclass(frozen=True, slots=True)
class AStarResult:
    actions: tuple[int, ...]
    expansions: int


@dataclass(frozen=True, slots=True)
class FoodPathAssessment:
    action: int
    safe: bool
    actions: tuple[int, ...] | None
    expansions: int
    tail_reachable: bool
    reachable_area: int


@dataclass(frozen=True, slots=True)
class PlannerDecision:
    admissible_actions: tuple[int, ...]
    # 与 admissible_actions 一一对应；每条路径都从当前状态出发并最终吃到当前食物。
    certified_paths: tuple[tuple[int, ...], ...]
    tier: DecisionTier
    food_assessments: tuple[FoodPathAssessment, ...]
    cycle_safe_actions: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.admissible_actions:
            raise ValueError("planner decision requires at least one admissible action")
        if tuple(sorted(set(self.admissible_actions))) != self.admissible_actions:
            raise ValueError("admissible actions must be unique and sorted")
        if len(self.certified_paths) != len(self.admissible_actions):
            raise ValueError("every admissible action requires exactly one certified path")
        if any(
            not path or path[0] != action
            for action, path in zip(
                self.admissible_actions,
                self.certified_paths,
                strict=True,
            )
        ):
            raise ValueError("each certified path must start with its admissible action")

    def path_for_action(self, action: int) -> tuple[int, ...]:
        try:
            index = self.admissible_actions.index(action)
        except ValueError as exc:
            raise ValueError(f"action {action} is not certified by this decision") from exc
        return self.certified_paths[index]


def _direction_between(start: Point, end: Point) -> Direction:
    delta = end.x - start.x, end.y - start.y
    directions = {
        (1, 0): Direction.RIGHT,
        (-1, 0): Direction.LEFT,
        (0, 1): Direction.DOWN,
        (0, -1): Direction.UP,
    }
    try:
        return directions[delta]
    except KeyError as exc:
        raise ValueError("points are not orthogonally adjacent") from exc
