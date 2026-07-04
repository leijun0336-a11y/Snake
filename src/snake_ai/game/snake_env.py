from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class Point:
    x: int
    y: int


class Direction(Enum):
    RIGHT = 0
    DOWN = 1
    LEFT = 2
    UP = 3


class SnakeEnv:
    """Small Gym-like Snake environment for low-dimensional DQN training."""

    action_size = 3
    state_size = 11

    def __init__(
        self,
        width: int = 20,
        height: int = 20,
        render_mode: bool = False,
        cell_size: int = 24,
        fps: int = 30,
        seed: int | None = None,
    ) -> None:
        if width < 5 or height < 5:
            raise ValueError("width and height must both be at least 5")

        self.width = width
        self.height = height
        self.render_mode = render_mode
        self.cell_size = cell_size
        self.fps = fps
        self.random = random.Random(seed)
        self.renderer = None

        if render_mode:
            from snake_ai.game.renderer import SnakeRenderer

            self.renderer = SnakeRenderer(width, height, cell_size, fps)

        self.direction = Direction.RIGHT
        self.snake: list[Point] = []
        self.food = Point(0, 0)
        self.score = 0
        self.steps_since_food = 0
        self.frame_iteration = 0
        self.reset()

    def reset(self) -> list[int]:
        center = Point(self.width // 2, self.height // 2)
        self.direction = Direction.RIGHT
        self.snake = [
            center,
            Point(center.x - 1, center.y),
            Point(center.x - 2, center.y),
        ]
        self.score = 0
        self.steps_since_food = 0
        self.frame_iteration = 0
        self._place_food()
        return self.get_state()

    def step(self, action: int) -> tuple[list[int], float, bool, dict[str, int]]:
        if action not in (0, 1, 2):
            raise ValueError("action must be 0 (straight), 1 (right), or 2 (left)")

        self.frame_iteration += 1
        self.steps_since_food += 1
        new_head = self._move(action)

        reward = 0.0
        done = False

        if self.is_collision(new_head) or self._is_too_long_without_food():
            done = True
            reward = -10.0
            return self.get_state(), reward, done, {"score": self.score}

        self.snake.insert(0, new_head)
        if new_head == self.food:
            self.score += 1
            self.steps_since_food = 0
            reward = 10.0
            self._place_food()
        else:
            self.snake.pop()

        if self.renderer is not None:
            self.renderer.render(self.snake, self.food, self.score)

        return self.get_state(), reward, done, {"score": self.score}

    def get_state(self) -> list[int]:
        head = self.snake[0]
        direction = self.direction

        straight = self._next_point(direction)
        right = self._next_point(self._turn(direction, 1))
        left = self._next_point(self._turn(direction, -1))

        return [
            int(self.is_collision(straight)),
            int(self.is_collision(right)),
            int(self.is_collision(left)),
            int(direction == Direction.LEFT),
            int(direction == Direction.RIGHT),
            int(direction == Direction.UP),
            int(direction == Direction.DOWN),
            int(self.food.x < head.x),
            int(self.food.x > head.x),
            int(self.food.y < head.y),
            int(self.food.y > head.y),
        ]

    def is_collision(self, point: Point) -> bool:
        hits_wall = point.x < 0 or point.x >= self.width or point.y < 0 or point.y >= self.height
        hits_body = point in self.snake[1:]
        return hits_wall or hits_body

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()

    def _place_food(self) -> None:
        available = [
            Point(x, y)
            for x in range(self.width)
            for y in range(self.height)
            if Point(x, y) not in self.snake
        ]
        if not available:
            self.food = self.snake[0]
            return
        self.food = self.random.choice(available)

    def _move(self, action: int) -> Point:
        if action == 1:
            self.direction = self._turn(self.direction, 1)
        elif action == 2:
            self.direction = self._turn(self.direction, -1)

        return self._next_point(self.direction)

    def _next_point(self, direction: Direction) -> Point:
        head = self.snake[0]
        if direction == Direction.RIGHT:
            return Point(head.x + 1, head.y)
        if direction == Direction.LEFT:
            return Point(head.x - 1, head.y)
        if direction == Direction.DOWN:
            return Point(head.x, head.y + 1)
        return Point(head.x, head.y - 1)

    @staticmethod
    def _turn(direction: Direction, turn: int) -> Direction:
        directions = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        index = directions.index(direction)
        return directions[(index + turn) % len(directions)]

    def _is_too_long_without_food(self) -> bool:
        return self.steps_since_food > self.width * self.height
