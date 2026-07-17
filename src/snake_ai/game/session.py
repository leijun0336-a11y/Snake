from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from snake_ai.game.controllers import ControlContext, Controller
from snake_ai.game.snake_env import Direction, Observation, Point, SnakeEnv


@dataclass(frozen=True)
class GameSnapshot:
    snake: tuple[Point, ...]
    food: Point
    direction: Direction
    score: int
    steps: int
    hunger_ratio: float
    elapsed_seconds: float
    done: bool
    termination_reason: str
    last_action: int | None


@dataclass(frozen=True)
class SessionStep:
    previous: GameSnapshot
    current: GameSnapshot
    ate_food: bool


class GameSession:
    def __init__(
        self,
        env: SnakeEnv,
        controller: Controller,
        *,
        tick_rate: int,
        seed: int,
        max_steps: int | None = None,
    ) -> None:
        if env.renderer is not None:
            raise ValueError("GameSession requires SnakeEnv(render_mode=False)")
        if tick_rate < 1:
            raise ValueError("tick_rate must be positive")
        if max_steps is not None and max_steps < 1:
            raise ValueError("max_steps must be positive when provided")
        self.env = env
        self.controller = controller
        self.tick_rate = tick_rate
        self.seed = seed
        self.max_steps = max_steps
        self.observation: Observation
        self.done = False
        self.termination_reason = "none"
        self.last_action: int | None = None
        self.previous_snapshot: GameSnapshot
        self.snapshot: GameSnapshot
        self.reset(seed)

    def reset(self, seed: int | None = None) -> GameSnapshot:
        if seed is not None:
            self.seed = seed
        self.controller.reset()
        self.observation = self.env.reset(seed=self.seed)
        self.done = False
        self.termination_reason = "none"
        self.last_action = None
        self.snapshot = self._make_snapshot()
        self.previous_snapshot = self.snapshot
        return self.snapshot

    def handle_event(self, event: Any) -> None:
        self.controller.handle_event(event)

    def choose_action(self) -> int:
        if self.done:
            raise RuntimeError("Cannot choose an action for a finished session")
        return self.controller.choose_action(ControlContext(self.observation, self.env.direction))

    def advance(self, action: int) -> SessionStep:
        if self.done:
            raise RuntimeError("Cannot advance a finished session")
        previous = self.snapshot
        previous_score = self.env.score
        self.observation, _, env_done, _ = self.env.step(action)
        self.done = env_done
        self.termination_reason = self.env.termination_reason
        if (
            not self.done
            and self.max_steps is not None
            and self.env.frame_iteration >= self.max_steps
        ):
            self.done = True
            self.termination_reason = "max_steps"
        self.last_action = action
        self.previous_snapshot = previous
        self.snapshot = self._make_snapshot()
        return SessionStep(previous, self.snapshot, self.env.score > previous_score)

    def tick(self) -> SessionStep:
        return self.advance(self.choose_action())

    def _make_snapshot(self) -> GameSnapshot:
        return GameSnapshot(
            snake=tuple(self.env.snake),
            food=self.env.food,
            direction=self.env.direction,
            score=self.env.score,
            steps=self.env.frame_iteration,
            hunger_ratio=self.env.hunger_ratio,
            elapsed_seconds=self.env.frame_iteration / self.tick_rate,
            done=self.done,
            termination_reason=self.termination_reason,
            last_action=self.last_action,
        )
