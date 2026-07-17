from __future__ import annotations

from typing import Any

from snake_ai.game.controllers import HumanController
from snake_ai.game.session import GameSession
from snake_ai.game.snake_env import SnakeEnv


class SoloMode:
    def __init__(self, session: GameSession) -> None:
        self.session = session

    @classmethod
    def create(cls, *, seed: int, tick_rate: int) -> SoloMode:
        env = SnakeEnv(
            width=6,
            height=6,
            render_mode=False,
            seed=seed,
            state_mode="vector",
            reward_profile="experiment8",
            starvation_enabled=False,
        )
        return cls(
            GameSession(
                env,
                HumanController(),
                tick_rate=tick_rate,
                seed=seed,
                max_steps=400,
            )
        )

    @property
    def finished(self) -> bool:
        return self.session.done

    @property
    def result_text(self) -> str:
        if self.session.snapshot.termination_reason == "board_completed":
            return "BOARD COMPLETE"
        if self.session.snapshot.termination_reason == "max_steps":
            return "RUN COMPLETE"
        return "GAME OVER"

    @property
    def result_reason_text(self) -> str:
        reasons = {
            "board_completed": "YOU FILLED THE BOARD",
            "max_steps": "THE 400 STEP LIMIT WAS REACHED",
            "collision_wall": "YOU HIT THE WALL",
            "collision_body": "YOU HIT YOUR OWN BODY",
        }
        return reasons.get(self.session.snapshot.termination_reason, "THE RUN ENDED")

    def handle_event(self, event: Any) -> None:
        self.session.handle_event(event)

    def tick(self) -> None:
        if not self.finished:
            self.session.tick()
