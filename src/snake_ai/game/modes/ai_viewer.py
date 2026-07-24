from __future__ import annotations

from typing import Any

from snake_ai.game.ai_profiles import AIProfile
from snake_ai.game.controllers import DQNController
from snake_ai.game.session import GameSession
from snake_ai.game.snake_env import SnakeEnv


class AIViewerMode:
    def __init__(self, session: GameSession, ai_id: str) -> None:
        self.session = session
        self.ai_id = ai_id

    @classmethod
    def create(
        cls,
        profile: AIProfile,
        controller: DQNController,
        *,
        seed: int,
        tick_rate: int,
    ) -> AIViewerMode:
        env = SnakeEnv(
            width=profile.width,
            height=profile.height,
            render_mode=False,
            seed=seed,
            state_mode=profile.state_mode,
            reward_profile=profile.reward_profile,
            starvation_enabled=False,
        )
        return cls(
            GameSession(
                env,
                controller,
                tick_rate=tick_rate,
                seed=seed,
                max_steps=400,
            ),
            profile.id,
        )

    @property
    def finished(self) -> bool:
        return self.session.done

    @property
    def result_text(self) -> str:
        if self.session.snapshot.termination_reason == "board_completed":
            return "AI COMPLETED THE BOARD"
        return "AI RUN COMPLETE"

    @property
    def result_reason_text(self) -> str:
        reasons = {
            "board_completed": "AI FILLED THE BOARD",
            "max_steps": "THE 400 STEP LIMIT WAS REACHED",
            "collision_wall": "AI HIT THE WALL",
            "collision_body": "AI HIT ITS OWN BODY",
        }
        return reasons.get(self.session.snapshot.termination_reason, "THE AI RUN ENDED")

    def handle_event(self, event: Any) -> None:
        self.session.handle_event(event)

    def tick(self) -> None:
        if not self.finished:
            self.session.tick()
