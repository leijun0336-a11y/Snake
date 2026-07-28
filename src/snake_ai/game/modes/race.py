from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from snake_ai.game.ai_profiles import AIProfile
from snake_ai.game.controllers import DQNController, HumanController
from snake_ai.game.food_policy import SeededRaceFoodPolicy
from snake_ai.game.session import GameSession
from snake_ai.game.snake_env import SnakeEnv

RaceWinner = Literal["player", "ai", "draw"]


@dataclass(frozen=True)
class RaceResult:
    winner: RaceWinner
    reason: str

    @property
    def display_text(self) -> str:
        if self.winner == "player":
            return "YOU WIN"
        if self.winner == "ai":
            return "AI WINS"
        return "DRAW"

    @property
    def reason_text(self) -> str:
        if self.reason == "board_completed":
            winner = "YOU" if self.winner == "player" else "AI"
            return f"{winner} REACHED FULL SCORE FIRST"
        if self.reason == "target_same_tick":
            return "BOTH REACHED FULL SCORE TOGETHER"
        if self.reason in ("collision_wall", "collision_body"):
            loser = "AI" if self.winner == "player" else "YOU"
            if self.reason == "collision_wall":
                collision = "HIT THE WALL"
            else:
                collision = "HIT ITS BODY" if loser == "AI" else "HIT YOUR OWN BODY"
            return f"{loser} {collision}"
        if self.reason.startswith("both_collided"):
            detail = self._score_comparison_text()
            return f"BOTH COLLIDED: {detail}"
        if self.reason.startswith("max_steps"):
            detail = self._score_comparison_text()
            return f"400 STEPS: {detail}"
        return "THE RACE ENDED"

    def _score_comparison_text(self) -> str:
        if self.reason.endswith("earlier_score"):
            return "EARLIER FINAL SCORE"
        if self.winner != "draw":
            return "HIGHER SCORE"
        return "SAME SCORE AND TIMING"


class RaceMode:
    def __init__(
        self,
        player_session: GameSession,
        ai_session: GameSession,
        *,
        max_steps: int = 400,
        ai_id: str = "AI",
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        self.player_session = player_session
        self.ai_session = ai_session
        self.ai_id = ai_id
        player_full_score = self._full_score(player_session)
        ai_full_score = self._full_score(ai_session)
        if player_full_score != ai_full_score:
            raise ValueError("Race sessions must have the same full score")
        self.full_score = player_full_score
        self.max_steps = max_steps
        self.race_steps = 0
        self.player_last_score_step = 0
        self.ai_last_score_step = 0
        self.result: RaceResult | None = None

    @classmethod
    def create(
        cls,
        profile: AIProfile,
        ai_controller: DQNController,
        *,
        race_seed: int,
        tick_rate: int,
        max_steps: int = 400,
    ) -> RaceMode:
        player_env = SnakeEnv(
            width=profile.width,
            height=profile.height,
            render_mode=False,
            seed=race_seed,
            state_mode="vector",
            reward_profile=profile.reward_profile,
            starvation_enabled=False,
            food_policy=SeededRaceFoodPolicy(race_seed),
        )
        ai_env = SnakeEnv(
            width=profile.width,
            height=profile.height,
            render_mode=False,
            seed=race_seed,
            state_mode=profile.state_mode,
            reward_profile=profile.reward_profile,
            starvation_enabled=False,
            food_policy=SeededRaceFoodPolicy(race_seed),
        )
        return cls(
            GameSession(
                player_env,
                HumanController(),
                tick_rate=tick_rate,
                seed=race_seed,
            ),
            GameSession(
                ai_env,
                ai_controller,
                tick_rate=tick_rate,
                seed=race_seed,
            ),
            max_steps=max_steps,
            ai_id=profile.id,
        )

    @property
    def finished(self) -> bool:
        return self.result is not None

    @property
    def result_text(self) -> str:
        return self.result.display_text if self.result is not None else ""

    @property
    def result_reason_text(self) -> str:
        return self.result.reason_text if self.result is not None else ""

    def handle_event(self, event: Any) -> None:
        self.player_session.handle_event(event)

    def tick(self) -> None:
        if self.finished:
            return

        # Decide both actions from the same logical tick before mutating either board.
        player_action = self.player_session.choose_action()
        ai_action = self.ai_session.choose_action()
        player_step = self.player_session.advance(player_action)
        ai_step = self.ai_session.advance(ai_action)
        self.race_steps += 1

        if player_step.ate_food:
            self.player_last_score_step = self.race_steps
        if ai_step.ate_food:
            self.ai_last_score_step = self.race_steps
        self.result = self._judge()

    def _judge(self) -> RaceResult | None:
        player = self.player_session.snapshot
        ai = self.ai_session.snapshot
        player_target = player.score >= self.full_score
        ai_target = ai.score >= self.full_score

        if player_target and ai_target:
            return RaceResult("draw", "target_same_tick")
        if player_target:
            return RaceResult("player", "board_completed")
        if ai_target:
            return RaceResult("ai", "board_completed")

        if player.done and ai.done:
            return self._compare_scores("both_collided")
        if player.done:
            return RaceResult("ai", player.termination_reason)
        if ai.done:
            return RaceResult("player", ai.termination_reason)

        if self.race_steps >= self.max_steps:
            return self._compare_scores("max_steps")
        return None

    def _compare_scores(self, reason: str) -> RaceResult:
        player_score = self.player_session.snapshot.score
        ai_score = self.ai_session.snapshot.score
        if player_score > ai_score:
            return RaceResult("player", reason)
        if ai_score > player_score:
            return RaceResult("ai", reason)
        if self.player_last_score_step < self.ai_last_score_step:
            return RaceResult("player", f"{reason}_earlier_score")
        if self.ai_last_score_step < self.player_last_score_step:
            return RaceResult("ai", f"{reason}_earlier_score")
        return RaceResult("draw", reason)

    @staticmethod
    def _full_score(session: GameSession) -> int:
        snapshot = session.snapshot
        board_area = session.env.width * session.env.height
        return snapshot.score + board_area - len(snapshot.snake)
