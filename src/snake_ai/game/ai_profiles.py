from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from snake_ai.config import CHECKPOINT_DIR


@dataclass(frozen=True)
class AIProfile:
    id: str
    display_name: str
    checkpoint_path: Path
    state_mode: str
    width: int
    height: int
    reward_profile: str


DEFAULT_AI_ID = "dqn_20260722_201922"

AI_PROFILES: dict[str, AIProfile] = {
    DEFAULT_AI_ID: AIProfile(
        id=DEFAULT_AI_ID,
        display_name="Snake AI",
        checkpoint_path=CHECKPOINT_DIR / "dqn_20260722_201922" / "best.pt",
        state_mode="hybrid",
        width=6,
        height=6,
        reward_profile="experiment8",
    )
}


def get_ai_profile(profile_id: str = DEFAULT_AI_ID) -> AIProfile:
    try:
        return AI_PROFILES[profile_id]
    except KeyError as exc:
        raise KeyError(f"Unknown AI profile: {profile_id}") from exc
