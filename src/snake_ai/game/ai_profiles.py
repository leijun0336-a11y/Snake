from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from snake_ai.config import CHECKPOINT_DIR


@dataclass(frozen=True)
class AIProfile:
    id: str
    display_name: str
    checkpoint_path: Path
    huggingface_repo_id: str
    huggingface_filename: str
    huggingface_revision: str
    state_mode: str
    width: int
    height: int
    reward_profile: str


DEFAULT_AI_ID = "dqn_20260728_140741"

AI_PROFILES: dict[str, AIProfile] = {
    DEFAULT_AI_ID: AIProfile(
        id=DEFAULT_AI_ID,
        display_name="Snake AI",
        checkpoint_path=CHECKPOINT_DIR / DEFAULT_AI_ID / "best.pt",
        huggingface_repo_id="leijun0336-a11y/Snake",
        huggingface_filename="best.pt",
        huggingface_revision="main",
        state_mode="hybrid",
        width=6,
        height=6,
        reward_profile="experiment8",
    )
}


def ensure_checkpoint(profile: AIProfile) -> Path:
    """Return the local checkpoint, downloading it once from the public Hub repo if needed."""

    if profile.checkpoint_path.is_file():
        return profile.checkpoint_path

    profile.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface-hub is required to download the AI checkpoint"
        ) from exc

    try:
        downloaded_path = Path(
            hf_hub_download(
                repo_id=profile.huggingface_repo_id,
                filename=profile.huggingface_filename,
                revision=profile.huggingface_revision,
                local_dir=profile.checkpoint_path.parent,
                token=False,
            )
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not download AI checkpoint from "
            f"https://huggingface.co/{profile.huggingface_repo_id}: {exc}"
        ) from exc

    if not downloaded_path.is_file():
        raise FileNotFoundError(f"Downloaded AI checkpoint does not exist: {downloaded_path}")
    return downloaded_path


def get_ai_profile(profile_id: str = DEFAULT_AI_ID) -> AIProfile:
    try:
        return AI_PROFILES[profile_id]
    except KeyError as exc:
        raise KeyError(f"Unknown AI profile: {profile_id}") from exc
