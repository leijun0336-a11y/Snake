from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RUNS_DIR = PROJECT_ROOT / "runs"


@dataclass(frozen=True)
class RewardConfig:
    """A named, immutable reward/environment-semantics profile."""

    name: str
    potential_reward: bool
    cost_rewards: bool
    progress_beta: float
    food_reward: float
    collision_penalty: float
    starvation_penalty: float
    win_reward: float
    step_penalty: float
    hunger_penalty_scale: float
    step_cost_scope: str
    terminal_cost_mode: str
    starvation_limit_mode: str
    starvation_comparison: str
    progress_mode: str = "legacy_food_target"
    historical_source_revision: str | None = None


# 与 chynl/snake 对齐后的当前基线。保持为默认 profile，避免旧调用静默改变行为。
REFERENCE_REWARD_CONFIG = RewardConfig(
    name="reference",
    potential_reward=False,
    cost_rewards=True,
    progress_beta=2.0,
    food_reward=10.0,
    collision_penalty=-100.0,
    starvation_penalty=-100.0,
    win_reward=90.0,
    step_penalty=-0.01,
    hunger_penalty_scale=0.0,
    step_cost_scope="ordinary_move",
    terminal_cost_mode="replace",
    starvation_limit_mode="board_area_plus_snake_length",
    starvation_comparison="gte",
)


# 第八次实验 dqn_20260712_130642 的原始语义。
# 源码证据对应后来提交为 62ff05d 的工作树；不能把这些历史边界“修正”为当前语义。
EXPERIMENT8_REWARD_CONFIG = RewardConfig(
    name="experiment8",
    potential_reward=True,
    cost_rewards=True,
    progress_beta=2.0,
    food_reward=10.0,
    collision_penalty=-100.0,
    starvation_penalty=-12.0,
    win_reward=20.0,
    step_penalty=-0.005,
    hunger_penalty_scale=0.02,
    step_cost_scope="all_legal_moves",
    terminal_cost_mode="accumulate",
    starvation_limit_mode="board_area",
    starvation_comparison="gt",
    historical_source_revision="62ff05d5a8d7b65472a984e56647f2c20bceb915",
)


REWARD_CONFIGS = {
    config.name: config
    for config in (REFERENCE_REWARD_CONFIG, EXPERIMENT8_REWARD_CONFIG)
}
REWARD_PROFILE_NAMES = tuple(REWARD_CONFIGS)


def get_reward_config(name: str) -> RewardConfig:
    """Return a known profile and fail explicitly for unknown names."""

    try:
        return REWARD_CONFIGS[name]
    except KeyError as exc:
        choices = ", ".join(REWARD_PROFILE_NAMES)
        raise ValueError(f"unknown reward profile {name!r}; expected one of: {choices}") from exc


@dataclass(frozen=True)
class EnvConfig:
    width: int = 20
    height: int = 20
    cell_size: int = 24
    fps: int = 30


@dataclass(frozen=True)
class TrainConfig:
    episodes: int = 15000
    max_steps_per_episode: int | None = 500
    batch_size: int = 128
    gamma: float = 0.99
    learning_rate: float = 1e-3
    replay_buffer_size: int = 100_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    epsilon_decay_episodes: int | None = None
    target_update_interval: int = 1000
    hidden_size: int = 256
    cnn_channels: int = 32
    cnn_output_channels: int = 8
    cnn_dilations: tuple[int, ...] = (1, 1, 2)
    cnn_pool_size: tuple[int, int] = (10, 10)
    seed: int = 42
