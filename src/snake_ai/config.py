from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RUNS_DIR = PROJECT_ROOT / "runs"


@dataclass(frozen=True)
class RewardConfig:
    """一组完整的奖励语义配置，用于复现不同训练实验的奖励行为。"""

    # 奖励配置名称，也是命令行选择 reward profile 时使用的标识。
    name: str
    # 是否启用势函数进度奖励：beta * (gamma * new_phi - old_phi)。
    # old_phi 和 new_phi 分别为移动前后棋盘状态的势函数值，gamma 为折扣因子。
    potential_reward: bool
    # 是否启用步成本和饥饿成本；开启时饿死使用 starvation_penalty，不同于撞死的数值。
    # 关闭时饿死和撞死的惩罚相同。
    cost_rewards: bool
    # 势函数进度奖励的缩放系数 beta，必须为非负数。
    progress_beta: float
    # 吃到食物时获得的奖励。
    food_reward: float
    # 蛇头撞墙或撞到身体时的终止惩罚。
    collision_penalty: float
    # 超过无进食步数上限时的终止惩罚；仅在 cost_rewards 开启时使用。
    starvation_penalty: float
    # 蛇占满棋盘时获得的终止奖励。
    win_reward: float
    # 每个适用移动产生的固定时间成本，应为非正数。
    step_penalty: float
    # 饥饿成本系数，实际成本为 -scale * hunger_ratio**2。
    # hunger_ratio= min(steps_since_food / starvation_limit, 1.0)
    hunger_penalty_scale: float
    # 步成本的作用范围：ordinary_move 不含吃食物，all_legal_moves 含所有合法移动。
    step_cost_scope: str
    # 饿死时的成本处理：replace 清除本步成本，accumulate 保留并叠加终止惩罚。
    terminal_cost_mode: str
    # 饿死的标准：board_area 使用棋盘面积，board_area_plus_snake_length 再加蛇长。
    starvation_limit_mode: str
    # 触发饿死的边界比较：gt 表示大于上限，gte 表示大于或等于上限。
    starvation_comparison: str
    # 进度奖励语义版本；legacy_food_target 表示本步始终以移动前的食物为目标计算。
    progress_mode: str = "legacy_food_target"
    # 历史配置所依据的源码git提交；当前配置或无特定来源时为 None。
    historical_source_revision: str | None = None


# 基线奖励配置
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


# 第八次实验 dqn_20260712_130642 的奖励配置(目前最优)。
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
    # 步成本作用范围为所有合法移动，包含吃食物的移动。
    step_cost_scope="all_legal_moves",
    # 蛇饿死时，保留这一普通移动产生的步成本和饥饿成本，再叠加饿死惩罚。
    terminal_cost_mode="accumulate",
    # 蛇连续未吃到食物的步数上限，等于棋盘的总格子数。
    starvation_limit_mode="board_area",
    # 饿死的边界比较为大于上限，等于上限时不算饿死。
    starvation_comparison="gt",
    # 用来记录这套历史奖励配置所依据的 Git 源码版本的commit id.
    historical_source_revision="62ff05d5a8d7b65472a984e56647f2c20bceb915",
)

# 把所有奖励配置整理成一个“按名称查找配置”的字典，并生成所有可用配置名称。
REWARD_CONFIGS = {
    config.name: config for config in (REFERENCE_REWARD_CONFIG, EXPERIMENT8_REWARD_CONFIG)
}
REWARD_PROFILE_NAMES = tuple(REWARD_CONFIGS)


# 根据奖励配置名称，返回对应的 RewardConfig 对象
def get_reward_config(name: str) -> RewardConfig:

    try:
        return REWARD_CONFIGS[name]
    except KeyError as exc:
        choices = ", ".join(REWARD_PROFILE_NAMES)
        raise ValueError(f"unknown reward profile {name!r}; expected one of: {choices}") from exc


@dataclass(frozen=True)
class EnvConfig:
    width: int = 6
    height: int = 6
    cell_size: int = 48
    fps: int = 20


@dataclass(frozen=True)
class TrainConfig:
    episodes: int = 15000
    max_steps_per_episode: int | None = 500
    batch_size: int = 128
    gamma: float = 0.99
    # TD target 聚合的连续真实奖励步数；1 为传统 one-step DQN。
    n_step: int = 1
    learning_rate: float = 1e-4
    replay_buffer_size: int = 100_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    # 是否使用指数衰减；False 表示使用线性衰减。
    epsilon_exp_decay: bool = False
    # 指数衰减模式下，每局结束后 epsilon 乘以的系数。
    epsilon_exp_factor: float = 0.995
    # 线性衰减到 epsilon_end 所用局数；None 表示训练时取最大局数的一半。
    epsilon_linear_episodes: int | None = None
    target_update_interval: int = 1000
    hidden_size: int = 256
    cnn_channels: int = 32
    cnn_output_channels: int = 8
    # 三个卷积块中对应的膨胀率，分别为 1、1、2。
    cnn_dilations: tuple[int, ...] = (1, 1, 2)
    seed: int = 42


@dataclass(frozen=True)
class PPOConfig:
    rollout_steps: int = 2048
    update_epochs: int = 4
    gae_lambda: float = 0.95
    clip_coefficient: float = 0.2
    value_clip_coefficient: float = 0.2
    entropy_coefficient: float = 0.01
    value_loss_coefficient: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = 0.02
    normalize_advantage: bool = True
