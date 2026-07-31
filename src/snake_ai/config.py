from dataclasses import dataclass
from pathlib import Path
from typing import Literal

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RUNS_DIR = PROJECT_ROOT / "runs"


@dataclass(frozen=True)
class RewardConfig:
    """一组奖励尺度和边界配置；势函数奖励统一采用当前严格 PBRS 语义。"""

    # 奖励配置名称，也是命令行选择 reward profile 时使用的标识。
    name: str
    # 是否启用势函数进度奖励：beta * (gamma * new_phi - old_phi)。
    # old_phi 和 new_phi 分别为移动前后棋盘状态的势函数值，gamma 为折扣因子。
    # phi(head, food) = 1.0 - (|food.x - head.x| + |food.y - head.y|) / ((width - 1) + (height - 1))
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
    # 超过无进食步数上限时的终止惩罚；仅在 cost_rewards 开启时使用。通常与网格大小和蛇长相关。
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
    # 进度奖励语义版本；下一状态使用其实际食物，终止状态的势函数固定为零。
    progress_mode: str = "pbrs_food_distance"
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


# 第八次实验 dqn_20260712_130642 的奖励尺度和边界配置（目前最优）。
# 源码证据对应后来提交为 62ff05d 的工作树；PBRS 计算统一采用当前实现。
EXPERIMENT8_REWARD_CONFIG = RewardConfig(
    name="experiment8",
    potential_reward=True,
    cost_rewards=True,
    progress_beta=1.0,
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
    # 该历史奖励配置所依据的 Git revision。
    historical_source_revision="62ff05d5a8d7b65472a984e56647f2c20bceb915",
)

# PPO 专用奖励配置 —— 大部分奖励项沿用 experiment8 / 25 的尺度，
# 但经 6x6/10x10 Hybrid 对照实验后，将吃食奖励提高到 4.0，增强成功信号。
# 碰撞惩罚保持 -4.0，使成功进食与失败碰撞的绝对量级对称。
# 其余缩放继续使 value function 的 MSE loss 保持在合理范围（≈1）。
# 搭配 --ppo-normalize-returns（默认开启）时，归一化 returns 可进一步稳定训练。
EXPERIMENT_PPO_REWARD_CONFIG = RewardConfig(
    name="experiment_ppo",
    potential_reward=True,
    cost_rewards=True,
    progress_beta=2.0 / 25,
    food_reward=4.0,
    collision_penalty=-100.0 / 25,
    starvation_penalty=-12.0 / 25,
    win_reward=20.0 / 25,
    step_penalty=-0.005 / 25,
    hunger_penalty_scale=0.02 / 25,
    step_cost_scope="all_legal_moves",
    terminal_cost_mode="accumulate",
    starvation_limit_mode="2x_board_area_plus_snake_length",
    starvation_comparison="gte",
    progress_mode="pbrs_food_distance",
)

# 把所有奖励配置整理成一个"按名称查找配置"的字典，并生成所有可用配置名称。
REWARD_CONFIGS = {
    config.name: config
    for config in (REFERENCE_REWARD_CONFIG, EXPERIMENT8_REWARD_CONFIG, EXPERIMENT_PPO_REWARD_CONFIG)
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
    episodes: int = 50000
    # reference profile 未显式覆盖时使用该单局步数上限；experiment8/experiment_ppo 默认不设上限。
    max_steps_per_episode: int | None = 500
    batch_size: int = 128
    gamma: float = 0.995
    # TD target 聚合的连续真实奖励步数；1 为传统 one-step DQN。
    n_step: int = 1
    learning_rate: float = 1e-4
    replay_buffer_size: int = 100_000
    learning_starts: int = 2_000
    # 默认使用 proportional PER；可通过 --no-PER 切换为均匀经验回放。
    per: bool = True
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_anneal_steps: int = 100_000
    per_epsilon: float = 1e-6
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    # 是否使用指数衰减；False 表示使用线性衰减。
    epsilon_exp_decay: bool = False
    # 指数衰减模式下，每局结束后 epsilon 乘以的系数。
    epsilon_exp_factor: float = 0.995
    # 线性衰减的计数单位；默认按环境 step 衰减。
    epsilon_decay_unit: Literal["step", "episode"] = "step"
    # 按环境 step 线性衰减到 epsilon_end 所用步数。
    epsilon_linear_steps: int = 300_000
    # 线性衰减到 epsilon_end 所用局数。
    epsilon_linear_episodes: int | None = 15_000
    target_update_interval: int = 1000
    hidden_size: int = 256
    cnn_channels: int = 32
    cnn_output_channels: int = 8
    # 三个卷积块中对应的膨胀率，分别为 1、1、2。
    cnn_dilations: tuple[int, ...] = (1, 1, 2)
    # 以蛇头为中心的局部特征窗口边长，必须为正奇数。
    local_crop_size: int = 3
    use_local_crop: bool = True
    seed: int = 42


@dataclass(frozen=True)
class PPOConfig:
    rollout_steps: int = 2048
    update_epochs: int = 4
    gae_lambda: float = 0.95
    # PPO 对重要性采样比值的 clip range，通常在 0.1~0.3 之间。
    clip_coefficient: float = 0.2
    # PPO 对价值函数的 clip range，通常在 0.1~0.3 之间。价值也会间接影响策略。
    value_clip_coefficient: float = 0.2
    # PPO 策略熵系数的初始值。
    entropy_coefficient: float = 0.05
    # PPO 策略熵系数衰减后的最终值。
    entropy_coefficient_end: float = 0.001
    # PPO 策略熵系数完成线性衰减所用的 episode 数；None 表示由训练入口推导。
    entropy_anneal_episodes: int | None = None
    value_loss_coefficient: float = 0.5
    max_grad_norm: float = 0.5
    # 每个 epoch 末尾检查，若 approx_kl > target_kl 则跳过剩余 epoch（即使还没跑满 update_epochs 轮）
    target_kl: float | None = 0.02
    normalize_advantage: bool = True
    normalize_returns: bool = True
