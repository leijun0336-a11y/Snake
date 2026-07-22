# 处理自引用问题
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from snake_ai.agents import DQNAgent, PPOAgent
from snake_ai.agents.ppo_agent import PPOMetrics
from snake_ai.config import (
    CHECKPOINT_DIR,
    REWARD_PROFILE_NAMES,
    RUNS_DIR,
    EnvConfig,
    PPOConfig,
    TrainConfig,
)
from snake_ai.game import SnakeEnv
from snake_ai.utils import set_seed, summarize_values
from snake_ai.validation import (
    DEFAULT_SELECTION_THRESHOLDS,
    SeedSetName,
    StagedValidationState,
    ValidationResult,
    evaluate_policy,
    make_episode_seeds,
    run_staged_validation,
)
from snake_ai.wandb_logging import WandbRun, start_wandb, training_metrics


TRAINING_START_DELAY_SECONDS = 5

try:
    from torch.utils.tensorboard import SummaryWriter
# 如果用户没装 TensorBoard，就让它等于 None
except ImportError:
    SummaryWriter = None


# 解析启动脚本时的命令行参数
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a DQN or PPO agent for Snake.")
    parser.add_argument("--algorithm", choices=("dqn", "ppo"), default="dqn")
    # 最大训练局数；早停没有触发时，训练最多跑到这个 episode。
    parser.add_argument("--max-episodes", type=int, default=TrainConfig.episodes)
    # 限制蛇的最大步数，防止无限绕圈。
    parser.add_argument("--max-steps-per-episode", type=int, default=None)
    # 是否带渲染训练, action="store_true"表示：启动脚本时写上--render则为True，不写默认为False
    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="log the reference training curves live to the Snake W&B project",
    )
    parser.add_argument("--width", type=int, default=EnvConfig.width)
    parser.add_argument("--height", type=int, default=EnvConfig.height)
    # 渲染游戏时每个格子的像素尺寸
    parser.add_argument("--cell-size", type=int, default=EnvConfig.cell_size)
    parser.add_argument("--fps", type=int, default=EnvConfig.fps)
    parser.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR)
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    # 选模型：全连接 Vector、卷积 Grid、混合 Hybrid
    parser.add_argument("--state-mode", choices=("vector", "grid", "hybrid"), default="hybrid")
    # 选奖励配置
    parser.add_argument("--reward-profile", choices=REWARD_PROFILE_NAMES, default="experiment8")
    # 是否使用势函数进度奖励，默认启用；可通过 --no-potential-reward 关闭(argparse内置方法)。
    parser.add_argument("--potential-reward", action=argparse.BooleanOptionalAction, default=True)
    # 是否关闭步成本和饥饿成本；关闭后饿死和碰撞的惩罚值统一。
    parser.add_argument("--no-cost-rewards", action="store_true")
    # 完全确定性会让部分 CUDA 卷积/池化算子明显变慢；默认优先训练速度。
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--gamma", type=float, default=TrainConfig.gamma)
    parser.add_argument(
        "--n-step",
        type=int,
        default=TrainConfig.n_step,
        help="number of real rewards in each TD target (default: 1)",
    )
    parser.add_argument("--learning-rate", type=float, default=TrainConfig.learning_rate)
    parser.add_argument("--replay-buffer-size", type=int, default=TrainConfig.replay_buffer_size)
    parser.add_argument("--epsilon-start", type=float, default=TrainConfig.epsilon_start)
    parser.add_argument("--epsilon-end", type=float, default=TrainConfig.epsilon_end)
    # 是否采用指数衰减；默认关闭，即采用线性衰减。
    parser.add_argument("--epsilon-exp-decay", action="store_true")
    # 指数衰减系数；仅在启用 --epsilon-exp-decay 时生效。
    parser.add_argument(
        "--epsilon-exp-factor", type=float, default=TrainConfig.epsilon_exp_factor, metavar="FACTOR"
    )
    # epsilon 线性降至下限所用局数；默认取最大训练局数的一半。
    parser.add_argument(
        "--epsilon-linear-episodes",
        type=int,
        default=TrainConfig.epsilon_linear_episodes,
        metavar="N",
    )
    parser.add_argument(
        "--target-update-interval", type=int, default=TrainConfig.target_update_interval
    )
    parser.add_argument("--hidden-size", type=int, default=TrainConfig.hidden_size)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    # 这些参数只影响 Grid/Hybrid 的卷积主干，Vector baseline 会忽略它们。
    parser.add_argument("--cnn-channels", type=int, default=TrainConfig.cnn_channels)
    parser.add_argument("--cnn-output-channels", type=int, default=TrainConfig.cnn_output_channels)
    parser.add_argument("--cnn-dilations", type=int, nargs="+", default=TrainConfig.cnn_dilations)
    parser.add_argument("--ppo-rollout-steps", type=int, default=PPOConfig.rollout_steps)
    parser.add_argument("--ppo-update-epochs", type=int, default=PPOConfig.update_epochs)
    parser.add_argument("--ppo-gae-lambda", type=float, default=PPOConfig.gae_lambda)
    parser.add_argument("--ppo-clip-coefficient", type=float, default=PPOConfig.clip_coefficient)
    parser.add_argument(
        "--ppo-value-clip-coefficient",
        type=float,
        default=PPOConfig.value_clip_coefficient,
    )
    parser.add_argument(
        "--ppo-entropy-coefficient", type=float, default=PPOConfig.entropy_coefficient
    )
    parser.add_argument(
        "--ppo-value-loss-coefficient",
        type=float,
        default=PPOConfig.value_loss_coefficient,
    )
    parser.add_argument("--ppo-max-grad-norm", type=float, default=PPOConfig.max_grad_norm)
    parser.add_argument("--ppo-target-kl", type=float, default=PPOConfig.target_kl)
    parser.add_argument(
        "--ppo-normalize-advantage",
        action=argparse.BooleanOptionalAction,
        default=PPOConfig.normalize_advantage,
    )
    parser.add_argument(
        "--ppo-normalize-returns",
        action=argparse.BooleanOptionalAction,
        default=PPOConfig.normalize_returns,
    )
    # 默认严格跑满最大训练局数；显式传入该参数后才启用早停。
    parser.add_argument("--early-stop", action="store_true")
    # 至少训练多少局后，才允许基于阶段验证的早停判断生效。
    parser.add_argument("--min-episodes", type=int, default=5000)
    parser.add_argument("--validation-interval", type=int, default=500)
    parser.add_argument("--validation-episodes", type=int, default=100)
    parser.add_argument("--confirmation-episodes", type=int, default=500)
    parser.add_argument("--validation-patience", type=int, default=8)
    parser.add_argument("--validation-max-steps", type=int, default=1000)
    # 如果设置了目标平均分，确认验证达到该值后停止训练。
    parser.add_argument("--target-mean-score", type=float, default=None)
    return parser.parse_args()


def resolve_max_steps_per_episode(args: argparse.Namespace) -> int | None:

    if args.max_steps_per_episode is not None and args.max_steps_per_episode < 1:
        raise ValueError("max_steps_per_episode must be at least 1")
    if args.max_steps_per_episode is not None:
        return args.max_steps_per_episode
    if args.reward_profile in ("experiment8", "experiment_ppo"):
        return None
    return TrainConfig.max_steps_per_episode


# 校验训练参数、解析派生默认值，并构造最终配置。
def build_configs(args: argparse.Namespace) -> tuple[TrainConfig, EnvConfig]:

    max_episodes = args.max_episodes
    if max_episodes < 1:
        raise ValueError("max episodes must be at least 1")
    max_steps_per_episode = resolve_max_steps_per_episode(args)
    if args.width < 5 or args.height < 5:
        raise ValueError("width and height must be at least 5")
    if args.cell_size < 1 or args.fps < 1:
        raise ValueError("cell_size and fps must be positive")
    if args.batch_size < 1 or args.replay_buffer_size < args.batch_size:
        raise ValueError("replay_buffer_size must be at least batch_size >= 1")
    if not 0.0 <= args.gamma <= 1.0:
        raise ValueError("gamma must be between 0 and 1")
    if args.n_step < 1:
        raise ValueError("n_step must be at least 1")
    if args.learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if not 0.0 <= args.epsilon_end <= args.epsilon_start <= 1.0:
        raise ValueError("epsilon values must satisfy 0 <= end <= start <= 1")
    if not 0.0 < args.epsilon_exp_factor <= 1.0:
        raise ValueError("epsilon_exp_factor must be in (0, 1]")
    if args.epsilon_linear_episodes is not None and args.epsilon_linear_episodes < 1:
        raise ValueError("epsilon_linear_episodes must be positive")
    epsilon_linear_episodes = (
        max(1, int(max_episodes * 0.5))
        if args.epsilon_linear_episodes is None
        else args.epsilon_linear_episodes
    )
    if args.target_update_interval < 1 or args.hidden_size < 1:
        raise ValueError("target_update_interval and hidden_size must be positive")
    if args.min_episodes < 1:
        raise ValueError("min_episodes must be at least 1")
    if args.min_episodes > max_episodes:
        raise ValueError("min_episodes must be less than or equal to max episodes")
    if args.validation_interval < 1:
        raise ValueError("validation_interval must be at least 1")
    if args.validation_episodes < 1 or args.confirmation_episodes < 1:
        raise ValueError("validation episode counts must be at least 1")
    if args.validation_patience < 1:
        raise ValueError("validation_patience must be at least 1")
    if args.validation_max_steps < 1:
        raise ValueError("validation_max_steps must be at least 1")
    if args.cnn_channels <= 0 or args.cnn_output_channels <= 0:
        raise ValueError("CNN channel sizes must be positive")
    if any(dilation <= 0 for dilation in args.cnn_dilations):
        raise ValueError("cnn_dilations must contain positive integers")
    if args.algorithm == "ppo":
        if args.ppo_rollout_steps < 1:
            raise ValueError("ppo_rollout_steps must be at least 1")
        if args.batch_size > args.ppo_rollout_steps:
            raise ValueError("batch_size must not exceed ppo_rollout_steps")
        if args.ppo_rollout_steps % args.batch_size != 0:
            raise ValueError("ppo_rollout_steps must be divisible by batch_size")
        if args.ppo_update_epochs < 1:
            raise ValueError("ppo_update_epochs must be at least 1")
        if not 0.0 <= args.ppo_gae_lambda <= 1.0:
            raise ValueError("ppo_gae_lambda must be between 0 and 1")
        if args.ppo_clip_coefficient <= 0.0 or args.ppo_value_clip_coefficient <= 0.0:
            raise ValueError("PPO clip coefficients must be positive")
        if args.ppo_entropy_coefficient < 0.0 or args.ppo_value_loss_coefficient < 0.0:
            raise ValueError("PPO loss coefficients must be non-negative")
        if args.ppo_max_grad_norm <= 0.0:
            raise ValueError("ppo_max_grad_norm must be positive")
        if args.ppo_target_kl is not None and args.ppo_target_kl <= 0.0:
            raise ValueError("ppo_target_kl must be positive")

    train_config = TrainConfig(
        episodes=max_episodes,
        max_steps_per_episode=max_steps_per_episode,
        batch_size=args.batch_size,
        gamma=args.gamma,
        n_step=args.n_step,
        learning_rate=args.learning_rate,
        replay_buffer_size=args.replay_buffer_size,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_exp_decay=args.epsilon_exp_decay,
        epsilon_exp_factor=args.epsilon_exp_factor,
        epsilon_linear_episodes=epsilon_linear_episodes,
        target_update_interval=args.target_update_interval,
        hidden_size=args.hidden_size,
        cnn_channels=args.cnn_channels,
        cnn_output_channels=args.cnn_output_channels,
        cnn_dilations=tuple(args.cnn_dilations),
        seed=args.seed,
    )
    env_config = EnvConfig(
        width=args.width,
        height=args.height,
        cell_size=args.cell_size,
        fps=args.fps,
    )
    return train_config, env_config


def build_ppo_config(args: argparse.Namespace) -> PPOConfig:
    return PPOConfig(
        rollout_steps=args.ppo_rollout_steps,
        update_epochs=args.ppo_update_epochs,
        gae_lambda=args.ppo_gae_lambda,
        clip_coefficient=args.ppo_clip_coefficient,
        value_clip_coefficient=args.ppo_value_clip_coefficient,
        entropy_coefficient=args.ppo_entropy_coefficient,
        value_loss_coefficient=args.ppo_value_loss_coefficient,
        max_grad_norm=args.ppo_max_grad_norm,
        target_kl=args.ppo_target_kl,
        normalize_advantage=args.ppo_normalize_advantage,
        normalize_returns=args.ppo_normalize_returns,
    )


# 在训练开始前，打印一份"训练会在什么条件下停止"的概览
def print_stop_overview(
    args: argparse.Namespace,
    train_config: TrainConfig,
) -> None:
    early_stop_enabled = args.early_stop
    if args.algorithm == "dqn":
        selection_start_text = (
            "epsilon floor (dynamic)"
            if train_config.epsilon_exp_decay
            else f"epsilon floor ({train_config.epsilon_linear_episodes})"
        )
    else:
        selection_start_text = f"episode {args.min_episodes}"

    print("")
    print("========== training stop overview ==========")
    print(f"max episodes without early stop : {train_config.episodes}")
    print(f"algorithm                       : {args.algorithm}")
    print(f"best selection starts at        : {selection_start_text}")
    print(
        "staged validation              : "
        f"{args.validation_episodes} quick / "
        f"{args.confirmation_episodes} confirmation episodes, "
        f"every {args.validation_interval} training episodes"
    )
    print(f"early stop enabled              : {early_stop_enabled}")
    if not early_stop_enabled:
        print("earliest early stop episode     : disabled")
        print("actual stop condition           : run until max episodes")
    else:
        print(f"early stop eligibility         : episode >= {args.min_episodes}")
        if args.target_mean_score is None:
            print("target mean score stop          : disabled")
        else:
            print(
                "target mean score stop          : "
                f"confirmation mean >= {args.target_mean_score:.2f}"
            )
        print(
            "patience stop                   : "
            f"no confirmed best update for "
            f"{args.validation_patience} eligible validation rounds"
        )
        print(
            "final patience check            : "
            f"{args.confirmation_episodes}-episode confirmation is required"
        )
    print("============================================")
    print("")


# 生成训练报告的文本内容，供 TensorBoard 显示
def format_train_report(
    run_name: str,
    episodes: int,
    total_time_sec: float,
    best_score: int,
    best_mean_score: float,
    scores: list[int],
    episode_steps: list[int],
    score_per_steps: list[float],
    mean_scores: list[float],
    episode_rewards: list[float],
    mean_rewards: list[float],
    losses: list[float],
    mean_loss_100s: list[float],
    algorithm_metrics: dict[str, list[float]],
) -> str:
    summaries = {
        "score": summarize_values(scores),
        "episode_steps": summarize_values(episode_steps),
        "score_per_step": summarize_values(score_per_steps),
        "mean_score_100": summarize_values(mean_scores),
        "episode_reward": summarize_values(episode_rewards),
        "mean_reward_100": summarize_values(mean_rewards),
        "loss": summarize_values(losses),
        "mean_loss_100": summarize_values(mean_loss_100s),
    }
    summaries.update(
        (name, summarize_values(values))
        for name, values in algorithm_metrics.items()
        if values
    )
    lines = [
        f"Run: `{run_name}`",
        f"Episodes: `{episodes}`",
        f"Total time: `{total_time_sec:.3f} sec`",
        f"Best score: `{best_score}`",
        f"Best mean_score_100: `{best_mean_score:.4f}`",
        "",
        "| Metric | Mean | Std | Min | Max | Last |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, summary in summaries.items():
        lines.append(
            f"| {name} | {summary['mean']:.4f} | {summary['std']:.4f} | "
            f"{summary['min']:.4f} | {summary['max']:.4f} | {summary['last']:.4f} |"
        )
    return "\n".join(lines)


def count_agent_parameters(agent: DQNAgent | PPOAgent) -> dict[str, int]:
    """Count policy capacity, optimized parameters, and all stored network parameters."""

    policy_parameters = sum(parameter.numel() for parameter in agent.policy_net.parameters())
    optimized_parameters = {
        id(parameter): parameter
        for group in agent.optimizer.param_groups
        for parameter in group["params"]
    }
    network_parameters = {
        id(parameter): parameter
        for network_name in ("policy_net", "target_net")
        if (network := getattr(agent, network_name, None)) is not None
        for parameter in network.parameters()
    }
    return {
        "policy": policy_parameters,
        "optimized": sum(parameter.numel() for parameter in optimized_parameters.values()),
        "all_networks": sum(parameter.numel() for parameter in network_parameters.values()),
    }


def main() -> None:
    args = parse_args()
    train_config, env_config = build_configs(args)
    ppo_config = build_ppo_config(args)
    set_seed(train_config.seed, deterministic=args.deterministic)

    # 假设你在 2026 年 7 月 7 日 下午 3 点 30 分 45 秒 执行了这行代码
    # 最后生成的 run_name 字符串就会是"dqn_20260707_153045"
    run_name = datetime.now().strftime(f"{args.algorithm}_%Y%m%d_%H%M%S")
    # 定义文件路径。命令行参数里的短横线 - 会自动转换成 Python 属性名里的下划线 _
    run_dir = args.runs_dir / run_name
    # checkpoint 也使用同一个 run_name 分目录保存，避免多次训练互相覆盖 best.pt/latest.pt。
    checkpoint_dir = args.checkpoint_dir / run_name
    # 创建每次训练过程的文件夹，注意一次训练包含多个episode.
    # runs侧重记录日志和指标，比如每一局分数 score，最近 100 局平均分 mean_score_100，loss
    run_dir.mkdir(parents=True, exist_ok=True)
    # 创建本次训练专属断点文件夹，断点侧重记录模型权重，用于加载模型和继续训练。
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"run_dir={run_dir}")
    print(f"checkpoint_dir={checkpoint_dir}")
    print_stop_overview(args, train_config)

    # 在指定的文件夹（这里是刚才创建的 run_dir）里新建训练日志文件；.train 后缀方便和评估日志区分。
    writer = SummaryWriter(run_dir, filename_suffix=".train") if SummaryWriter is not None else None

    # 训练指标和阶段验证指标分开保存。
    csv_path = run_dir / "train_metrics.csv"
    validation_csv_path = run_dir / "validation_metrics.csv"

    env = SnakeEnv(
        width=env_config.width,
        height=env_config.height,
        # 是否渲染
        render_mode=args.render,
        # 每个格子渲染成多少像素
        cell_size=env_config.cell_size,
        # 渲染帧率，画面刷新速度
        fps=env_config.fps,
        seed=train_config.seed,
        # 环境直接在 reset()/step() 中返回对应 observation，避免训练循环重复构造状态。
        state_mode=args.state_mode,
        reward_profile=args.reward_profile,
        potential_reward=args.potential_reward,
        cost_rewards=not args.no_cost_rewards,
        reward_gamma=train_config.gamma,
    )
    state_size = (
        env.grid_state_shape if args.state_mode in ("grid", "hybrid") else env.state_size
    )
    common_agent_kwargs = {
        "state_size": state_size,
        "action_size": env.action_size,
        "hidden_size": train_config.hidden_size,
        "learning_rate": train_config.learning_rate,
        "gamma": train_config.gamma,
        "batch_size": train_config.batch_size,
        "state_mode": args.state_mode,
        "auxiliary_size": env.state_size,
        "cnn_channels": train_config.cnn_channels,
        "cnn_output_channels": train_config.cnn_output_channels,
        "cnn_dilations": train_config.cnn_dilations,
        "seed": train_config.seed,
    }
    if args.algorithm == "ppo":
        agent = PPOAgent(
            **common_agent_kwargs,
            rollout_steps=ppo_config.rollout_steps,
            update_epochs=ppo_config.update_epochs,
            gae_lambda=ppo_config.gae_lambda,
            clip_coefficient=ppo_config.clip_coefficient,
            value_clip_coefficient=ppo_config.value_clip_coefficient,
            entropy_coefficient=ppo_config.entropy_coefficient,
            value_loss_coefficient=ppo_config.value_loss_coefficient,
            max_grad_norm=ppo_config.max_grad_norm,
            target_kl=ppo_config.target_kl,
            normalize_advantage=ppo_config.normalize_advantage,
            normalize_returns=ppo_config.normalize_returns,
        )
    else:
        agent = DQNAgent(
            **common_agent_kwargs,
            n_step=train_config.n_step,
            replay_buffer_size=train_config.replay_buffer_size,
            epsilon_start=train_config.epsilon_start,
            epsilon_end=train_config.epsilon_end,
            epsilon_exp_decay=train_config.epsilon_exp_decay,
            epsilon_exp_factor=train_config.epsilon_exp_factor,
            epsilon_linear_episodes=train_config.epsilon_linear_episodes,
            target_update_interval=train_config.target_update_interval,
        )
    validation_env = SnakeEnv(
        width=env_config.width,
        height=env_config.height,
        render_mode=False,
        cell_size=env_config.cell_size,
        fps=env_config.fps,
        seed=train_config.seed,
        starvation_enabled=False,
        state_mode=args.state_mode,
    )
    quick_seeds = make_episode_seeds(
        train_config.seed,
        "quick",
        args.validation_episodes,
    )
    confirmation_seeds = make_episode_seeds(
        train_config.seed,
        "confirmation",
        args.confirmation_episodes,
    )

    resolved_training = asdict(train_config)
    if args.algorithm == "ppo":
        for dqn_only_field in (
            "n_step",
            "replay_buffer_size",
            "epsilon_start",
            "epsilon_end",
            "epsilon_exp_decay",
            "epsilon_exp_factor",
            "epsilon_linear_episodes",
            "target_update_interval",
        ):
            resolved_training.pop(dqn_only_field)

    # 同时写入 run/checkpoint 目录，并嵌入每个 checkpoint。以后不再依赖日志反推奖励。
    run_config = {
        "schema_version": 1,
        "algorithm": args.algorithm,
        "run_name": run_name,
        "command_argv": [str(value) for value in sys.argv],
        "environment": {
            "width": env_config.width,
            "height": env_config.height,
            "cell_size": env_config.cell_size,
            "fps": env_config.fps,
            "state_mode": args.state_mode,
            "starvation_enabled": env.starvation_enabled,
        },
        "reward": env.get_reward_settings(),
        "training": resolved_training,
        **({"ppo": asdict(ppo_config)} if args.algorithm == "ppo" else {}),
        "early_stop": {
            "enabled": args.early_stop,
            "min_episodes": args.min_episodes,
            "validation_patience": args.validation_patience,
            "target_mean_score": args.target_mean_score,
        },
        "validation": {
            "interval": args.validation_interval,
            "quick_episodes": args.validation_episodes,
            "confirmation_episodes": args.confirmation_episodes,
            "max_steps": args.validation_max_steps,
            "quick_seed_start": quick_seeds[0],
            "confirmation_seed_start": confirmation_seeds[0],
            "selection_thresholds": asdict(DEFAULT_SELECTION_THRESHOLDS),
            "starvation_enabled": False,
            "selection_start_episode": (
                args.min_episodes
                if args.algorithm == "ppo"
                else train_config.epsilon_linear_episodes
            ),
        },
        "deterministic": args.deterministic,
    }
    config_text = json.dumps(run_config, ensure_ascii=False, indent=2) + "\n"
    (run_dir / "config.json").write_text(config_text, encoding="utf-8")
    (checkpoint_dir / "config.json").write_text(config_text, encoding="utf-8")
    print(f"reward_profile={env.reward_profile}")
    print(
        "max_steps_per_episode="
        + (
            "unlimited"
            if train_config.max_steps_per_episode is None
            else str(train_config.max_steps_per_episode)
        )
    )
    parameter_counts = count_agent_parameters(agent)
    print(f"policy_parameters={parameter_counts['policy']:,}")
    print(f"optimized_parameters={parameter_counts['optimized']:,}")
    print(f"all_network_parameters={parameter_counts['all_networks']:,}")

    # 每局得分历史，用于计算 mean_score_100 和生成 train/report。
    scores: list[int] = []
    # 每局存活步数历史，用于生成 train/report。
    episode_steps_history: list[int] = []
    # 每局吃食效率历史，用于生成 train/report。
    score_per_steps: list[float] = []
    # 每局结束时的最近 100 局平均分历史，用于生成 train/report。
    mean_scores: list[float] = []
    # 每局累计环境奖励历史，用于观察 reward 信号和游戏分数 score 的差异。
    episode_rewards: list[float] = []
    # 每局结束时的最近 100 局平均累计奖励历史，用于生成 train/report。
    mean_rewards: list[float] = []
    # 每局平均 loss 历史，用于生成 train/report 中的 loss 摘要。
    losses_history: list[float] = []
    # 每局平均 loss 历史，用于计算滚动的 mean_loss_100。
    mean_losses: list[float] = []
    # 每局结束时的 mean_loss_100 历史，用于生成 train/report。
    mean_loss_100s: list[float] = []
    algorithm_metric_histories: dict[str, list[float]] = (
        {"epsilon": [], "replay_buffer_size": []}
        if args.algorithm == "dqn"
        else {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "approx_kl": [],
            "clip_fraction": [],
            "explained_variance": [],
        }
    )
    # 历史单局最高分，仅用于终端输出观察。
    best_score = -1
    # 历史最高训练 mean_score_100 只用于观察，不再决定 best.pt。
    best_mean_score = float("-inf")
    validation_state = StagedValidationState()
    # 训练开始时间，用于计算 train/report 中的总耗时。
    train_start_time = time.perf_counter()

    # 在 CSV 文件中写入训练指标表头
    with (
        csv_path.open("w", newline="", encoding="utf-8") as file,
        validation_csv_path.open("w", newline="", encoding="utf-8") as validation_file,
    ):
        metrics = csv.writer(file)
        common_metrics_header = [
                "episode",
                "score",
                "score_per_step",
                "mean_score_100",
                "episode_reward",
                "mean_reward_100",
                "food_reward",
                "progress_reward",
                "step_penalty",
                "hunger_penalty",
                "terminal_reward",
                "termination_reason",
                "episode_steps",
        ]
        if args.algorithm == "dqn":
            algorithm_metrics_header = [
                "epsilon",
                "loss",
                "mean_loss_100",
                "replay_buffer_size",
            ]
        else:
            algorithm_metrics_header = [
                "loss",
                "mean_loss_100",
                "policy_loss",
                "value_loss",
                "entropy",
                "approx_kl",
                "clip_fraction",
                "explained_variance",
                "ppo_epochs",
                "ppo_samples",
                "ppo_update_steps",
                "rollout_buffer_size",
            ]
        metrics.writerow(common_metrics_header + algorithm_metrics_header)
        validation_metrics = csv.writer(validation_file)
        validation_metrics.writerow(
            [
                "training_episode",
                "stage",
                "seed_set",
                "validation_episodes",
                "max_steps",
                "mean_score",
                "score_std",
                "min_score",
                "max_score",
                "full_score",
                "full_games",
                "full_rate",
                "mean_steps",
                "timeout_games",
                "timeout_rate",
                "passed_stage",
                "promoted_to_best",
                "best_training_episode",
                "evaluation_time_sec",
            ]
        )

        def run_validation(
            training_episode: int,
            seed_set: SeedSetName,
        ) -> ValidationResult:
            if seed_set == "quick":
                seeds = quick_seeds
            elif seed_set == "confirmation":
                seeds = confirmation_seeds
            else:
                raise ValueError(f"unsupported training validation seed set: {seed_set}")
            print(
                f"validation_start episode={training_episode} "
                f"stage={seed_set} episodes={len(seeds)}"
            )
            result = evaluate_policy(
                agent,
                validation_env,
                seeds,
                seed_set=seed_set,
                max_steps=args.validation_max_steps,
            )
            print(
                f"validation_end episode={training_episode} stage={seed_set} "
                f"mean={result.mean_score:.3f} std={result.score_std:.3f} "
                f"full_rate={result.full_rate * 100:.2f}% "
                f"timeout_rate={result.timeout_rate * 100:.2f}% "
                f"time={result.total_time_sec:.1f}s"
            )
            return result

        def record_validation(
            training_episode: int,
            stage: str,
            result: ValidationResult,
            *,
            passed_stage: bool,
            promoted_to_best: bool,
        ) -> None:
            validation_metrics.writerow(
                [
                    training_episode,
                    stage,
                    result.seed_set,
                    result.episodes,
                    result.max_steps,
                    f"{result.mean_score:.6f}",
                    f"{result.score_std:.6f}",
                    result.min_score,
                    result.max_score,
                    result.full_score,
                    result.full_games,
                    f"{result.full_rate:.6f}",
                    f"{result.mean_steps:.6f}",
                    result.timeout_games,
                    f"{result.timeout_rate:.6f}",
                    passed_stage,
                    promoted_to_best,
                    validation_state.best_training_episode,
                    f"{result.total_time_sec:.6f}",
                ]
            )
            validation_file.flush()
            if writer is not None:
                tag_stage = "quick" if result.seed_set == "quick" else "confirmation"
                writer.add_scalar(
                    f"validation/{tag_stage}_mean_score",
                    result.mean_score,
                    training_episode,
                )
                writer.add_scalar(
                    f"validation/{tag_stage}_full_rate",
                    result.full_rate,
                    training_episode,
                )
                writer.add_scalar(
                    f"validation/{tag_stage}_timeout_rate",
                    result.timeout_rate,
                    training_episode,
                )

        def save_best(
            training_episode: int,
            quick_result: ValidationResult,
            confirmation_result: ValidationResult,
        ) -> None:
            metadata = {
                **run_config,
                "selection": {
                    "training_episode": training_episode,
                    "quick": quick_result.to_dict(),
                    "confirmation": confirmation_result.to_dict(),
                    "selection_metric": "mean_score_then_full_rate",
                    "selection_thresholds": asdict(DEFAULT_SELECTION_THRESHOLDS),
                },
            }
            agent.save(checkpoint_dir / "best.pt", metadata=metadata)

        wandb_run: WandbRun | None = None
        training_succeeded = False
        try:
            if args.wandb:
                wandb_run = start_wandb(
                    run_name=run_name,
                    run_dir=run_dir,
                    config=run_config,
                )

            print(f"training_starts_in={TRAINING_START_DELAY_SECONDS}s", flush=True)
            time.sleep(TRAINING_START_DELAY_SECONDS)
            train_start_time = time.perf_counter()

            for episode in range(1, train_config.episodes + 1):
                state = env.reset()
                done = False

                # 记录一个episode中所有步的loss
                losses: list[float] = []
                ppo_updates: list[PPOMetrics] = []
                # 记录一个 episode 中所有 step 的 reward 总和，和 score 分开观察。
                episode_reward = 0.0
                reward_components = {
                    "food": 0.0,
                    "progress": 0.0,
                    "step": 0.0,
                    "hunger": 0.0,
                    "terminal": 0.0,
                }

                # 一次episode训练
                while not done and (
                    train_config.max_steps_per_episode is None
                    or env.frame_iteration < train_config.max_steps_per_episode
                ):
                    # 训练时的动作采样
                    action = agent.act(state, training=True)
                    # 环境反馈，info是环境额外返回的信息字典，不直接参与DQN更新
                    next_state, reward, done, info = env.step(action)
                    # 累加本局每一步的环境奖励，形成单局累计 reward。
                    episode_reward += reward
                    for component in reward_components:
                        reward_components[component] += float(info[f"reward_{component}"])
                    # 加入经验回放池
                    agent.remember(state, action, reward, next_state, done)
                    # 智能体更新Q值，如果经验回放池没达到batch_size则返回None
                    update_result = agent.learn()
                    if isinstance(update_result, PPOMetrics):
                        ppo_updates.append(update_result)
                        losses.append(update_result.loss)
                    elif update_result is not None:
                        losses.append(update_result)
                    state = next_state

                # 自然终止时队列通常已在 remember() 中冲刷；若这里只是达到最大步数，
                # 则按实际跨度 k 保存尾部，并允许从最后的非终止状态 bootstrap。
                agent.finish_episode()

                # 每个 episode 执行一次 epsilon 衰减；默认按最大训练局数的 50% 线性退火。
                if args.algorithm == "dqn":
                    agent.decay_epsilon(episode)
                # 从环境返回的额外信息info中提取游戏分数字段的值
                score = int(info["score"])
                # 记录游戏分数到列表中
                scores.append(score)
                # 记录最近最多100次获得的游戏的平均分
                # 注意游戏的分数和环境的奖励是不同的概念，一个是指标，一个是训练信号
                mean_score = sum(scores[-100:]) / min(len(scores), 100)
                # 记录单局累计 reward，并计算最近最多 100 局的平均累计 reward。
                episode_rewards.append(episode_reward)
                mean_reward = sum(episode_rewards[-100:]) / min(len(episode_rewards), 100)
                mean_rewards.append(mean_reward)
                # episode_steps 用来区分"很快撞死"和"走了很久但没吃到食物"。
                episode_steps = int(info["steps"])
                # 吃食效率 = 吃到的食物数 / 存活步数，衡量策略是否直奔目标
                score_per_step = score / episode_steps if episode_steps > 0 else 0.0
                # 一个episode中产生的所有损失求平均。
                episode_loss = sum(losses) / len(losses) if losses else None
                if args.algorithm == "dqn":
                    mean_loss = episode_loss if episode_loss is not None else 0.0
                    mean_losses.append(mean_loss)
                elif episode_loss is not None:
                    mean_losses.append(episode_loss)
                    mean_loss = episode_loss
                else:
                    mean_loss = 0.0
                mean_loss_100 = (
                    sum(mean_losses[-100:]) / min(len(mean_losses), 100)
                    if mean_losses
                    else 0.0
                )
                episode_steps_history.append(episode_steps)
                score_per_steps.append(score_per_step)
                mean_scores.append(mean_score)
                if args.algorithm == "dqn" or episode_loss is not None:
                    losses_history.append(mean_loss)
                    mean_loss_100s.append(mean_loss_100)
                if args.algorithm == "dqn":
                    algorithm_metric_histories["epsilon"].append(agent.epsilon)
                    algorithm_metric_histories["replay_buffer_size"].append(
                        float(len(agent.replay_buffer))
                    )
                for update in ppo_updates:
                    algorithm_metric_histories["policy_loss"].append(update.policy_loss)
                    algorithm_metric_histories["value_loss"].append(update.value_loss)
                    algorithm_metric_histories["entropy"].append(update.entropy)
                    algorithm_metric_histories["approx_kl"].append(update.approx_kl)
                    algorithm_metric_histories["clip_fraction"].append(update.clip_fraction)
                    algorithm_metric_histories["explained_variance"].append(
                        update.explained_variance
                    )
                ppo_episode_metrics: dict[str, float | int] = {}
                if ppo_updates:
                    ppo_episode_metrics = {
                        "policy_loss": sum(item.policy_loss for item in ppo_updates)
                        / len(ppo_updates),
                        "value_loss": sum(item.value_loss for item in ppo_updates)
                        / len(ppo_updates),
                        "entropy": sum(item.entropy for item in ppo_updates) / len(ppo_updates),
                        "approx_kl": sum(item.approx_kl for item in ppo_updates)
                        / len(ppo_updates),
                        "clip_fraction": sum(item.clip_fraction for item in ppo_updates)
                        / len(ppo_updates),
                        "explained_variance": sum(
                            item.explained_variance for item in ppo_updates
                        )
                        / len(ppo_updates),
                        "ppo_epochs": ppo_updates[-1].epochs,
                        "ppo_samples": sum(item.samples for item in ppo_updates),
                        "ppo_update_steps": agent.update_steps,
                        "rollout_buffer_size": len(agent.rollout),
                    }

                # 存下得分最高时的参数
                if score > best_score:
                    best_score = score

                # 训练 mean_score_100 只保留为诊断指标。
                if mean_score > best_mean_score:
                    best_mean_score = mean_score

                # 将训练指标写入 TensorBoard
                if writer is not None:
                    # 单局得分
                    writer.add_scalar("train/score", score, episode)
                    # 吃食效率 = 吃到的食物数 / 存活步数
                    writer.add_scalar("train/score_per_step", score_per_step, episode)
                    # 历史最高分
                    # 最近100局滑动平均分
                    writer.add_scalar("train/mean_score_100", mean_score, episode)
                    writer.add_scalar("train/best_mean_score_100", best_mean_score, episode)
                    # 本局累计环境奖励
                    writer.add_scalar("train/episode_reward", episode_reward, episode)
                    # 最近100局滑动平均累计环境奖励
                    writer.add_scalar("train/mean_reward_100", mean_reward, episode)
                    for component, component_value in reward_components.items():
                        writer.add_scalar(f"train/reward_{component}", component_value, episode)
                    # 本局存活步数
                    writer.add_scalar("train/episode_steps", episode_steps, episode)
                    if episode_loss is not None or args.algorithm == "dqn":
                        writer.add_scalar("train/loss", mean_loss, episode)
                        writer.add_scalar("train/mean_loss_100", mean_loss_100, episode)
                    if args.algorithm == "dqn":
                        writer.add_scalar("train/epsilon", agent.epsilon, episode)
                        writer.add_scalar(
                            "train/replay_buffer_size", len(agent.replay_buffer), episode
                        )
                    else:
                        for metric_name, metric_value in ppo_episode_metrics.items():
                            writer.add_scalar(f"train/{metric_name}", metric_value, episode)

                if wandb_run is not None:
                    wandb_run.log(
                        training_metrics(
                            episode=episode,
                            scores=scores,
                            mean_score_100=mean_score,
                            episode_reward=episode_reward,
                            mean_reward_100=mean_reward,
                            episode_steps=episode_steps_history,
                            loss=mean_loss,
                            mean_loss_100=mean_loss_100,
                            algorithm=args.algorithm,
                            epsilon=(agent.epsilon if args.algorithm == "dqn" else None),
                            replay_buffer_size=(
                                len(agent.replay_buffer) if args.algorithm == "dqn" else None
                            ),
                            ppo_metrics=ppo_episode_metrics,
                        )
                    )

                common_metrics_row = [
                        episode,
                        score,
                        f"{score_per_step:.6f}",
                        f"{mean_score:.4f}",
                        f"{episode_reward:.4f}",
                        f"{mean_reward:.4f}",
                        f"{reward_components['food']:.6f}",
                        f"{reward_components['progress']:.6f}",
                        f"{reward_components['step']:.6f}",
                        f"{reward_components['hunger']:.6f}",
                        f"{reward_components['terminal']:.6f}",
                        str(info["termination_reason"]),
                        episode_steps,
                ]
                if args.algorithm == "dqn":
                    algorithm_metrics_row = [
                        f"{agent.epsilon:.6f}",
                        mean_loss,
                        f"{mean_loss_100:.4f}",
                        len(agent.replay_buffer),
                    ]
                else:
                    algorithm_metrics_row = [
                        "" if episode_loss is None else f"{mean_loss:.6f}",
                        "" if episode_loss is None else f"{mean_loss_100:.6f}",
                        *[
                            ppo_episode_metrics.get(name, "")
                            for name in (
                                "policy_loss",
                                "value_loss",
                                "entropy",
                                "approx_kl",
                                "clip_fraction",
                                "explained_variance",
                                "ppo_epochs",
                                "ppo_samples",
                                "ppo_update_steps",
                                "rollout_buffer_size",
                            )
                        ],
                    ]
                metrics.writerow(common_metrics_row + algorithm_metrics_row)

                # 强制将内存缓冲区（Buffer）中的数据立刻写入到实际的硬盘文件中，防止数据丢失
                file.flush()

                # 存下最近一次episode更新出来的参数
                agent.save(checkpoint_dir / "latest.pt", metadata=run_config)

                decision = run_staged_validation(
                    episode=episode,
                    state=validation_state,
                    evaluator=lambda seed_set: run_validation(episode, seed_set),
                    interval=args.validation_interval,
                    early_stop_enabled=args.early_stop,
                    min_episodes=args.min_episodes,
                    patience=args.validation_patience,
                    target_mean_score=args.target_mean_score,
                    epsilon=(agent.epsilon if args.algorithm == "dqn" else None),
                    epsilon_end=(agent.epsilon_end if args.algorithm == "dqn" else None),
                    selection_ready=(
                        episode >= args.min_episodes if args.algorithm == "ppo" else None
                    ),
                )
                for event in decision.events:
                    record_validation(
                        episode,
                        event.stage,
                        event.result,
                        passed_stage=event.passed_stage,
                        promoted_to_best=event.promoted_to_best,
                    )

                if decision.best_updated:
                    if (
                        validation_state.best_quick is None
                        or validation_state.best_confirmation is None
                        or validation_state.best_training_episode is None
                    ):
                        raise RuntimeError("best update is missing validation results")
                    save_best(
                        validation_state.best_training_episode,
                        validation_state.best_quick,
                        validation_state.best_confirmation,
                    )
                    print(
                        f"best_updated episode={validation_state.best_training_episode} "
                        f"mean={validation_state.best_confirmation.mean_score:.3f} "
                        f"full_rate="
                        f"{validation_state.best_confirmation.full_rate * 100:.2f}%"
                    )

                if decision.stop_reason is not None:
                    print(
                        f"early_stop={decision.stop_reason} episode={episode} "
                        f"rounds_without_improvement="
                        f"{validation_state.rounds_without_improvement} "
                        f"best_episode={validation_state.best_training_episode}"
                    )

                # 每个episode输出一次指标信息
                algorithm_status = (
                    f"epsilon={agent.epsilon:.3f} replay={len(agent.replay_buffer)}"
                    if args.algorithm == "dqn"
                    else f"updates={agent.update_steps} rollout={len(agent.rollout)}"
                )
                print(
                    f"episode={episode:4d} score={score:3d} steps={episode_steps:4d} "
                    f"reward={episode_reward:6.1f} mean100={mean_score:6.2f} "
                    f"{algorithm_status} loss={mean_loss:.4f} best_score={best_score}"
                )

                if decision.stop_reason is not None:
                    break
            training_succeeded = True
        finally:  # 无论上面是否跑完，这段代码都必须执行。
            env.close()
            validation_env.close()
            try:
                if writer is not None:
                    if scores:
                        writer.add_text(
                            "train/report",
                            format_train_report(
                                run_name=run_name,
                                episodes=len(scores),
                                total_time_sec=time.perf_counter() - train_start_time,
                                best_score=best_score,
                                best_mean_score=best_mean_score,
                                scores=scores,
                                episode_steps=episode_steps_history,
                                score_per_steps=score_per_steps,
                                mean_scores=mean_scores,
                                episode_rewards=episode_rewards,
                                mean_rewards=mean_rewards,
                                losses=losses_history,
                                mean_loss_100s=mean_loss_100s,
                                algorithm_metrics=algorithm_metric_histories,
                            ),
                            global_step=len(scores),
                        )
                    writer.close()
            finally:
                if wandb_run is not None:
                    try:
                        if scores:
                            wandb_run.summary.update(
                                {
                                    "episodes": len(scores),
                                    "best_score": best_score,
                                    "best_mean_score_100": best_mean_score,
                                    "total_time_sec": time.perf_counter() - train_start_time,
                                }
                            )
                    finally:
                        wandb_run.finish(exit_code=0 if training_succeeded else 1)

    if validation_state.best_training_episode is None:
        print(
            "best_not_created=selection_did_not_start "
            "latest.pt remains available; no fallback checkpoint was selected"
        )


if __name__ == "__main__":
    main()
