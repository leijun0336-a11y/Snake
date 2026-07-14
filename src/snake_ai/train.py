# 处理自引用问题
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from snake_ai.agents import DQNAgent
from snake_ai.config import (
    CHECKPOINT_DIR,
    REWARD_PROFILE_NAMES,
    RUNS_DIR,
    EnvConfig,
    TrainConfig,
)
from snake_ai.game import SnakeEnv
from snake_ai.utils import set_seed

try:
    from torch.utils.tensorboard import SummaryWriter
# 如果用户没装 TensorBoard，就让它等于 None
except ImportError:  
    SummaryWriter = None

# 解析启动脚本时的命令行参数
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a DQN agent for Snake.")
    # 最大训练局数；早停没有触发时，训练最多跑到这个 episode。
    parser.add_argument("--max-episodes", type=int, default=TrainConfig.episodes)
    parser.add_argument(
        "--max-steps-per-episode",
        type=int,
        default=None,
        help=(
            "Override the profile's training episode step limit. The reference "
            "profile defaults to 500; experiment8 is intentionally unlimited."
        ),
    )
    # 是否带渲染训练, action="store_true"表示：启动脚本时写上--render则为True，不写默认为False
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--width", type=int, default=EnvConfig.width)
    parser.add_argument("--height", type=int, default=EnvConfig.height)
    parser.add_argument("--cell-size", type=int, default=EnvConfig.cell_size)
    parser.add_argument("--fps", type=int, default=EnvConfig.fps)
    parser.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR)
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument(
        "--state-mode", choices=("vector", "grid", "hybrid"), default="vector"
    )
    parser.add_argument(
        "--reward-profile",
        choices=REWARD_PROFILE_NAMES,
        default="reference",
        help=(
            "Named reward semantics. experiment8 exactly restores the historical "
            "dqn_20260712_130642 reward and starvation behavior."
        ),
    )
    parser.add_argument(
        "--potential-reward",
        action="store_true",
        help="Enable potential-based food progress shaping (disabled by default).",
    )
    parser.add_argument(
        "--no-cost-rewards",
        action="store_true",
        help="Disable per-step and hunger costs and use the baseline timeout penalty.",
    )
    # 完全确定性会让部分 CUDA 卷积/池化算子明显变慢；默认优先训练速度。
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--gamma", type=float, default=TrainConfig.gamma)
    parser.add_argument("--learning-rate", type=float, default=TrainConfig.learning_rate)
    parser.add_argument(
        "--replay-buffer-size", type=int, default=TrainConfig.replay_buffer_size
    )
    parser.add_argument("--epsilon-start", type=float, default=TrainConfig.epsilon_start)
    parser.add_argument("--epsilon-end", type=float, default=TrainConfig.epsilon_end)
    parser.add_argument("--epsilon-decay", type=float, default=TrainConfig.epsilon_decay)
    parser.add_argument(
        "--epsilon-decay-episodes",
        type=int,
        default=TrainConfig.epsilon_decay_episodes,
        help=(
            "Linearly decay epsilon to epsilon-end over this many episodes. "
            "Defaults to 50%% of the current max episodes. Set to 0 to use exponential "
            "epsilon-decay instead."
        ),
    )
    parser.add_argument(
        "--target-update-interval",
        type=int,
        default=TrainConfig.target_update_interval,
    )
    parser.add_argument("--hidden-size", type=int, default=TrainConfig.hidden_size)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    # 这些参数只影响 Grid/Hybrid 的卷积主干，Vector baseline 会忽略它们。
    parser.add_argument("--cnn-channels", type=int, default=TrainConfig.cnn_channels)
    parser.add_argument(
        "--cnn-output-channels", type=int, default=TrainConfig.cnn_output_channels
    )
    parser.add_argument(
        "--cnn-dilations", type=int, nargs="+", default=TrainConfig.cnn_dilations
    )
    parser.add_argument(
        "--cnn-pool-size", type=int, nargs=2, default=TrainConfig.cnn_pool_size
    )
    # 默认严格跑满最大训练局数；显式传入该参数后才启用早停。
    parser.add_argument(
        "--early-stop",
        action="store_true",
        help="Enable patience/target based early stopping (disabled by default).",
    )
    # 至少训练多少局后，才允许早停判断生效。
    parser.add_argument("--min-episodes", type=int, default=5000)
    # 超过最小训练局数后，允许连续多少局没有有效提升。
    parser.add_argument("--patience", type=int, default=500)
    # mean_score_100 至少提升多少，才算一次有效提升。
    parser.add_argument("--min-delta", type=float, default=0.5)
    # 如果设置了目标平均分，达到该 mean_score_100 后直接停止训练。
    parser.add_argument("--target-mean-score", type=float, default=None)
    return parser.parse_args()


# 统计列表的均值、标准差、最小值、最大值和最后一个值，返回字典
def summarize_values(values: list[int] | list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min": float(min(values)),
        "max": float(max(values)),
        "last": float(values[-1]),
    }


def resolve_max_steps_per_episode(args: argparse.Namespace) -> int | None:
    """Validate profile invariants and resolve its training episode step limit."""

    if args.max_steps_per_episode is not None and args.max_steps_per_episode < 1:
        raise ValueError("max_steps_per_episode must be at least 1")
    if args.reward_profile == "experiment8":
        if args.potential_reward:
            raise ValueError(
                "experiment8 already enables potential reward; omit --potential-reward"
            )
        if args.no_cost_rewards:
            raise ValueError(
                "experiment8 requires its historical cost rewards; "
                "--no-cost-rewards is incompatible"
            )
        if args.max_steps_per_episode is not None:
            raise ValueError(
                "experiment8 requires no independent training episode step limit; "
                "omit --max-steps-per-episode"
            )
        if args.gamma != 0.99:
            raise ValueError("experiment8 requires gamma=0.99 for its potential reward")
        return None
    return (
        TrainConfig.max_steps_per_episode
        if args.max_steps_per_episode is None
        else args.max_steps_per_episode
    )


def print_stop_overview(args: argparse.Namespace, max_episodes: int) -> None:
    early_stop_enabled = args.early_stop
    patience_earliest_episode = max(args.min_episodes, args.patience + 1)
    target_earliest_episode = 100 if args.target_mean_score is not None else None
    earliest_candidates = [patience_earliest_episode]
    if target_earliest_episode is not None:
        earliest_candidates.append(target_earliest_episode)
    reachable_earliest_candidates = [
        episode for episode in earliest_candidates if episode <= max_episodes
    ]
    earliest_stop_text = (
        str(min(reachable_earliest_candidates))
        if reachable_earliest_candidates
        else "not reachable within max episodes"
    )

    print("")
    print("========== training stop overview ==========")
    print(f"max episodes without early stop : {max_episodes}")
    print(f"early stop enabled              : {early_stop_enabled}")
    if not early_stop_enabled:
        print("earliest early stop episode     : disabled")
        print("actual stop condition           : run until max episodes")
    else:
        print(f"earliest early stop episode     : {earliest_stop_text}")
        if target_earliest_episode is None:
            print("target mean score stop          : disabled")
        else:
            print(
                "target mean score stop          : "
                f"episode >= {target_earliest_episode}, "
                f"mean_score_100 >= {args.target_mean_score:.2f}"
            )
        print(
            "patience stop                   : "
            f"episode >= {args.min_episodes}, "
            f"no effective improvement for {args.patience} episodes"
        )
        print(f"earliest patience stop episode  : {patience_earliest_episode}")
        if patience_earliest_episode > max_episodes:
            print("patience stop within max episode: impossible with current settings")
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
    epsilons: list[float],
    replay_buffer_sizes: list[int],
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
        "epsilon": summarize_values(epsilons),
        "replay_buffer_size": summarize_values(replay_buffer_sizes),
    }
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


def main() -> None:
    args = parse_args()
    max_episodes = args.max_episodes
    if max_episodes < 1:
        raise ValueError("max episodes must be at least 1")
    max_steps_per_episode = resolve_max_steps_per_episode(args)
    if args.width < 4 or args.height < 4:
        raise ValueError("width and height must be at least 4")
    if args.cell_size < 1 or args.fps < 1:
        raise ValueError("cell_size and fps must be positive")
    if args.batch_size < 1 or args.replay_buffer_size < args.batch_size:
        raise ValueError("replay_buffer_size must be at least batch_size >= 1")
    if not 0.0 <= args.gamma <= 1.0:
        raise ValueError("gamma must be between 0 and 1")
    if args.learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if not 0.0 <= args.epsilon_end <= args.epsilon_start <= 1.0:
        raise ValueError("epsilon values must satisfy 0 <= end <= start <= 1")
    if not 0.0 < args.epsilon_decay <= 1.0:
        raise ValueError("epsilon_decay must be in (0, 1]")
    if args.epsilon_decay_episodes is not None and args.epsilon_decay_episodes < 0:
        raise ValueError("epsilon_decay_episodes must be non-negative")
    epsilon_decay_episodes = (
        max(1, int(max_episodes * 0.5))
        if args.epsilon_decay_episodes is None
        else args.epsilon_decay_episodes
    )
    if args.target_update_interval < 1 or args.hidden_size < 1:
        raise ValueError("target_update_interval and hidden_size must be positive")
    if args.min_episodes < 1:
        raise ValueError("min_episodes must be at least 1")
    if args.min_episodes > max_episodes:
        raise ValueError("min_episodes must be less than or equal to max episodes")
    if args.cnn_channels <= 0 or args.cnn_output_channels <= 0:
        raise ValueError("CNN channel sizes must be positive")
    if any(dilation <= 0 for dilation in args.cnn_dilations):
        raise ValueError("cnn_dilations must contain positive integers")
    if any(size <= 0 for size in args.cnn_pool_size):
        raise ValueError("cnn_pool_size must contain positive integers")

    train_config = TrainConfig(
        episodes=max_episodes,
        max_steps_per_episode=max_steps_per_episode,
        batch_size=args.batch_size,
        gamma=args.gamma,
        learning_rate=args.learning_rate,
        replay_buffer_size=args.replay_buffer_size,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        epsilon_decay_episodes=(
            None if epsilon_decay_episodes == 0 else epsilon_decay_episodes
        ),
        target_update_interval=args.target_update_interval,
        hidden_size=args.hidden_size,
        cnn_channels=args.cnn_channels,
        cnn_output_channels=args.cnn_output_channels,
        cnn_dilations=tuple(args.cnn_dilations),
        cnn_pool_size=tuple(args.cnn_pool_size),
        seed=args.seed,
    )
    set_seed(train_config.seed, deterministic=args.deterministic)
    env_config = EnvConfig(
        width=args.width,
        height=args.height,
        cell_size=args.cell_size,
        fps=args.fps,
    )

    # 假设你在 2026 年 7 月 7 日 下午 3 点 30 分 45 秒 执行了这行代码
    # 最后生成的 run_name 字符串就会是"dqn_20260707_153045"
    run_name = datetime.now().strftime("dqn_%Y%m%d_%H%M%S")
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
    print_stop_overview(args, train_config.episodes)

    # 在指定的文件夹（这里是刚才创建的 run_dir）里新建训练日志文件；.train 后缀方便和评估日志区分。
    writer = SummaryWriter(run_dir, filename_suffix=".train") if SummaryWriter is not None else None
    
    # train_metrics.csv 是训练指标表格文件；和 eval_metrics.csv 配对，文件名能直接看出用途。
    csv_path = run_dir / "train_metrics.csv"

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
        potential_reward=True if args.potential_reward else None,
        cost_rewards=False if args.no_cost_rewards else None,
        reward_gamma=train_config.gamma,
    )
    agent = DQNAgent(
        # 状态维度
        state_size=(
            env.grid_state_shape
            if args.state_mode in ("grid", "hybrid")
            else env.state_size
        ),
        # 动作维度
        action_size=env.action_size, 
        hidden_size=train_config.hidden_size,
        learning_rate=train_config.learning_rate,
        # 折扣率
        gamma=train_config.gamma,  
        # 经验池大小
        replay_buffer_size=train_config.replay_buffer_size,  
        batch_size=train_config.batch_size,
        # 初始epsilon值(Epsilon-Greedy)
        epsilon_start=train_config.epsilon_start,  
        # epsilon值的下界
        epsilon_end=train_config.epsilon_end,  
        # epsilon的指数衰减系数；线性衰减关闭时使用。
        epsilon_decay=train_config.epsilon_decay,
        # 线性衰减到 epsilon_end 需要的 episode 数。
        epsilon_decay_episodes=train_config.epsilon_decay_episodes,
        # 隔多少步更新一次目标网络
        # Q训练网络更新一次视为一步，当经验池满后，则等价于贪吃蛇走一步
        target_update_interval=train_config.target_update_interval,   
        state_mode=args.state_mode,
        # Hybrid 拼接的是完整 get_state()，因此辅助向量维度等于环境 state_size。
        auxiliary_size=env.state_size,
        cnn_channels=train_config.cnn_channels,
        cnn_output_channels=train_config.cnn_output_channels,
        cnn_dilations=train_config.cnn_dilations,
        cnn_pool_size=train_config.cnn_pool_size,
        seed=train_config.seed,
    )

    # 同时写入 run/checkpoint 目录，并嵌入每个 checkpoint。以后不再依赖日志反推奖励。
    run_config = {
        "schema_version": 1,
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
        "training": asdict(train_config),
        "early_stop": {
            "enabled": args.early_stop,
            "min_episodes": args.min_episodes,
            "patience": args.patience,
            "min_delta": args.min_delta,
            "target_mean_score": args.target_mean_score,
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
    # 每局结束后的 epsilon 历史，用于生成 train/report。
    epsilons: list[float] = []
    # 每局结束时的经验池大小历史，用于生成 train/report。
    replay_buffer_sizes: list[int] = []
    # 历史单局最高分，仅用于终端输出观察。
    best_score = -1
    # 历史最高 mean_score_100，用于决定何时保存 best.pt。
    best_mean_score = float("-inf")
    # 早停判断中的历史最高有效 mean_score_100，小于 min_delta 的提升不重置耐心计数。
    early_stop_best_mean_score = float("-inf")
    # 最近一次达到有效提升的 episode，用于判断是否长时间没有明显进步。
    last_improve_episode = 0
    # 训练开始时间，用于计算 train/report 中的总耗时。
    train_start_time = time.perf_counter()

    # 在 CSV 文件中写入训练指标表头
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        metrics = csv.writer(file)
        metrics.writerow(
            [
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
                "epsilon",
                "loss",
                "mean_loss_100",
                "replay_buffer_size",
            ]
        )

        try:
            for episode in range(1, train_config.episodes + 1):
                state = env.reset()
                done = False
                
                # 记录一个episode中所有步的loss
                losses: list[float] = []
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
                while (
                    not done
                    and (
                        train_config.max_steps_per_episode is None
                        or env.frame_iteration < train_config.max_steps_per_episode
                    )
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
                    loss = agent.learn()
                    if loss is not None:
                        losses.append(loss)
                    state = next_state

                # 每个 episode 执行一次 epsilon 衰减；默认按最大训练局数的 50% 线性退火。
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
                # episode_steps 用来区分“很快撞死”和“走了很久但没吃到食物”。
                episode_steps = int(info["steps"])
                # 吃食效率 = 吃到的食物数 / 存活步数，衡量策略是否直奔目标
                score_per_step = score / episode_steps if episode_steps > 0 else 0.0
                # 一个episode中产生的所有损失求平均。
                mean_loss = sum(losses) / len(losses) if losses else 0.0
                mean_losses.append(mean_loss)
                mean_loss_100 = sum(mean_losses[-100:]) / min(len(mean_losses), 100)
                episode_steps_history.append(episode_steps)
                score_per_steps.append(score_per_step)
                mean_scores.append(mean_score)
                losses_history.append(mean_loss)
                mean_loss_100s.append(mean_loss_100)
                epsilons.append(agent.epsilon)
                replay_buffer_sizes.append(len(agent.replay_buffer))

                # 存下得分最高时的参数
                if score > best_score:
                    best_score = score

                # 保存最近 100 局平均分最高时的参数；不足 100 局时用已有局数的平均分。
                if mean_score > best_mean_score:
                    best_mean_score = mean_score
                    agent.save(checkpoint_dir / "best.pt", metadata=run_config)

                # 早停使用 mean_score_100 判断收敛，不使用 reward，避免奖励塑形影响模型选择。
                if mean_score > early_stop_best_mean_score + args.min_delta:
                    early_stop_best_mean_score = mean_score
                    last_improve_episode = episode

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
                        writer.add_scalar(
                            f"train/reward_{component}", component_value, episode
                        )
                    # 本局存活步数
                    writer.add_scalar("train/episode_steps", episode_steps, episode)
                    # 当前探索率
                    writer.add_scalar("train/epsilon", agent.epsilon, episode)
                    # 本局平均loss
                    writer.add_scalar("train/loss", mean_loss, episode)
                    # 最近100局滑动平均loss
                    writer.add_scalar("train/mean_loss_100", mean_loss_100, episode)
                    # 经验回放池当前容量
                    writer.add_scalar("train/replay_buffer_size", len(agent.replay_buffer), episode)

                metrics.writerow(
                    [
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
                        f"{agent.epsilon:.6f}",
                        mean_loss,
                        f"{mean_loss_100:.4f}",
                        len(agent.replay_buffer),
                    ]
                )
                
                # 强制将内存缓冲区（Buffer）中的数据立刻写入到实际的硬盘文件中，防止数据丢失
                file.flush()

                # 存下最近一次episode更新出来的参数
                agent.save(checkpoint_dir / "latest.pt", metadata=run_config)

                # 每个episode输出一次指标信息
                print(
                    f"episode={episode:4d} score={score:3d} steps={episode_steps:4d} "
                    f"reward={episode_reward:6.1f} "
                    f"mean100={mean_score:6.2f} epsilon={agent.epsilon:.3f} "
                    f"loss={mean_loss:.4f} best_score={best_score}"
                )

                # early_stop_enabled 统一控制下面两种早停条件是否生效。
                early_stop_enabled = args.early_stop
                # 达到目标 mean_score_100 后停止；至少等 100 局，避免前期均值窗口太短。
                reached_target = (
                    args.target_mean_score is not None
                    and episode >= 100
                    and mean_score >= args.target_mean_score
                )
                # 超过最小训练局数后，如果太久没有有效提升，则认为进入平台期。
                patience_exhausted = (
                    episode >= args.min_episodes
                    and episode - last_improve_episode >= args.patience
                )
                if early_stop_enabled and reached_target:
                    print(
                        f"early_stop=target_mean_score "
                        f"episode={episode} mean100={mean_score:.2f} "
                        f"target={args.target_mean_score:.2f}"
                    )
                    break
                if early_stop_enabled and patience_exhausted:
                    print(
                        f"early_stop=patience episode={episode} "
                        f"best_mean100={best_mean_score:.2f} "
                        f"last_improve_episode={last_improve_episode} "
                        f"patience={args.patience}"
                    )
                    break
        finally:  # 无论上面是否跑完，这段代码都必须执行。
            env.close()
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
                            epsilons=epsilons,
                            replay_buffer_sizes=replay_buffer_sizes,
                        ),
                        global_step=len(scores),
                    )
                writer.close()


if __name__ == "__main__":
    main()
