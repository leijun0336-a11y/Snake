# 运行方法: uv run src/snake_ai/evaluate.py
# 如果要可视化 --tensorboard
# 默认会渲染，不想渲染可以加上 --no-render

# 处理自引用问题
from __future__ import annotations  

import argparse
import csv
from pathlib import Path
from typing import Any, TextIO

import torch

from snake_ai.agents import DQNAgent, PPOAgent
from snake_ai.config import CHECKPOINT_DIR, RUNS_DIR, EnvConfig, TrainConfig
from snake_ai.game import SnakeEnv
from snake_ai.utils import set_seed, summarize_values
from snake_ai.validation import ValidationEpisode, evaluate_policy, make_episode_seeds


EVAL_METRICS_HEADER = (
    "episode",
    "score",
    "steps",
    "score_per_step",
    "max_snake_length",
)

try:
    from torch.utils.tensorboard import SummaryWriter
# 如果用户没装 TensorBoard，就让它等于 None。
except ImportError:
    SummaryWriter = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained Snake DQN or PPO agent.")
    # 不传时默认加载 checkpoints 下时间最新的 DQN/PPO 实验。
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--algorithm", choices=("dqn", "ppo"), default=None)
    parser.add_argument("--episodes", type=int, default=1000)
    # 防止转圈。
    parser.add_argument("--max-steps",type=int,default=1000)
    # 不渲染，默认渲染。
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--cell-size", type=int, default=EnvConfig.cell_size)
    parser.add_argument("--fps", type=int, default=EnvConfig.fps)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    # 是否把本次评估的分数写入 CSV 和 TensorBoard。
    parser.add_argument("--tensorboard", action="store_true")
    # 指定评估指标输出目录；不指定时默认绑定到最近一次训练的 runs/dqn_* 目录。
    parser.add_argument("--eval-output-dir", type=Path, default=None)
    parser.add_argument(
        "--state-mode", choices=("vector", "grid", "hybrid"), default=None
    )
    return parser.parse_args()

def _experiment_dirs(root: Path, algorithm: str | None = None) -> list[Path]:
    prefixes = (algorithm,) if algorithm is not None else ("dqn", "ppo")
    return [
        path
        for prefix in prefixes
        for path in root.glob(f"{prefix}_*")
        if path.is_dir()
    ]


def _experiment_timestamp(path: Path) -> str:
    return path.name.split("_", maxsplit=1)[1]


# 默认提取 DQN/PPO 中最近一次实验的 run 目录，若没有则报错。
def find_latest_run_dir(runs_dir: Path = RUNS_DIR, algorithm: str | None = None) -> Path:
    run_dirs = _experiment_dirs(runs_dir, algorithm)
    if not run_dirs:
        raise FileNotFoundError(f"No training run directory found in {runs_dir}")
    return max(run_dirs, key=_experiment_timestamp)


# 默认提取最近一次实验的权重，权重默认取latest.pt，若没有latest.pt则取best.pt
def find_latest_checkpoint(
    checkpoint_dir: Path = CHECKPOINT_DIR,
    algorithm: str | None = None,
) -> Path:
    checkpoint_dirs = _experiment_dirs(checkpoint_dir, algorithm)
    if not checkpoint_dirs:
        legacy_checkpoint = checkpoint_dir / "latest.pt"
        if legacy_checkpoint.exists():
            return legacy_checkpoint
        raise FileNotFoundError(f"No checkpoint directory found in {checkpoint_dir}")

    latest_dir = max(checkpoint_dirs, key=_experiment_timestamp)
    latest_checkpoint = latest_dir / "latest.pt"
    if not latest_checkpoint.exists():
        raise FileNotFoundError(
            f"No latest.pt found in latest checkpoint directory {latest_dir}"
        )
    return latest_checkpoint


# 确定评估结果的tensorboard文件存储在哪个文件夹下。
def find_run_dir_for_checkpoint(checkpoint_path: Path, runs_dir: Path = RUNS_DIR) -> Path:

    run_name = checkpoint_path.parent.name
    if run_name.startswith(("dqn_", "ppo_")):
        return runs_dir / run_name
    return find_latest_run_dir(runs_dir)


# 每次评估都创建一份全新的 CSV，避免中断重跑或切换 checkpoint 后混入旧数据。
def open_eval_metrics_csv(csv_path: Path) -> tuple[TextIO, Any]:
    csv_file = csv_path.open("w", newline="", encoding="utf-8")
    metrics = csv.writer(csv_file)
    metrics.writerow(EVAL_METRICS_HEADER)
    csv_file.flush()
    return csv_file, metrics


# 把一次评估的统计结果整理成 Markdown 格式的文本报告，放入tensorboard的 text 标签页中。
def format_eval_report(
    checkpoint_path: Path,
    episodes: int,
    total_time_sec: float,
    scores: list[int],
    steps: list[int],
    score_per_steps: list[float],
    max_lengths: list[int],
    full_score: int,
) -> str:
    summaries = {
        "score": summarize_values(scores),
        "steps": summarize_values(steps),
        "score_per_step": summarize_values(score_per_steps),
        "max_snake_length": summarize_values(max_lengths),
    }
    full_games = sum(1 for score in scores if score >= full_score)
    full_rate = full_games / episodes if episodes > 0 else 0.0
    lines = [
        f"Checkpoint: `{checkpoint_path}`",
        f"Episodes: `{episodes}`",
        f"Total time: `{total_time_sec:.3f} sec`",
        "",
        "| Metric | Mean | Std | Min | Max |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, summary in summaries.items():
        lines.append(
            f"| {name} | {summary['mean']:.4f} | {summary['std']:.4f} | "
            f"{summary['min']:.4f} | {summary['max']:.4f} |"
        )
    lines.extend(
        [
            "",
            "| Full Score Metric | Value |",
            "|---|---:|",
            f"| full_score | {full_score} |",
            f"| full_games | {full_games} / {episodes} |",
            f"| full_rate | {full_rate * 100:.2f}% |",
        ]
    )
    return "\n".join(lines)


# 读取 checkpoint 中记录的状态输入模式。
def get_checkpoint_state_mode(checkpoint_path: Path) -> str:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    return str(checkpoint.get("state_mode", "vector"))


def get_checkpoint_algorithm(checkpoint_path: Path) -> str:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    algorithm = checkpoint.get("algorithm")
    if algorithm is None:
        run_config = checkpoint.get("run_config")
        if isinstance(run_config, dict):
            algorithm = run_config.get("algorithm")
    if algorithm is None:
        return "dqn"
    algorithm = str(algorithm)
    if algorithm not in ("dqn", "ppo"):
        raise ValueError(f"Unsupported checkpoint algorithm={algorithm!r}")
    return algorithm


def get_checkpoint_environment(checkpoint_path: Path) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    run_config = checkpoint.get("run_config")
    if not isinstance(run_config, dict):
        return {}
    environment = run_config.get("environment")
    return dict(environment) if isinstance(environment, dict) else {}


# 校验评估参数并构造最终配置。
def build_configs(
    args: argparse.Namespace,
    checkpoint_path: Path | None = None,
) -> tuple[TrainConfig, EnvConfig]:

    checkpoint_environment = (
        get_checkpoint_environment(checkpoint_path) if checkpoint_path is not None else {}
    )
    width = args.width if args.width is not None else checkpoint_environment.get("width", EnvConfig.width)
    height = (
        args.height if args.height is not None else checkpoint_environment.get("height", EnvConfig.height)
    )
    if args.episodes < 1:
        raise ValueError("episodes must be at least 1")
    if args.max_steps < 1:
        raise ValueError("max_steps must be at least 1")
    if width < 5 or height < 5:
        raise ValueError("width and height must be at least 5")
    if args.cell_size < 1 or args.fps < 1:
        raise ValueError("cell_size and fps must be positive")

    train_config = TrainConfig(seed=args.seed)
    env_config = EnvConfig(
        width=int(width),
        height=int(height),
        cell_size=args.cell_size,
        fps=args.fps,
    )
    return train_config, env_config


def main() -> None:
    args = parse_args()
    # 如果命令行没有指定 checkpoint，就默认评估最近一次训练的 latest.pt。
    checkpoint_path = args.checkpoint or find_latest_checkpoint(algorithm=args.algorithm)
    train_config, env_config = build_configs(args, checkpoint_path)
    set_seed(train_config.seed)
    algorithm = get_checkpoint_algorithm(checkpoint_path)
    if args.algorithm is not None and algorithm != args.algorithm:
        raise ValueError(
            f"Checkpoint algorithm={algorithm!r} does not match --algorithm {args.algorithm!r}"
        )
    # 默认使用 checkpoint 记录的模式，避免手动选择错误的网络输入结构。
    state_mode = args.state_mode or get_checkpoint_state_mode(checkpoint_path)

    # ---- 确定评估日志输出目录 ----
    output_dir: Path | None = None
    if args.tensorboard:
        output_dir = args.eval_output_dir or find_run_dir_for_checkpoint(checkpoint_path)
        output_dir.mkdir(parents=True, exist_ok=True)

    # ---- TensorBoard ----
    writer = (
        SummaryWriter(output_dir, filename_suffix=".eval")
        if (SummaryWriter is not None and output_dir is not None)
        else None
    )

    # ---- CSV ----
    csv_path = output_dir / "eval_metrics.csv" if output_dir is not None else None

    print(f"checkpoint={checkpoint_path}")
    print(f"algorithm={algorithm}")
    print(f"state_mode={state_mode}")
    if output_dir is not None:
        print(f"output_dir={output_dir}")

    env = SnakeEnv(
        width=env_config.width,
        height=env_config.height,
        # 默认开启渲染
        render_mode=not args.no_render,
        # 一个格子的像素个数
        cell_size=env_config.cell_size,
        fps=env_config.fps,
        seed=train_config.seed,
        # 对齐 chynl/snake 的 benchmark：评估时不使用逐食物 starvation。
        starvation_enabled=False,
        # 与训练保持一致，由环境直接返回 checkpoint 所需模式的 observation。
        state_mode=state_mode,
    )
    common_agent_kwargs = dict(
        # 状态维度
        state_size=(
            env.grid_state_shape if state_mode in ("grid", "hybrid") else env.state_size
        ),
        # 动作维度
        action_size=env.action_size,
        hidden_size=train_config.hidden_size,
        state_mode=state_mode,
        # load() 会在必要时使用 checkpoint 中的 CNN 参数重建当前默认网络。
        auxiliary_size=env.state_size,
        cnn_channels=train_config.cnn_channels,
        cnn_output_channels=train_config.cnn_output_channels,
        cnn_dilations=train_config.cnn_dilations,
        seed=train_config.seed,
    )
    if algorithm == "ppo":
        agent = PPOAgent(**common_agent_kwargs)
    else:
        agent = DQNAgent(
            **common_agent_kwargs,
            epsilon_start=0.0,
            epsilon_end=0.0,
        )

    # 加载模型参数用于测试
    agent.load(checkpoint_path)

    scores: list[int] = []
    steps: list[int] = []
    max_lengths: list[int] = []
    score_per_steps: list[float] = []
    # 累计和用于绘制稳定的 running mean；第 N 个点表示前 N 局的真实平均值。
    running_totals = {
        "score": 0.0,
        "steps": 0.0,
        "score_per_step": 0.0,
        "max_snake_length": 0.0,
    }
    csv_file = None
    metrics = None

    # 记录评估过程中的结果，打印并写入csv和tensorboard.
    def record_episode(episode: int, result: ValidationEpisode) -> None:
        # episode：当前是第几局评估，例如第 1 局、第 2 局。
        # 这一局的评估结果，包含得分、步数、最大蛇长、是否超时等信息。
        
        scores.append(result.score)
        steps.append(result.steps)
        max_lengths.append(result.max_snake_length)
        score_per_steps.append(result.score_per_step)
        running_totals["score"] += result.score
        running_totals["steps"] += result.steps
        running_totals["score_per_step"] += result.score_per_step
        running_totals["max_snake_length"] += result.max_snake_length

        print(
            f"episode={episode:4d}  score={result.score:3d}  "
            f"steps={result.steps:4d}  max_len={result.max_snake_length}  "
            f"eff={result.score_per_step:.4f}  timed_out={result.timed_out}"
        )

        if writer is not None:
            writer.add_scalar("eval/score", result.score, episode)
            writer.add_scalar("eval/steps", result.steps, episode)
            writer.add_scalar("eval/score_per_step", result.score_per_step, episode)
            writer.add_scalar(
                "eval/max_snake_length",
                result.max_snake_length,
                episode,
            )
            writer.add_scalar("eval/timed_out", int(result.timed_out), episode)
            for metric_name, total in running_totals.items():
                writer.add_scalar(
                    f"eval_running_mean/{metric_name}",
                    total / episode,
                    episode,
                )

        if metrics is not None:
            metrics.writerow(
                [
                    episode,
                    result.score,
                    result.steps,
                    f"{result.score_per_step:.6f}",
                    result.max_snake_length,
                ]
            )
            csv_file.flush()

    try:
        if csv_path is not None:
            csv_file, metrics = open_eval_metrics_csv(csv_path)

        seeds = make_episode_seeds(train_config.seed, "final", args.episodes)
        result = evaluate_policy(
            agent,
            env,
            seeds,
            seed_set="final",
            max_steps=args.max_steps,
            on_episode=record_episode,
        )
        if writer is not None:
            writer.add_scalar("eval_summary/full_score", result.full_score, result.episodes)
            writer.add_scalar("eval_summary/full_games", result.full_games, result.episodes)
            writer.add_scalar("eval_summary/full_rate", result.full_rate, result.episodes)
            writer.add_scalar(
                "eval_summary/timeout_rate",
                result.timeout_rate,
                result.episodes,
            )
            writer.add_text(
                "eval/report",
                format_eval_report(
                    checkpoint_path=checkpoint_path,
                    episodes=result.episodes,
                    total_time_sec=result.total_time_sec,
                    scores=scores,
                    steps=steps,
                    score_per_steps=score_per_steps,
                    max_lengths=max_lengths,
                    full_score=result.full_score,
                ),
                global_step=result.episodes,
            )
    finally:
        env.close()
        if writer is not None:
            writer.close()
        if csv_file is not None:
            csv_file.close()

    print(
        f"average_score={result.mean_score:.2f}  best_score={result.max_score}  "
        f"avg_steps={result.mean_steps:.1f}  "
        f"avg_eff={result.mean_score_per_step:.4f}  "
        f"avg_max_len={result.mean_max_snake_length:.2f}  "
        f"full_score={result.full_score}  "
        f"full_games={result.full_games}/{result.episodes}  "
        f"full_rate={result.full_rate * 100:.2f}%  "
        f"timeout_rate={result.timeout_rate * 100:.2f}%"
    )


if __name__ == "__main__":
    main()
