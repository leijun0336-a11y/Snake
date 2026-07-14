# 运行方法: uv run src/snake_ai/evaluate.py
# 如果要可视化 --tensorboard
# 默认会渲染，不想渲染可以加上 --no-render

# 处理自引用问题
from __future__ import annotations  

import argparse
import csv
import time
from pathlib import Path

import torch

from snake_ai.agents import DQNAgent
from snake_ai.config import CHECKPOINT_DIR, RUNS_DIR, EnvConfig, TrainConfig
from snake_ai.game import SnakeEnv
from snake_ai.utils import set_seed, summarize_values

try:
    from torch.utils.tensorboard import SummaryWriter
# 如果用户没装 TensorBoard，就让它等于 None。
except ImportError:
    SummaryWriter = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained Snake DQN agent.")
    # 不传时默认加载 checkpoints/<最新 dqn_*>/latest.pt；显式传入时使用用户指定路径。
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--episodes", type=int, default=1000)
    # 防止转圈。
    parser.add_argument("--max-steps",type=int,default=1000)
    # 不渲染，默认渲染。
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--width", type=int, default=EnvConfig.width)
    parser.add_argument("--height", type=int, default=EnvConfig.height)
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

# 默认提取最近一次实验的 runs/dqn_YYYYMMDD_HHMMSS 目录，若没有则报错。
def find_latest_run_dir(runs_dir: Path = RUNS_DIR) -> Path:
    # 训练脚本会生成 runs/dqn_YYYYMMDD_HHMMSS 目录，这里按目录名时间取最新的一个。
    run_dirs = [path for path in runs_dir.glob("dqn_*") if path.is_dir()]
    if not run_dirs:
        raise FileNotFoundError(f"No training run directory found in {runs_dir}")
    return max(run_dirs, key=lambda path: path.name)


# 默认提取最近一次实验的权重，权重默认取latest.pt，若没有latest.pt则取best.pt
def find_latest_checkpoint(checkpoint_dir: Path = CHECKPOINT_DIR) -> Path:
    checkpoint_dirs = [path for path in checkpoint_dir.glob("dqn_*") if path.is_dir()]
    if not checkpoint_dirs:
        legacy_checkpoint = checkpoint_dir / "latest.pt"
        if legacy_checkpoint.exists():
            return legacy_checkpoint
        raise FileNotFoundError(f"No checkpoint directory found in {checkpoint_dir}")

    latest_dir = max(checkpoint_dirs, key=lambda path: path.name)
    latest_checkpoint = latest_dir / "latest.pt"
    if not latest_checkpoint.exists():
        raise FileNotFoundError(
            f"No latest.pt found in latest checkpoint directory {latest_dir}"
        )
    return latest_checkpoint


# 确定评估结果的tensorboard文件存储在哪个文件夹下。
def find_run_dir_for_checkpoint(checkpoint_path: Path, runs_dir: Path = RUNS_DIR) -> Path:

    run_name = checkpoint_path.parent.name
    if run_name.startswith("dqn_"):
        return runs_dir / run_name
    return find_latest_run_dir(runs_dir)


# 把一次评估的统计结果整理成 Markdown 格式的文本报告，放入tensorboard的 text 标签页中。
def format_eval_report(
    checkpoint_path: Path,
    episodes: int,
    total_time_sec: float,
    scores: list[int],
    steps: list[int],
    score_per_steps: list[float],
    max_lengths: list[int],
) -> str:
    summaries = {
        "score": summarize_values(scores),
        "steps": summarize_values(steps),
        "score_per_step": summarize_values(score_per_steps),
        "max_snake_length": summarize_values(max_lengths),
    }
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
    return "\n".join(lines)


# 读取 checkpoint 中记录的状态输入模式。
def get_checkpoint_state_mode(checkpoint_path: Path) -> str:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    return str(checkpoint.get("state_mode", "vector"))


# 校验评估参数并构造最终配置。
def build_configs(args: argparse.Namespace) -> tuple[TrainConfig, EnvConfig]:


    if args.episodes < 1:
        raise ValueError("episodes must be at least 1")
    if args.max_steps < 1:
        raise ValueError("max_steps must be at least 1")
    if args.width < 5 or args.height < 5:
        raise ValueError("width and height must be at least 5")
    if args.cell_size < 1 or args.fps < 1:
        raise ValueError("cell_size and fps must be positive")

    train_config = TrainConfig(seed=args.seed)
    env_config = EnvConfig(
        width=args.width,
        height=args.height,
        cell_size=args.cell_size,
        fps=args.fps,
    )
    return train_config, env_config


def main() -> None:
    args = parse_args()
    train_config, env_config = build_configs(args)
    set_seed(train_config.seed)
    
    # 如果命令行没有指定 checkpoint，就默认评估最近一次训练的 latest.pt。
    checkpoint_path = args.checkpoint or find_latest_checkpoint()
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
    agent = DQNAgent(
        # 状态维度
        state_size=(
            env.grid_state_shape if state_mode in ("grid", "hybrid") else env.state_size
        ),
        # 动作维度
        action_size=env.action_size,
        hidden_size=train_config.hidden_size,
        # 起始epsilon值(Epsilon-Greedy在评估时关闭)
        epsilon_start=0.0,
        # epsilon值的下限(评估时Epsilon-Greedy关闭)
        epsilon_end=0.0,
        state_mode=state_mode,
        # load() 会在必要时使用 checkpoint 中的 CNN 参数重建当前默认网络。
        auxiliary_size=env.state_size,
        cnn_channels=train_config.cnn_channels,
        cnn_output_channels=train_config.cnn_output_channels,
        cnn_dilations=train_config.cnn_dilations,
        cnn_pool_size=train_config.cnn_pool_size,
        seed=train_config.seed,
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
    eval_start_time = time.perf_counter()
    try:
        if csv_path is not None:
            csv_file = csv_path.open("a", newline="", encoding="utf-8")
            metrics = csv.writer(csv_file)
            # 首次创建 CSV 时写入表头
            if csv_path.stat().st_size == 0:
                metrics.writerow(
                    [
                        "episode",
                        "score",
                        "steps",
                        "score_per_step",
                        "max_snake_length",
                    ]
                )

        for episode in range(1, args.episodes + 1):
            state = env.reset()
            done = False
            info = {"score": 0}
            # 每局开始时蛇身长度为3(初始值)，后续吃到食物会增长
            max_snake_length = len(env.snake)
            evaluation_steps = 0

            # 评估时只进行动作采样和环境反馈。
            while not done and evaluation_steps < args.max_steps:
                action = agent.act(state, training=False)
                state, _, done, info = env.step(action)
                evaluation_steps += 1
                # 每步记录蛇身长度，追踪本局峰值
                current_length = int(info["snake_length"])
                if current_length > max_snake_length:
                    max_snake_length = current_length

            # 记录这次episode的各项指标
            score = int(info["score"])
            episode_steps = int(info["steps"])
            scores.append(score)
            steps.append(episode_steps)
            max_lengths.append(max_snake_length)
            # 吃食效率 = 吃到的食物数 / 存活步数
            score_per_step = score / episode_steps if episode_steps > 0 else 0.0
            score_per_steps.append(score_per_step)
            running_totals["score"] += score
            running_totals["steps"] += episode_steps
            running_totals["score_per_step"] += score_per_step
            running_totals["max_snake_length"] += max_snake_length

            # 每个episode输出一次信息。
            print(
                f"episode={episode:4d}  score={score:3d}  "
                f"steps={episode_steps:4d}  max_len={max_snake_length}  "
                f"eff={score_per_step:.4f}"
            )

            if writer is not None:
                # 单局得分
                writer.add_scalar("eval/score", score, episode)
                # 存活步数
                writer.add_scalar("eval/steps", episode_steps, episode)
                # 吃食效率
                writer.add_scalar("eval/score_per_step", score_per_step, episode)
                # 本局最大蛇身长度
                writer.add_scalar("eval/max_snake_length", max_snake_length, episode)
                # 累计均值比独立 episode 折线稳定，最后一点等于本次评估总体均值。
                for metric_name, total in running_totals.items():
                    writer.add_scalar(
                        f"eval_running_mean/{metric_name}",
                        total / episode,
                        episode,
                    )

            if csv_path is not None:
                metrics.writerow(
                    [
                        episode,
                        score,
                        episode_steps,
                        f"{score_per_step:.6f}",
                        max_snake_length,
                    ]
                )
                csv_file.flush()
        if scores and writer is not None:
            total_time_sec = time.perf_counter() - eval_start_time
            writer.add_text(
                "eval/report",
                format_eval_report(
                    checkpoint_path=checkpoint_path,
                    episodes=len(scores),
                    total_time_sec=total_time_sec,
                    scores=scores,
                    steps=steps,
                    score_per_steps=score_per_steps,
                    max_lengths=max_lengths,
                ),
                global_step=len(scores),
            )
    finally:
        env.close()
        if writer is not None:
            writer.close()
        if csv_file is not None:
            csv_file.close()

    # 输出本次评估的汇总统计
    if scores:
        avg_score = sum(scores) / len(scores)
        avg_steps = sum(steps) / len(steps)
        avg_eff = sum(score_per_steps) / len(score_per_steps)
        avg_max_len = sum(max_lengths) / len(max_lengths)
        print(
            f"average_score={avg_score:.2f}  best_score={max(scores)}  "
            f"avg_steps={avg_steps:.1f}  avg_eff={avg_eff:.4f}  "
            f"avg_max_len={avg_max_len:.2f}"
        )


if __name__ == "__main__":
    main()
