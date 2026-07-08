# 运行方法: uv run src/snake_ai/evaluate.py
# 如果要可视化 --tensorboard
# 默认会渲染，不想渲染可以加上 --no-render

# 处理自引用问题
from __future__ import annotations  

import argparse
import csv
from datetime import datetime
from pathlib import Path

from snake_ai.agents import DQNAgent
from snake_ai.config import CHECKPOINT_DIR, RUNS_DIR, EnvConfig, TrainConfig
from snake_ai.game import SnakeEnv
from snake_ai.utils import set_seed

try:
    from torch.utils.tensorboard import SummaryWriter
# 如果用户没装 TensorBoard，就让它等于 None。
except ImportError:
    SummaryWriter = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained Snake DQN agent.")
    # 不传时默认加载 checkpoints/<最新 dqn_*>/best.pt；显式传入时使用用户指定路径。
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--width", type=int, default=EnvConfig.width)
    parser.add_argument("--height", type=int, default=EnvConfig.height)
    # 是否把本次评估的分数写入 CSV 和 TensorBoard。
    parser.add_argument("--tensorboard", action="store_true")
    # 指定评估指标输出目录；不指定时默认绑定到最近一次训练的 runs/dqn_* 目录。
    parser.add_argument("--eval-output-dir", type=Path, default=None)
    return parser.parse_args()


def find_latest_run_dir(runs_dir: Path = RUNS_DIR) -> Path:
    # 训练脚本会生成 runs/dqn_YYYYMMDD_HHMMSS 目录，这里按目录名时间取最新的一个。
    run_dirs = [path for path in runs_dir.glob("dqn_*") if path.is_dir()]
    if not run_dirs:
        raise FileNotFoundError(f"No training run directory found in {runs_dir}")
    return max(run_dirs, key=lambda path: path.name)


def find_latest_best_checkpoint(checkpoint_dir: Path = CHECKPOINT_DIR) -> Path:
    # 新训练会保存到 checkpoints/dqn_YYYYMMDD_HHMMSS/best.pt，这里按目录名时间取最新模型。
    checkpoint_dirs = [path for path in checkpoint_dir.glob("dqn_*") if path.is_dir()]
    if not checkpoint_dirs:
        legacy_checkpoint = checkpoint_dir / "best.pt"
        if legacy_checkpoint.exists():
            return legacy_checkpoint
        raise FileNotFoundError(f"No checkpoint directory found in {checkpoint_dir}")

    latest_dir = max(checkpoint_dirs, key=lambda path: path.name)
    latest_checkpoint = latest_dir / "best.pt"
    if not latest_checkpoint.exists():
        raise FileNotFoundError(f"No best.pt found in latest checkpoint directory {latest_dir}")
    return latest_checkpoint


def get_next_eval_step(csv_path: Path) -> int:
    # 多次评估会追加到同一个 CSV；这里用已有行数推算下一次 TensorBoard 的起始 step。
    if not csv_path.exists():
        return 1

    with csv_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        global_steps = [
            int(row["global_step"]) for row in reader if row.get("global_step", "").isdigit()
        ]

    return max(global_steps, default=0) + 1


def save_eval_metrics(
    scores: list[int],
    output_dir: Path,
    checkpoint: Path,
    eval_run: str,
    start_step: int,
) -> Path:
    # 把每一局的 score 追加写入 CSV，避免多次评估时覆盖历史测试结果。
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "eval_metrics.csv"
    average_score = sum(scores) / len(scores) if scores else 0.0
    best_score = max(scores) if scores else 0
    should_write_header = not csv_path.exists()

    with csv_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if should_write_header:
            writer.writerow(
                [
                    "eval_run",
                    "global_step",
                    "episode",
                    "score",
                    "checkpoint",
                    "average_score",
                    "best_score",
                ]
            )
        for episode, score in enumerate(scores, start=1):
            global_step = start_step + episode - 1
            writer.writerow(
                [
                    eval_run,
                    global_step,
                    episode,
                    score,
                    checkpoint,
                    f"{average_score:.4f}",
                    best_score,
                ]
            )

    return csv_path


def write_eval_tensorboard(
    scores: list[int],
    output_dir: Path,
    eval_run: str,
    start_step: int,
) -> None:
    # 评估结果写入同一个 run 目录，但只写一张 figure，避免均分/最高分被 TensorBoard 拆成单独图。
    if SummaryWriter is None:
        raise ImportError("TensorBoard is not installed. Please install tensorboard first.")

    # matplotlib 只在需要写 TensorBoard 评估图时导入，普通评估不额外加载绘图库。
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    average_score = sum(scores) / len(scores) if scores else 0.0
    best_score = max(scores) if scores else 0
    final_step = start_step + len(scores) - 1
    episodes = list(range(1, len(scores) + 1))

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(episodes, scores, marker="o", linewidth=1.8, label="score")
    ax.axhline(average_score, color="tab:orange", linestyle="--", label=f"avg={average_score:.2f}")
    ax.axhline(best_score, color="tab:green", linestyle=":", label=f"best={best_score}")
    ax.set_title("Evaluation Scores")
    ax.set_xlabel("Evaluation Episode")
    ax.set_ylabel("Score")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    # 评估局数变多时，不要每一局都显示刻度，否则 TensorBoard 里的图会挤成一团。
    if episodes:
        tick_interval = max(1, len(episodes) // 10)
        xticks = episodes[::tick_interval]
        if xticks[-1] != episodes[-1]:
            xticks.append(episodes[-1])
        ax.set_xticks(xticks)

    # TensorBoard 的 scalar 平滑会让短序列比例尺看起来异常；figure 这里手动给真实分数留边界。
    if scores:
        min_score = min(scores)
        max_score = max(scores)
        padding = max(1.0, (max_score - min_score) * 0.1)
        ax.set_ylim(min_score - padding, max_score + padding)

    fig.tight_layout()

    # 给评估 event 文件加上简短后缀，避免和训练 event 文件都叫 events.out.tfevents.* 而难以区分。
    writer = SummaryWriter(output_dir, filename_suffix=".eval")
    try:
        writer.add_figure("eval/scores", fig, final_step)
    finally:
        writer.close()
        plt.close(fig)


def main() -> None:
    args = parse_args()
    train_config = TrainConfig()
    set_seed(train_config.seed)
    env_config = EnvConfig(width=args.width, height=args.height)
    # 如果命令行没有指定 checkpoint，就默认评估最近一次训练的 best.pt。
    checkpoint_path = args.checkpoint or find_latest_best_checkpoint()
    print(f"checkpoint={checkpoint_path}")
    env = SnakeEnv(
        width=env_config.width,
        height=env_config.height,
        # 默认开启渲染
        render_mode=not args.no_render,  
        # 一个格子的像素个数
        cell_size=env_config.cell_size,  
        fps=env_config.fps,
        seed=train_config.seed,
    )
    agent = DQNAgent(
        # 状态维度
        state_size=env.state_size,  
        # 动作维度
        action_size=env.action_size,  
        hidden_size=train_config.hidden_size,
        # 起始epsilon值(Epsilon-Greedy在评估时关闭)
        epsilon_start=0.0,  
        # epsilon值的下限(评估时Epsilon-Greedy关闭)
        epsilon_end=0.0,  
        seed=train_config.seed,
    )
    
    # 加载模型参数用于测试
    agent.load(checkpoint_path)  
    
    # 记录每个episode的分数
    scores: list[int] = []  
    try:
        for episode in range(1, args.episodes + 1):  
            state = env.reset()
            done = False
            info = {"score": 0}
            
            # 评估时只进行动作采样和环境反馈。
            while not done:
                action = agent.act(state, training=False)
                state, _, done, info = env.step(action)
            # 记录这次episode的分数
            scores.append(int(info["score"]))
            
            # 每个episode输出一次信息。
            print(f"episode={episode} score={info['score']}")
    finally:
        env.close()

    # 这里输出的是本次评估的平均分和最高分
    if scores:
        print(f"average_score={sum(scores) / len(scores):.2f} best_score={max(scores)}")

    if args.tensorboard:
        # 没有手动指定目录时，评估指标默认写入最近一次训练目录，方便和训练指标放在一起看。
        output_dir = args.eval_output_dir or find_latest_run_dir()
        eval_run = datetime.now().strftime("eval_%Y%m%d_%H%M%S")
        csv_path = output_dir / "eval_metrics.csv"
        start_step = get_next_eval_step(csv_path)
        metrics_path = save_eval_metrics(scores, output_dir, checkpoint_path, eval_run, start_step)
        write_eval_tensorboard(scores, output_dir, eval_run, start_step)
        print(f"eval_metrics={metrics_path}")
        print(f"eval_tensorboard_dir={output_dir}")


if __name__ == "__main__":
    main()
