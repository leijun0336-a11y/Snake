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
# 如果用户没装 TensorBoard，就让它等于 None
except ImportError:  
    SummaryWriter = None

# 解析启动脚本时的命令行参数
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a DQN agent for Snake.")
    parser.add_argument("--episodes", type=int, default=TrainConfig.epidddsodes)
    # 是否带渲染训练, action="store_true"表示：启动脚本时写上--render则为True，不写默认为False
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--width", type=int, default=EnvConfig.width)
    parser.add_argument("--height", type=int, default=EnvConfig.height)
    parser.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR)
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_config = TrainConfig(episodes=args.episodes)
    set_seed(train_config.seed)
    env_config = EnvConfig(width=args.width, height=args.height)

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
    )
    agent = DQNAgent(
        # 状态维度
        state_size=env.state_size,  
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
        # epsilon的衰减系数
        epsilon_decay=train_config.epsilon_decay,  
        # 隔多少步更新一次目标网络
        # Q训练网络更新一次视为一步，当经验池满后，则等价于贪吃蛇走一步
        target_update_interval=train_config.target_update_interval,   
        seed=train_config.seed,
    )

    scores: list[int] = []
    mean_losses: list[float] = []
    best_score = -1

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        metrics = csv.writer(file)
        metrics.writerow(
            [
                "episode",
                "score",
                "score_per_step",
                "best_score",
                "mean_score_100",
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

                # 一次episode训练
                while not done:
                    # 训练时的动作采样
                    action = agent.act(state, training=True)
                    # 环境反馈，info是环境额外返回的信息字典，不直接参与DQN更新
                    next_state, reward, done, info = env.step(action)
                    # 加入经验回放池
                    agent.remember(state, action, reward, next_state, done)
                    # 智能体更新Q值，如果经验回放池没达到batch_size则返回None
                    loss = agent.learn()
                    if loss is not None:
                        losses.append(loss)
                    state = next_state

                # 每个episode执行一次epsilon衰减
                agent.decay_epsilon()
                # 从环境返回的额外信息info中提取游戏分数字段的值
                score = int(info["score"])
                # 记录游戏分数到列表中
                scores.append(score)
                # 记录最近最多100次获得的游戏的平均分
                # 注意游戏的分数和环境的奖励是不同的概念，一个是指标，一个是训练信号
                mean_score = sum(scores[-100:]) / min(len(scores), 100)
                # 吃食效率 = 吃到的食物数 / 存活步数，衡量策略是否直奔目标
                score_per_step = score / episode_steps if episode_steps > 0 else 0.0
                # 一个episode中产生的所有损失求平均。
                mean_loss = sum(losses) / len(losses) if losses else 0.0
                mean_losses.append(mean_loss)
                mean_loss_100 = sum(mean_losses[-100:]) / min(len(mean_losses), 100)
                # episode_steps 用来区分“很快撞死”和“走了很久但没吃到食物”。
                episode_steps = env.frame_iteration

                # 存下得分最高时的参数
                if score > best_score:
                    best_score = score
                    agent.save(checkpoint_dir / "best.pt")

                if writer is not None:
                    # 单局得分
                    writer.add_scalar("train/score", score, episode)
                    # 吃食效率 = 吃到的食物数 / 存活步数
                    writer.add_scalar("train/score_per_step", score_per_step, episode)
                    # 历史最高分
                    writer.add_scalar("train/best_score", best_score, episode)
                    # 最近100局滑动平均分
                    writer.add_scalar("train/mean_score_100", mean_score, episode)
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
                        best_score,
                        f"{mean_score:.4f}",
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
                agent.save(checkpoint_dir / "latest.pt")

                # 每个episode输出一次指标信息
                print(
                    f"episode={episode:4d} score={score:3d} "
                    f"mean100={mean_score:6.2f} epsilon={agent.epsilon:.3f} "
                    f"loss={mean_loss:.4f} best={best_score}"
                )
        finally:  # 无论上面是否跑完，这段代码都必须执行。
            env.close()
            if writer is not None:
                writer.close()


if __name__ == "__main__":
    main()
