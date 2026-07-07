# 处理自引用问题
from __future__ import annotations  

import argparse
from pathlib import Path

from snake_ai.agents import DQNAgent
from snake_ai.config import CHECKPOINT_DIR, EnvConfig, TrainConfig
from snake_ai.game import SnakeEnv
from snake_ai.utils import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained Snake DQN agent.")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_DIR / "best.pt")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--width", type=int, default=EnvConfig.width)
    parser.add_argument("--height", type=int, default=EnvConfig.height)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_config = TrainConfig()
    set_seed(train_config.seed)
    env_config = EnvConfig(width=args.width, height=args.height)
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
    agent.load(args.checkpoint)  
    
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


if __name__ == "__main__":
    main()
