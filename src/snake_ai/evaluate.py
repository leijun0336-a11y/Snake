from __future__ import annotations

import argparse
from pathlib import Path

from snake_ai.agents import DQNAgent
from snake_ai.config import CHECKPOINT_DIR, EnvConfig, TrainConfig
from snake_ai.game import SnakeEnv


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
    env_config = EnvConfig(width=args.width, height=args.height)
    env = SnakeEnv(
        width=env_config.width,
        height=env_config.height,
        render_mode=not args.no_render,
        cell_size=env_config.cell_size,
        fps=env_config.fps,
        seed=train_config.seed,
    )
    agent = DQNAgent(
        state_size=env.state_size,
        action_size=env.action_size,
        hidden_size=train_config.hidden_size,
        epsilon_start=0.0,
        epsilon_end=0.0,
        seed=train_config.seed,
    )
    agent.load(args.checkpoint)

    scores: list[int] = []
    try:
        for episode in range(1, args.episodes + 1):
            state = env.reset()
            done = False
            info = {"score": 0}
            while not done:
                action = agent.act(state, training=False)
                state, _, done, info = env.step(action)
            scores.append(int(info["score"]))
            print(f"episode={episode} score={info['score']}")
    finally:
        env.close()

    if scores:
        print(f"average_score={sum(scores) / len(scores):.2f} best_score={max(scores)}")


if __name__ == "__main__":
    main()
