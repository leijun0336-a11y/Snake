from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

from snake_ai.agents import DQNAgent
from snake_ai.config import CHECKPOINT_DIR, RUNS_DIR, EnvConfig, TrainConfig
from snake_ai.game import SnakeEnv

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover
    SummaryWriter = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a DQN agent for Snake.")
    parser.add_argument("--episodes", type=int, default=TrainConfig.episodes)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--width", type=int, default=EnvConfig.width)
    parser.add_argument("--height", type=int, default=EnvConfig.height)
    parser.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR)
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_config = TrainConfig(episodes=args.episodes)
    env_config = EnvConfig(width=args.width, height=args.height)

    run_name = datetime.now().strftime("dqn_%Y%m%d_%H%M%S")
    run_dir = args.runs_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(run_dir) if SummaryWriter is not None else None
    csv_path = run_dir / "metrics.csv"

    env = SnakeEnv(
        width=env_config.width,
        height=env_config.height,
        render_mode=args.render,
        cell_size=env_config.cell_size,
        fps=env_config.fps,
        seed=train_config.seed,
    )
    agent = DQNAgent(
        state_size=env.state_size,
        action_size=env.action_size,
        hidden_size=train_config.hidden_size,
        learning_rate=train_config.learning_rate,
        gamma=train_config.gamma,
        replay_buffer_size=train_config.replay_buffer_size,
        batch_size=train_config.batch_size,
        epsilon_start=train_config.epsilon_start,
        epsilon_end=train_config.epsilon_end,
        epsilon_decay=train_config.epsilon_decay,
        target_update_interval=train_config.target_update_interval,
        seed=train_config.seed,
    )

    scores: list[int] = []
    best_score = -1

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        metrics = csv.writer(file)
        metrics.writerow(["episode", "score", "mean_score_100", "epsilon", "loss"])

        try:
            for episode in range(1, train_config.episodes + 1):
                state = env.reset()
                done = False
                losses: list[float] = []

                while not done:
                    action = agent.act(state, training=True)
                    next_state, reward, done, info = env.step(action)
                    agent.remember(state, action, reward, next_state, done)
                    loss = agent.learn()
                    if loss is not None:
                        losses.append(loss)
                    state = next_state

                agent.decay_epsilon()
                score = int(info["score"])
                scores.append(score)
                mean_score = sum(scores[-100:]) / min(len(scores), 100)
                mean_loss = sum(losses) / len(losses) if losses else 0.0

                if writer is not None:
                    writer.add_scalar("score", score, episode)
                    writer.add_scalar("mean_score_100", mean_score, episode)
                    writer.add_scalar("epsilon", agent.epsilon, episode)
                    writer.add_scalar("loss", mean_loss, episode)

                metrics.writerow([episode, score, f"{mean_score:.4f}", f"{agent.epsilon:.6f}", mean_loss])
                file.flush()

                if score > best_score:
                    best_score = score
                    agent.save(args.checkpoint_dir / "best.pt")

                agent.save(args.checkpoint_dir / "latest.pt")

                print(
                    f"episode={episode:4d} score={score:3d} "
                    f"mean100={mean_score:6.2f} epsilon={agent.epsilon:.3f} "
                    f"loss={mean_loss:.4f} best={best_score}"
                )
        finally:
            env.close()
            if writer is not None:
                writer.close()


if __name__ == "__main__":
    main()
