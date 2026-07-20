from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from snake_ai.agents import DQNAgent
from snake_ai.game import SnakeEnv
from snake_ai.planning_10x10 import (
    DecisionTier,
    PlannedDQNPolicy10x10,
    Planner10x10Config,
    PlanningState,
    StrictSafePlanner10x10,
)
from snake_ai.planning_10x10.metrics import PlannedEpisodeResult, PlannerEpisodeMetrics
from snake_ai.utils import set_seed
from snake_ai.validation import make_episode_seeds


PROTOCOLS = ("legacy-final", "training-semantics")


@dataclass(frozen=True, slots=True)
class PlannedEvaluationSummary:
    checkpoint: str
    protocol: str
    episodes: int
    max_steps: int
    mean_score: float
    score_std: float
    max_score: int
    completion_rate: float
    timeout_rate: float
    override_rate: float
    hamiltonian_cycle_rate: float
    planner_mean_ms: float
    planner_max_ms: float
    total_time_sec: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the strict 10x10 tail-safe A* + Hamiltonian DQN policy."
    )
    # 必须显式指定，禁止从最新目录、best.pt 或其他模型自动回退。
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--protocol", choices=PROTOCOLS, default="training-semantics")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-astar-expansions", type=int, default=500)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def resolve_max_steps(protocol: str, requested: int | None) -> int:
    if protocol not in PROTOCOLS:
        raise ValueError(f"unknown evaluation protocol: {protocol}")
    value = requested if requested is not None else (1_000 if protocol == "legacy-final" else 5_000)
    if value < 1:
        raise ValueError("max_steps must be positive")
    return value


def validate_checkpoint(checkpoint_path: Path) -> dict[str, Any]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"10x10 checkpoint does not exist: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError("10x10 planner checkpoint must contain a metadata dictionary")
    if checkpoint.get("architecture_version") != DQNAgent.ARCHITECTURE_VERSION:
        raise ValueError("10x10 planner requires an architecture_version=3 checkpoint")
    if checkpoint.get("state_size") != (SnakeEnv.grid_channels, 10, 10):
        raise ValueError("10x10 planner checkpoint must use state_size=(9, 10, 10)")
    if checkpoint.get("state_mode") != "hybrid":
        raise ValueError("10x10 planner checkpoint must use hybrid state mode")
    if checkpoint.get("action_size") != SnakeEnv.action_size:
        raise ValueError("10x10 planner checkpoint must use exactly three actions")

    run_config = checkpoint.get("run_config")
    if not isinstance(run_config, dict):
        raise ValueError("10x10 planner checkpoint is missing run_config")
    environment = run_config.get("environment")
    reward = run_config.get("reward")
    if not isinstance(environment, dict) or not isinstance(reward, dict):
        raise ValueError("10x10 planner checkpoint has incomplete environment/reward metadata")
    expected_environment = {
        "width": 10,
        "height": 10,
        "state_mode": "hybrid",
        "starvation_enabled": True,
    }
    actual_environment = {key: environment.get(key) for key in expected_environment}
    if actual_environment != expected_environment:
        raise ValueError(f"10x10 planner checkpoint environment mismatch: {actual_environment}")
    if reward.get("profile") != "experiment8":
        raise ValueError("10x10 planner checkpoint must use reward profile 'experiment8'")
    starvation_semantics = {
        "starvation_limit_mode": reward.get("starvation_limit_mode"),
        "starvation_comparison": reward.get("starvation_comparison"),
    }
    if starvation_semantics != {
        "starvation_limit_mode": "board_area",
        "starvation_comparison": "gt",
    }:
        raise ValueError(
            f"10x10 planner checkpoint starvation semantics mismatch: {starvation_semantics}"
        )
    return checkpoint


def load_policy(
    checkpoint_path: Path,
    *,
    max_astar_expansions: int,
) -> PlannedDQNPolicy10x10:
    validate_checkpoint(checkpoint_path)
    agent = DQNAgent(
        state_size=(SnakeEnv.grid_channels, 10, 10),
        action_size=SnakeEnv.action_size,
        epsilon_start=0.0,
        epsilon_end=0.0,
        state_mode="hybrid",
        auxiliary_size=SnakeEnv.state_size,
    )
    agent.load(checkpoint_path)
    planner = StrictSafePlanner10x10(Planner10x10Config(max_astar_expansions=max_astar_expansions))
    return PlannedDQNPolicy10x10(agent, planner)


def evaluate(
    policy: PlannedDQNPolicy10x10,
    checkpoint_path: Path,
    *,
    protocol: str,
    episodes: int,
    max_steps: int,
    seed: int,
    render: bool,
) -> tuple[PlannedEvaluationSummary, tuple[PlannedEpisodeResult, ...]]:
    if episodes < 1:
        raise ValueError("episodes must be positive")
    if protocol not in PROTOCOLS:
        raise ValueError(f"unknown evaluation protocol: {protocol}")
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    starvation_enabled = protocol == "training-semantics"
    env = SnakeEnv(
        width=10,
        height=10,
        render_mode=render,
        seed=seed,
        state_mode="hybrid",
        reward_profile="experiment8",
        starvation_enabled=starvation_enabled,
    )
    seeds = make_episode_seeds(seed, "final", episodes)
    results: list[PlannedEpisodeResult] = []
    all_planner_ns: list[int] = []
    total_decisions = 0
    total_overrides = 0
    total_hamiltonian_decisions = 0
    started = time.perf_counter()

    try:
        for episode, episode_seed in enumerate(seeds, start=1):
            policy.reset()
            observation = env.reset(seed=episode_seed)
            done = False
            metrics = PlannerEpisodeMetrics()
            info: dict[str, int | float | str] = {
                "score": 0,
                "steps": 0,
                "snake_length": len(env.snake),
                "termination_reason": "none",
            }

            while not done and env.frame_iteration < max_steps:
                planning_state = PlanningState.from_env(env)
                plan_started = time.perf_counter_ns()
                planned_action = policy.choose_action(observation, planning_state)
                elapsed_ns = time.perf_counter_ns() - plan_started
                metrics.record(planned_action, elapsed_ns)
                all_planner_ns.append(elapsed_ns)
                observation, _, done, info = env.step(planned_action.action)

            timed_out = not done and env.frame_iteration >= max_steps
            termination_reason = "max_steps" if timed_out else str(info["termination_reason"])
            result = PlannedEpisodeResult(
                episode=episode,
                seed=episode_seed,
                score=int(info["score"]),
                steps=int(info["steps"]),
                snake_length=int(info["snake_length"]),
                termination_reason=termination_reason,
                completed=termination_reason == "board_completed",
                timed_out=timed_out,
                planner_decisions=metrics.decisions,
                planner_overrides=metrics.overrides,
                safe_food_decisions=metrics.tiers[DecisionTier.SAFE_FOOD.value],
                hamiltonian_cycle_decisions=metrics.tiers[DecisionTier.HAMILTONIAN_CYCLE.value],
                planner_total_ms=metrics.total_ms,
                planner_max_ms=metrics.max_ms,
            )
            results.append(result)
            total_decisions += metrics.decisions
            total_overrides += metrics.overrides
            total_hamiltonian_decisions += result.hamiltonian_cycle_decisions
            print(
                f"episode={episode:4d} score={result.score:3d} steps={result.steps:4d} "
                f"reason={result.termination_reason} overrides={result.planner_overrides}"
            )
    finally:
        env.close()

    elapsed = time.perf_counter() - started
    scores = [result.score for result in results]
    summary = PlannedEvaluationSummary(
        checkpoint=str(checkpoint_path),
        protocol=protocol,
        episodes=episodes,
        max_steps=max_steps,
        mean_score=statistics.fmean(scores),
        score_std=statistics.pstdev(scores) if episodes > 1 else 0.0,
        max_score=max(scores),
        completion_rate=sum(result.completed for result in results) / episodes,
        timeout_rate=sum(result.timed_out for result in results) / episodes,
        override_rate=total_overrides / total_decisions if total_decisions else 0.0,
        hamiltonian_cycle_rate=(
            total_hamiltonian_decisions / total_decisions if total_decisions else 0.0
        ),
        planner_mean_ms=(statistics.fmean(all_planner_ns) / 1_000_000 if all_planner_ns else 0.0),
        planner_max_ms=(max(all_planner_ns) / 1_000_000 if all_planner_ns else 0.0),
        total_time_sec=elapsed,
    )
    return summary, tuple(results)


def write_results(
    output_dir: Path,
    summary: PlannedEvaluationSummary,
    results: tuple[PlannedEpisodeResult, ...],
    *,
    planner_config: Planner10x10Config,
) -> None:
    # 每次评估必须使用新目录，避免覆盖或混入历史结果。
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "config.json").write_text(
        json.dumps(
            {
                "summary": summary.to_dict(),
                "planner": asdict(planner_config),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    with (output_dir / "episodes.csv").open("w", newline="", encoding="utf-8") as file:
        fieldnames = list(results[0].to_dict()) if results else []
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(result.to_dict() for result in results)


def main() -> None:
    args = parse_args()
    max_steps = resolve_max_steps(args.protocol, args.max_steps)
    set_seed(args.seed)
    planner_config = Planner10x10Config(max_astar_expansions=args.max_astar_expansions)
    policy = load_policy(
        args.checkpoint,
        max_astar_expansions=planner_config.max_astar_expansions,
    )
    summary, results = evaluate(
        policy,
        args.checkpoint,
        protocol=args.protocol,
        episodes=args.episodes,
        max_steps=max_steps,
        seed=args.seed,
        render=args.render,
    )
    if args.output_dir is not None:
        write_results(args.output_dir, summary, results, planner_config=planner_config)
    print(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
