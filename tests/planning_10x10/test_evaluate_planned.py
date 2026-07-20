from __future__ import annotations

from pathlib import Path

import pytest
import torch

from snake_ai.agents import DQNAgent
from snake_ai.evaluate_planned_10x10 import (
    PlannedEvaluationSummary,
    evaluate,
    resolve_max_steps,
    validate_checkpoint,
    write_results,
)
from snake_ai.game.ai_profiles import DEFAULT_AI_ID, get_ai_profile
from snake_ai.planning_10x10 import (
    PlannedDQNPolicy10x10,
    Planner10x10Config,
    StrictSafePlanner10x10,
)
from snake_ai.planning_10x10.metrics import PlannedEpisodeResult


def save_10x10_checkpoint(path: Path) -> None:
    agent = DQNAgent(
        state_size=(9, 10, 10),
        action_size=3,
        state_mode="hybrid",
        auxiliary_size=20,
        seed=1,
        device="cpu",
    )
    agent.save(
        path,
        metadata={
            "environment": {
                "width": 10,
                "height": 10,
                "state_mode": "hybrid",
                "starvation_enabled": True,
            },
            "reward": {
                "profile": "experiment8",
                "starvation_limit_mode": "board_area",
                "starvation_comparison": "gt",
            },
        },
    )


def test_protocol_defaults_are_explicit() -> None:
    assert resolve_max_steps("legacy-final", None) == 1_000
    assert resolve_max_steps("training-semantics", None) == 5_000
    assert resolve_max_steps("training-semantics", 321) == 321


@pytest.mark.parametrize("protocol", ["unknown", "", "TRAINING-SEMANTICS"])
def test_evaluate_rejects_unknown_protocol(protocol: str) -> None:
    agent = DQNAgent(
        state_size=(9, 10, 10),
        action_size=3,
        state_mode="hybrid",
        auxiliary_size=20,
        seed=1,
        device="cpu",
    )
    policy = PlannedDQNPolicy10x10(agent, StrictSafePlanner10x10())

    with pytest.raises(ValueError, match="unknown evaluation protocol"):
        evaluate(
            policy,
            Path("unused.pt"),
            protocol=protocol,
            episodes=1,
            max_steps=1,
            seed=1,
            render=False,
        )


def test_checkpoint_validation_accepts_only_explicit_matching_metadata(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "ten_by_ten.pt"
    save_10x10_checkpoint(checkpoint_path)

    checkpoint = validate_checkpoint(checkpoint_path)

    assert checkpoint["state_size"] == (9, 10, 10)
    assert checkpoint["state_mode"] == "hybrid"


def test_checkpoint_validation_rejects_mismatch_without_selecting_another_file(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "wrong.pt"
    save_10x10_checkpoint(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint["run_config"]["environment"]["width"] = 6
    torch.save(checkpoint, checkpoint_path)
    (tmp_path / "best.pt").touch()

    with pytest.raises(ValueError, match="environment mismatch"):
        validate_checkpoint(checkpoint_path)


def test_checkpoint_validation_rejects_changed_starvation_semantics(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "wrong-starvation.pt"
    save_10x10_checkpoint(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint["run_config"]["reward"]["starvation_comparison"] = "gte"
    torch.save(checkpoint, checkpoint_path)

    with pytest.raises(ValueError, match="starvation semantics mismatch"):
        validate_checkpoint(checkpoint_path)


def test_random_dqn_and_strict_planner_complete_a_short_smoke_without_collision() -> None:
    agent = DQNAgent(
        state_size=(9, 10, 10),
        action_size=3,
        state_mode="hybrid",
        auxiliary_size=20,
        seed=4,
        device="cpu",
    )
    policy = PlannedDQNPolicy10x10(agent, StrictSafePlanner10x10())

    summary, results = evaluate(
        policy,
        Path("random-untrained.pt"),
        protocol="training-semantics",
        episodes=2,
        max_steps=30,
        seed=4,
        render=False,
    )

    assert summary.episodes == 2
    assert len(results) == 2
    assert all(result.termination_reason == "max_steps" for result in results)
    assert all(result.steps == 30 for result in results)
    assert all(
        result.safe_food_decisions + result.hamiltonian_cycle_decisions == result.planner_decisions
        for result in results
    )
    assert 0.0 <= summary.hamiltonian_cycle_rate <= 1.0


def test_result_writer_uses_new_directory_and_refuses_overwrite(tmp_path: Path) -> None:
    output_dir = tmp_path / "planned-eval"
    result = PlannedEpisodeResult(
        episode=1,
        seed=3_000_042,
        score=1,
        steps=10,
        snake_length=4,
        termination_reason="max_steps",
        completed=False,
        timed_out=True,
        planner_decisions=10,
        planner_overrides=2,
        safe_food_decisions=8,
        hamiltonian_cycle_decisions=2,
        planner_total_ms=3.0,
        planner_max_ms=0.5,
    )
    summary = PlannedEvaluationSummary(
        checkpoint="model.pt",
        protocol="legacy-final",
        episodes=1,
        max_steps=10,
        mean_score=1.0,
        score_std=0.0,
        max_score=1,
        completion_rate=0.0,
        timeout_rate=1.0,
        override_rate=0.2,
        hamiltonian_cycle_rate=0.2,
        planner_mean_ms=0.3,
        planner_max_ms=0.5,
        total_time_sec=1.0,
    )

    write_results(output_dir, summary, (result,), planner_config=Planner10x10Config())

    assert (output_dir / "config.json").is_file()
    assert (output_dir / "episodes.csv").is_file()
    with pytest.raises(FileExistsError):
        write_results(output_dir, summary, (result,), planner_config=Planner10x10Config())


def test_existing_six_by_six_default_profile_is_unchanged() -> None:
    profile = get_ai_profile(DEFAULT_AI_ID)

    assert profile.id == "experiment_20260715"
    assert (profile.width, profile.height) == (6, 6)
    assert profile.state_mode == "hybrid"
    assert profile.reward_profile == "experiment8"
    assert profile.checkpoint_path.name == "best.pt"
