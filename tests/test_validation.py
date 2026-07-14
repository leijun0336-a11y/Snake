from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from snake_ai.game import SnakeEnv
from snake_ai.validation import (
    SeedSetName,
    StagedValidationState,
    ValidationResult,
    epsilon_at_floor,
    evaluate_policy,
    make_episode_seeds,
    next_validation_patience,
    passes_confirmation,
    passes_quick_screen,
    run_staged_validation,
    should_run_periodic_validation,
    validation_patience_exhausted,
)


class StraightAgent:
    def __init__(self) -> None:
        self.policy_net = torch.nn.Linear(1, 1)

    def act(self, state: object, training: bool = False) -> int:
        assert training is False
        return 0


def make_result(
    seed_set: SeedSetName,
    *,
    episodes: int,
    mean_score: float,
    full_rate: float,
) -> ValidationResult:
    return ValidationResult(
        seed_set=seed_set,
        episodes=episodes,
        max_steps=1000,
        full_score=33,
        mean_score=mean_score,
        score_std=1.0,
        min_score=0,
        max_score=33,
        full_games=round(full_rate * episodes),
        full_rate=full_rate,
        mean_steps=100.0,
        mean_score_per_step=0.1,
        mean_max_snake_length=10.0,
        timeout_games=0,
        timeout_rate=0.0,
        total_time_sec=1.0,
    )


def test_seed_sets_are_stable_and_disjoint() -> None:
    quick = make_episode_seeds(42, "quick", 100)
    confirmation = make_episode_seeds(42, "confirmation", 500)
    final = make_episode_seeds(42, "final", 1000)

    assert quick == make_episode_seeds(42, "quick", 100)
    assert set(quick).isdisjoint(confirmation)
    assert set(quick).isdisjoint(final)
    assert set(confirmation).isdisjoint(final)


def test_quick_and_confirmation_thresholds() -> None:
    quick_best = make_result("quick", episodes=100, mean_score=28.0, full_rate=0.60)
    confirm_best = make_result(
        "confirmation",
        episodes=500,
        mean_score=28.0,
        full_rate=0.60,
    )

    assert passes_quick_screen(
        replace(quick_best, mean_score=28.25, full_rate=0.55),
        quick_best,
    )
    assert passes_quick_screen(
        replace(quick_best, mean_score=27.90, full_rate=0.62),
        quick_best,
    )
    assert not passes_quick_screen(
        replace(quick_best, mean_score=28.10, full_rate=0.61),
        quick_best,
    )
    assert passes_confirmation(
        replace(confirm_best, mean_score=28.15, full_rate=0.55),
        confirm_best,
    )
    assert passes_confirmation(
        replace(confirm_best, mean_score=27.85, full_rate=0.615),
        confirm_best,
    )


def test_comparison_rejects_different_seed_sets_or_episode_counts() -> None:
    quick = make_result("quick", episodes=100, mean_score=28.0, full_rate=0.60)
    confirmation = make_result(
        "confirmation",
        episodes=500,
        mean_score=28.0,
        full_rate=0.60,
    )

    with pytest.raises(ValueError, match="quick comparison"):
        passes_quick_screen(confirmation, quick)
    with pytest.raises(ValueError, match="same episode count"):
        passes_quick_screen(replace(quick, episodes=99), quick)


def test_epsilon_gate_schedule_and_patience() -> None:
    assert not epsilon_at_floor(0.0101, 0.01)
    assert epsilon_at_floor(0.010000001, 0.01)
    assert should_run_periodic_validation(10_500, 10_000, 500)
    assert not should_run_periodic_validation(10_250, 10_000, 500)

    rounds = next_validation_patience(
        0,
        promoted=False,
        early_stop_eligible=False,
    )
    assert rounds == 0
    rounds = next_validation_patience(
        rounds,
        promoted=False,
        early_stop_eligible=True,
    )
    assert rounds == 1
    assert not validation_patience_exhausted(rounds, 8)
    assert (
        next_validation_patience(
            rounds,
            promoted=True,
            early_stop_eligible=True,
        )
        == 0
    )
    assert validation_patience_exhausted(8, 8)


def test_staged_validation_initializes_only_at_epsilon_floor() -> None:
    state = StagedValidationState()
    quick = make_result("quick", episodes=100, mean_score=28.0, full_rate=0.60)
    confirmation = make_result(
        "confirmation",
        episodes=500,
        mean_score=28.0,
        full_rate=0.60,
    )
    calls: list[SeedSetName] = []

    def evaluator(seed_set: SeedSetName) -> ValidationResult:
        calls.append(seed_set)
        return quick if seed_set == "quick" else confirmation

    before_floor = run_staged_validation(
        episode=9_999,
        epsilon=0.0101,
        epsilon_end=0.01,
        state=state,
        evaluator=evaluator,
        interval=500,
        early_stop_enabled=True,
        min_episodes=5_000,
        patience=8,
        target_mean_score=None,
    )
    initialized = run_staged_validation(
        episode=10_000,
        epsilon=0.01,
        epsilon_end=0.01,
        state=state,
        evaluator=evaluator,
        interval=500,
        early_stop_enabled=True,
        min_episodes=5_000,
        patience=8,
        target_mean_score=None,
    )

    assert before_floor.events == ()
    assert calls == ["quick", "confirmation"]
    assert initialized.best_updated
    assert state.best_training_episode == 10_000


def test_patience_runs_final_confirmation_before_stopping() -> None:
    best_quick = make_result("quick", episodes=100, mean_score=28.0, full_rate=0.60)
    best_confirmation = make_result(
        "confirmation",
        episodes=500,
        mean_score=28.0,
        full_rate=0.60,
    )
    state = StagedValidationState(
        selection_start_episode=10_000,
        best_quick=best_quick,
        best_confirmation=best_confirmation,
        best_training_episode=10_000,
    )
    bad_quick = replace(best_quick, mean_score=27.0, full_rate=0.50)
    bad_confirmation = replace(
        best_confirmation,
        mean_score=27.0,
        full_rate=0.50,
    )
    calls: list[SeedSetName] = []

    def evaluator(seed_set: SeedSetName) -> ValidationResult:
        calls.append(seed_set)
        return bad_quick if seed_set == "quick" else bad_confirmation

    decision = run_staged_validation(
        episode=10_500,
        epsilon=0.01,
        epsilon_end=0.01,
        state=state,
        evaluator=evaluator,
        interval=500,
        early_stop_enabled=True,
        min_episodes=5_000,
        patience=1,
        target_mean_score=None,
    )

    assert calls == ["quick", "confirmation"]
    assert [event.stage for event in decision.events] == [
        "quick",
        "early_stop_confirmation",
    ]
    assert decision.stop_reason == "validation_patience"


def test_confirmed_promotion_resets_validation_patience() -> None:
    best_quick = make_result("quick", episodes=100, mean_score=28.0, full_rate=0.60)
    best_confirmation = make_result(
        "confirmation",
        episodes=500,
        mean_score=28.0,
        full_rate=0.60,
    )
    state = StagedValidationState(
        selection_start_episode=10_000,
        best_quick=best_quick,
        best_confirmation=best_confirmation,
        best_training_episode=10_000,
        rounds_without_improvement=7,
    )
    better_quick = replace(best_quick, mean_score=28.3)
    better_confirmation = replace(best_confirmation, mean_score=28.2)

    def evaluator(seed_set: SeedSetName) -> ValidationResult:
        return better_quick if seed_set == "quick" else better_confirmation

    decision = run_staged_validation(
        episode=10_500,
        epsilon=0.01,
        epsilon_end=0.01,
        state=state,
        evaluator=evaluator,
        interval=500,
        early_stop_enabled=True,
        min_episodes=5_000,
        patience=8,
        target_mean_score=None,
    )

    assert decision.best_updated
    assert decision.stop_reason is None
    assert state.best_training_episode == 10_500
    assert state.rounds_without_improvement == 0


def test_confirmed_best_can_trigger_target_after_min_episodes() -> None:
    best_quick = make_result("quick", episodes=100, mean_score=29.0, full_rate=0.65)
    best_confirmation = make_result(
        "confirmation",
        episodes=500,
        mean_score=29.0,
        full_rate=0.65,
    )
    state = StagedValidationState(
        selection_start_episode=10_000,
        best_quick=best_quick,
        best_confirmation=best_confirmation,
        best_training_episode=10_000,
    )
    worse_quick = replace(best_quick, mean_score=27.0, full_rate=0.50)

    decision = run_staged_validation(
        episode=10_500,
        epsilon=0.01,
        epsilon_end=0.01,
        state=state,
        evaluator=lambda _: worse_quick,
        interval=500,
        early_stop_enabled=True,
        min_episodes=10_500,
        patience=8,
        target_mean_score=28.5,
    )

    assert decision.stop_reason == "target_validation"


def test_evaluate_policy_reseeds_each_episode_and_restores_model_mode() -> None:
    agent = StraightAgent()
    env = SnakeEnv(
        width=6,
        height=6,
        seed=1,
        starvation_enabled=False,
        potential_reward=False,
    )
    seen_seeds: list[int] = []

    result = evaluate_policy(
        agent,
        env,
        (101, 102),
        seed_set="quick",
        max_steps=1,
        on_episode=lambda _, episode: seen_seeds.append(episode.seed),
    )

    assert seen_seeds == [101, 102]
    assert result.episodes == 2
    assert result.timeout_games == 2
    assert result.full_score == 33
    assert agent.policy_net.training is True
    env.close()


def test_environment_reset_seed_reproduces_initial_food() -> None:
    env = SnakeEnv(width=6, height=6, seed=1)

    env.reset(seed=1234)
    first_food = env.food
    env.reset(seed=1234)

    assert env.food == first_food
    env.close()
