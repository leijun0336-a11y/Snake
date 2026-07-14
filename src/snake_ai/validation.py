from __future__ import annotations

import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal, Protocol

from snake_ai.game import SnakeEnv


SeedSetName = Literal["quick", "confirmation", "final"]

SEED_SET_STRIDE = 1_000_000
COMPARISON_EPSILON = 1e-12
SEED_SET_INDEX: dict[SeedSetName, int] = {
    "quick": 1,
    "confirmation": 2,
    "final": 3,
}


class GreedyAgent(Protocol):
    policy_net: Any

    def act(self, state: Any, training: bool = False) -> int: ...


@dataclass(frozen=True)
class ValidationEpisode:
    seed: int
    score: int
    steps: int
    score_per_step: float
    max_snake_length: int
    timed_out: bool


@dataclass(frozen=True)
class ValidationResult:
    seed_set: SeedSetName
    episodes: int
    max_steps: int
    full_score: int
    mean_score: float
    score_std: float
    min_score: int
    max_score: int
    full_games: int
    full_rate: float
    mean_steps: float
    mean_score_per_step: float
    mean_max_snake_length: float
    timeout_games: int
    timeout_rate: float
    total_time_sec: float

    def to_dict(self) -> dict[str, int | float | str]:
        return asdict(self)


@dataclass(frozen=True)
class SelectionThresholds:
    quick_mean_delta: float = 0.25
    quick_mean_tolerance: float = 0.10
    quick_full_rate_delta: float = 0.02
    confirmation_mean_delta: float = 0.15
    confirmation_mean_tolerance: float = 0.15
    confirmation_full_rate_delta: float = 0.015


DEFAULT_SELECTION_THRESHOLDS = SelectionThresholds()


@dataclass(frozen=True)
class ValidationEvent:
    stage: str
    result: ValidationResult
    passed_stage: bool
    promoted_to_best: bool


@dataclass
class StagedValidationState:
    selection_start_episode: int | None = None
    best_quick: ValidationResult | None = None
    best_confirmation: ValidationResult | None = None
    best_training_episode: int | None = None
    rounds_without_improvement: int = 0


@dataclass(frozen=True)
class StagedValidationDecision:
    events: tuple[ValidationEvent, ...] = ()
    best_updated: bool = False
    stop_reason: Literal["target_validation", "validation_patience"] | None = None


def make_episode_seeds(
    base_seed: int,
    seed_set: SeedSetName,
    episodes: int,
) -> tuple[int, ...]:
    """Build stable, non-overlapping per-episode seed sets."""

    if episodes < 1:
        raise ValueError("validation episodes must be at least 1")
    if episodes >= SEED_SET_STRIDE:
        raise ValueError(
            f"validation episodes must be less than {SEED_SET_STRIDE} to keep seed sets disjoint"
        )
    start = base_seed + SEED_SET_INDEX[seed_set] * SEED_SET_STRIDE
    return tuple(range(start, start + episodes))


def epsilon_at_floor(epsilon: float, epsilon_end: float, tolerance: float = 1e-8) -> bool:
    return epsilon <= epsilon_end + tolerance


def should_run_periodic_validation(
    episode: int,
    selection_start_episode: int,
    interval: int,
) -> bool:
    if interval < 1:
        raise ValueError("validation interval must be at least 1")
    return episode > selection_start_episode and (episode - selection_start_episode) % interval == 0


def next_validation_patience(
    current_rounds: int,
    *,
    promoted: bool,
    early_stop_eligible: bool,
) -> int:
    if current_rounds < 0:
        raise ValueError("validation patience count must be non-negative")
    if not early_stop_eligible:
        return current_rounds
    return 0 if promoted else current_rounds + 1


def validation_patience_exhausted(current_rounds: int, patience: int) -> bool:
    if patience < 1:
        raise ValueError("validation patience must be at least 1")
    return current_rounds >= patience


def run_staged_validation(
    *,
    episode: int,
    epsilon: float,
    epsilon_end: float,
    state: StagedValidationState,
    evaluator: Callable[[SeedSetName], ValidationResult],
    interval: int,
    early_stop_enabled: bool,
    min_episodes: int,
    patience: int,
    target_mean_score: float | None,
    thresholds: SelectionThresholds = DEFAULT_SELECTION_THRESHOLDS,
) -> StagedValidationDecision:
    """Advance best selection and early stopping by one training episode."""

    if state.selection_start_episode is None:
        if not epsilon_at_floor(epsilon, epsilon_end):
            return StagedValidationDecision()
        quick = evaluator("quick")
        confirmation = evaluator("confirmation")
        state.selection_start_episode = episode
        _promote(state, episode, quick, confirmation)
        stop_reason = (
            "target_validation"
            if _target_reached(
                confirmation,
                episode,
                early_stop_enabled,
                min_episodes,
                target_mean_score,
            )
            else None
        )
        return StagedValidationDecision(
            events=(
                ValidationEvent("initial_quick", quick, True, True),
                ValidationEvent("initial_confirmation", confirmation, True, True),
            ),
            best_updated=True,
            stop_reason=stop_reason,
        )

    if not should_run_periodic_validation(
        episode,
        state.selection_start_episode,
        interval,
    ):
        return StagedValidationDecision()
    if state.best_quick is None or state.best_confirmation is None:
        raise RuntimeError("best validation baselines were not initialized")

    quick = evaluator("quick")
    quick_passed = passes_quick_screen(quick, state.best_quick, thresholds)
    confirmation: ValidationResult | None = None
    promoted = False
    best_updated = False
    events: list[ValidationEvent] = []

    if quick_passed:
        confirmation = evaluator("confirmation")
        promoted = passes_confirmation(
            confirmation,
            state.best_confirmation,
            thresholds,
        )
        if promoted:
            _promote(state, episode, quick, confirmation)
            best_updated = True

    events.append(ValidationEvent("quick", quick, quick_passed, promoted))
    if confirmation is not None:
        events.append(ValidationEvent("confirmation", confirmation, promoted, promoted))

    early_stop_eligible = early_stop_enabled and episode >= min_episodes
    state.rounds_without_improvement = next_validation_patience(
        state.rounds_without_improvement,
        promoted=promoted,
        early_stop_eligible=early_stop_eligible,
    )

    if _target_reached(
        state.best_confirmation,
        episode,
        early_stop_enabled,
        min_episodes,
        target_mean_score,
    ):
        return StagedValidationDecision(
            events=tuple(events),
            best_updated=best_updated,
            stop_reason="target_validation",
        )

    if not (
        early_stop_eligible
        and validation_patience_exhausted(
            state.rounds_without_improvement,
            patience,
        )
    ):
        return StagedValidationDecision(
            events=tuple(events),
            best_updated=best_updated,
        )

    # If quick screening skipped confirmation, perform the mandatory final check.
    if confirmation is None:
        confirmation = evaluator("confirmation")
        promoted = passes_confirmation(
            confirmation,
            state.best_confirmation,
            thresholds,
        )
        if promoted:
            _promote(state, episode, quick, confirmation)
            state.rounds_without_improvement = 0
            best_updated = True
        events.append(
            ValidationEvent(
                "early_stop_confirmation",
                confirmation,
                promoted,
                promoted,
            )
        )

    stop_reason: Literal["target_validation", "validation_patience"] | None
    if _target_reached(
        state.best_confirmation,
        episode,
        early_stop_enabled,
        min_episodes,
        target_mean_score,
    ):
        stop_reason = "target_validation"
    else:
        stop_reason = None if promoted else "validation_patience"
    return StagedValidationDecision(
        events=tuple(events),
        best_updated=best_updated,
        stop_reason=stop_reason,
    )


def passes_quick_screen(
    candidate: ValidationResult,
    incumbent: ValidationResult,
    thresholds: SelectionThresholds = DEFAULT_SELECTION_THRESHOLDS,
) -> bool:
    _require_comparable(candidate, incumbent, "quick")
    mean_delta = candidate.mean_score - incumbent.mean_score
    full_rate_delta = candidate.full_rate - incumbent.full_rate
    return _gte(mean_delta, thresholds.quick_mean_delta) or (
        _gte(mean_delta, -thresholds.quick_mean_tolerance)
        and _gte(full_rate_delta, thresholds.quick_full_rate_delta)
    )


def passes_confirmation(
    candidate: ValidationResult,
    incumbent: ValidationResult,
    thresholds: SelectionThresholds = DEFAULT_SELECTION_THRESHOLDS,
) -> bool:
    _require_comparable(candidate, incumbent, "confirmation")
    mean_delta = candidate.mean_score - incumbent.mean_score
    full_rate_delta = candidate.full_rate - incumbent.full_rate
    return _gte(mean_delta, thresholds.confirmation_mean_delta) or (
        abs(mean_delta) <= thresholds.confirmation_mean_tolerance + COMPARISON_EPSILON
        and _gte(full_rate_delta, thresholds.confirmation_full_rate_delta)
    )


def evaluate_policy(
    agent: GreedyAgent,
    env: SnakeEnv,
    seeds: Sequence[int],
    *,
    seed_set: SeedSetName,
    max_steps: int,
    on_episode: Callable[[int, ValidationEpisode], None] | None = None,
) -> ValidationResult:
    """Evaluate a frozen greedy policy without mutating training state."""

    if not seeds:
        raise ValueError("validation seeds must not be empty")
    if max_steps < 1:
        raise ValueError("validation max_steps must be at least 1")

    scores: list[int] = []
    steps: list[int] = []
    score_per_steps: list[float] = []
    max_snake_lengths: list[int] = []
    timeout_games = 0
    full_score: int | None = None
    policy_was_training = bool(agent.policy_net.training)
    agent.policy_net.eval()
    started = time.perf_counter()

    try:
        for index, episode_seed in enumerate(seeds, start=1):
            state = env.reset(seed=int(episode_seed))
            if full_score is None:
                full_score = env.width * env.height - len(env.snake)
            done = False
            info: dict[str, int | float | str] = {
                "score": 0,
                "steps": 0,
                "snake_length": len(env.snake),
            }
            max_snake_length = len(env.snake)
            evaluation_steps = 0

            while not done and evaluation_steps < max_steps:
                action = agent.act(state, training=False)
                state, _, done, info = env.step(action)
                evaluation_steps += 1
                max_snake_length = max(
                    max_snake_length,
                    int(info["snake_length"]),
                )

            score = int(info["score"])
            episode_steps = int(info["steps"])
            timed_out = not done and evaluation_steps >= max_steps
            score_per_step = score / episode_steps if episode_steps > 0 else 0.0
            episode_result = ValidationEpisode(
                seed=int(episode_seed),
                score=score,
                steps=episode_steps,
                score_per_step=score_per_step,
                max_snake_length=max_snake_length,
                timed_out=timed_out,
            )
            scores.append(score)
            steps.append(episode_steps)
            score_per_steps.append(score_per_step)
            max_snake_lengths.append(max_snake_length)
            timeout_games += int(timed_out)
            if on_episode is not None:
                on_episode(index, episode_result)
    finally:
        agent.policy_net.train(policy_was_training)

    episodes = len(scores)
    if full_score is None:
        raise RuntimeError("validation did not run any episodes")
    full_games = sum(score >= full_score for score in scores)
    return ValidationResult(
        seed_set=seed_set,
        episodes=episodes,
        max_steps=max_steps,
        full_score=full_score,
        mean_score=statistics.fmean(scores),
        score_std=statistics.pstdev(scores) if episodes > 1 else 0.0,
        min_score=min(scores),
        max_score=max(scores),
        full_games=full_games,
        full_rate=full_games / episodes,
        mean_steps=statistics.fmean(steps),
        mean_score_per_step=statistics.fmean(score_per_steps),
        mean_max_snake_length=statistics.fmean(max_snake_lengths),
        timeout_games=timeout_games,
        timeout_rate=timeout_games / episodes,
        total_time_sec=time.perf_counter() - started,
    )


def _require_comparable(
    candidate: ValidationResult,
    incumbent: ValidationResult,
    expected_seed_set: SeedSetName,
) -> None:
    if candidate.seed_set != expected_seed_set or incumbent.seed_set != expected_seed_set:
        raise ValueError(f"{expected_seed_set} comparison requires two {expected_seed_set} results")
    if (
        candidate.episodes != incumbent.episodes
        or candidate.max_steps != incumbent.max_steps
        or candidate.full_score != incumbent.full_score
    ):
        raise ValueError(
            "validation results must use the same episode count, max_steps, and full_score"
        )


def _gte(left: float, right: float) -> bool:
    return left + COMPARISON_EPSILON >= right


def _promote(
    state: StagedValidationState,
    episode: int,
    quick: ValidationResult,
    confirmation: ValidationResult,
) -> None:
    state.best_quick = quick
    state.best_confirmation = confirmation
    state.best_training_episode = episode
    state.rounds_without_improvement = 0


def _target_reached(
    confirmation: ValidationResult | None,
    episode: int,
    early_stop_enabled: bool,
    min_episodes: int,
    target_mean_score: float | None,
) -> bool:
    return (
        early_stop_enabled
        and confirmation is not None
        and episode >= min_episodes
        and target_mean_score is not None
        and confirmation.mean_score >= target_mean_score
    )
