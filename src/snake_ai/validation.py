from __future__ import annotations

import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal, Protocol

from snake_ai.game import SnakeEnv


# 指定的验证集名称。
SeedSetName = Literal["quick", "confirmation", "final"]

SEED_SET_STRIDE = 1_000_000
COMPARISON_EPSILON = 1e-12
# 现有选拔阈值最初按 6x6 棋盘制定；初始蛇长为 3，因此满分为 33。
# 其他棋盘先按“平均分 / 满分”计算完成比例，再换算到这个基准分数尺度。
# 这样阈值仍可保持原值，同时 full_score=33 时能够走原始计算路径。
SELECTION_REFERENCE_FULL_SCORE = 33
SEED_SET_INDEX: dict[SeedSetName, int] = {
    "quick": 1,
    "confirmation": 2,
    "final": 3,
}


#  规定“符合要求的智能体”必须具备哪些属性和方法。
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

    # 把当前的 dataclass 对象转换成普通 Python 字典。
    # 保存 checkpoint 的 metadata 需要使用普通的数据结构。
    def to_dict(self) -> dict[str, int | float | str]:
        return asdict(self)


# 验证选拔标准的配置。均分相关阈值使用 6x6 满分 33 的等价分数尺度；
# 满分率相关阈值本身已经是比例，不需要随棋盘尺寸换算。
@dataclass(frozen=True)
class SelectionThresholds:
    quick_mean_delta: float = 0.25
    quick_mean_tolerance: float = 0.10
    quick_full_rate_delta: float = 0.02
    confirmation_mean_delta: float = 0.15
    confirmation_mean_tolerance: float = 0.15
    confirmation_full_rate_delta: float = 0.015


DEFAULT_SELECTION_THRESHOLDS = SelectionThresholds()

# 验证阶段的事件记录
@dataclass(frozen=True)
class ValidationEvent:
    stage: str
    result: ValidationResult
    passed_stage: bool
    promoted_to_best: bool


# 验证阶段擂主的记录
@dataclass
class StagedValidationState:
    selection_start_episode: int | None = None
    best_quick: ValidationResult | None = None
    best_confirmation: ValidationResult | None = None
    best_training_episode: int | None = None
    rounds_without_improvement: int = 0


@dataclass(frozen=True)
class StagedValidationDecision:
    # 本轮分阶段验证产生的事件集合。比如当前这一局既进行了快速验证，又进行了确认验证，那么 events 就会包含两个 ValidationEvent 对象。
    # 一个长度不限的元组，并且每个元素都是 ValidationEvent.
    # ... 不是省略方法实现，而是元组类型语法，表示前面的元素类型可以重复任意次。
    events: tuple[ValidationEvent, ...] = ()
    best_updated: bool = False
    stop_reason: Literal["target_validation", "validation_patience"] | None = None

# 不同的验证阶段使用不同的随机种子
def make_episode_seeds(
    base_seed: int,
    seed_set: SeedSetName,
    episodes: int,
) -> tuple[int, ...]:

    if episodes < 1:
        raise ValueError("validation episodes must be at least 1")
    if episodes >= SEED_SET_STRIDE:
        raise ValueError(
            f"validation episodes must be less than {SEED_SET_STRIDE} to keep seed sets disjoint"
        )
    
    # 种子计算方法：基础种子值+之前定义的种子集合索引*步长
    start = base_seed + SEED_SET_INDEX[seed_set] * SEED_SET_STRIDE
    # 返回一个元组，包含从 start 开始的连续整数，长度为 episodes
    # 例如，如果 base_seed=1000, seed_set='quick', episodes=5, 那么返回的元组将是 (1000000, 1000001, 1000002, 1000003, 1000004)
    # 一次验证中每个episode使用不同的种子，因为一局中两条蛇吃到的食物数量可能不同，导致消耗的随机数也不同。因此每个episode都要同步一下。
    return tuple(range(start, start + episodes))


# 判断当前 epsilon 是否已经下降到最低值附近。如果epsilon到达下限则开始验证和选拔最佳模型。
def epsilon_at_floor(epsilon: float, epsilon_end: float, tolerance: float = 1e-8) -> bool:
    return epsilon <= epsilon_end + tolerance


# 判断当前训练 episode 是否到了“定期验证”的时间点。
def should_run_periodic_validation(
    episode: int,
    selection_start_episode: int,
    interval: int,
) -> bool:
    if interval < 1:
        raise ValueError("validation interval must be at least 1")
    return episode > selection_start_episode and (episode - selection_start_episode) % interval == 0


# 更新“连续多少轮验证没有产生更好的模型”的计数器。
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


# 判断“连续没有改进的验证轮数”是否已经达到早停上限。
def validation_patience_exhausted(current_rounds: int, patience: int) -> bool:
    if patience < 1:
        raise ValueError("validation patience must be at least 1")
    return current_rounds >= patience


# 分阶段验证调度器
# 是否进行快速验证
# 是否继续进行确认验证
# 当前模型是否成为新的最佳模型
# 是否因为达到目标或长期无改进而停止训练
def run_staged_validation(
    *,            # `*` 后面的参数必须通过“参数名=值”的方式传入。
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

    # 如果本轮 quick 快速筛选没有通过，因此跳过了 confirmation 验证，
    # 但训练又准备因为长期无改进而早停，那么停止前必须补做一次 confirmation 验证。
    # 相当于即将早停前给最后一次机会。
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


# 判断当前候选模型是否通过快速筛选，获得进入 confirmation 确认验证的资格。
# 被调度器函数调用。
def passes_quick_screen(
    candidate: ValidationResult,
    incumbent: ValidationResult,
    thresholds: SelectionThresholds = DEFAULT_SELECTION_THRESHOLDS,
) -> bool:
    _require_comparable(candidate, incumbent, "quick")
    mean_delta = _selection_mean_delta(candidate, incumbent)
    full_rate_delta = candidate.full_rate - incumbent.full_rate
    return _gte(mean_delta, thresholds.quick_mean_delta) or (
        _gte(mean_delta, -thresholds.quick_mean_tolerance)
        and _gte(full_rate_delta, thresholds.quick_full_rate_delta)
    )


# 判断候选模型是否通过正式确认验证，并取代当前最佳模型。
# 被调度器函数调用。
def passes_confirmation(
    candidate: ValidationResult,
    incumbent: ValidationResult,
    thresholds: SelectionThresholds = DEFAULT_SELECTION_THRESHOLDS,
) -> bool:
    _require_comparable(candidate, incumbent, "confirmation")
    mean_delta = _selection_mean_delta(candidate, incumbent)
    full_rate_delta = candidate.full_rate - incumbent.full_rate
    return _gte(mean_delta, thresholds.confirmation_mean_delta) or (
        abs(mean_delta) <= thresholds.confirmation_mean_tolerance + COMPARISON_EPSILON
        and _gte(full_rate_delta, thresholds.confirmation_full_rate_delta)
    )


# 让当前模型在固定的一组随机种子下，以纯贪心策略运行多局游戏，并汇总评估指标。
# 被调度器函数调用。
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


# 比赛前检查双方是否使用了相同赛制。
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


def _selection_mean_delta(
    candidate: ValidationResult,
    incumbent: ValidationResult,
) -> float:
    """返回按棋盘满分归一化后的 6x6 等价均分差。"""
    full_score = candidate.full_score
    if full_score <= 0:
        raise ValueError("validation full_score must be positive")
    if full_score == SELECTION_REFERENCE_FULL_SCORE:
        # 保留历史 6x6 的原始减法路径，避免额外除法和乘法引入任何浮点差异。
        return candidate.mean_score - incumbent.mean_score
    completion_delta = (
        candidate.mean_score / full_score
        - incumbent.mean_score / full_score
    )
    return completion_delta * SELECTION_REFERENCE_FULL_SCORE


def _gte(left: float, right: float) -> bool:
    return left + COMPARISON_EPSILON >= right


# 把当前候选模型登记为新的最佳模型。
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
