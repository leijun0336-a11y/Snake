from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any

from snake_ai.planning_10x10.dqn_policy import PlannedAction


@dataclass(frozen=True, slots=True)
class PlannedEpisodeResult:
    episode: int
    seed: int
    score: int
    steps: int
    snake_length: int
    termination_reason: str
    completed: bool
    timed_out: bool
    planner_decisions: int
    planner_overrides: int
    safe_food_decisions: int
    hamiltonian_cycle_decisions: int
    single_safe_action_decisions: int
    multi_safe_action_decisions: int
    three_safe_action_decisions: int
    admissible_action_total: int
    raw_dqn_safe_decisions: int
    dqn_non_default_choices: int
    planner_total_ms: float
    planner_max_ms: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PlannerEpisodeMetrics:
    decisions: int = 0
    overrides: int = 0
    total_ns: int = 0
    max_ns: int = 0
    tiers: Counter[str] = field(default_factory=Counter)
    single_safe_action_decisions: int = 0
    multi_safe_action_decisions: int = 0
    three_safe_action_decisions: int = 0
    admissible_action_total: int = 0
    raw_dqn_safe_decisions: int = 0
    dqn_non_default_choices: int = 0

    def record(self, action: PlannedAction, elapsed_ns: int) -> None:
        admissible = action.planner_decision.admissible_actions
        self.decisions += 1
        self.overrides += int(action.overridden)
        self.total_ns += elapsed_ns
        self.max_ns = max(self.max_ns, elapsed_ns)
        self.tiers[str(action.planner_decision.tier)] += 1
        self.admissible_action_total += len(admissible)
        self.single_safe_action_decisions += int(len(admissible) == 1)
        self.multi_safe_action_decisions += int(len(admissible) >= 2)
        self.three_safe_action_decisions += int(len(admissible) == 3)
        self.raw_dqn_safe_decisions += int(action.raw_dqn_action in admissible)
        # 最小动作编号只是一个与 Q 无关的确定性对照；偏离它说明 Q 排序改变了选择。
        self.dqn_non_default_choices += int(len(admissible) >= 2 and action.action != admissible[0])

    @property
    def total_ms(self) -> float:
        return self.total_ns / 1_000_000

    @property
    def max_ms(self) -> float:
        return self.max_ns / 1_000_000
