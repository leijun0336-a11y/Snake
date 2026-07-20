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

    def record(self, action: PlannedAction, elapsed_ns: int) -> None:
        self.decisions += 1
        self.overrides += int(action.overridden)
        self.total_ns += elapsed_ns
        self.max_ns = max(self.max_ns, elapsed_ns)
        self.tiers[str(action.planner_decision.tier)] += 1

    @property
    def total_ms(self) -> float:
        return self.total_ns / 1_000_000

    @property
    def max_ms(self) -> float:
        return self.max_ns / 1_000_000
