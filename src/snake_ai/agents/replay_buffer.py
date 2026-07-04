from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Deque


@dataclass(frozen=True)
class Transition:
    state: list[float]
    action: int
    reward: float
    next_state: list[float]
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int | None = None) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.memory: Deque[Transition] = deque(maxlen=capacity)
        self.random = random.Random(seed)

    def push(
        self,
        state: list[float],
        action: int,
        reward: float,
        next_state: list[float],
        done: bool,
    ) -> None:
        self.memory.append(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> list[Transition]:
        if batch_size > len(self.memory):
            raise ValueError("batch_size cannot be larger than buffer length")
        return self.random.sample(list(self.memory), batch_size)

    def __len__(self) -> int:
        return len(self.memory)
