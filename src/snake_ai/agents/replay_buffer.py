# 处理自引用问题
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)  # frozen=True，创建后字段不能被重新赋值。

# 定义一条"经验"中包含哪些数据，成为经验类
class Transition:
    # 当前状态，也就是执行动作前的状态。
    state: Any
    # 在当前状态下执行的动作。
    action: int
    # 执行动作后环境返回的奖励。
    reward: float
    # 执行动作后环境返回的下一个状态。
    next_state: Any
    # 执行动作后这一局是否结束。
    done: bool


SafeMask = tuple[bool, ...]


@dataclass(frozen=True)
class MaskedTransition:
    """仅供 Masked DQN 使用的经验；普通经验结构保持不变。"""

    state: Any
    action: int
    reward: float
    next_state: Any
    done: bool
    safe_mask: SafeMask
    next_safe_mask: SafeMask


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int | None = None) -> None:
        # capacity 表示经验池最多保存多少条经验，必须是正数。
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        # 固定容量环形列表可按随机索引直接采样，避免每次 learn() 都复制整个 deque。
        self.memory: list[Transition | None] = [None] * capacity
        self.position = 0
        self.size = 0
        # 独立的随机数生成器，用于随机采样经验；传入 seed 后采样过程更容易复现。
        self.random = random.Random(seed)

    def push(
        self,
        state: Any,
        action: int,
        reward: float,
        next_state: Any,
        done: bool,
    ) -> None:
        # 写满后从 position 指向的位置覆盖最旧经验，不复制 NumPy state 本身。
        self.memory[self.position] = Transition(state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> list[Transition]:
        # 采样数量不能超过当前经验池里已有的经验数量。
        if batch_size > self.size:
            raise ValueError("batch_size cannot be larger than buffer length")
        # 只生成 batch_size 个随机索引，采样成本不再随整个经验池容量线性增长。
        indices = self.random.sample(range(self.size), batch_size)
        batch: list[Transition] = []
        for index in indices:
            transition = self.memory[index]
            if transition is None:
                raise RuntimeError("Replay buffer contains an uninitialized slot")
            batch.append(transition)
        return batch

    def __len__(self) -> int:
        # 让 len(buffer) 返回已写入经验数，而不是预分配列表容量。
        return self.size


class MaskedReplayBuffer:
    """与普通 ReplayBuffer 隔离，避免改变历史训练经验的字段和采样路径。"""

    def __init__(self, capacity: int, action_size: int, seed: int | None = None) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if action_size <= 0:
            raise ValueError("action_size must be positive")
        self.capacity = capacity
        self.action_size = action_size
        self.memory: list[MaskedTransition | None] = [None] * capacity
        self.position = 0
        self.size = 0
        self.random = random.Random(seed)

    def push(
        self,
        state: Any,
        action: int,
        reward: float,
        next_state: Any,
        done: bool,
        safe_mask: SafeMask,
        next_safe_mask: SafeMask,
    ) -> None:
        current_mask = self._validate_mask(safe_mask, "safe_mask")
        following_mask = self._validate_mask(next_safe_mask, "next_safe_mask")
        if not 0 <= action < self.action_size:
            raise ValueError("action is out of range")
        if not current_mask[action]:
            raise ValueError("masked replay cannot store an uncertified action")

        self.memory[self.position] = MaskedTransition(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            safe_mask=current_mask,
            next_safe_mask=following_mask,
        )
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> list[MaskedTransition]:
        if batch_size > self.size:
            raise ValueError("batch_size cannot be larger than buffer length")
        indices = self.random.sample(range(self.size), batch_size)
        batch: list[MaskedTransition] = []
        for index in indices:
            transition = self.memory[index]
            if transition is None:
                raise RuntimeError("Masked replay buffer contains an uninitialized slot")
            batch.append(transition)
        return batch

    def __len__(self) -> int:
        return self.size

    def _validate_mask(self, mask: SafeMask, name: str) -> SafeMask:
        normalized = tuple(bool(value) for value in mask)
        if len(normalized) != self.action_size:
            raise ValueError(f"{name} must contain exactly {self.action_size} values")
        if not any(normalized):
            raise ValueError(f"{name} must certify at least one action")
        return normalized
