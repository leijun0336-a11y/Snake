# 处理自引用问题
from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from itertools import islice
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
    # 这条聚合经验实际跨越的环境步数；1 表示传统 one-step TD。
    n_steps: int = 1


class NStepAccumulator:
    """把连续 one-step 交互聚合成可写入 replay 的 n-step 经验。"""

    def __init__(self, n_step: int, gamma: float) -> None:
        if n_step < 1:
            raise ValueError("n_step must be at least 1")
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be between 0 and 1")
        self.n_step = n_step
        self.gamma = gamma
        self.pending: deque[Transition] = deque()

    def append(
        self,
        state: Any,
        action: int,
        reward: float,
        next_state: Any,
        done: bool,
    ) -> tuple[Transition, ...]:
        """加入一步交互，返回本次已经成熟的聚合经验。"""

        self.pending.append(Transition(state, action, reward, next_state, done))
        ready: list[Transition] = []

        if len(self.pending) >= self.n_step:
            ready.append(self._pop_aggregate(self.n_step))

        # 自然终止后没有未来状态可 bootstrap，要把不足 n 步的尾部全部冲刷出来。
        if done:
            ready.extend(self.flush())
        return tuple(ready)

    def flush(self) -> tuple[Transition, ...]:
        """冲刷 episode 尾部；截断样本保留 done=False 和实际跨度 k。"""

        ready: list[Transition] = []
        while self.pending:
            ready.append(self._pop_aggregate(len(self.pending)))
        return tuple(ready)

    def _pop_aggregate(self, horizon: int) -> Transition:
        window = tuple(islice(self.pending, horizon))
        if not window:
            raise RuntimeError("cannot aggregate an empty n-step window")

        discounted_reward = sum(
            (self.gamma**offset) * transition.reward for offset, transition in enumerate(window)
        )
        first = window[0]
        last = window[-1]
        self.pending.popleft()
        return Transition(
            state=first.state,
            action=first.action,
            reward=discounted_reward,
            next_state=last.next_state,
            done=last.done,
            n_steps=horizon,
        )

    def __len__(self) -> int:
        return len(self.pending)


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
        n_steps: int = 1,
    ) -> None:
        if n_steps < 1:
            raise ValueError("n_steps must be at least 1")
        # 写满后从 position 指向的位置覆盖最旧经验，不复制 NumPy state 本身。
        self.memory[self.position] = Transition(
            state,
            action,
            reward,
            next_state,
            done,
            n_steps,
        )
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
