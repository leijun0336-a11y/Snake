# 处理自引用问题
from __future__ import annotations

import random
# 两端都能快速弹出和插入数据的列表。
from collections import deque
from dataclasses import dataclass
# 用于给双端队列声明元素类型。
from typing import Deque


@dataclass(frozen=True)  # frozen=True，创建后字段不能被重新赋值。

# 定义一条"经验"中包含哪些数据，成为经验类
class Transition:
    # 当前状态，也就是执行动作前的状态。
    state: list[float]
    # 在当前状态下执行的动作。
    action: int
    # 执行动作后环境返回的奖励。
    reward: float
    # 执行动作后环境返回的下一个状态。
    next_state: list[float]
    # 执行动作后这一局是否结束。
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int | None = None) -> None:
        # capacity 表示经验池最多保存多少条经验，必须是正数。
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        # 用固定最大长度的队列保存经验；满了以后，再加入新经验会自动丢弃最旧经验。
        self.memory: Deque[Transition] = deque(maxlen=capacity)
        # 独立的随机数生成器，用于随机采样经验；传入 seed 后采样过程更容易复现。
        self.random = random.Random(seed)

    def push(
        self,
        state: list[float],
        action: int,
        reward: float,
        next_state: list[float],
        done: bool,
    ) -> None:
        # 把一次交互产生的 state/action/reward/next_state/done 打包成 Transition 存入经验池。
        self.memory.append(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> list[Transition]:
        # 采样数量不能超过当前经验池里已有的经验数量。
        if batch_size > len(self.memory):
            raise ValueError("batch_size cannot be larger than buffer length")
        # 从经验池中随机抽取 batch_size 条经验，用于一次 DQN 训练更新。返回一个列表。
        return self.random.sample(list(self.memory), batch_size)

    def __len__(self) -> int:
        # 让 len(buffer) 可以直接返回当前经验池中的经验条数。
        return len(self.memory)
