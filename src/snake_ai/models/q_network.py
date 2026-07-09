from __future__ import annotations

import torch
from torch import nn


class QNetwork(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dueling: bool = True,
    ) -> None:
        super().__init__()
        self.dueling = dueling

        if not dueling:
            self.net = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, output_size),
            )
            return

        self.feature = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        # 分别生成状态值和优势值
        self.value_stream = nn.Linear(hidden_size, 1)
        self.advantage_stream = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.dueling:
            return self.net(x)

        features = self.feature(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # 用 Dueling DQN 的方式计算 Q 值：Q(s, a) = V(s) + A(s, a) - mean(A(s, a'))
        # 输出形状完全和普通DQN一样，都是 [batch_size, action_size]，只是计算方式不同。
        return value + advantage - advantage.mean(dim=1, keepdim=True)
