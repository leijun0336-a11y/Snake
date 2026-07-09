from __future__ import annotations

import torch
from torch import nn


class DilatedResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation),
        )
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class QNetwork(nn.Module):
    def __init__(
        self,
        input_size: int | tuple[int, int, int],
        hidden_size: int,
        output_size: int,
        dueling: bool = True,
        state_mode: str = "vector",
        direction_size: int = 4,
    ) -> None:
        super().__init__()
        if state_mode not in ("vector", "grid"):
            raise ValueError("state_mode must be 'vector' or 'grid'")

        self.dueling = dueling
        self.state_mode = state_mode

        if state_mode == "vector":
            if not isinstance(input_size, int):
                raise TypeError("vector state mode expects an integer input_size")
            self.feature = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
            )
            feature_size = hidden_size
        else:
            if not isinstance(input_size, tuple):
                raise TypeError("grid state mode expects input_size=(channels, height, width)")
            channels, _, _ = input_size
            self.cnn = nn.Sequential(
                nn.Conv2d(channels, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                DilatedResidualBlock(32, dilation=1),
                DilatedResidualBlock(32, dilation=2),
                DilatedResidualBlock(32, dilation=4),
                nn.Conv2d(32, 16, kernel_size=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((5, 5)),
                nn.Flatten(),
            )
            feature_size = 16 * 5 * 5 + direction_size
            self.feature = nn.Sequential(
                nn.Linear(feature_size, hidden_size),
                nn.ReLU(),
            )
            feature_size = hidden_size

        if not dueling:
            self.head = nn.Linear(feature_size, output_size)
            return

        self.value_stream = nn.Linear(feature_size, 1)
        self.advantage_stream = nn.Linear(feature_size, output_size)

    def forward(
        self, x: torch.Tensor | tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        if self.state_mode == "grid":
            grid, direction = x
            features = self.cnn(grid)
            features = torch.cat((features, direction), dim=1)
            features = self.feature(features)
        else:
            features = self.feature(x)

        if not self.dueling:
            return self.head(features)

        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


# SnakeEnv.get_state() 当前返回 19 维向量，用于 vector baseline。
# SnakeEnv.get_grid_state() 返回 (grid, direction)：
# - grid shape = [5, height, width]
# - channel 0: 边界格子
# - channel 1: 蛇身，不含蛇头
# - channel 2: 蛇头
# - channel 3: 食物
# - channel 4: 蛇身顺序，蛇头为 1.0，越靠近尾巴数值越小
# - direction shape = [4]，顺序为 left, right, up, down
