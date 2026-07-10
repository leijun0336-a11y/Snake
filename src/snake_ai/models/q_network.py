from __future__ import annotations

import torch
from torch import nn


class DilatedResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        # 对 3x3 卷积使用 padding=dilation，可扩大感受野并保持特征图高宽不变。
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
        auxiliary_size: int = 19,
        cnn_channels: int = 32,
        cnn_output_channels: int = 16,
        cnn_dilations: tuple[int, ...] = (1, 2, 4),
        cnn_pool_size: tuple[int, int] = (5, 5),
    ) -> None:
        super().__init__()
        if state_mode not in ("vector", "grid", "hybrid"):
            raise ValueError("state_mode must be 'vector', 'grid', or 'hybrid'")
        if cnn_channels <= 0 or cnn_output_channels <= 0:
            raise ValueError("CNN channel sizes must be positive")
        if not cnn_dilations or any(dilation <= 0 for dilation in cnn_dilations):
            raise ValueError("cnn_dilations must contain positive integers")
        if len(cnn_pool_size) != 2 or any(size <= 0 for size in cnn_pool_size):
            raise ValueError("cnn_pool_size must contain two positive integers")
        if auxiliary_size <= 0:
            raise ValueError("auxiliary_size must be positive")

        self.dueling = dueling
        self.state_mode = state_mode

        if state_mode == "vector":
            # Vector baseline 只处理环境提供的 19 维人工状态，不经过 CNN。
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
            # Grid 和 Hybrid 共用同一套空洞卷积主干，保证对照实验只改变输入融合方式。
            if not isinstance(input_size, tuple):
                raise TypeError("grid and hybrid modes expect input_size=(channels, height, width)")
            channels, _, _ = input_size
            cnn_layers: list[nn.Module] = [
                nn.Conv2d(channels, cnn_channels, kernel_size=3, padding=1),
                nn.ReLU(),
            ]
            # 根据 dilation 配置动态创建残差块，避免把 1、2、4 和通道数重复写死。
            cnn_layers.extend(
                DilatedResidualBlock(cnn_channels, dilation=dilation)
                for dilation in cnn_dilations
            )
            cnn_layers.extend(
                [
                    # 1x1 卷积只压缩通道，不改变空间尺寸。
                    nn.Conv2d(cnn_channels, cnn_output_channels, kernel_size=1),
                    nn.ReLU(),
                    # 固定池化输出尺寸，让不同地图大小都能接入同一个全连接层。
                    nn.AdaptiveAvgPool2d(cnn_pool_size),
                    nn.Flatten(),
                ]
            )
            self.cnn = nn.Sequential(*cnn_layers)

            # 展平维度由压缩通道数和池化尺寸推导，调整 CNN 配置时无需手工同步。
            cnn_feature_size = cnn_output_channels * cnn_pool_size[0] * cnn_pool_size[1]
            # Hybrid 比纯 Grid 多拼接完整的 19 维人工状态。
            feature_size = (
                cnn_feature_size + auxiliary_size
                if state_mode == "hybrid"
                else cnn_feature_size
            )
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
        if self.state_mode == "hybrid":
            grid, auxiliary_state = x
            features = self.cnn(grid)
            # 拼接发生在 CNN 展平之后，人工状态不会被误当作图像通道卷积。
            features = torch.cat((features, auxiliary_state), dim=1)
            features = self.feature(features)
        elif self.state_mode == "grid":
            # 纯 Grid 模式只依赖网格特征，不再额外提供 direction one-hot。
            features = self.feature(self.cnn(x))
        else:
            features = self.feature(x)

        if not self.dueling:
            return self.head(features)

        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


# SnakeEnv.get_state() 当前返回 19 维向量，用于 vector baseline。
# SnakeEnv.get_grid_state() 返回纯 grid：
# - grid shape = [5, height, width]
# - channel 0: 边界格子
# - channel 1: 蛇身，不含蛇头
# - channel 2: 蛇头
# - channel 3: 食物
# - channel 4: 蛇身顺序，蛇头为 1.0，越靠近尾巴数值越小
# SnakeEnv.get_hybrid_state() 返回 (grid, vector_state)，其中 vector_state shape = [19]。


#  19 维低维状态向量输入：

# | 序号 | 维度 | 含义 |
# |------|------|------|
# | 1 | `danger_straight` | 直行下一步是否危险。 |
# | 2 | `danger_right` | 右转下一步是否危险。 |
# | 3 | `danger_left` | 左转下一步是否危险。 |
# | 4 | `direction_left` | 当前是否向左移动。 |
# | 5 | `direction_right` | 当前是否向右移动。 |
# | 6 | `direction_up` | 当前是否向上移动。 |
# | 7 | `direction_down` | 当前是否向下移动。 |
# | 8 | `food_left` | 食物是否在蛇头左侧。 |
# | 9 | `food_right` | 食物是否在蛇头右侧。 |
# | 10 | `food_up` | 食物是否在蛇头上方。 |
# | 11 | `food_down` | 食物是否在蛇头下方。 |
# | 12 | `food_dx_norm` | 食物相对蛇头的 x 距离，归一化到 `[-1, 1]`。 |
# | 13 | `food_dy_norm` | 食物相对蛇头的 y 距离，归一化到 `[-1, 1]`。 |
# | 14 | `wall_distance_straight` | 直行方向到墙的归一化距离。 |
# | 15 | `wall_distance_right` | 右转方向到墙的归一化距离。 |
# | 16 | `wall_distance_left` | 左转方向到墙的归一化距离。 |
# | 17 | `body_distance_straight` | 直行方向最近身体的归一化距离；没有身体时为 `1.0`。 |
# | 18 | `body_distance_right` | 右转方向最近身体的归一化距离；没有身体时为 `1.0`。 |
# | 19 | `body_distance_left` | 左转方向最近身体的归一化距离；没有身体时为 `1.0`。 |
