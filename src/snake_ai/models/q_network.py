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


# SnakeEnv.get_state() 当前返回 19 维状态向量：
# 1. danger_straight: 直行下一步是否危险。
# 2. danger_right: 右转下一步是否危险。
# 3. danger_left: 左转下一步是否危险。
# 4. direction_left: 当前是否向左移动。
# 5. direction_right: 当前是否向右移动。
# 6. direction_up: 当前是否向上移动。
# 7. direction_down: 当前是否向下移动。
# 8. food_left: 食物是否在蛇头左侧。
# 9. food_right: 食物是否在蛇头右侧。
# 10. food_up: 食物是否在蛇头上方。
# 11. food_down: 食物是否在蛇头下方。
# 12. food_dx_norm: 食物相对蛇头的 x 距离，归一化到 [-1, 1]。
# 13. food_dy_norm: 食物相对蛇头的 y 距离，归一化到 [-1, 1]。
# 14. wall_distance_straight: 直行方向到墙的归一化距离。
# 15. wall_distance_right: 右转方向到墙的归一化距离。
# 16. wall_distance_left: 左转方向到墙的归一化距离。
# 17. body_distance_straight: 直行方向最近身体的归一化距离；没有身体时为 1.0。
# 18. body_distance_right: 右转方向最近身体的归一化距离；没有身体时为 1.0。
# 19. body_distance_left: 左转方向最近身体的归一化距离；没有身体时为 1.0。
