# 处理自引用问题
from __future__ import annotations

import torch
from torch import nn


class QNetwork(nn.Module):
    # input_size是输入状态的维度，当前为11，output_size是输出动作的数量，当前为3。
    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
# 注：输入状态的维度包括：(以左上角为(0, 0)，向右x增大，向下y增大)
# 1. 前方是否危险(相对方向位置特征，0或1.0)
# 2. 右方是否危险
# 3. 左方是否危险
# 4. 当前方向是否向左(绝对方向特征，0或1.0)
# 5. 当前方向是否向右
# 6. 当前方向是否向上
# 7. 当前方向是否向下
# 8. 食物是否在左侧(绝对位置食物特征，0或1.0)
# 9. 食物是否在右侧
# 10. 食物是否在上方
# 11. 食物是否在下方

# 具体可参考snake_env.py文件中Snake_env类的get_state()方法

# 输出的是一个三维向量，向量内数值可以是任意实数
# 第0个数：直行的估计价值；第一个数：右转的估计价值；第2个数：左转的估计价值。
