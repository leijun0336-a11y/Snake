from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


# 返回不超过 preferred、并且能够整除 channels 的最大 GroupNorm 组数。
def _group_count(channels: int, preferred: int) -> int:
    
    # GroupNorm 要求通道数能够被组数整除。这个辅助函数让自定义 8、12、16、32
    # 等通道数都能找到合法配置，而不是把组数硬编码成 8 或 4。
    for groups in range(min(channels, preferred), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ResidualBlock(nn.Module):
    """保持空间尺寸不变的残差块，第一层负责扩展感受野，第二层融合连续邻域。"""

    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        # GroupNorm 不依赖 batch 统计量，因此训练时的 replay batch 和 act() 的
        # 单状态 batch=1 使用相同规则，比 BatchNorm 更适合 DQN 的非平稳数据。
        groups = _group_count(channels, 8)
        self.block = nn.Sequential(
            # padding=dilation 保证 3x3 空洞卷积前后的 H、W 不变。
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.GroupNorm(groups, channels),
            nn.ReLU(),
            # 第二层固定为普通 3x3 卷积，重新混合连续格子，缓解连续使用
            # 大 dilation 时可能出现的栅格采样问题。
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
        )
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x 与 self.block(x) 形状均为 [B,C,H,W]，可直接进行残差相加。
        return self.activation(x + self.block(x))


class QNetwork(nn.Module):
    """支持 Vector、Grid 和 Hybrid 三种状态输入的 Dueling Q 网络。"""

    # Grid 的 2、3、4、5 号通道分别是朝左、右、上、下的蛇头。
    # 将四个通道相加后，每个样本只应有一个值为 1 的蛇头位置。
    HEAD_CHANNEL_SLICE = slice(2, 6)
    # 局部分支固定观察蛇头周围 5x5，即上下左右各两格。
    LOCAL_CROP_SIZE = 5

    def __init__(
        self,
        input_size: int | tuple[int, int, int],
        hidden_size: int,
        output_size: int,
        dueling: bool = True,
        state_mode: str = "vector",
        auxiliary_size: int = 20,
        cnn_channels: int = 32,
        cnn_output_channels: int = 8,
        cnn_dilations: tuple[int, ...] = (1, 1, 2),
    ) -> None:
        super().__init__()
        # 尽早检查架构参数，避免在第一次前向传播时才出现难以定位的形状错误。
        if state_mode not in ("vector", "grid", "hybrid"):
            raise ValueError("state_mode must be 'vector', 'grid', or 'hybrid'")
        if cnn_channels <= 0 or cnn_output_channels <= 0:
            raise ValueError("CNN channel sizes must be positive")
        if not cnn_dilations or any(dilation <= 0 for dilation in cnn_dilations):
            raise ValueError("cnn_dilations must contain positive integers")
        if auxiliary_size <= 0:
            raise ValueError("auxiliary_size must be positive")

        self.dueling = dueling
        self.state_mode = state_mode

        if state_mode == "vector":
            # Vector 输入形状为 [B,20]，继续使用两层 MLP，作为低参数量基线。
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
            # Grid 输入形状为 [B,C,H,W]；Hybrid 的第一项同样是该 Grid。
            if not isinstance(input_size, tuple):
                raise TypeError("grid and hybrid modes expect input_size=(channels, height, width)")
            channels, height, width = input_size
            if channels < self.HEAD_CHANNEL_SLICE.stop:
                raise ValueError("grid input must contain the four directional head channels")

            # ===============主干部分==============
            # 共享 CNN 主干不进行下采样：9通道输入先映射到32通道，默认再经过
            # dilation=(1,1,2) 的三个残差块，输出仍为 [B,32,H,W]。
            backbone: list[nn.Module] = [
                nn.Conv2d(channels, cnn_channels, kernel_size=3, padding=1, bias=False),
                # GroupNorm 的两个参数分别是组数和通道数，组数必须能整除通道数。
                nn.GroupNorm(_group_count(cnn_channels, 8), cnn_channels),
                nn.ReLU(),
            ]
            backbone.extend(
                ResidualBlock(cnn_channels, dilation=dilation)
                for dilation in cnn_dilations
            )
            self.cnn = nn.Sequential(*backbone)

            
            # =============分支部分==============
            # 全局和局部分支分别使用自己的 1x1 投影层。两者不共享权重，使全局分支可以侧重布局，局部分支可以侧重蛇头附近的墙和身体。
            branch_groups = _group_count(cnn_output_channels, 4)
            self.global_projection = nn.Sequential(
                nn.Conv2d(cnn_channels, cnn_output_channels, kernel_size=1, bias=False),
                nn.GroupNorm(branch_groups, cnn_output_channels),
                nn.ReLU(),
            )
            self.local_projection = nn.Sequential(
                nn.Conv2d(cnn_channels, cnn_output_channels, kernel_size=1, bias=False),
                nn.GroupNorm(branch_groups, cnn_output_channels),
                nn.ReLU(),
            )
            # 网格尺寸已经由 --width/--height 决定。恒等映射保留完整 HxW，
            # 同时维持统一的分支调用方式，不再把不同棋盘强制缩放到固定尺寸。
            self.global_pool = nn.Identity()

            # 全局维度随实际棋盘面积变化；局部分支始终为 8*5*5=200（默认通道数）。
            global_size = cnn_output_channels * height * width
            local_size = cnn_output_channels * self.LOCAL_CROP_SIZE**2
            fused_size = global_size + local_size
            
            # Hybrid 继续拼接固定长度的人工状态，最终维度为 8*H*W+220（默认配置）。
            if state_mode == "hybrid":
                fused_size += auxiliary_size
            self.feature = nn.Sequential(
                nn.Linear(fused_size, hidden_size),
                nn.ReLU(),
            )
            feature_size = hidden_size

        # ===============全连接层部分的选择==============
        if not dueling:
            # 关闭 Dueling 时直接从共享特征预测每个动作的 Q 值。
            self.head = nn.Linear(feature_size, output_size)
        elif state_mode == "vector":
            # 保持 Vector 基线的轻量单层 Value/Advantage 分支，避免同时改变基线容量。
            self.value_stream = nn.Linear(feature_size, 1)
            self.advantage_stream = nn.Linear(feature_size, output_size)
        else:
            # Grid/Hybrid 的两个分支各自增加一层非线性：默认 256->128->1/3。
            stream_size = max(hidden_size // 2, 1)
            self.value_stream = nn.Sequential(
                nn.Linear(feature_size, stream_size),
                nn.ReLU(),
                nn.Linear(stream_size, 1),
            )
            self.advantage_stream = nn.Sequential(
                nn.Linear(feature_size, stream_size),
                nn.ReLU(),
                nn.Linear(stream_size, output_size),
            )

    def _crop_around_head(
        self, features: torch.Tensor, grid: torch.Tensor
    ) -> torch.Tensor:
        """按每个样本的蛇头坐标，批量提取可反向传播的 5x5 局部特征。"""
        # grid[:,2:6] 是四个方向蛇头通道；求和后得到 [B,H,W] 的蛇头 mask。
        head_mask = grid[:, self.HEAD_CHANNEL_SLICE].sum(dim=1)
        batch_size = head_mask.shape[0]
        # 展平后取 argmax，得到每个样本蛇头在 H*W 中的一维索引。
        flat_positions = head_mask.flatten(start_dim=1).argmax(dim=1)

        # 四周补两格后，即使蛇头位于角落，也能获得完整的 5x5 窗口。
        radius = self.LOCAL_CROP_SIZE // 2
        padded = F.pad(features, (radius, radius, radius, radius))
        # unfold 为棋盘上的每个原始位置生成一个展平的 5x5 候选窗口：
        # [B,C,H+4,W+4] -> [B,C*25,H*W]。
        all_patches = F.unfold(padded, kernel_size=self.LOCAL_CROP_SIZE)
        # 把蛇头索引扩展到所有 C*25 个特征，用一次 gather 在 GPU 上批量取出
        # 每个样本对应的窗口，避免 Python 循环和逐样本 CPU/GPU 同步。
        gather_index = flat_positions.view(batch_size, 1, 1).expand(
            -1, all_patches.shape[1], -1
        )
        crops = all_patches.gather(2, gather_index).squeeze(2)
        # [B,C*25] 还原成 [B,C,5,5]，随后由调用方展平为200维。
        return crops.reshape(
            batch_size,
            features.shape[1],
            self.LOCAL_CROP_SIZE,
            self.LOCAL_CROP_SIZE,
        )

    # 把两个CNN分支拉成一条向量。
    def _spatial_features(self, grid: torch.Tensor) -> torch.Tensor:
        # grid 的高宽由 --width/--height 决定。
        
        # 共享特征保持实际 HxW 分辨率，两个分支从同一语义特征图出发。
        shared = self.cnn(grid)
        # 全局分支：[B,32,H,W] -> [B,8,H,W] -> [B,8*H*W]。
        global_features = self.global_pool(self.global_projection(shared)).flatten(1)
        # 局部分支：[B,32,H,W] -> [B,8,H,W] -> 蛇头5x5 -> [B,200]。
        local_map = self.local_projection(shared)
        local_features = self._crop_around_head(local_map, grid).flatten(1)
        # 默认得到 [B,8*H*W+200]，同时保留全局布局和蛇头附近的精确格子信息。
        return torch.cat((global_features, local_features), dim=1)

    def forward(
        self, x: torch.Tensor | tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        if self.state_mode == "hybrid":
            # Hybrid: 动态空间特征 + 20维人工特征 -> hidden_size 维共享决策特征。
            grid, auxiliary_state = x
            features = torch.cat(
                (self._spatial_features(grid), auxiliary_state), dim=1
            )
            # mlp到hidden_size维共享特征。
            features = self.feature(features)
        elif self.state_mode == "grid":
            # 纯 Grid 不再提供 hunger 旁路，只使用9通道空间状态。
            features = self.feature(self._spatial_features(x))
        else:
            # Vector: 20维人工状态直接通过 MLP。
            features = self.feature(x)

        if not self.dueling:
            return self.head(features)
        
        # dueling DQN 的两个分支的输出。
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # 减去动作优势均值可消除 V 与 A 的不可辨识常数偏移。
        return value + advantage - advantage.mean(dim=1, keepdim=True)
