from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from snake_ai.models.q_network import ResidualBlock, _group_count


class QNetworkOld(nn.Module):
    """版本2 Q网络，仅用于加载和评估历史checkpoint。"""

    HEAD_CHANNEL_SLICE = slice(2, 6)
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
        cnn_pool_size: tuple[int, int] = (10, 10),
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
                raise TypeError(
                    "grid and hybrid modes expect input_size=(channels, height, width)"
                )
            channels, height, width = input_size
            if channels < self.HEAD_CHANNEL_SLICE.stop:
                raise ValueError("grid input must contain the four directional head channels")

            backbone: list[nn.Module] = [
                nn.Conv2d(channels, cnn_channels, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(_group_count(cnn_channels, 8), cnn_channels),
                nn.ReLU(),
            ]
            backbone.extend(
                ResidualBlock(cnn_channels, dilation=dilation)
                for dilation in cnn_dilations
            )
            self.cnn = nn.Sequential(*backbone)

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
            self.global_pool = self._build_pool_layer(height, width, cnn_pool_size)

            global_size = cnn_output_channels * cnn_pool_size[0] * cnn_pool_size[1]
            local_size = cnn_output_channels * self.LOCAL_CROP_SIZE**2
            fused_size = global_size + local_size
            if state_mode == "hybrid":
                fused_size += auxiliary_size
            self.feature = nn.Sequential(
                nn.Linear(fused_size, hidden_size),
                nn.ReLU(),
            )
            feature_size = hidden_size

        if not dueling:
            self.head = nn.Linear(feature_size, output_size)
        elif state_mode == "vector":
            self.value_stream = nn.Linear(feature_size, 1)
            self.advantage_stream = nn.Linear(feature_size, output_size)
        else:
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

    @staticmethod
    def _build_pool_layer(
        height: int,
        width: int,
        pool_size: tuple[int, int],
    ) -> nn.Module:
        pool_height, pool_width = pool_size
        if height % pool_height == 0 and width % pool_width == 0:
            return nn.AvgPool2d(
                kernel_size=(height // pool_height, width // pool_width),
                stride=(height // pool_height, width // pool_width),
            )
        return nn.AdaptiveAvgPool2d(pool_size)

    def _crop_around_head(
        self,
        features: torch.Tensor,
        grid: torch.Tensor,
    ) -> torch.Tensor:
        head_mask = grid[:, self.HEAD_CHANNEL_SLICE].sum(dim=1)
        batch_size = head_mask.shape[0]
        flat_positions = head_mask.flatten(start_dim=1).argmax(dim=1)

        radius = self.LOCAL_CROP_SIZE // 2
        padded = F.pad(features, (radius, radius, radius, radius))
        all_patches = F.unfold(padded, kernel_size=self.LOCAL_CROP_SIZE)
        gather_index = flat_positions.view(batch_size, 1, 1).expand(
            -1,
            all_patches.shape[1],
            -1,
        )
        crops = all_patches.gather(2, gather_index).squeeze(2)
        return crops.reshape(
            batch_size,
            features.shape[1],
            self.LOCAL_CROP_SIZE,
            self.LOCAL_CROP_SIZE,
        )

    def _spatial_features(self, grid: torch.Tensor) -> torch.Tensor:
        shared = self.cnn(grid)
        global_features = self.global_pool(self.global_projection(shared)).flatten(1)
        local_map = self.local_projection(shared)
        local_features = self._crop_around_head(local_map, grid).flatten(1)
        return torch.cat((global_features, local_features), dim=1)

    def forward(
        self,
        x: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        if self.state_mode == "hybrid":
            grid, auxiliary_state = x
            features = torch.cat(
                (self._spatial_features(grid), auxiliary_state),
                dim=1,
            )
            features = self.feature(features)
        elif self.state_mode == "grid":
            features = self.feature(self._spatial_features(x))
        else:
            features = self.feature(x)

        if not self.dueling:
            return self.head(features)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        return value + advantage - advantage.mean(dim=1, keepdim=True)

