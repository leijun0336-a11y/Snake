import torch
from torch import nn
from torch.nn import functional as F

from snake_ai.models.q_network import QNetwork


def _unfold_crop(
    features: torch.Tensor,
    grid: torch.Tensor,
    crop_size: int = QNetwork.LOCAL_CROP_SIZE,
) -> torch.Tensor:
    """保留修改前的裁剪公式，仅用于验证新实现严格等价。"""
    head_mask = grid[:, QNetwork.HEAD_CHANNEL_SLICE].sum(dim=1)
    batch_size = head_mask.shape[0]
    positions = head_mask.flatten(1).argmax(dim=1)
    radius = crop_size // 2
    patches = F.unfold(
        F.pad(features, (radius, radius, radius, radius)),
        kernel_size=crop_size,
    )
    index = positions[:, None, None].expand(-1, patches.shape[1], -1)
    return patches.gather(2, index).squeeze(2).reshape(
        batch_size,
        features.shape[1],
        crop_size,
        crop_size,
    )


def _grid_with_heads(height: int, width: int) -> torch.Tensor:
    """生成覆盖棋盘全部位置的 batch，每个样本恰有一个蛇头。"""
    grid = torch.zeros((height * width, 9, height, width))
    for position in range(height * width):
        y, x = divmod(position, width)
        grid[position, 2 + position % 4, y, x] = 1.0
    return grid


def _network(height: int = 6, width: int = 6) -> QNetwork:
    return QNetwork(
        input_size=(9, height, width),
        hidden_size=32,
        output_size=3,
        state_mode="grid",
        cnn_channels=8,
        cnn_output_channels=4,
        cnn_dilations=(1,),
    )


def test_direct_crop_matches_unfold_at_every_board_position() -> None:
    torch.manual_seed(1)
    height = width = 6
    network = _network(height, width)
    grid = _grid_with_heads(height, width)
    features = torch.randn((height * width, 4, height, width))

    expected = _unfold_crop(features, grid)
    actual = network._crop_around_head(features, grid)

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_direct_crop_preserves_non_uniform_gradients() -> None:
    torch.manual_seed(2)
    height, width = 6, 7
    network = _network(height, width)
    grid = _grid_with_heads(height, width)[[0, width - 1, width * (height - 1), -1, 17]]
    reference_features = torch.randn((len(grid), 4, height, width), requires_grad=True)
    direct_features = reference_features.detach().clone().requires_grad_(True)
    upstream = torch.linspace(-1.0, 1.0, len(grid) * 4 * 25).reshape(len(grid), 4, 5, 5)

    expected = _unfold_crop(reference_features, grid)
    actual = network._crop_around_head(direct_features, grid)
    (expected * upstream).sum().backward()
    (actual * upstream).sum().backward()

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        direct_features.grad,
        reference_features.grad,
        rtol=0.0,
        atol=0.0,
    )


class _ReferenceQNetwork(QNetwork):
    """模拟修改前的 Identity 和 unfold 路径，用于端到端兼容性测试。"""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.global_pool = nn.Identity()

    def _crop_around_head(
        self,
        features: torch.Tensor,
        grid: torch.Tensor,
    ) -> torch.Tensor:
        return _unfold_crop(features, grid)

    def _spatial_features(self, grid: torch.Tensor) -> torch.Tensor:
        shared = self.cnn(grid)
        global_features = self.global_pool(self.global_projection(shared)).flatten(1)
        local_features = self._crop_around_head(
            self.local_projection(shared), grid
        ).flatten(1)
        return torch.cat((global_features, local_features), dim=1)


def test_full_q_values_and_state_dict_remain_compatible() -> None:
    torch.manual_seed(3)
    reference = _ReferenceQNetwork(
        input_size=(9, 6, 6),
        hidden_size=32,
        output_size=3,
        state_mode="grid",
        cnn_channels=8,
        cnn_output_channels=4,
        cnn_dilations=(1,),
    )
    current = _network()
    reference_state = reference.state_dict()

    assert reference_state.keys() == current.state_dict().keys()
    assert "local_offsets" not in reference_state
    current.load_state_dict(reference_state, strict=True)

    grid = _grid_with_heads(6, 6)[[0, 7, 20, 35]]
    with torch.no_grad():
        expected = reference(grid)
        actual = current(grid)

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
