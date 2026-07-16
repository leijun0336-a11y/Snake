from __future__ import annotations

import argparse
import statistics
from collections.abc import Callable

import torch
from torch import nn
from torch.nn import functional as F

from snake_ai.models.q_network import QNetwork


class UnfoldQNetwork(QNetwork):
    """修改前的网络路径，仅用于同架构性能对照。"""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.global_pool = nn.Identity()

    def _crop_around_head(
        self,
        features: torch.Tensor,
        grid: torch.Tensor,
    ) -> torch.Tensor:
        head_mask = grid[:, self.HEAD_CHANNEL_SLICE].sum(dim=1)
        positions = head_mask.flatten(1).argmax(dim=1)
        radius = self.LOCAL_CROP_SIZE // 2
        patches = F.unfold(
            F.pad(features, (radius, radius, radius, radius)),
            kernel_size=self.LOCAL_CROP_SIZE,
        )
        index = positions[:, None, None].expand(-1, patches.shape[1], -1)
        return patches.gather(2, index).squeeze(2).reshape(
            len(grid), features.shape[1], self.LOCAL_CROP_SIZE, self.LOCAL_CROP_SIZE
        )

    def _spatial_features(self, grid: torch.Tensor) -> torch.Tensor:
        shared = self.cnn(grid)
        global_features = self.global_pool(self.global_projection(shared)).flatten(1)
        local_features = self._crop_around_head(
            self.local_projection(shared), grid
        ).flatten(1)
        return torch.cat((global_features, local_features), dim=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark unfold and direct feature crops.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--channels", type=int, default=8)
    parser.add_argument("--sizes", type=int, nargs="+", default=(6, 20))
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=500)
    return parser.parse_args()


def make_grid(batch_size: int, size: int, device: torch.device) -> torch.Tensor:
    grid = torch.randn((batch_size, 9, size, size), device=device)
    grid[:, 2:6] = 0.0
    samples = torch.arange(batch_size, device=device)
    positions = samples % (size * size)
    grid[samples, 2 + samples % 4, positions // size, positions % size] = 1.0
    return grid


def measure(
    step: Callable[[], None],
    warmup: int,
    iterations: int,
) -> tuple[float, float]:
    for _ in range(warmup):
        step()
    torch.cuda.synchronize()

    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    timings: list[float] = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        step()
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end))

    peak_mib = (torch.cuda.max_memory_allocated() - baseline) / 1024**2
    return statistics.median(timings), peak_mib


def benchmark_crop(
    network: QNetwork,
    grid: torch.Tensor,
    channels: int,
    warmup: int,
    iterations: int,
) -> tuple[float, float]:
    features = torch.randn(
        (len(grid), channels, grid.shape[2], grid.shape[3]),
        device=grid.device,
        requires_grad=True,
    )
    upstream = torch.randn((len(grid), channels, 5, 5), device=grid.device)

    def step() -> None:
        features.grad = None
        network._crop_around_head(features, grid).backward(upstream)

    return measure(step, warmup, iterations)


def benchmark_network(
    network: QNetwork,
    grid: torch.Tensor,
    warmup: int,
    iterations: int,
) -> tuple[float, float]:
    def step() -> None:
        network.zero_grad(set_to_none=True)
        network(grid).sum().backward()

    return measure(step, warmup, iterations)


def benchmark_dqn_update(
    network_class: type[QNetwork],
    state: dict[str, torch.Tensor],
    grid: torch.Tensor,
    warmup: int,
    iterations: int,
) -> tuple[float, float]:
    policy = build_network(network_class, grid.shape[2], state, grid.device)
    target = build_network(network_class, grid.shape[2], state, grid.device).eval()
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    next_grid = torch.roll(grid, shifts=1, dims=0)
    actions = torch.arange(len(grid), device=grid.device) % 3
    rewards = torch.linspace(-1.0, 1.0, len(grid), device=grid.device)
    dones = (actions == 0).float()

    def step() -> None:
        optimizer.zero_grad(set_to_none=True)
        current_q = policy(grid).gather(1, actions[:, None]).squeeze(1)
        with torch.no_grad():
            next_actions = policy(next_grid).argmax(dim=1)
            next_q = target(next_grid).gather(1, next_actions[:, None]).squeeze(1)
            expected_q = rewards + 0.99 * next_q * (1.0 - dones)
        F.smooth_l1_loss(current_q, expected_q).backward()
        optimizer.step()

    return measure(step, warmup, iterations)


def build_network(
    network_class: type[QNetwork],
    size: int,
    state: dict[str, torch.Tensor],
    device: torch.device,
) -> QNetwork:
    network = network_class(
        input_size=(9, size, size),
        hidden_size=128,
        output_size=3,
        state_mode="grid",
        cnn_channels=32,
        cnn_output_channels=8,
        cnn_dilations=(1, 1, 2),
    ).to(device)
    network.load_state_dict(state, strict=True)
    return network


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; this benchmark does not fall back to CPU.")
    if args.batch_size < 1 or args.channels < 1:
        raise ValueError("batch-size and channels must be positive")
    if args.warmup < 0 or args.iterations < 1 or any(size < 5 for size in args.sizes):
        raise ValueError("warmup must be non-negative, iterations positive, and sizes at least 5")

    device = torch.device("cuda")
    print("mode size variant crop_ms crop_peak_MiB network_ms network_peak_MiB dqn_steps_s")
    for deterministic in (False, True):
        torch.use_deterministic_algorithms(deterministic)
        for size in args.sizes:
            grid = make_grid(args.batch_size, size, device)
            torch.manual_seed(42)
            template = QNetwork(
                input_size=(9, size, size),
                hidden_size=128,
                output_size=3,
                state_mode="grid",
                cnn_channels=32,
                cnn_output_channels=8,
                cnn_dilations=(1, 1, 2),
            )
            state = {key: value.clone() for key, value in template.state_dict().items()}

            for label, network_class in (("unfold", UnfoldQNetwork), ("direct", QNetwork)):
                network = build_network(network_class, size, state, device)
                crop_ms, crop_memory = benchmark_crop(
                    network, grid, args.channels, args.warmup, args.iterations
                )
                network_ms, network_memory = benchmark_network(
                    network, grid, args.warmup, args.iterations
                )
                dqn_ms, _ = benchmark_dqn_update(
                    network_class, state, grid, args.warmup, args.iterations
                )
                mode = "deterministic" if deterministic else "default"
                print(
                    f"{mode} {size} {label} {crop_ms:.4f} {crop_memory:.2f} "
                    f"{network_ms:.4f} {network_memory:.2f} {1000.0 / dqn_ms:.2f}"
                )


if __name__ == "__main__":
    main()
