from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RUNS_DIR = PROJECT_ROOT / "runs"


@dataclass(frozen=True)
class EnvConfig:
    width: int = 20
    height: int = 20
    cell_size: int = 24
    fps: int = 30


@dataclass(frozen=True)
class TrainConfig:
    episodes: int = 15000
    batch_size: int = 128
    gamma: float = 0.99
    learning_rate: float = 1e-3
    replay_buffer_size: int = 100_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    target_update_interval: int = 1000
    hidden_size: int = 256
    cnn_channels: int = 32
    cnn_output_channels: int = 8
    cnn_dilations: tuple[int, ...] = (1, 1, 2)
    cnn_pool_size: tuple[int, int] = (10, 10)
    seed: int = 42
