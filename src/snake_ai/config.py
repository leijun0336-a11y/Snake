from dataclasses import dataclass
from pathlib import Path


# 获取当前脚本往上数第 3 层的父目录，并将其作为“项目根目录”。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
# 定义模型权重（检查点）的存放路径。
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
# 定义运行日志或实验结果的存放路径。
RUNS_DIR = PROJECT_ROOT / "runs"


@dataclass(frozen=True) # frozen=True表示只读
class EnvConfig:
    # 游戏网格的宽度，单位是格子数量。
    width: int = 20
    # 游戏网格的高度，单位是格子数量。
    height: int = 20
    # 渲染时每个格子的像素大小，只影响 pygame 显示效果。
    cell_size: int = 24
    # 渲染帧率，只影响可视化速度，不影响无渲染训练。帧率越高，蛇跑得越快。
    fps: int = 30


@dataclass(frozen=True)
class TrainConfig:
    # 最大训练局数；如果早停没有触发，训练会在达到这个上限后结束。
    episodes: int = 5000
    # 每次从经验池中采样多少条经验用于一次神经网络更新。
    batch_size: int = 64
    # 折扣因子，越接近 1 越重视未来奖励。
    gamma: float = 0.99
    # Adam 优化器的学习率，控制网络参数每次更新的步幅。
    learning_rate: float = 1e-3
    # 经验回放池最大容量，存放 state/action/reward/next_state/done。
    replay_buffer_size: int = 100_000
    # 初始探索率，训练前期按这个概率随机选择动作。
    epsilon_start: float = 1.0
    # 最低探索率，防止后期完全不探索。
    epsilon_end: float = 0.01
    # 每局结束后的探索率衰减系数。
    epsilon_decay: float = 0.995
    # 每隔多少次学习(buffer满后蛇每走一步就learn()一次)步骤，把 policy network 的参数同步到 target network。
    target_update_interval: int = 1000
    # Q 网络隐藏层神经元数量。
    hidden_size: int = 128
    # Grid/Hybrid CNN 主干通道数。
    cnn_channels: int = 32
    # 1x1 卷积压缩后的通道数。
    cnn_output_channels: int = 16
    # 空洞残差块的 dilation 序列。
    cnn_dilations: tuple[int, ...] = (1, 2, 4)
    # 自适应平均池化输出的高和宽。
    cnn_pool_size: tuple[int, int] = (5, 5)
    # 随机种子，用于让初始化、探索和食物生成尽量可复现。
    seed: int = 42
