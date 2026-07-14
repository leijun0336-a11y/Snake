from __future__ import annotations

import os
import random
import statistics

import numpy as np
import torch


def summarize_values(values: list[int] | list[float]) -> dict[str, float]:
    """返回数值序列的均值、标准差、最小值、最大值和最后一个值。"""

    return {
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min": float(min(values)),
        "max": float(max(values)),
        "last": float(values[-1]),
    }


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Set random seeds for Python, NumPy, PyTorch, and CUDA.

    deterministic=True 会尽量让 CUDA 计算也使用确定性算法，但可能略微降低速度。
    不同硬件、驱动、CUDA 或 PyTorch 版本之间仍可能存在细微差异。
    """
    # Python 内置 random 模块的随机种子。
    random.seed(seed)
    # Python 哈希随机化种子；需在进程启动前设置才对所有哈希行为完全生效。
    os.environ["PYTHONHASHSEED"] = str(seed)
    # NumPy 随机种子。
    np.random.seed(seed)
    # PyTorch CPU 随机种子。
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        # 当前 GPU 的随机种子。
        torch.cuda.manual_seed(seed)
        # 所有 GPU 的随机种子，多卡训练时有用。
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        # 让 cuDNN 尽量选择确定性算法。
        torch.backends.cudnn.deterministic = True
        # 关闭 cuDNN 自动搜索最快算法，避免因算法选择带来不确定性。
        torch.backends.cudnn.benchmark = False
        # 严格要求确定性算法；若算子没有确定性实现则立即报错，避免静默继续训练。
        torch.use_deterministic_algorithms(True)
    else:
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
