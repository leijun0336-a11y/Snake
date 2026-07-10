# 问题
把“走到即将离开的尾巴位置”误判为死亡。

# 解决：
如果 new_head == food：
    这一步会吃食物，尾巴不会移动
    所以不能撞到任何身体部分

如果 new_head != food：
    这一步没吃食物，尾巴会移动走
    所以允许走到当前尾巴的位置


# 问题
每次训练会直接把上次训练的神经网络权重覆盖，无法保留多个权重，不利于复现以前的实验。

解决：修改文件夹结构，按照时间编号实验文件夹，不同的实验下有不同的权重和评估指标的记录。

# 问题
训练 grid 版本奇慢无比

原因：

grid 模式慢的主因不是状态构造或 ReplayBuffer，而是 CNN 训练路径触发了 PyTorch deterministic CUDA 的慢路径。

vector 模式只使用 19 维状态和 MLP，不经过卷积和池化；grid 模式会把 `[5, height, width]` 网格送入 CNN，每个训练 step 都要做卷积、池化和反向传播。之前 `set_seed()` 默认开启：

```python
torch.use_deterministic_algorithms(True, warn_only=True)
```

这会强制 CUDA 尽量使用确定性算法。grid CNN 中的 `AdaptiveAvgPool2d` 反向传播会触发类似警告：

```text
adaptive_avg_pool2d_backward_cuda does not have a deterministic implementation
```

因此训练会避开或限制很多最快的 CUDA 算法，速度可能被大幅拖慢；vector 模式没有这个算子，所以几乎不受影响。

解决：

1. 训练入口新增 `--deterministic` 参数，默认关闭 deterministic CUDA，优先保证训练速度。只有需要严格复现实验时再显式开启：

```bash
bash scripts/train_autodl.sh --state-mode grid --deterministic
```

2. 在默认 `20x20 -> 5x5` 这种整除池化场景下，把 `AdaptiveAvgPool2d` 替换成等价的 `AvgPool2d`，避开 `adaptive_avg_pool2d_backward_cuda` 的慢路径和警告。

3. 保留数据路径优化作为辅助优化：Grid 状态使用连续 `float32` NumPy 数组；ReplayBuffer 使用固定容量环形列表并按随机索引采样，避免 `random.sample(list(self.memory), 64)` 在经验池很大时每步复制整个 buffer。
