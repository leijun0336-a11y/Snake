# 问题记录

本文记录已经定位或解决过的问题。当前行为以 `src/snake_ai/`、`README.md` 和测试为准。

## 走到即将离开的尾巴位置被误判为碰撞

**状态：已解决。**

环境会区分本步是否吃到食物：

- `new_head == food`：蛇会增长，尾巴不会移动，因此蛇头不能进入任何现有身体格；
- `new_head != food`：尾巴会在本步离开，蛇头可以进入移动前的尾巴格。

当前判断由 `SnakeEnv._is_collision_after_move()` 统一处理。

## 多次训练互相覆盖 checkpoint

**状态：已解决。**

每次训练使用带算法和时间戳的独立目录：

```text
checkpoints/<algorithm>_<timestamp>/
runs/<algorithm>_<timestamp>/
```

其中 `latest.pt` 是最近一次模型快照，`best.pt` 由阶段验证选出。当前 checkpoint
不包含 optimizer、replay/rollout buffer 和随机数状态，因此不能视为完整的断点续训文件。

## Grid 模式训练缓慢

**状态：旧实现问题；原诊断不再代表当前网络。**

旧版本曾使用 19 维 Vector 状态、5 通道 Grid 状态和 `AdaptiveAvgPool2d`。严格
deterministic CUDA 下，旧池化反向传播可能进入慢路径并产生
`adaptive_avg_pool2d_backward_cuda` 警告。

当前实现已经发生以下变化：

- Vector 状态为 20 维；
- Grid 状态为 `[9, height, width]`；
- CNN 不再使用 `AdaptiveAvgPool2d`，全局特征直接保留完整 `H × W` 分辨率；
- 局部分支默认裁剪蛇头周围 `3 × 3` 特征，可通过参数调整或关闭；
- Grid 状态使用连续 `float32` NumPy 数组；
- ReplayBuffer 使用固定容量结构，PER 使用 Sum Tree；
- `--deterministic` 默认关闭，仅在需要严格复现时显式启用。

当前性能问题应基于实际 GPU、棋盘尺寸、batch size 和 deterministic 开关重新测量，
不能继续归因于已经移除的自适应池化。
