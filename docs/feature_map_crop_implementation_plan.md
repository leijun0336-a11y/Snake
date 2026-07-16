# 特征图裁剪实现调整方案

## 结论

建议修改，但严格限定为以下两项计算结果等价的速度优化，不改变网络的输入、参数、特征维度、融合方式或 Q 值计算语义：

1. 删除全局分支中冗余的 `nn.Identity`。
2. 将局部特征分支的 `F.unfold + gather` 替换为只收集目标 5×5 区域的直接索引。

其中，删除 `nn.Identity` 是安全的。`nn.Identity` 不包含参数和 buffer，前向传播只是原样返回输入；删除后输出值、梯度、形状和 `state_dict` 均不变。它只能减少一次 Python 模块调用及 hook 分发，性能收益预计很小，主要优化仍来自局部裁剪。

当前流程是：

~~~text
pad -> unfold 全部 H×W 个 5×5 窗口 -> gather 蛇头窗口
~~~

实际只需要一个窗口。6×6 棋盘会生成 36 个候选窗口后丢弃 35 个；20×20 会生成 400 个后只保留 1 个。这个中间张量和对应反向传播没有必要。

需要注意：目前只能确认存在结构性浪费，尚不能确认它就是 deterministic 训练慢数倍的唯一原因。新方案仍使用 gather，而 PyTorch 的 CUDA gather backward 本身也会受 deterministic 设置影响，因此必须用 AutoDL 同一环境实测。

## 等价性边界

本方案允许删除没有计算作用的包装层，以及用等价索引替换不必要的中间张量；不允许改变任何会影响网络数值语义的设计。

必须保持：

- `global_projection(shared)` 的卷积、GroupNorm 和 ReLU 完全不变。
- 全局特征仍保留完整的 H×W 空间信息并直接展平，不能增加全局池化或改变棋盘尺寸。
- 局部特征仍来自 `local_projection(shared)`，仍以蛇头为中心裁剪 5×5，边界仍补零。
- 全局和局部特征的拼接顺序、融合层输入维度及最终 Q 值计算完全不变。
- 网络参数名称、参数值和 `state_dict` 键保持不变，已有 checkpoint 能继续加载。

不要求保留仅用于组织代码、但不参与计算的 `global_pool: nn.Identity` 模块节点。当前运行时代码没有依赖该节点执行额外逻辑，但 `tests/test_dqn_agent.py` 会直接断言它是 `nn.Identity`，实现时必须同步把该测试改为验证“没有自适应池化且全局特征维度仍为 `C×H×W`”。若项目外部代码曾访问 `model.global_pool` 或在其上注册 hook，也需要改为访问 `global_projection`。这些变化只涉及模块树、测试和调试入口，不影响网络计算；旧网络 `q_network_old.py` 的真实池化层及相关测试必须保持不变。

## 拟采用实现

### 1. 删除全局分支的恒等映射

当前代码：

~~~python
self.global_pool = nn.Identity()

global_features = self.global_pool(
    self.global_projection(shared)
).flatten(1)
~~~

调整为：

~~~python
global_features = self.global_projection(shared).flatten(1)
~~~

这不是删除全局分支，也不是把全局特征改成池化结果；只是删除 `Identity(x) == x` 的冗余调用。`global_projection` 的输出仍以 `[B, C, H, W]` 形式完整保留，并在随后展平为 `[B, C×H×W]`。

### 2. 优化局部特征裁剪

保留以下语义：

- 仍从 local_projection(shared) 的可训练特征图裁剪。
- 仍取蛇头中心的 5×5 区域，边界补零。
- 梯度仍能回传到 local_projection 和共享 CNN。

将 unfold 改为“展平特征图后直接收集 25 个位置”：

~~~python
head_y, head_x = ...
padded = F.pad(features, (2, 2, 2, 2))
padded_flat = padded.flatten(2)

top_left = head_y * padded_width + head_x
indices = top_left[:, None] + local_offsets[None, :]
crop = padded_flat.gather(
    2,
    indices[:, None, :].expand(-1, channels, -1),
)
crop = crop.reshape(batch_size, channels, 5, 5)
~~~

local_offsets 是固定的 25 个相对位置，使用 persistent=False 的 buffer 缓存，随模型设备移动但不写入 checkpoint。

形状变化：

~~~text
当前中间结果：[B, C×25, H×W]
修改后结果：  [B, C, 25]
~~~

两项调整都不改变网络参数、融合维度和 `state_dict` 键，因此现有 checkpoint 可继续加载。新增的 `local_offsets` 使用 `persistent=False`，也不会写入 checkpoint。

## 不采用的方案

- 不使用逐样本 Python 循环切片：会产生大量小计算图节点和 GPU kernel 调度。
- 不使用 grid_sample：CUDA backward 在严格 deterministic 模式下没有确定性实现。
- 不把 `nn.Identity` 替换为任何真实的池化层：池化会改变全局特征及融合层输入，属于架构变更。
- 暂不把局部 patch 前移到环境 observation：这会从“学习后的特征裁剪”变成“原始输入裁剪”，属于架构变更并会影响旧 checkpoint。
- 不修改 q_network_old.py；它是隔离的旧网络实现。

## 代码与测试

计划修改：

1. src/snake_ai/models/q_network.py：删除 `global_pool = nn.Identity()` 及对应调用，并用直接索引替换 `F.unfold`。
2. tests/test_dqn_agent.py：删除对新网络 `global_pool` 属性的 `nn.Identity` 断言，继续验证完整 H×W 特征维度和不存在 `AdaptiveAvgPool2d`；保留旧网络的池化断言。
3. tests/test_q_network.py：新增全局分支、局部裁剪和 checkpoint 兼容性测试。
4. scripts/benchmark_feature_crop.py：新增 AutoDL GPU microbenchmark。

测试范围：

- 6×6 棋盘全部 36 个蛇头位置，包括四角和边缘。
- 随机 batch 下新旧裁剪输出一致。
- 使用非均匀上游梯度验证 features.grad 一致。
- 对相同 `shared` 输入验证删除 `Identity` 前后的全局特征逐元素完全一致，反向梯度完全一致。
- Grid/Hybrid 前向输出形状不变。
- 使用相同权重和输入验证完整 Q 值一致。
- 修改前后的 `state_dict` 键完全一致，旧 checkpoint 可加载。

## 性能验证

必须在与正式训练相同的 AutoDL GPU、PyTorch 和 CUDA 环境下比较：

- 形状：B=128、C=8，分别测试 6×6 和 20×20。
- 模式：deterministic 开启与关闭各测试一次。
- 指标：裁剪 forward+backward 延迟、CUDA 峰值显存、完整网络 forward+backward 延迟、完整 DQN learn steps/sec。
- 每组先 warmup，再至少重复 500 次并报告中位数。

`nn.Identity` 的调用开销可能低于 GPU 计时噪声，因此不把它单独达到某个加速比例作为验收条件；只要求删除后没有性能回退，并以完整网络和 DQN 训练吞吐为最终依据。

验收条件：

- 输出、梯度和 checkpoint 兼容性测试全部通过；删除 `Identity` 前后必须严格等价。
- 新裁剪的 deterministic forward+backward 延迟低于旧实现。
- 完整训练 steps/sec 有稳定提升，且 non-deterministic 模式无明显回退。

如果裁剪 microbenchmark 改善但完整训练没有改善，则保留 benchmark 结论，不继续把局部 patch 移到 observation；后续再单独定位卷积或其他 deterministic 算子。

## 官方依据

- PyTorch Unfold：会提取所有滑动局部块，并形成包含全部窗口的输出。
  https://docs.pytorch.org/docs/stable/generated/torch.nn.Unfold.html
- PyTorch deterministic algorithms：CUDA 上带梯度的 gather 会使用确定性算法；grid_sample backward 在 CUDA 上没有确定性实现。
  https://docs.pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
