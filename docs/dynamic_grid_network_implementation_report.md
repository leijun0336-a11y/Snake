# 动态网格 Hybrid Q 网络实现报告

## 实现结果

已按确认方案完成动态网格改造。Grid/Hybrid 网络不再把空间特征固定池化到
10×10，而是直接使用 `--width`、`--height` 对应的实际 `H×W` 构建全连接层。

## 主要实现

- 全局分支改为 `1×1 Conv 32→8 + GroupNorm + ReLU + Identity`，删除
  `AdaptiveAvgPool2d`、普通平均池化和 `_build_pool_layer()`。
- 动态计算全局维度 `8*H*W`；Hybrid 融合维度为 `8*H*W+220`。
- 保留共享CNN、独立全局/局部投影、蛇头中心5×5局部分支和Dueling Q头。
- 删除 `cnn_pool_size` 配置、`--cnn-pool-size` 命令行参数及相关参数传递。
- checkpoint架构版本升级为3，不再保存池化目标尺寸。
- 默认网络可以兼容加载版本2的10×10 checkpoint；版本2的6×6和20×20
  checkpoint需要显式选择 `q_network_old`。
- 新增 `models/q_network_old.py`，完整保留版本2池化结构，仅用于历史权重评估；
  `DQNAgent.learn()` 和 `save()` 会拒绝旧网络。
- `evaluate.py` 新增 `--network {q_network,q_network_old}`，默认使用 `q_network`。
- `--deterministic` 现在使用严格确定性算法；遇到无确定性实现的算子会直接报错，
  关闭该参数时会显式恢复非确定性算法模式。
- README和实验8启动脚本已同步更新；实验8脚本现在表示在动态网格架构上复用其
  奖励和训练参数，不再声称复现旧版自适应池化架构。

## 动态维度示例

| 网格 | 全局维度 | Hybrid融合维度 |
|---|---:|---:|
| 6×6 | 288 | 508 |
| 10×10 | 800 | 1020 |
| 20×20 | 3200 | 3420 |

不同尺寸会构建不同输入形状的全连接层，因此每种网格应使用独立checkpoint。
本次实现不支持同一个模型在一次训练中混合不同宽高。

## 验证结果

- 完整测试：`76 passed`。
- Ruff静态检查：通过。
- Python字节码编译：通过。
- 新增覆盖：6×6、10×10、20×20、矩形网格，Grid/Hybrid前向形状，
  Identity与动态融合维度，新网络无AdaptivePool，版本2 checkpoint的新旧网络选择，
  严格确定性开关。
- 已使用第八次实验真实 `latest.pt` 完成 `q_network_old` 无渲染短评估。

本次没有启动正式训练。

旧网络兼容层的完整接入点、依赖边界和后续移除步骤见
[`old_q_network_isolation_report.md`](old_q_network_isolation_report.md)。
