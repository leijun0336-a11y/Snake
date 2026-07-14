# 旧版 Q 网络隔离与移除报告

## 目的与当前状态

本报告记录版本 2 旧 Q 网络兼容层的全部接入点、依赖关系和移除步骤。
后续不再需要评估旧 checkpoint 时，可以按“逆操作清单”删除兼容层，而不影响
当前版本 3 动态网格网络的训练和评估。

当前约束如下：

- `q_network` 是默认网络，用于版本 3 动态网格训练和评估。
- `q_network_old` 仅用于评估版本 2 的历史 checkpoint，不能训练或保存。
- 第八次实验的 6×6 checkpoint 必须通过 `q_network_old` 加载。
- 旧网络保留 `AdaptiveAvgPool2d`，但只在 `torch.no_grad()` 评估路径中使用。

第八次实验真实 checkpoint 的关键结构信息为：

- `architecture_version = 2`
- `state_size = (9, 6, 6)`
- `cnn_pool_size = (10, 10)`
- `network_mode = hybrid`

## 隔离边界

### 旧网络主体

`src/snake_ai/models/q_network_old.py`

- 定义 `QNetworkOld`。
- 保留版本 2 的固定/自适应池化和旧全连接输入维度。
- 不包含训练入口，也不会被默认网络自动选中。

该文件目前复用了 `q_network.py` 中的 `ResidualBlock` 和 `_group_count`。
因此它在结构上已隔离，但并非完全自包含。删除旧网络时不需要修改这两个公共实现；
反过来，在旧兼容层仍需保留期间，不应随意改变这两个符号的计算语义。若未来要大幅
重构它们，应先把版本 2 对应实现复制到 `q_network_old.py` 内部并重新做真实权重测试。

### 选择与加载适配层

`src/snake_ai/evaluate.py`

- 提供 `--network {q_network,q_network_old}`。
- 默认值为 `q_network`。
- 将字符串选择传给 `DQNAgent`，并在评估摘要中打印网络类型。

`src/snake_ai/agents/dqn_agent.py`

- 导入并按 `network_type` 构建 `QNetwork` 或 `QNetworkOld`。
- 保存旧网络池化尺寸 `old_cnn_pool_size`，供版本 2 权重还原。
- 默认新网络拒绝加载空间尺寸被池化改变的版本 2 checkpoint。
- 旧网络允许加载这类 checkpoint。
- `learn()` 和 `save()` 显式拒绝 `q_network_old`，防止误训练和生成新旧混合权重。

`src/snake_ai/models/__init__.py`

- 对外导出 `QNetworkOld`。

## 完整接入清单

移除旧兼容层前，应检查以下文件：

| 类别 | 文件 | 旧网络相关内容 |
|---|---|---|
| 模型 | `src/snake_ai/models/q_network_old.py` | 版本 2 网络完整实现 |
| 模型导出 | `src/snake_ai/models/__init__.py` | `QNetworkOld` 导入与导出 |
| Agent | `src/snake_ai/agents/dqn_agent.py` | 网络选择、旧池化尺寸、加载及训练保护 |
| 评估入口 | `src/snake_ai/evaluate.py` | `--network` 参数、打印和参数传递 |
| Agent 测试 | `tests/test_dqn_agent.py` | 旧权重加载、非法网络类型测试 |
| 评估测试 | `tests/test_evaluate.py` | 默认网络和旧网络参数解析测试 |
| 使用说明 | `README.md` | 第八次实验评估命令和参数说明 |
| 动态网络文档 | `docs/dynamic_grid_network_plan.md` | 历史 checkpoint 兼容说明 |
| 实现报告 | `docs/dynamic_grid_network_implementation_report.md` | 本次隔离实现及验证结果 |

## 移除前置条件

只有同时满足以下条件时，才建议删除：

1. 不再需要重新评估版本 2 的 6×6、20×20 或其他“网格尺寸不等于池化尺寸”的权重。
2. 第八次实验的最终指标和必要轨迹已经导出并归档。
3. 历史 checkpoint、实验配置和结果目录已保留；删除兼容代码不等于删除实验数据。
4. 最好先把当前兼容实现单独提交到版本控制，以便日后临时恢复。

建议保留以下历史数据，不要随兼容代码一起删除：

- `checkpoints/dqn_20260712_130642/`
- `runs/dqn_20260712_130642/`（如果该目录存在）
- 第八次实验的配置、指标、图表和报告

## 逆操作清单

按以下顺序移除，可避免中间状态出现失效导入。

### 1. 恢复单网络评估入口

在 `src/snake_ai/evaluate.py` 中：

1. 删除 `--network` 参数定义。
2. 删除评估摘要中的网络类型打印。
3. 创建 `DQNAgent` 时不再传入 `network_type`。

完成后，评估入口将始终使用默认的当前 `QNetwork`。

### 2. 删除 Agent 兼容分支

在 `src/snake_ai/agents/dqn_agent.py` 中：

1. 删除 `QNetworkOld` 导入。
2. 删除 `NETWORK_TYPES`、`network_type` 构造参数及其校验和成员变量。
3. 删除 `old_cnn_pool_size`。
4. 删除 `learn()`、`save()` 中针对旧网络的拒绝分支。
5. 将 `_build_network()` 简化为只构建并返回 `QNetwork`。
6. 从新 checkpoint 元数据中删除 `network_type` 字段；该项也可暂时保留，
   因为加载端不依赖它，但删除后结构更干净。
7. 将版本 2 的池化尺寸检查恢复为无条件检查：当 checkpoint 的
   `cnn_pool_size` 与实际网格尺寸不同，直接报告旧架构不兼容。
8. 从 `architecture_changed` 判断和加载后状态更新中删除旧池化尺寸分支。

注意：不要删除版本 2 的全部加载逻辑。网格尺寸本身就是 10×10、且旧池化尺寸也是
10×10 的版本 2 checkpoint，仍可由当前动态网络兼容加载。

### 3. 删除模型导出和模型文件

1. 从 `src/snake_ai/models/__init__.py` 删除 `QNetworkOld` 的导入和导出。
2. 删除 `src/snake_ai/models/q_network_old.py`。

### 4. 删除或改写测试

在 `tests/test_dqn_agent.py` 中：

- 删除 `QNetworkOld` 导入。
- 删除旧网络成功加载尺寸不匹配版本 2 checkpoint 的测试。
- 删除非法 `network_type` 的测试。
- 保留默认网络拒绝不兼容版本 2 checkpoint 的测试。

在 `tests/test_evaluate.py` 中：

- 删除选择 `q_network_old` 的参数解析测试。
- 删除对默认 `args.network == "q_network"` 的断言。

### 5. 清理文档

从 `README.md`、动态网络方案和实现报告中删除：

- `--network q_network_old` 示例。
- `--network` 参数说明。
- 旧网络可加载版本 2 尺寸不匹配 checkpoint 的说明。

历史实验结论可以保留，但应注明旧权重需要恢复当时提交后才能重新运行。

## 移除后的搜索检查

在项目根目录执行：

```powershell
rg -n --hidden --glob '!.git' "q_network_old|QNetworkOld|old_cnn_pool_size|network_type|--network" src tests README.md docs
```

在当前项目结构下，移除完成后应无结果。如果以后新增了其他同名通用概念，需要人工
判断是否与旧网络兼容层有关。

## 移除后的验证

```powershell
uv run pytest -q -p no:cacheprovider
uv run ruff check src tests
uv run python -m compileall -q src tests
uv run python -m snake_ai.evaluate --help
```

验证要求：

- 全部测试通过。
- Ruff 和字节码编译通过。
- `evaluate --help` 不再显示 `--network`。
- 版本 3 checkpoint 仍能正常评估。
- 版本 2 的 10×10 兼容测试仍通过。
- 第八次实验的 6×6 版本 2 checkpoint 加载失败属于删除后的预期行为，错误信息应明确。

## 当前兼容层验证记录

兼容层加入后已完成：

- 完整测试：`76 passed`。
- Ruff 静态检查：通过。
- Python 字节码编译：通过。
- 使用第八次实验真实 `latest.pt` 完成一次无渲染短评估并成功加载。

当前评估命令示例：

```powershell
uv run python -m snake_ai.evaluate `
  --checkpoint checkpoints\dqn_20260712_130642\latest.pt `
  --network q_network_old `
  --width 6 --height 6 --episodes 1 --no-render
```

本报告记录的是代码边界和逆操作，不代替版本控制。旧网络真正删除前，建议把本次
兼容实现的提交哈希补充到本报告中。
