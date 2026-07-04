# Snake AI

用 DQN 训练一个会玩贪吃蛇的强化学习智能体。当前版本是初始闭环版：包含游戏环境、pygame 渲染、DQN 智能体、训练入口、评估入口和基础测试。

## 项目结构

```text
src/snake_ai/
├── game/          # 贪吃蛇环境和渲染
├── agents/        # DQN agent 和 replay buffer
├── models/        # Q 网络
├── train.py       # 训练入口
├── evaluate.py    # 评估入口
└── config.py      # 默认配置
```

## 本地开发

如果本地暂时不下载依赖，可以先只看代码和修改代码。之后在 AutoDL 或合适环境中执行：

```bash
uv sync
```

运行测试：

```bash
uv run pytest
```

## 训练

无渲染训练：

```bash
uv run python -m snake_ai.train
```

小规模测试训练：

```bash
uv run python -m snake_ai.train --episodes 20
```

带渲染训练，主要用于观察，不建议正式训练时使用：

```bash
uv run python -m snake_ai.train --render --episodes 5
```

训练产物：

- `checkpoints/best.pt`
- `checkpoints/latest.pt`
- `runs/<run_name>/metrics.csv`
- TensorBoard 日志

## 评估

```bash
uv run python -m snake_ai.evaluate --checkpoint checkpoints/best.pt
```

无窗口评估：

```bash
uv run python -m snake_ai.evaluate --no-render --episodes 10
```

## AutoDL

上传项目后可以运行：

```bash
bash scripts/train_autodl.sh --episodes 2000
```

如果服务器镜像暂时不用 uv，也可以安装为可编辑包后运行：

```bash
pip install -e .
python -m snake_ai.train --episodes 2000
```
