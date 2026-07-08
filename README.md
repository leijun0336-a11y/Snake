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
├── config.py      # 默认配置
└── utils.py       # 随机种子与可复现性工具
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

`tests/` 里的测试用于检查代码逻辑是否正确，例如环境重置、移动、撞墙、经验池采样等；它不需要训练好的模型。`evaluate.py` 用于加载训练好的模型并评估 AI 的实际分数，需要先有 `checkpoints/<run_name>/best.pt` 或其他 checkpoint。

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

- `checkpoints/<run_name>/best.pt`
- `checkpoints/<run_name>/latest.pt`
- `runs/<run_name>/train_metrics.csv`
- TensorBoard 训练日志，event 文件名带有 `.train` 后缀，指标使用 `train/` 前缀分组

训练日志会记录 `score`、`best_score`、`mean_score_100`、`episode_steps`、`epsilon`、`loss`、`mean_loss_100` 和 `replay_buffer_size`。

## TensorBoard 可视化

训练或评估写入 TensorBoard 日志后，在项目根目录启动：

```bash
uv run tensorboard --logdir runs
```

然后在浏览器打开：

```text
http://localhost:6006
```

`runs/<run_name>/` 里可能同时包含训练和评估 event 文件：

- `.train` 后缀：训练指标，例如 `train/score`、`train/loss`、`train/mean_score_100`
- `.eval` 后缀：评估图，例如 `eval/scores`

如果在 AutoDL 或远程服务器上运行，需要使用平台自带的 TensorBoard 面板，或把服务器的 `6006` 端口转发到本地。

## 可复现性

训练和评估入口都会调用 `snake_ai.utils.set_seed()`，统一设置 Python、NumPy、PyTorch 和 CUDA 的随机种子，并默认开启 PyTorch/cuDNN 的确定性设置。这样能让同一环境下的实验尽量可复现。

需要注意：不同 GPU、CUDA、驱动或 PyTorch 版本之间仍可能存在细微差异，因此这里追求的是“尽量可复现”，不是跨所有机器的绝对一致。

## 评估

```bash
uv run python -m snake_ai.evaluate
```

默认会加载最近一次训练目录中的 `checkpoints/<run_name>/best.pt`。

指定 checkpoint 评估：

```bash
uv run python -m snake_ai.evaluate --checkpoint checkpoints/dqn_20260707_151533/best.pt
```

无窗口评估：

```bash
uv run python -m snake_ai.evaluate --no-render --episodes 10
```

保存评估指标并写入 TensorBoard：

```bash
uv run python -m snake_ai.evaluate --no-render --episodes 10 --tensorboard
```

默认会把评估产物绑定到最近一次训练目录 `runs/dqn_*` 下：

- `eval_metrics.csv`：追加记录每一次评估中每一局的 score
- TensorBoard 评估日志：event 文件名带有 `.eval` 后缀，记录一张 `eval/scores` 评估图，包含所有 score、平均分参考线和最高分参考线

多次运行评估时，新的测试指标会继续追加到同一个训练目录，不会覆盖旧的评估记录。

如果想手动指定输出目录：

```bash
uv run python -m snake_ai.evaluate --no-render --episodes 10 --tensorboard --eval-output-dir runs/dqn_20260707_151533
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
