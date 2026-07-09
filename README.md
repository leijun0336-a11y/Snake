# Snake AI

用 Double DQN + Dueling DQN 训练一个会玩贪吃蛇的强化学习智能体。项目包含游戏环境、pygame 渲染、DQN agent、训练入口、评估入口、TensorBoard 日志和基础测试。

## 项目结构

```text
src/snake_ai/
├── game/          # 贪吃蛇环境和渲染
├── agents/        # DQN agent 和 replay buffer
├── models/        # Q 网络
├── train.py       # 训练入口
├── evaluate.py    # 评估入口
├── config.py      # 默认配置
└── utils.py       # 随机种子与可复现工具

scripts/
├── train_autodl.sh
├── evaluate.sh
├── train_autodl.ps1
└── evaluate.ps1
```

## 本地开发

安装依赖：

```bash
uv sync
```

运行测试：

```bash
uv run pytest
```

`tests/` 用于检查环境重置、移动、撞墙、经验池采样等基础逻辑；评估脚本需要先有训练产出的 checkpoint。

## 启动脚本

Linux / AutoDL：

```bash
bash scripts/train_autodl.sh
bash scripts/evaluate.sh
```

Windows PowerShell：

```powershell
.\scripts\train_autodl.ps1
.\scripts\evaluate.ps1
```

## 训练

直接训练：

```bash
uv run python -m snake_ai.train
```

训练产物：

- `checkpoints/<run_name>/best.pt`
- `checkpoints/<run_name>/latest.pt`
- `runs/<run_name>/train_metrics.csv`
- TensorBoard 训练日志，event 文件名带有 `.train` 后缀，包含逐局标量和 `train/report` 文本摘要

训练日志会记录 `score`、`mean_score_100`、`episode_reward`、`mean_reward_100`、`episode_steps`、`epsilon`、`loss`、`mean_loss_100` 和 `replay_buffer_size`。其中 `loss` 是 Huber loss。

## 评估

直接评估：

```bash
uv run python -m snake_ai.evaluate
```

默认会加载最近一次训练目录中的 `checkpoints/<run_name>/best.pt`。

评估产物默认绑定到被评估 checkpoint 对应的 `runs/<run_name>` 目录：

- `eval_metrics.csv`：追加记录每一局的 `score`、`steps`、`score_per_step` 和 `max_snake_length`
- TensorBoard 评估日志，event 文件名带有 `.eval` 后缀，包含逐局标量和 `eval/report` 文本摘要

多次运行评估时，新的测试指标会继续追加到同一个训练目录，不会覆盖旧记录。

## 参数说明

训练参数：

- `--episodes`：训练 episode 数量。
- `--render`：训练时打开 pygame 渲染窗口。
- `--width`：游戏网格宽度。
- `--height`：游戏网格高度。
- `--checkpoint-dir`：checkpoint 输出目录。
- `--runs-dir`：训练日志输出目录。

评估参数：

- `--checkpoint`：指定要加载的 checkpoint。
- `--episodes`：评估 episode 数量。
- `--no-render`：评估时不打开 pygame 渲染窗口。
- `--width`：游戏网格宽度。
- `--height`：游戏网格高度。
- `--tensorboard`：写入 TensorBoard 评估日志和 `eval_metrics.csv`。
- `--eval-output-dir`：指定评估指标输出目录。

TensorBoard 参数：

- `--logdir`：指定 TensorBoard 读取的日志目录。

## 环境 info

`SnakeEnv.step()` 返回的 `info` 字典包含环境即时指标：

- `score`：当前局吃到的食物数。
- `steps`：当前局已经走过的步数。
- `snake_length`：当前蛇身长度。
- `steps_since_food`：距离上次吃到食物已经走过的步数。

评估脚本会基于这些即时指标继续计算 `score_per_step`、`max_snake_length`、平均分等汇总指标。

## TensorBoard

启动 TensorBoard：

```bash
uv run tensorboard
```

然后在浏览器打开：

```text
http://localhost:6006
```

`runs/<run_name>/` 里可能同时包含训练和评估 event 文件：

- `.train` 后缀：训练指标，例如 `train/score`、`train/episode_reward`、`train/loss`、`train/mean_score_100`
- `.eval` 后缀：评估指标，例如 `eval/score`、`eval/steps`、`eval/score_per_step`、`eval/max_snake_length`

## 可复现性

训练和评估入口都会调用 `snake_ai.utils.set_seed()`，统一设置 Python、NumPy、PyTorch 和 CUDA 的随机种子，并默认开启 PyTorch/cuDNN 的确定性设置。

不同 GPU、CUDA、驱动或 PyTorch 版本之间仍可能存在细微差异，因此这里追求的是尽量可复现，不是跨所有机器的绝对一致。

## tmux

在远程服务器上训练时，SSH 断开会导致进程终止。用 tmux 可以让训练在后台持续运行。

```bash
sudo apt-get update && sudo apt-get install -y tmux
```

常用操作：

| 操作 | 命令 |
|------|------|
| 新建会话 | `tmux new -s snake` |
| 查看所有会话 | `tmux ls` |
| 接入已有会话 | `tmux attach -t snake` |
| 断开并保留后台 | `Ctrl+B` 然后按 `D` |
| 删除会话 | `tmux kill-session -t snake` |

## AutoDL

上传项目后可以运行：

```bash
bash scripts/train_autodl.sh
```

如果服务器镜像暂时不用 `uv`，也可以安装为可编辑包后运行：

```bash
pip install -e .
python -m snake_ai.train
```
