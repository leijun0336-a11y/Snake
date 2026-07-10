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

训练多通道网格 CNN state：

```bash
uv run python -m snake_ai.train --state-mode grid
```

训练网格 CNN 与 19 维人工状态融合的 Hybrid state：

```bash
uv run python -m snake_ai.train --state-mode hybrid
```

训练产物：

- `checkpoints/<run_name>/best.pt`
- `checkpoints/<run_name>/latest.pt`
- `runs/<run_name>/train_metrics.csv`
- TensorBoard 训练日志，event 文件名带有 `.train` 后缀，包含逐局标量和 `train/report` 文本摘要

训练日志会记录 `score`、`mean_score_100`、`episode_reward`、`mean_reward_100`、`episode_steps`、`epsilon`、`loss`、`mean_loss_100` 和 `replay_buffer_size`。其中 `loss` 是 Huber loss。

训练默认启用早停策略。默认最多训练 `--max-episodes=5000` 局，早停基于 `mean_score_100`，不基于 reward：先至少训练 `--min-episodes=1000` 局；之后如果连续 `--patience=500` 局没有超过 `--min-delta=0.5` 级别的有效提升，就停止训练。`best.pt` 仍然保存历史最高 `mean_score_100` 对应的权重。也可以通过 `--target-mean-score` 设置达到目标平均分后停止。

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

- `--max-episodes`：最大训练 episode 数量；早停没有触发时，训练达到该上限后结束。
- `--episodes`：兼容旧用法，等价于覆盖 `--max-episodes`。
- `--render`：训练时打开 pygame 渲染窗口。
- `--width`：游戏网格宽度。
- `--height`：游戏网格高度。
- `--checkpoint-dir`：checkpoint 输出目录。
- `--runs-dir`：训练日志输出目录。
- `--state-mode`：状态输入模式，可选 `vector`、`grid`、`hybrid`；默认 `vector`。
- `--cnn-channels`：Grid/Hybrid CNN 主干通道数，默认 `32`。
- `--cnn-output-channels`：1x1 卷积压缩后的通道数，默认 `16`。
- `--cnn-dilations`：空洞残差块的 dilation 序列，默认 `1 2 4`。
- `--cnn-pool-size`：自适应平均池化输出的高和宽，默认 `5 5`。
- `--no-early-stop`：关闭训练早停。
- `--min-episodes`：早停生效前至少训练的 episode 数量。
- `--patience`：超过最小训练局数后，允许连续多少个 episode 没有有效提升。
- `--min-delta`：`mean_score_100` 至少提升多少才算一次有效提升。
- `--target-mean-score`：达到指定 `mean_score_100` 后停止训练。

评估参数：

- `--checkpoint`：指定要加载的 checkpoint。
- `--episodes`：评估 episode 数量。
- `--no-render`：评估时不打开 pygame 渲染窗口。
- `--width`：游戏网格宽度。
- `--height`：游戏网格高度。
- `--tensorboard`：写入 TensorBoard 评估日志和 `eval_metrics.csv`。
- `--eval-output-dir`：指定评估指标输出目录。
- `--state-mode`：指定评估状态输入模式；不指定时会从 checkpoint 自动读取。

TensorBoard 参数：

- `--logdir`：指定 TensorBoard 读取的日志目录。

## 环境 info

`SnakeEnv.step()` 返回的 `info` 字典包含环境即时指标：

- `score`：当前局吃到的食物数。
- `steps`：当前局已经走过的步数。
- `snake_length`：当前蛇身长度。
- `steps_since_food`：距离上次吃到食物已经走过的步数。

评估脚本会基于这些即时指标继续计算 `score_per_step`、`max_snake_length`、平均分等汇总指标。

## 环境 state

`SnakeEnv.get_state()` 返回 19 维低维状态向量，作为 Q 网络输入：

| 序号 | 维度 | 含义 |
|------|------|------|
| 1 | `danger_straight` | 直行下一步是否危险。 |
| 2 | `danger_right` | 右转下一步是否危险。 |
| 3 | `danger_left` | 左转下一步是否危险。 |
| 4 | `direction_left` | 当前是否向左移动。 |
| 5 | `direction_right` | 当前是否向右移动。 |
| 6 | `direction_up` | 当前是否向上移动。 |
| 7 | `direction_down` | 当前是否向下移动。 |
| 8 | `food_left` | 食物是否在蛇头左侧。 |
| 9 | `food_right` | 食物是否在蛇头右侧。 |
| 10 | `food_up` | 食物是否在蛇头上方。 |
| 11 | `food_down` | 食物是否在蛇头下方。 |
| 12 | `food_dx_norm` | 食物相对蛇头的 x 距离，归一化到 `[-1, 1]`。 |
| 13 | `food_dy_norm` | 食物相对蛇头的 y 距离，归一化到 `[-1, 1]`。 |
| 14 | `wall_distance_straight` | 直行方向到墙的归一化距离。 |
| 15 | `wall_distance_right` | 右转方向到墙的归一化距离。 |
| 16 | `wall_distance_left` | 左转方向到墙的归一化距离。 |
| 17 | `body_distance_straight` | 直行方向最近身体的归一化距离；没有身体时为 `1.0`。 |
| 18 | `body_distance_right` | 右转方向最近身体的归一化距离；没有身体时为 `1.0`。 |
| 19 | `body_distance_left` | 左转方向最近身体的归一化距离；没有身体时为 `1.0`。 |

`SnakeEnv.get_grid_state()` 返回纯多通道网格状态，作为 Grid CNN Q 网络输入：

| 部分 | 形状 | 含义 |
|------|------|------|
| grid | `[5, height, width]` | 多通道符号网格。 |

grid 通道说明：

| 通道 | 含义 |
|------|------|
| 0 | 边界格子。 |
| 1 | 蛇身，不含蛇头。 |
| 2 | 蛇头。 |
| 3 | 食物。 |
| 4 | 蛇身顺序，蛇头为 `1.0`，越靠近尾巴数值越小。 |

`SnakeEnv.get_hybrid_state()` 返回 `(grid, vector_state)`：`grid` 形状仍为
`[5, height, width]`，`vector_state` 是 `get_state()` 返回的完整 19 维人工状态。
Hybrid Q 网络先提取并展平 CNN 特征，再与 19 维状态拼接。

三种模式用于对照实验：

| 模式 | Q 网络输入 | 用途 |
|------|------------|------|
| `vector` | 19 维人工状态 | 保留原始 MLP baseline。 |
| `grid` | 纯 5 通道网格 | 检验空洞 CNN 从网格端到端提取特征的能力，不额外拼接方向向量。 |
| `hybrid` | 5 通道网格 + 19 维人工状态 | 结合全图布局与人工特征，提高有限算力下的学习效率。 |

Grid 和 Hybrid 共用轻量空洞 CNN，默认结构为：

```text
5 通道网格
  -> 3x3 Conv（32 通道）
  -> DilatedResidualBlock（dilation=1, 2, 4）
  -> 1x1 Conv（16 通道）
  -> AdaptiveAvgPool2d(5, 5)
  -> Flatten（400 维）
```

Grid 模式把 400 维 CNN 特征直接送入共享全连接层；Hybrid 模式先拼接 19 维人工状态，形成 419 维特征，再送入共享全连接层。两者最后都连接 Dueling 的 `V(s)` 和 `A(s,a)` 分支。

CNN 的主干通道数、压缩通道数、dilation 序列和池化尺寸已经参数化。训练保存的 checkpoint 会记录这些架构参数，评估时自动按 checkpoint 重建网络。不同 state mode 的 checkpoint 不能混用；旧的“grid + 4 维方向向量”checkpoint 与当前纯 Grid CNN 结构不兼容，会明确报错而不会静默加载。

## 指标说明

训练指标：

| 指标 | TensorBoard tag | CSV 字段 | 含义 |
|------|-----------------|----------|------|
| 单局得分 | `train/score` | `score` | 当前 episode 吃到的食物数。 |
| 吃食效率 | `train/score_per_step` | `score_per_step` | `score / episode_steps`。 |
| 近 100 局平均得分 | `train/mean_score_100` | `mean_score_100` | 最近最多 100 个 episode 的 `score` 平均值。 |
| 历史最高近 100 局平均得分 | `train/best_mean_score_100` | 无 | 历史最高 `mean_score_100`，用于保存 `best.pt`。 |
| 单局累计奖励 | `train/episode_reward` | `episode_reward` | 当前 episode 内所有 step 的 reward 总和。 |
| 近 100 局平均累计奖励 | `train/mean_reward_100` | `mean_reward_100` | 最近最多 100 个 episode 的 `episode_reward` 平均值。 |
| 单局步数 | `train/episode_steps` | `episode_steps` | 当前 episode 的存活步数。 |
| 探索率 | `train/epsilon` | `epsilon` | 当前 epsilon-greedy 探索率。 |
| 单局平均损失 | `train/loss` | `loss` | 当前 episode 内所有学习 step 的平均 Huber loss。 |
| 近 100 局平均损失 | `train/mean_loss_100` | `mean_loss_100` | 最近最多 100 个 episode 的 `loss` 平均值。 |
| 经验池大小 | `train/replay_buffer_size` | `replay_buffer_size` | 当前 replay buffer 中的经验数量。 |
| 训练摘要 | `train/report` | 无 | Text 页中的汇总表，包含均值、标准差、最小值、最大值和最后值。 |

评估指标：

| 指标 | TensorBoard tag | CSV 字段 | 含义 |
|------|-----------------|----------|------|
| 单局得分 | `eval/score` | `score` | 当前评估 episode 吃到的食物数。 |
| 单局步数 | `eval/steps` | `steps` | 当前评估 episode 的存活步数。 |
| 吃食效率 | `eval/score_per_step` | `score_per_step` | `score / steps`。 |
| 最大蛇身长度 | `eval/max_snake_length` | `max_snake_length` | 当前评估 episode 中蛇身达到过的最大长度。 |
| 评估摘要 | `eval/report` | 无 | Text 页中的汇总表，包含均值、标准差、最小值和最大值。 |

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
