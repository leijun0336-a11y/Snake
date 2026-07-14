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

训练网格 CNN 与 20 维人工状态融合的 Hybrid state：

```bash
uv run python -m snake_ai.train --state-mode hybrid
```

训练产物：

- `checkpoints/<run_name>/best.pt`
- `checkpoints/<run_name>/latest.pt`
- `runs/<run_name>/train_metrics.csv`
- `runs/<run_name>/validation_metrics.csv`
- TensorBoard 训练日志，event 文件名带有 `.train` 后缀，包含逐局标量和 `train/report` 文本摘要

训练日志会记录 `score`、`mean_score_100`、总奖励、各奖励分量、终止原因、`episode_steps`、`epsilon`、`loss`、`mean_loss_100` 和 `replay_buffer_size`。其中 `loss` 是 Huber loss。

环境默认使用与 `chynl/snake` 一致的奖励：吃食物 `+10`、碰撞 `-100`、超时 `-100`、填满地图的最后一步总奖励 `+100`；仍存活但没有吃到食物的普通移动奖励为 `-0.01`。基于曼哈顿距离的势函数奖励默认关闭，饥饿成本默认也为 `0`。势函数奖励可以显式启用，普通移动成本可以独立关闭：

```bash
# 启用势函数奖励
uv run python -m snake_ai.train --potential-reward

# 关闭普通移动成本，只保留事件奖励
uv run python -m snake_ai.train --no-cost-rewards

# 启用势函数，同时关闭普通移动成本
uv run python -m snake_ai.train --potential-reward --no-cost-rewards
```

奖励行为使用命名 profile 固定，避免再次靠源码默认值猜测历史配置：

- `reference`：默认值，即上述与 `chynl/snake` 对齐的配置；训练 episode 默认最多 500 步。
- `experiment8`：严格恢复 `dqn_20260712_130642` 的奖励、叠加顺序和 starvation 语义；自动开启 legacy potential，starvation 在超过固定棋盘面积后终止，且没有独立训练步数上限。

在 AutoDL 上严格复现实验八的完整训练参数时，不要手工拼命令，直接运行不接受额外参数的固定脚本：

```bash
bash scripts/train_experiment8_autodl.sh
```

每个新 run 都会在 run 与 checkpoint 目录写入相同的 `config.json`，并把这份完整配置嵌入 checkpoint 的 `run_config` 字段。

`latest.pt` 始终保存最近训练状态；`best.pt` 只由独立的贪心阶段验证决定。epsilon 降到下限时先运行 100 局快速集和 500 局确认集并初始化 best；之后默认每 500 个训练 episode 运行快速验证，只有通过筛选并在确认集上达到更新门槛时才覆盖 best。验证使用独立环境、固定且互不重叠的逐局种子，不写 replay buffer，也不改变训练状态。

训练默认不启用早停。显式传入 `--early-stop` 后，epsilon 到达下限且满足 `--min-episodes` 才开始累计验证 patience；连续 `--validation-patience` 轮没有确认提升时，停止前会补做或复用一次确认验证。`--target-mean-score` 同样依据确认验证均分，不再使用训练 `mean_score_100`。

## 评估

直接评估：

```bash
uv run python -m snake_ai.evaluate
```

默认会加载最近一次训练目录中的 `checkpoints/<run_name>/latest.pt`。如需评估其他权重，使用 `--checkpoint` 显式指定。

评估版本2历史网络（例如第八次6×6实验）时，显式选择只读旧网络实现：

```bash
uv run python -m snake_ai.evaluate \
  --checkpoint checkpoints/dqn_20260712_130642/latest.pt \
  --network q_network_old \
  --width 6 --height 6 --no-render
```

评估产物默认绑定到被评估 checkpoint 对应的 `runs/<run_name>` 目录：

- `eval_metrics.csv`：追加记录每一局的 `score`、`steps`、`score_per_step` 和 `max_snake_length`
- TensorBoard 评估日志，event 文件名带有 `.eval` 后缀，包含逐局标量和 `eval/report` 文本摘要

多次运行评估时，新的测试指标会继续追加到同一个训练目录，不会覆盖旧记录。

## 参数说明

训练参数：

- `--max-episodes`：最大训练 episode 数量；早停没有触发时，训练达到该上限后结束。
- `--max-steps-per-episode`：覆盖 profile 的训练 episode 总步数上限；`reference` 默认 `500`，`experiment8` 为严格复现而禁止设置此参数并保持无限。
- `--render`：训练时打开 pygame 渲染窗口。
- `--width`：游戏网格宽度。
- `--height`：游戏网格高度。
- `--checkpoint-dir`：checkpoint 输出目录。
- `--runs-dir`：训练日志输出目录。
- `--state-mode`：状态输入模式，可选 `vector`、`grid`、`hybrid`；默认 `vector`。
- `--reward-profile`：奖励与 starvation 行为，可选 `reference`、`experiment8`；默认 `reference`。
- `--potential-reward`：启用基于食物距离的势函数奖励；默认关闭。
- `--no-cost-rewards`：关闭普通移动与饥饿成本；超时仍使用与碰撞相同的 `-100` 终止惩罚。
- `--cnn-channels`：Grid/Hybrid CNN 主干通道数，默认 `32`。
- `--cnn-output-channels`：全局/局部分支各自使用的 1x1 卷积压缩通道数，默认 `8`。
- `--cnn-dilations`：共享残差块第一层卷积的 dilation 序列，默认 `1 1 2`；每个块的第二层固定使用普通 3x3 卷积。
- `--early-stop`：启用训练早停；默认关闭。
- `--min-episodes`：早停生效前至少训练的 episode 数量。
- `--validation-interval`：epsilon 到达下限后，每隔多少训练 episode 进行快速验证，默认 `500`。
- `--validation-episodes`：快速验证局数，默认 `100`。
- `--confirmation-episodes`：确认验证局数，默认 `500`。
- `--validation-patience`：连续多少轮阶段验证没有确认提升后进入早停最终确认，默认 `8`。
- `--validation-max-steps`：每个阶段验证 episode 的最大步数，默认 `1000`。
- `--target-mean-score`：确认验证均分达到该值后停止训练。

评估参数：

- `--checkpoint`：指定要加载的 checkpoint。
- `--episodes`：评估 episode 数量，默认 `1000`。
- `--max-steps`：每个评估 episode 的总步数上限，默认 `1000`；评估时不启用逐食物 starvation，以对齐 `chynl/snake` benchmark。
- `--no-render`：评估时不打开 pygame 渲染窗口。
- `--width`：游戏网格宽度。
- `--height`：游戏网格高度。
- `--tensorboard`：写入 TensorBoard 评估日志和 `eval_metrics.csv`。
- `--eval-output-dir`：指定评估指标输出目录。
- `--state-mode`：指定评估状态输入模式；不指定时会从 checkpoint 自动读取。
- `--network`：选择 `q_network` 或 `q_network_old`，默认 `q_network`；旧网络仅用于评估版本2历史checkpoint。

独立评估使用最终测试种子集；每个 episode 都会按自己的固定种子重置环境，因此相同 checkpoint、参数和种子能够重复得到相同结果。快速集、确认集和最终测试集互不重叠。

TensorBoard 参数：

- `--logdir`：指定 TensorBoard 读取的日志目录。

## 环境 info

`SnakeEnv.step()` 返回的 `info` 字典包含环境即时指标：

- `score`：当前局吃到的食物数。
- `steps`：当前局已经走过的步数。
- `snake_length`：当前蛇身长度。
- `steps_since_food`：距离上次吃到食物已经走过的步数。
- `reward_food`、`reward_progress`、`reward_step`、`reward_hunger`、`reward_terminal`：当前 step 的奖励分量。
- `reward_total`：当前 step 的奖励总和。
- `termination_reason`：`none`、`collision_wall`、`collision_body`、`starvation` 或 `board_completed`。

评估脚本会基于这些即时指标继续计算 `score_per_step`、`max_snake_length`、平均分等汇总指标。

## 环境 state

`SnakeEnv.get_state()` 返回 20 维低维状态向量，作为 Q 网络输入：

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
| 20 | `hunger_ratio` | `steps_since_food / (width * height + snake_length)`，截断到 `[0, 1]`；评估关闭 starvation 时固定为 `0`。 |

`SnakeEnv.get_grid_state()` 返回纯多通道网格状态，作为 Grid CNN Q 网络输入：

| 部分 | 形状 | 含义 |
|------|------|------|
| grid | `[9, height, width]` | `float32` NumPy 多通道空间网格。 |

grid 通道说明：

| 通道 | 含义 |
|------|------|
| 0 | `boundary`：合法棋盘的最外圈格子。它是边缘风险提示，不是棋盘外的真实墙体。 |
| 1 | `snake_body`：蛇身，不含蛇头，包含蛇尾。 |
| 2 | `head_left`：仅当蛇向左时，在蛇头格置 `1`。 |
| 3 | `head_right`：仅当蛇向右时，在蛇头格置 `1`。 |
| 4 | `head_up`：仅当蛇向上时，在蛇头格置 `1`。 |
| 5 | `head_down`：仅当蛇向下时，在蛇头格置 `1`。 |
| 6 | `snake_tail`：仅在蛇尾格置 `1`。 |
| 7 | `food`：仅在食物格置 `1`。 |
| 8 | `body_order`：蛇头为 `1.0`，沿蛇身递减，蛇尾为 `1 / snake_length`。 |

Grid 的 9 个通道只表达空间信息，纯 Grid 不再接收 `hunger_ratio` 常量平面或标量旁路。Vector 的第 20 维仍然是 `hunger_ratio`；Hybrid 因为会拼接完整 20 维人工状态，所以仍能观察饥饿进度。

`SnakeEnv.get_hybrid_state()` 返回 `(grid, vector_state)`：`grid` 是形状为
`[9, height, width]` 的 `float32` NumPy 数组，`vector_state` 是 `get_state()`
返回的完整 20 维人工状态。
Hybrid Q 网络先提取并展平 CNN 特征，再与 20 维状态拼接。

三种模式用于对照实验：

| 模式 | Q 网络输入 | 用途 |
|------|------------|------|
| `vector` | 20 维人工状态 | 低维 MLP baseline。 |
| `grid` | 纯 9 通道空间网格 | 从网格端到端学习；方向编码在四个蛇头通道内，不拼接人工向量或 hunger。 |
| `hybrid` | 9 通道网格 + 20 维人工状态 | 结合全局/局部空间布局与人工特征，20 维向量中包含 hunger。 |

Grid 和 Hybrid 共用带 GroupNorm 的双分支 CNN，默认结构为：

```text
9 通道网格 [B, 9, H, W]
  -> 3x3 Conv（32 通道）+ GroupNorm + ReLU
  -> ResidualBlock（第一层 dilation=1，第二层 dilation=1）
  -> ResidualBlock（第一层 dilation=1，第二层 dilation=1）
  -> ResidualBlock（第一层 dilation=2，第二层 dilation=1）
  -> 共享特征图 [B, 32, H, W]
       ├─ 全局分支：1x1 Conv 32->8 + GroupNorm + Identity -> [B, 8*H*W]
       └─ 局部分支：1x1 Conv 32->8 + GroupNorm + 蛇头中心 5x5 裁剪 -> [B, 200]
  -> Grid 拼接为 [B, 8*H*W+200]
```

Grid 模式把 `8*H*W+200` 维空间特征送入隐藏层；Hybrid 再拼接 20 维人工状态，形成 `8*H*W+220` 维。默认隐藏层为 `256`，Grid/Hybrid 的 Dueling 分支分别为 `256 -> 128 -> 1` 和 `256 -> 128 -> 3`，最后按 `Q = V + A - mean(A)` 合成三个动作的 Q 值。网络参数量会随棋盘面积变化。

局部分支将四个方向蛇头通道相加以定位蛇头，先在共享特征图外围补 2 格，再通过 `F.unfold + gather` 批量提取每个样本的 5x5 窗口，因此蛇头位于边角时形状仍固定，并且整个路径可以反向传播。

CNN 的主干通道数、分支压缩通道数和 dilation 序列已经参数化；空间宽高直接取自 `--width` 和 `--height`，不再使用自适应池化。训练保存的 checkpoint 会记录这些架构参数和 `architecture_version=3`。默认 `q_network` 可直接兼容版本2的10×10 checkpoint；版本2的6×6和20×20 checkpoint需要在评估时显式选择 `q_network_old`。旧网络保留历史池化结构，但DQNAgent会禁止其训练和保存。

Grid/Hybrid 的状态数据使用连续 NumPy 数组保存。动作选择时通过
`torch.from_numpy()` 读取单个状态；经验回放采样后先用一次 `np.stack()`
组成连续 batch，再整体转换并传入计算设备，避免递归遍历 Python 嵌套列表。
环境的 `reset()` 和 `step()` 会直接返回所选 `state_mode` 对应的 observation，
不会先生成无用的 vector state。ReplayBuffer 使用固定容量环形列表，通过随机
索引直接采样，避免每次学习时复制整个经验池。

## 指标说明

训练指标：

| 指标 | TensorBoard tag | CSV 字段 | 含义 |
|------|-----------------|----------|------|
| 单局得分 | `train/score` | `score` | 当前 episode 吃到的食物数。 |
| 吃食效率 | `train/score_per_step` | `score_per_step` | `score / episode_steps`。 |
| 近 100 局平均得分 | `train/mean_score_100` | `mean_score_100` | 最近最多 100 个 episode 的 `score` 平均值。 |
| 历史最高近 100 局平均得分 | `train/best_mean_score_100` | 无 | 仅用于观察训练曲线，不参与 `best.pt` 选择。 |
| 单局累计奖励 | `train/episode_reward` | `episode_reward` | 当前 episode 内所有 step 的 reward 总和。 |
| 近 100 局平均累计奖励 | `train/mean_reward_100` | `mean_reward_100` | 最近最多 100 个 episode 的 `episode_reward` 平均值。 |
| 食物事件奖励 | `train/reward_food` | `food_reward` | 当前 episode 的食物事件奖励总和。 |
| 势函数奖励 | `train/reward_progress` | `progress_reward` | 当前 episode 的食物距离势函数奖励总和。 |
| 时间成本 | `train/reward_step` | `step_penalty` | 当前 episode 的逐步时间成本总和。 |
| 饥饿成本 | `train/reward_hunger` | `hunger_penalty` | 当前 episode 的渐进饥饿成本总和。 |
| 终止事件奖励 | `train/reward_terminal` | `terminal_reward` | 当前 episode 的碰撞、超时或完成地图奖励。 |
| 终止原因 | 无 | `termination_reason` | 碰撞墙、碰撞身体、饥饿超时或完成地图。 |
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
| 累计平均得分 | `eval_running_mean/score` | 无 | 从第 1 局到当前局的 `score` 真实平均值。 |
| 累计平均步数 | `eval_running_mean/steps` | 无 | 从第 1 局到当前局的 `steps` 真实平均值。 |
| 累计平均吃食效率 | `eval_running_mean/score_per_step` | 无 | 从第 1 局到当前局的 `score_per_step` 真实平均值。 |
| 累计平均最大蛇长 | `eval_running_mean/max_snake_length` | 无 | 从第 1 局到当前局的 `max_snake_length` 真实平均值。 |
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

评估还会生成 `eval_running_mean` 分组。该分组中的四张累计均值图用于主要模型对比；曲线最后一个点严格等于本次全部评估 episode 的总体均值，不受 TensorBoard `Smoothing` 设置影响。

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
