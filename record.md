# 第八次实验可复现记录：dqn_20260712_130642

## 1. 记录目的

本文件冻结第八次 6×6 Hybrid DQN 实验中已经取得良好结果的有效配置、权重和指标，作为后续实验发生性能倒退时的恢复基线。

这次实验应优先保留的模型是：

```text
checkpoints/dqn_20260712_130642/latest.pt
```

不要用同目录的 `best.pt` 替代。独立评估表明 `latest.pt` 明显更好。

## 2. 重要的版本说明

本次训练目录创建于 `2026-07-12 13:06:42`。训练持续约 6.32 小时，而相关代码改动在训练过程中才陆续提交。因此，没有一个单独的 Git commit 能完整代表训练进程启动时加载的代码。

本记录按证据可靠性区分配置来源：

| 证据级别 | 含义 |
|---|---|
| 已验证 | 直接来自 checkpoint 元数据、训练 CSV 或 TensorBoard event |
| 高可信恢复 | checkpoint 未保存该字段，但能由训练日志奖励分量和训练时源码确定 |
| 推断 | 只能根据默认配置和运行结果推断，无法从产物中严格证明 |

后续不要直接使用当前源码默认值来描述本次实验。当前奖励和终止规则已经改变。

## 3. 产物位置和哈希

### 3.1 模型

| 文件 | 用途 | SHA256 |
|---|---|---|
| `checkpoints/dqn_20260712_130642/latest.pt` | 推荐恢复和评估的模型 | `DC4CBEE6CA130575DCDE840FA461B7BA77C9215607930F5A230E26610F4EC58B` |
| `checkpoints/dqn_20260712_130642/best.pt` | 训练 mean_score_100 选出的旧 best，仅用于对照 | `11F4905BA015DD811BF810BBB72260B977CE13BD1B8EAB20415ED947A113EDDA` |

### 3.2 日志

| 文件 | SHA256 |
|---|---|
| `runs/dqn_20260712_130642/train_metrics.csv` | `D190BA26E5887D8C0BC019CCE327107CFDE1EB15739DEB3D18CD0A8ACD943C30` |
| `runs/dqn_20260712_130642/eval_metrics.csv` | `1103034EEAB8BC73B0A82035A0E05941D5336B1DA7771F604E62532234829E37` |

注意：`eval_metrics.csv` 包含两次评估追加的数据，共 2000 行，前 1000 局是 `best.pt`，后 1000 局是 `latest.pt`。不能对整个 CSV 直接求均值来代表 latest。

`latest.pt` 的权威评估报告来自：

```text
runs/dqn_20260712_130642/
events.out.tfevents.1783859350.autodl-container-8eda4eaad0-2866bf7e.107843.0.eval
```

## 4. 环境配置

| 配置 | 值 | 证据 |
|---|---:|---|
| 棋盘宽度 | 6 | checkpoint `state_size=(9, 6, 6)` |
| 棋盘高度 | 6 | checkpoint `state_size=(9, 6, 6)` |
| 初始蛇长 | 3 | 环境实现 |
| 动作数 | 3 | checkpoint `action_size=3` |
| 状态模式 | `hybrid` | checkpoint |
| Grid 通道 | 9 | checkpoint |
| 辅助向量维度 | 20 | checkpoint `auxiliary_size=20` |
| 食物位置 | 随机 | 训练时环境实现 |
| 训练逐食物 starvation limit | `width * height = 36` | 训练时源码；不是当前的 `36 + snake_length` |
| 训练 episode 总步数上限 | 无独立 500 步上限 | 训练时源码；不是当前默认值 |

### 4.1 与当前环境的关键差异

当前代码已经改为：

```text
starvation_limit = width * height + snake_length
max_steps_per_training_episode = 500
```

第八次实验训练时使用的是：

```text
starvation_limit = width * height
max_steps_per_training_episode = unlimited
```

如果后续实验明显退化，恢复时必须把这两个差异纳入排查，不能只恢复 reward 数值。

## 5. 奖励函数配置

这是第八次实验实际使用的奖励设计，不是当前默认奖励。

| 奖励项 | 配置 | 实际语义 | 证据 |
|---|---:|---|---|
| 食物奖励 | `+10.0` | 每吃一个食物增加 10 | CSV `food_reward` |
| 碰撞墙壁 | `-100.0` | 终止奖励 | CSV 所有 `collision_wall` 行 |
| 碰撞身体 | `-100.0` | 终止奖励 | CSV 所有 `collision_body` 行 |
| 饥饿超时 | `-12.0` | 终止奖励 | CSV 所有 `starvation` 行 |
| 填满棋盘 | 额外 `+20.0` | 与最后一个食物 `+10` 叠加 | CSV 所有 `board_completed` 行 |
| 普通 step cost | `-0.005` | 当时会在合法移动上累计，包括吃食/通关步 | CSV `step_penalty` |
| Hunger cost | `-0.02 * hunger_ratio²` | 未吃食时累计 | CSV `hunger_penalty` |
| Potential shaping | 开启 | `beta * (gamma * next_phi - current_phi)` | CSV `progress_reward` 非零 |
| Potential beta | `2.0` | 曼哈顿距离势函数系数 | 训练时源码 |
| Reward gamma | `0.99` | 势函数和 DQN 折扣 | 训练配置 |

其中：

```text
phi = 1 - normalized_manhattan_distance_to_food
progress_reward = 2.0 * (0.99 * next_phi - current_phi)
hunger_ratio = min(steps_since_food / 36, 1.0)
```

### 5.1 典型终止奖励证据

训练 CSV 中终止分布：

| 终止原因 | 局数 | terminal_reward |
|---|---:|---:|
| collision_wall | 8672 | -100 |
| collision_body | 4825 | -100 |
| starvation | 167 | -12 |
| board_completed | 1336 | +20 |

完成棋盘时的单步总奖励并不固定为 100。它由以下分量构成：

```text
最后一个食物 +10
完成棋盘额外 +20
potential shaping
step cost -0.005
```

这与当前“完成棋盘最后一步总计 +100”的奖励设计不同。

## 6. DQN 与训练超参数

### 6.1 checkpoint 直接保存的配置

| 配置 | 值 |
|---|---:|
| `epsilon_start` | 1.0 |
| `epsilon_end` | 0.01 |
| `epsilon_exp_decay` | False |
| `epsilon_exp_factor` | 0.995，线性衰减模式下不作为主调度 |
| `epsilon_linear_episodes` | 7500 |
| 最终 epsilon | 0.01 |
| `learn_steps` | 941170 |
| hidden size | 256 |
| Dueling | True |
| architecture version | 2 |
| policy network 参数量 | 386564 |

### 6.2 网络结构

| 配置 | 值 |
|---|---|
| 状态模式 | Hybrid |
| Grid 输入 | `9×6×6` |
| 人工向量 | 20 维 |
| CNN 主干通道 | 32 |
| CNN 输出通道 | 8 |
| 残差块 dilation | `(1, 1, 2)` |
| 全局池化输出 | `(10, 10)` |
| 局部分支 crop | `5×5` |
| Dueling value/advantage | 开启 |

### 6.3 训练配置

| 配置 | 值 | 可信度 |
|---|---:|---|
| Episodes | 15000 | 已验证 |
| Batch size | 128 | 高可信恢复 |
| Gamma | 0.99 | 高可信恢复 |
| Learning rate | 0.001 | 高可信恢复 |
| Replay buffer | 100000 | 已由日志达到上限验证 |
| Target hard update interval | 1000 learn steps | 高可信恢复 |
| Loss | Huber / Smooth L1 | 训练时实现 |
| Optimizer | Adam | 训练时实现 |
| Gradient clipping | max norm 10 | 训练时实现 |
| 每环境 step 学习次数 | 1 | 训练时实现 |
| Epsilon 线性衰减区间 | 前 7500 局 | checkpoint 已验证 |
| 后半程 epsilon | 0.01 | checkpoint 和 CSV 已验证 |
| Early stop | 未触发；实际跑满 15000 局 | 已验证 |
| Seed | 大概率为默认 42 | 推断，checkpoint 未保存 seed |
| Deterministic CUDA | 无法确认 | 未保存 |

## 7. 历史训练命令语义

下面命令表达本次实验的训练意图，但不能直接在当前代码上原样复现旧奖励和旧 starvation，因为当前默认值已经变化：

```bash
uv run python -m snake_ai.train \
  --width 6 \
  --height 6 \
  --state-mode hybrid \
  --max-episodes 15000 \
  --epsilon-linear-episodes 7500
```

当时还需要满足：

```text
potential reward = enabled
early stop = disabled / 未触发
collision = -100
starvation = -12
win extra reward = +20
step = -0.005
hunger scale = 0.02
starvation limit = 36
training episode max steps = unlimited
```

当前代码已新增不可变的 `experiment8` reward profile，恢复旧奖励数值、叠加顺序、固定棋盘面积 starvation、严格 `> limit` 的历史边界，以及无独立训练步数上限。AutoDL 严格配置复现使用：

```bash
bash scripts/train_experiment8_autodl.sh
```

该脚本不接受额外参数，避免无意覆盖历史配置。普通训练仍默认使用 `reference` profile，不会静默改变已经与对方对齐的基线。

## 8. 训练结果

- Run：`dqn_20260712_130642`
- Episodes：15000
- 总训练时间：22766.937 秒，约 6.32 小时
- Best score：33
- Best mean_score_100：23.9100，出现在 episode 11576
- 最后一局 score：18
- 最后 mean_score_100：22.3800

| Metric | Mean | Std | Min | Max | Last |
|---|---:|---:|---:|---:|---:|
| score | 11.0931 | 10.9183 | 0.0000 | 33.0000 | 18.0000 |
| episode_steps | 62.7531 | 58.4306 | 3.0000 | 284.0000 | 95.0000 |
| score_per_step | 0.1489 | 0.0833 | 0.0000 | 0.7500 | 0.1895 |
| mean_score_100 | 11.0172 | 8.9144 | 0.0000 | 23.9100 | 22.3800 |
| episode_reward | 29.6697 | 140.6148 | -101.5719 | 375.7348 | 92.8959 |
| mean_reward_100 | 28.7161 | 107.9000 | -100.5940 | 205.7294 | 177.7125 |
| loss | 1.1715 | 0.6112 | 0.0000 | 12.1110 | 1.1146 |
| mean_loss_100 | 1.1711 | 0.5591 | 0.0000 | 2.6866 | 1.1566 |
| epsilon | 0.2575 | 0.3195 | 0.0100 | 0.9999 | 0.0100 |
| replay_buffer_size | 73162.4223 | 35154.3334 | 13.0000 | 100000.0000 | 100000.0000 |

## 9. latest.pt 评估结果

评估协议：

```text
checkpoint = latest.pt
episodes = 1000
max total steps per episode = 1000
per-food starvation = disabled
epsilon = 0
```

| Metric | Mean | Std | Min | Max |
|---|---:|---:|---:|---:|
| score | 27.9180 | 7.9546 | 2.0000 | 33.0000 |
| steps | 147.6990 | 95.8641 | 8.0000 | 1000.0000 |
| score_per_step | 0.2027 | 0.0339 | 0.0260 | 0.3478 |
| max_snake_length | 30.9180 | 7.9546 | 5.0000 | 36.0000 |

满分统计：

```text
满分定义 = score == 33
满分局数 = 619 / 1000
满分率 = 61.90%
```

对照模型 `best.pt` 的结果为平均分 26.466、满分表现更差，因此恢复时必须优先使用 `latest.pt`。

## 10. 性能倒退时的恢复清单

如果后续实验开倒车，按以下顺序检查：

1. 确认使用的权重 SHA256 是 `DC4CB...F4EC58B`，即第八次实验 `latest.pt`。
2. 评估必须只读取 latest 对应的 1000 局，不能把 `eval_metrics.csv` 的两批数据混合。
3. 将棋盘恢复为 6×6、状态恢复为 Hybrid、网络恢复为 architecture version 2。
4. 将 epsilon 恢复为 15000 局训练、前 7500 局线性衰减、后 7500 局保持 0.01。
5. 将奖励恢复为本文件第 5 节，而不是当前简化奖励。
6. 将训练 starvation limit 恢复为固定 36，并使用历史条件 `steps_since_food > 36`；即第 37 个连续未吃食合法移动才终止。
7. 移除当前每局 500 step 的训练截断。
8. 继续使用 `latest.pt`，不要仅根据训练 mean_score_100 选择 `best.pt`。
9. 用相同的 1000 局评估协议复测，目标基线是平均分 27.918、满分率 61.90%。

## 11. 建议的长期保护措施

历史 checkpoint 没有保存 reward、starvation、seed、optimizer 和完整训练配置。当前新训练已经会在 run/checkpoint 目录写入 `config.json`，并把解析后的配置嵌入 checkpoint 的 `run_config` 字段；optimizer 与完整续训状态仍未保存。长期完整续训还应继续补充：

```text
env_config
reward_config
train_config
evaluation_protocol
episode
seed
git_commit
full_command
optimizer_state
```

在该功能实现前，本文件、对应 checkpoint、CSV 和 TensorBoard event 应作为一个不可拆分的实验归档保留。
