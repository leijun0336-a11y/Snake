# Snake AI

基于DQN的强化学习小项目，贪吃蛇AI。

## 🚀 快速启动游戏(需要安装uv)

```bash
uv run --extra cpu snake-play
```

> 这是游戏的唯一推荐启动命令。首次运行会自动安装项目依赖和 CPU 版 PyTorch，
> 下载时间可能稍长；之后再次启动会直接复用已有环境。


## 最新三 seed 实验结果

以下图片汇总 `dqn_20260728_140741`、`dqn_20260728_140830` 和
`dqn_20260728_140919`。曲线中的每个 episode 均按三个训练 seed 的对应数据取算术平均。

### 训练曲线

三个实验均训练 50000 局。

图中浅色曲线表示三个 seed 在同一 episode 上的原始指标平均值，深色曲线表示该平均曲线的滚动均值：

- **Score**：单局吃到的食物数量；`score rolling50` 和 `mean score 100` 分别是最近
  50 局和 100 局的平均分。
- **Reward**：单局所有环境奖励之和；`mean reward 100` 是最近 100 局平均值。固定分解关系为
  `episode_reward = food_reward + progress_reward + step_penalty + hunger_penalty + terminal_reward`
  （各项均先在单局内累加），因此 Reward 不只由 Score 决定。
- **Episode Steps**：单局执行的环境步数；`steps rolling50` 是最近 50 局平均步数。
- **Loss**：该局内所有 DQN 学习更新的平均 Huber loss；`mean loss 100` 是最近 100 局的
  episode loss 平均值。Loss 与 Score 或 Reward 之间不存在固定换算关系。
- **Epsilon**：epsilon-greedy 的随机探索概率。本组实验按环境步数线性衰减，定量关系为
  `epsilon = max(0.01, 1.0 - 0.99 × environment_steps / 300000)`，因此它与 episode
  之间没有固定线性关系。
- **Replay Buffer Size**：经验池中保存的 transition 数量。本组使用 `n_step=1`、容量
  100000，因此关系为 `buffer_size = min(environment_steps, 100000)`。

![最新三 seed DQN 训练曲线](docs/images/dqn_20260728_three_seed_training.png)

### `best.pt` 评估曲线

每个实验的 `best.pt` 均独立评估 2000 局，图中每个评估 episode 按三个 checkpoint
的对应结果取算术平均。

| Seed | Run | Score 平均值 | 最大蛇长平均值 | 满分率 | 超时率 |
|---:|---|---:|---:|---:|---:|
| 42 | `dqn_20260728_140741` | 30.2840 | 33.2840 | 77.70% | 0.45% |
| 3407 | `dqn_20260728_140830` | 29.7275 | 32.7275 | 76.15% | 0.40% |
| 2027 | `dqn_20260728_140919` | 29.8775 | 32.8775 | 77.75% | 0.55% |
| 三 seed 平均 | — | 29.9630 | 32.9630 | 77.20% | 0.47% |

表格及评估图中的指标含义如下：

- **Score 平均值**：2000 个评估局得分的算术平均值。
- **最大蛇长平均值**：每局达到的最大蛇长再取平均。本环境中蛇只增长、不缩短，初始蛇长为
  3，因此每局恒有 `最大蛇长 = Score + 3`，进而恒有
  `最大蛇长平均值 = Score 平均值 + 3`。
- **满分率**：得分达到 `6 × 6 - 3 = 33` 的局数除以 2000；等价条件是最大蛇长达到
  36。它由得分分布决定，不能只通过 Score 平均值计算。
- **超时率**：模型在 1000 步上限内没有自然终止的局数除以 2000。超时局的 Steps 为
  1000，但超时率不能从平均步数直接反推。
- **Score 图**：浅色线是三个 checkpoint 在同一评估 episode 上的得分平均值，深色线是其
  最近 100 局滚动均值。
- **Episode Steps 图**：浅色线是三个 checkpoint 在同一评估 episode 上的步数平均值，
  深色线是其最近 100 局滚动均值。
- 表格最后一行是三个 seed 的算术平均；由于每个 seed 都评估 2000 局，它也等于把全部
  6000 局合并后计算的总体结果。图中的末端滚动均值只覆盖最后 100 个评估 episode，
  不等于表格中的 2000 局总体平均值。

![最新三 seed best.pt 评估曲线](docs/images/dqn_20260728_three_seed_best_evaluation.png)


## 贪吃蛇 AI 的实现方法

### 游戏规则

- 棋盘大小为 `6 × 6`，蛇的初始长度为 3，食物随机出现在空格中。
- AI 每一步只能选择**直行、右转、左转**，不能直接反向。
- 吃到食物后得 1 分，蛇身增长一格；占满 36 个格子即完成棋盘，因此满分为
  `36 - 3 = 33`。
- 撞墙、撞到自身都会结束游戏。训练时如果连续超过 36 步没有吃到食物，也会因饥饿结束，
  以减少无意义的循环。

### 状态与动作

AI 使用 Hybrid 状态，同时观察：

- **棋盘状态**：9 个通道表示边界、蛇身、蛇头方向、蛇尾、食物和身体顺序；
- **人工特征**：20 个数值描述三个动作是否危险、当前方向、食物方位、到墙和蛇身的距离，
  以及饥饿程度。

CNN 同时提取完整棋盘的全局特征和蛇头周围 `3 × 3` 的局部特征，再与人工特征拼接。
网络最终为三个动作分别输出一个 Q 值，选择预期长期收益最高的动作。

### 奖励设计


| 事件 | 奖励 |
|---|---:|
| 吃到食物 | `+10` |
| 占满棋盘 | 额外 `+20` |
| 撞墙或撞到身体 | `-100` |
| 长时间未进食 | `-12` |
| 每次合法移动 | `-0.005` |
| 饥饿成本 | `-0.02 × hunger_ratio²` |

其中 `hunger_ratio` 是“连续未进食步数 ÷ 36”，最大取 1。
此外还使用基于食物距离的势函数奖励：靠近食物时获得少量正反馈，远离食物时获得少量负反馈。
单步总奖励是吃食、距离变化、移动成本、饥饿成本和终止奖励之和。这样既保留“吃到食物”这一
主要目标，也能在尚未吃到食物时给 AI 提供方向提示。

### DQN 算法

1. **探索**：训练初期通过 epsilon-greedy 随机尝试动作，之后逐步降低随机概率。
2. **经验回放**：把 `(状态、动作、奖励、下一状态)` 保存到经验池；PER 优先抽取 TD error
   较大的经验。
3. **Double DQN**：策略网络选择下一动作，目标网络评估该动作，减少 Q 值过高估计。
4. **Dueling DQN**：分别估计状态价值和各动作的相对优势，再组合成三个动作的 Q 值。
5. **模型选拔**：训练过程中使用固定随机种子进行 quick 和 confirmation 验证，表现更好的
   checkpoint 保存为 `best.pt`，最终再进行独立评估。

6. **Q 网络**：把当前状态转换为三个动作的 Q 值。Q 值表示在当前状态选择某个动作后，
   预计能够获得的长期累计收益。训练时使用两套结构相同的网络：

   - **Policy Network**：持续学习，并负责选择动作；
   - **Target Network**：定期从 Policy Network 同步参数，用于计算更稳定的训练目标。

   网络先用 CNN 提取完整棋盘的全局特征和蛇头周围 `3 × 3` 的局部特征，再与 20 维人工特征
   融合。Dueling Head 将融合特征拆分为状态价值 `V(s)` 和动作优势 `A(s,a)`，最后计算：

   ```text
   Q(s,a) = V(s) + A(s,a) - mean(A(s,·))
   ```

   ```mermaid
   flowchart TB
       BOARD["Board State<br/>9 × 6 × 6"] --> CNN["CNN Backbone"]
       CNN --> GLOBAL["Global Features"]
       CNN --> LOCAL["3 × 3 Local Features"]

       GLOBAL --> FUSION["Feature Fusion"]
       LOCAL --> FUSION
       VECTOR["Vector Features<br/>20 values"] --> FUSION

       FUSION --> FC["Fully Connected Layer<br/>256 units"]
       FC --> VALUE["Value Stream<br/>V(s)"]
       FC --> ADVANTAGE["Advantage Stream<br/>A(s,a)"]

       VALUE --> COMBINE["Dueling Combination<br/>Q = V + A - mean(A)"]
       ADVANTAGE --> COMBINE
       COMBINE --> QVALUES["Three Q-values<br/>Straight | Right | Left"]
       QVALUES --> ACTION["Choose the Highest Q-value"]
   ```

   *Figure: Simplified architecture of the Hybrid Dueling Q-Network.*
