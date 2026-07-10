# Snake AI 奖励信号重构方案

## 1. 重构目标

当前环境的奖励只有：

- 吃到食物：`+10`
- 碰撞或饥饿超时：`-10`
- 其他步骤：`0`

这会带来三个问题：

1. 追食物的中间步骤没有学习信号。
2. 安全绕圈在超时前几乎没有代价。
3. 约 99% 的 replay transition 都是零奖励，Grid CNN 很难从中学习蛇头、方向和食物之间的关系。

奖励重构的目标是：

- 保持“吃到食物”作为最主要目标；
- 为接近食物提供幅度较小、连续的学习信号；
- 让低效移动和长期绕圈逐渐产生代价；
- 避免靠近—远离食物反复刷奖励；
- 保持各奖励组件可观测、可记录、可单独调参。

## 2. 总奖励公式

每一步奖励拆分为：

$$
r_t = r_{event} + r_{progress} + r_{step} + r_{hunger}
$$

第一版推荐参数如下：

| 奖励组件 | 推荐值 |
|---|---:|
| 吃到食物 | `+10.0` |
| 撞墙或撞身体 | `-10.0` |
| 饥饿超时 | `-12.0` |
| 填满地图 | `+20.0` |
| 普通时间成本 | `-0.005 / step` |
| 距离势函数系数 beta | `2.0` |
| 最大饥饿惩罚系数 | `0.02` |
| 折扣因子 gamma | `0.99` |

第一版不要同时引入自由空间奖励、蛇身长度奖励、存活奖励和重复位置惩罚。奖励组件越复杂，越难确认模型利用了哪一部分，以及是否出现 reward hacking。

## 3. 事件奖励

事件奖励继续作为最终目标的主要信号：

```python
FOOD_REWARD = 10.0
COLLISION_PENALTY = -10.0
STARVATION_PENALTY = -12.0
WIN_REWARD = 20.0
```

饥饿超时惩罚略大于普通碰撞，因为当前模型最主要的失败模式是安全绕圈直到饿死。

不建议将超时惩罚直接提高到 `-50` 或 `-100`。过大的负奖励会让 Q 网络主要围绕避免死亡训练，进一步忽视食物。

## 4. 食物距离势函数奖励

### 4.1 不使用简单的距离增减奖励

不要直接使用下面的规则：

```python
if new_distance < old_distance:
    reward += 1.0
else:
    reward -= 1.0
```

原因是：

- `+1/-1` 相对于吃食物的 `+10` 太大；
- 模型可能通过靠近一步、远离一步反复利用奖励；
- 优化目标会从“吃到食物”变成“制造距离变化”。

应使用 Potential-based Reward Shaping。

### 4.2 定义归一化距离

第一版使用曼哈顿距离：

$$
d(s) = |x_{food} - x_{head}| + |y_{food} - y_{head}|
$$

地图最大曼哈顿距离：

$$
D_{max} = (width - 1) + (height - 1)
$$

20×20 地图中 `D_max = 38`。

归一化距离：

$$
d_{norm}(s) = \frac{d(s)}{D_{max}}
$$

定义势函数：

$$
\Phi(s) = 1 - d_{norm}(s)
$$

其含义为：

- 蛇头到达食物时，`Phi = 1`；
- 离食物越远，`Phi` 越接近 `0`。

### 4.3 势函数奖励

使用与 DQN 相同的 `gamma`：

$$
r_{progress} = \beta(\gamma\Phi(s') - \Phi(s))
$$

推荐：

```python
gamma = 0.99
progress_beta = 2.0
```

伪代码：

```python
old_distance = manhattan(old_head, old_food)
new_distance = manhattan(new_head, old_food)

old_phi = 1.0 - old_distance / max_distance
new_phi = 1.0 - new_distance / max_distance

progress_reward = progress_beta * (
    gamma * new_phi - old_phi
)
```

必须使用移动前的同一颗食物 `old_food` 计算 `old_phi` 和 `new_phi`。吃到食物后环境会立即生成新食物，不能用新食物计算当前动作的 progress reward，否则成功吃食的 transition 会混入随机位置噪声。

在推荐参数下，典型幅度大约为：

- 接近食物一步：`+0.03～+0.05`
- 远离食物一步：`-0.05～-0.07`
- 与食物距离不变：轻微负奖励

食物事件仍然奖励 `+10`，因此势函数只负责提供方向梯度，不会取代最终任务。

### 4.4 碰撞时的处理

动作直接造成碰撞时，建议：

```python
progress_reward = 0.0
event_reward = -10.0
```

不要为非法的新坐标继续计算距离势函数。碰撞事件本身已经提供了足够明确的反馈。

## 5. 时间成本

每个正常的非终止步骤加入：

```python
step_reward = -0.005
```

作用是：

- 20 步吃到食物优于 200 步吃到食物；
- 无收益的循环不再是零成本；
- 鼓励更高的 `score_per_step`。

不建议第一版使用 `-0.1 / step`。如果一局移动 400 步，会累计 `-40`，超过食物和死亡事件的量级，可能诱导智能体尽快结束 episode。

`-0.005` 的量级相对安全：

- 20 步累计约 `-0.1`；
- 400 步累计约 `-2.0`。

它能够区分路径效率，但不会压过 `+10` 的食物奖励。

## 6. 渐进式饥饿惩罚

当前模型直到连续 401 步没吃食物时才突然收到终止惩罚。在此之前，绕圈没有额外代价。

定义饥饿比例：

$$
h = \min\left(\frac{steps\_since\_food}{starvation\_limit}, 1\right)
$$

每一步加入二次增长惩罚：

$$
r_{hunger} = -0.02h^2
$$

实现形式：

```python
hunger_ratio = min(
    steps_since_food / starvation_limit,
    1.0,
)
hunger_reward = -0.02 * hunger_ratio**2
```

20×20 地图中，`starvation_limit = 400`。对应的单步惩罚为：

| 连续未吃食物步数 | 单步饥饿惩罚 |
|---:|---:|
| 0 | `0` |
| 100 | `-0.00125` |
| 200 | `-0.005` |
| 300 | `-0.01125` |
| 400 | `-0.02` |

正常寻找食物时影响很小，但长期绕圈会越来越不划算。吃到食物后将 `steps_since_food` 清零，饥饿惩罚也随之清零。

## 7. 将饥饿进度加入 observation

这是奖励重构的必要配套修改。

当前 episode 是否即将结束取决于 `steps_since_food`，但智能体观察不到它。同一个网格画面可能表示：

- 刚刚吃完食物；
- 已经连续 399 步没有吃到食物。

这两个状态对智能体完全一样，却有不同的未来转移和终止风险，不满足严格的 Markov 性。

### Grid 模式

增加一个常数通道：

```python
hunger_channel[:, :] = (
    steps_since_food / starvation_limit
)
```

Grid observation 从：

```text
[5, height, width]
```

变为：

```text
[6, height, width]
```

第六通道的所有位置都是相同的饥饿比例，CNN 可以将其作为全局标量使用。

### Vector 和 Hybrid 模式

在向量状态中追加：

```python
steps_since_food_norm = (
    steps_since_food / starvation_limit
)
```

改变 observation 维度后必须重新训练，旧 checkpoint 不再兼容。

## 8. 推荐的奖励计算顺序

```python
old_head = snake[0]
old_food = food

new_head = calculate_new_head(action)

if collision(new_head):
    reward_components = {
        "food": 0.0,
        "progress": 0.0,
        "step": 0.0,
        "hunger": 0.0,
        "terminal": -10.0,
    }
    reward = sum(reward_components.values())
    done = True

else:
    move_snake(new_head)

    old_phi = food_potential(old_head, old_food)
    new_phi = food_potential(new_head, old_food)

    progress_reward = 2.0 * (
        gamma * new_phi - old_phi
    )
    step_reward = -0.005

    if new_head == old_food:
        food_reward = 10.0
        hunger_reward = 0.0
        steps_since_food = 0
        place_new_food()
    else:
        food_reward = 0.0
        steps_since_food += 1

        hunger_ratio = min(
            steps_since_food / starvation_limit,
            1.0,
        )
        hunger_reward = -0.02 * hunger_ratio**2

    if steps_since_food > starvation_limit:
        terminal_reward = -12.0
        done = True
    else:
        terminal_reward = 0.0

    reward_components = {
        "food": food_reward,
        "progress": progress_reward,
        "step": step_reward,
        "hunger": hunger_reward,
        "terminal": terminal_reward,
    }
    reward = sum(reward_components.values())
```

当前环境是在碰撞判断前增加 `steps_since_food`。实现重构时应统一计数顺序，并增加第 400/401 步的边界测试，避免超时条件出现 off-by-one。

## 9. 第一版暂时不要加入的奖励

### 9.1 不奖励单纯存活

不要加入：

```python
reward += 0.01
```

这会直接鼓励当前已经出现的绕圈策略。

### 9.2 不奖励自由空间

例如：

```python
reward += reachable_area * coefficient
```

模型可能为了保持大空间而永远不接近食物，或者沿地图边缘持续循环。

### 9.3 不直接惩罚重复坐标

蛇在正常规划时也可能需要重新经过旧位置。访问历史如果没有包含在 observation 中，还会产生新的非 Markov 奖励。

### 9.4 不给予大额固定距离奖励

`靠近 +1、远离 -1` 会让 shaping 奖励和真正吃到食物处于相同量级，使智能体优化距离变化而不是游戏得分。

### 9.5 第一版不使用 BFS 距离

把蛇身当作静态障碍计算 BFS 距离看起来更准确，但蛇尾会移动，当前无路不代表未来无路。BFS 距离还可能在局面变化时发生剧烈跳变。

第一版先使用稳定的曼哈顿距离。确认奖励链路正常后，再把 BFS 或动态可达距离作为独立对照实验。

## 10. 分别记录各奖励组件

不能只记录总 `episode_reward`。建议在 `info`、CSV 和 TensorBoard 中分别记录：

```text
episode_food_reward
episode_progress_reward
episode_step_penalty
episode_hunger_penalty
episode_terminal_penalty
episode_total_reward
```

同时记录终止原因：

```text
collision_wall
collision_body
starvation
board_completed
```

另外建议增加：

```text
greedy_action_straight_rate
greedy_action_right_rate
greedy_action_left_rate
starvation_rate
mean_steps_since_food
```

这样可以及时发现固定左转、固定右转或其他动作塌缩。

## 11. 对照实验计划

不要一次引入无法区分作用的大量奖励。建议分三组进行。

### 实验 A：原始奖励基线

```text
food       +10
collision  -10
其他         0
```

保留现有结果作为基线。

### 实验 B：只加入势函数

```text
food       +10
collision  -10
progress   beta=2.0
gamma      0.99
```

用于验证 CNN 是否开始利用食物位置。

### 实验 C：加入时间和饥饿成本

```text
step       -0.005
hunger     -0.02 * hunger_ratio^2
starvation -12
```

用于验证是否能消除安全绕圈直到超时的策略。

每个实验至少运行 3 个不同随机种子。只运行 seed 42 无法排除左右动作对称性被初始化偶然打破的影响。

模型选择应使用 `epsilon=0` 的独立评估均分，不能继续使用带探索的训练分数。

## 12. 推荐的最终配置

```text
gamma                     = 0.99

food_reward               = +10.0
collision_penalty         = -10.0
starvation_penalty        = -12.0
win_reward                = +20.0

progress_potential        = 1 - normalized_manhattan_distance
progress_beta             = 2.0
progress_reward           = beta * (gamma * next_phi - current_phi)

step_penalty              = -0.005
hunger_penalty            = -0.02 * hunger_ratio^2
starvation_limit          = width * height

Grid observation          = 原 5 通道 + hunger_ratio 常数通道
checkpoint selection      = epsilon=0 的独立评估均分
```

这套方案中：

- 食物和死亡事件决定最终目标；
- 势函数提供追食物的方向梯度；
- 时间成本鼓励更短路径；
- 饥饿惩罚消除长期安全绕圈；
- 饥饿 observation 修复超时条件对智能体不可见的问题；
- 分组件日志用于定位 reward hacking 和策略塌缩。
