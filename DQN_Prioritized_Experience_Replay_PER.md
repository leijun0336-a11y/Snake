# DQN 的优先经验回放（Prioritized Experience Replay，PER）

DQN 的 **Priority Replay Buffer**，通常叫：

\[
\boxed{\text{Prioritized Experience Replay，PER}}
\]

中文一般译为**优先经验回放**。

它解决的问题是：普通 DQN 从经验池中**均匀随机抽样**，但不同 transition 的学习价值并不相同。

---

# 1. 普通 Replay Buffer 的问题

普通 DQN 将经验存入经验池：

\[
(s_t,a_t,r_t,s_{t+1},d_t)
\]

训练时随机抽取一个 batch：

```python
batch = random.sample(replay_buffer, batch_size)
```

每条经验被抽中的概率相同：

\[
P(i)=\frac{1}{N}
\]

但是，有些经验模型早已学会，TD error 很小：

\[
|\delta_i|\approx 0
\]

重复训练它们带来的帮助有限。

另外一些经验预测错误很大：

\[
|\delta_i|\gg 0
\]

例如：

- 第一次吃到食物；
- 突然死亡；
- 从危险状态成功逃脱；
- Q 值预测明显错误。

这些经验通常更值得优先学习。

因此 PER 的核心思想是：

\[
\boxed{\text{TD error 越大的经验，被抽到的概率越高}}
\]

---

# 2. 什么是 TD error

对于普通 DQN：

\[
y_i
=
r_i+
\gamma(1-d_i)
\max_{a'}Q_{\text{target}}(s'_i,a')
\]

TD error 为：

\[
\delta_i
=
y_i-Q_{\text{online}}(s_i,a_i)
\]

绝对值：

\[
|\delta_i|
\]

表示当前网络对这条 transition 的预测误差。

例如：

```text
经验 A：TD error = 0.02
经验 B：TD error = 5.00
```

说明：

- 经验 A 基本已经学会；
- 经验 B 的预测仍然很不准确。

PER 会让经验 B 更容易被采样。

---

# 3. 优先级怎么定义

最常见的定义是：

\[
p_i=|\delta_i|+\varepsilon
\]

其中：

- \(p_i\)：第 \(i\) 条经验的优先级；
- \(|\delta_i|\)：TD error 的绝对值；
- \(\varepsilon\)：很小的正常数，防止优先级变成 0。

例如：

\[
\varepsilon=10^{-6}
\]

为什么取绝对值？

因为：

- 很大的正 TD error 值得学习；
- 很大的负 TD error 也值得学习。

我们关心的是“预测错了多少”，而不是预测偏高还是偏低。

---

# 4. 优先级如何转成采样概率

采样概率定义为：

\[
P(i)
=
\frac{p_i^\alpha}
{\sum_{j=1}^{N}p_j^\alpha}
\]

其中 \(\alpha\) 控制优先采样的强度。

## 当 \(\alpha=0\)

\[
p_i^0=1
\]

因此：

\[
P(i)=\frac{1}{N}
\]

退化为普通均匀经验回放。

## 当 \(\alpha\) 越大

高优先级经验被采到的概率越高。

常见取值：

\[
\alpha=0.6
\]

例如有三条经验：

\[
p=[1,2,7]
\]

假设 \(\alpha=1\)，采样概率为：

\[
P=[0.1,0.2,0.7]
\]

第三条经验有 70% 的概率被采到。

如果 \(\alpha=0\)，则三条经验都是：

\[
P=\left[\frac13,\frac13,\frac13\right]
\]

---

# 5. PER 为什么会引入偏差

普通经验回放是均匀采样：

\[
P(i)=\frac1N
\]

但 PER 故意让某些经验更容易出现。

这意味着训练数据分布被改变了。

例如，经验 B 本来只占经验池的 1%，但因为 TD error 很大，它可能在 batch 中出现 10% 的次数。

如果直接按普通 loss 训练，网络会过度关注这些经验，产生采样偏差。

所以需要使用 **重要性采样权重** 修正。

---

# 6. Importance Sampling Weight

第 \(i\) 条经验的重要性采样权重为：

\[
w_i
=
\left(
N\cdot P(i)
\right)^{-\beta}
\]

其中：

- \(N\)：经验池中经验总数；
- \(P(i)\)：第 \(i\) 条经验的采样概率；
- \(\beta\)：偏差修正强度。

通常再除以 batch 中最大权重：

\[
\tilde w_i
=
\frac{w_i}{\max_j w_j}
\]

使权重位于：

\[
0<\tilde w_i\leq1
\]

最终 loss 写成：

\[
L
=
\frac1B
\sum_{i=1}^{B}
\tilde w_i
\cdot
\ell(\delta_i)
\]

如果使用 Huber loss：

\[
L
=
\frac1B
\sum_i
\tilde w_i
\cdot
\operatorname{Huber}(\delta_i)
\]

---

# 7. \(\beta\) 的作用

\(\beta\) 控制重要性采样修正程度。

## 当 \(\beta=0\)

\[
w_i=1
\]

完全不修正偏差。

## 当 \(\beta=1\)

进行完整的重要性采样修正。

常见做法是训练开始时：

\[
\beta_{\text{start}}=0.4
\]

然后逐渐增加到：

\[
\beta_{\text{end}}=1.0
\]

例如：

\[
\beta_t
=
\beta_0+
(1-\beta_0)
\frac{t}{T}
\]

并限制最大为 1。

为什么不是一开始就设置成 1？

训练前期更希望突出高 TD error 样本，提高学习效率；训练后期为了保证收敛稳定，再逐渐加强偏差修正。

---

# 8. \(\alpha\) 和 \(\beta\) 的区别

这两个参数很容易混淆。

| 参数 | 控制内容 |
|---|---|
| \(\alpha\) | 采样时多大程度偏向高优先级经验 |
| \(\beta\) | 训练时多大程度修正非均匀采样造成的偏差 |

可以简单理解为：

\[
\boxed{\alpha\text{负责制造偏向，}\beta\text{负责修正偏向}}
\]

常见初始配置：

```python
per_alpha = 0.6
per_beta_start = 0.4
per_beta_end = 1.0
per_epsilon = 1e-6
```

---

# 9. PER 的完整训练流程

## 第一步：与环境交互

得到：

\[
(s_t,a_t,r_t,s_{t+1},d_t)
\]

并存入经验池。

新经验通常还没有 TD error，因此一般将其优先级设为当前经验池最大优先级：

\[
p_{\text{new}}=\max_i p_i
\]

这样可以确保新经验至少有机会被采样一次。

## 第二步：按照优先级抽样

根据：

\[
P(i)
=
\frac{p_i^\alpha}
{\sum_jp_j^\alpha}
\]

抽取 batch。

采样函数需要返回：

```python
transitions, indices, weights
```

其中：

- `transitions`：采样到的经验；
- `indices`：这些经验在 buffer 中的位置；
- `weights`：重要性采样权重。

## 第三步：计算 TD target

普通 DQN：

\[
y_i
=
r_i+
\gamma(1-d_i)
\max_{a'}Q_{\text{target}}(s'_i,a')
\]

Double DQN：

\[
a_i^*
=
\arg\max_{a'}
Q_{\text{online}}(s'_i,a')
\]

\[
y_i
=
r_i+
\gamma(1-d_i)
Q_{\text{target}}(s'_i,a_i^*)
\]

## 第四步：计算新的 TD error

\[
\delta_i=y_i-Q_{\text{online}}(s_i,a_i)
\]

## 第五步：加权计算 loss

例如使用 Smooth L1 Loss：

```python
elementwise_loss = F.smooth_l1_loss(
    current_q,
    target_q,
    reduction="none",
)

loss = (weights * elementwise_loss).mean()
```

不能直接使用：

```python
nn.SmoothL1Loss()
```

默认的 `reduction="mean"` 会先把各样本 loss 平均，无法再分别乘重要性权重。

应该设置：

```python
reduction="none"
```

## 第六步：更新网络

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

## 第七步：更新经验优先级

使用最新 TD error：

\[
p_i\leftarrow|\delta_i|+\varepsilon
\]

对应代码：

```python
new_priorities = td_errors.detach().abs() + per_epsilon
replay_buffer.update_priorities(indices, new_priorities)
```

完整循环就是：

```text
存储经验
  ↓
按旧优先级采样
  ↓
计算最新 TD error
  ↓
加权更新 DQN
  ↓
用最新 TD error 更新优先级
```

---

# 10. 一个简化代码示例

下面用普通数组实现基本逻辑，便于理解。

```python
import numpy as np


class PrioritizedReplayBuffer:
    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        epsilon: float = 1e-6,
    ):
        self.capacity = capacity
        self.alpha = alpha
        self.epsilon = epsilon

        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0

    def add(
        self,
        state,
        action,
        reward,
        next_state,
        terminated,
    ):
        transition = (
            state,
            action,
            reward,
            next_state,
            terminated,
        )

        # 新经验使用当前最大优先级
        if len(self.buffer) == 0:
            max_priority = 1.0
        else:
            max_priority = self.priorities[
                :len(self.buffer)
            ].max()

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition

        self.priorities[self.position] = max_priority

        self.position = (
            self.position + 1
        ) % self.capacity

    def sample(
        self,
        batch_size: int,
        beta: float,
    ):
        buffer_size = len(self.buffer)

        priorities = self.priorities[:buffer_size]

        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(
            buffer_size,
            size=batch_size,
            replace=False,
            p=probabilities,
        )

        transitions = [
            self.buffer[index]
            for index in indices
        ]

        weights = (
            buffer_size * probabilities[indices]
        ) ** (-beta)

        weights /= weights.max()

        return transitions, indices, weights

    def update_priorities(
        self,
        indices,
        td_errors,
    ):
        for index, td_error in zip(
            indices,
            td_errors,
        ):
            priority = abs(float(td_error)) + self.epsilon
            self.priorities[index] = priority

    def __len__(self):
        return len(self.buffer)
```

训练部分：

```python
transitions, indices, weights = replay_buffer.sample(
    batch_size=batch_size,
    beta=beta,
)

states, actions, rewards, next_states, terminated = zip(
    *transitions
)

weights = torch.as_tensor(
    weights,
    dtype=torch.float32,
    device=device,
)

current_q = online_net(states).gather(
    dim=1,
    index=actions.unsqueeze(1),
).squeeze(1)

with torch.no_grad():
    next_q = target_net(next_states).max(dim=1).values

    target_q = (
        rewards
        + gamma
        * (1.0 - terminated)
        * next_q
    )

td_errors = target_q - current_q

elementwise_loss = F.smooth_l1_loss(
    current_q,
    target_q,
    reduction="none",
)

loss = (
    weights * elementwise_loss
).mean()

optimizer.zero_grad()
loss.backward()
optimizer.step()

replay_buffer.update_priorities(
    indices,
    td_errors.detach().cpu().numpy(),
)
```

---

# 11. 为什么实际实现常用 Sum Tree

上面的数组实现每次都要计算所有优先级之和：

\[
\sum_i p_i^\alpha
\]

经验池如果有 100000 条数据，频繁计算和采样会比较慢。

高效实现通常使用 **Sum Tree**。

叶子节点存放每条经验的优先级，父节点保存两个子节点之和：

```text
                    10
                /        \
               3          7
             /  \       /  \
            1    2     3    4
```

总优先级是根节点：

\[
1+2+3+4=10
\]

采样时生成：

\[
x\sim U(0,10)
\]

然后根据区间沿树向下寻找对应叶子。

Sum Tree 的主要复杂度：

| 操作 | 复杂度 |
|---|---:|
| 添加经验 | \(O(\log N)\) |
| 更新优先级 | \(O(\log N)\) |
| 按优先级采样 | \(O(\log N)\) |

而简单数组实现更新和采样可能需要：

\[
O(N)
\]

对于较大的 replay buffer，Sum Tree 更合适。

---

# 12. Proportional PER 和 Rank-based PER

PER 主要有两类。

## Proportional prioritization

使用 TD error 大小：

\[
p_i=|\delta_i|+\varepsilon
\]

这是最常用的方式。

优点：

- 实现直观；
- TD error 越大，优先级越高。

缺点：

- 极端 TD error 可能占据过大概率；
- 对噪声和异常样本较敏感。

## Rank-based prioritization

先按照 TD error 大小排序，再根据排名分配优先级：

\[
p_i=\frac{1}{\operatorname{rank}(i)}
\]

例如：

| TD error 排名 | 优先级 |
|---:|---:|
| 1 | 1 |
| 2 | \(1/2\) |
| 3 | \(1/3\) |
| 4 | \(1/4\) |

它更不容易被某个极端 TD error 支配，但实现稍复杂。

多数工程实现使用 proportional PER。

---

# 13. PER 的优点

PER 的主要优点是提高样本利用效率。

对于贪吃蛇，以下经验可能更频繁地被学习：

- 吃到食物；
- 撞墙死亡；
- 撞到身体；
- 危险状态下做出错误动作；
- 网络预测与实际结果差异较大的状态。

这对奖励稀疏任务通常有帮助。

普通回放中，一次吃食物的经验可能很快被大量普通移动经验淹没；PER 可以因为其 TD error 较大而多次采样它。

---

# 14. PER 的缺点

## 过度关注异常经验

如果某些 transition 因为奖励噪声或环境随机性长期有较大 TD error，PER 会反复采样它们。

模型可能被少数异常样本支配。

## 降低经验多样性

高优先级样本会频繁出现，低优先级样本可能长期得不到训练。

因此 \(\alpha\) 不宜过大。

通常：

\[
\alpha=0.5\sim0.7
\]

而不是直接设置成 1。

## 实现复杂

相比普通 deque，需要额外维护：

- 优先级；
- 采样概率；
- 重要性权重；
- 索引；
- 优先级更新；
- Sum Tree。

## 计算开销更大

每次训练后都要更新优先级，采样过程也比均匀随机采样复杂。

---

# 15. 对贪吃蛇的推荐参数

可以先使用：

```python
replay_buffer_size = 100_000

per_alpha = 0.6
per_beta_start = 0.4
per_beta_end = 1.0
per_epsilon = 1e-6

batch_size = 128
loss = "huber"
```

\(\beta\) 随训练步数线性增加：

```python
progress = min(
    global_step / total_training_steps,
    1.0,
)

beta = (
    beta_start
    + progress * (1.0 - beta_start)
)
```

还可以对优先级设置上限，避免极端样本完全控制采样：

```python
priority = np.clip(
    abs(td_error) + epsilon,
    1e-6,
    100.0,
)
```

---

# 16. PER 能否与其他 DQN 改进结合

可以。PER 与以下方法修改的位置不同：

```text
Double DQN：修改 TD target 的动作选择方式
Dueling DQN：修改网络结构
n-step DQN：修改 TD target 的奖励范围
PER：修改经验采样方式
```

因此可以组合成：

\[
\boxed{
\text{Double DQN}
+
\text{Dueling Network}
+
\text{PER}
}
\]

这也是 Rainbow DQN 中的一部分。

使用 Double DQN 时，优先级仍然来自 Double DQN 的 TD error：

\[
\delta_i
=
r_i+
\gamma Q_{\text{target}}
\left(
s'_i,
\arg\max_aQ_{\text{online}}(s'_i,a)
\right)
-
Q_{\text{online}}(s_i,a_i)
\]

---

# 17. PER 对绕圈问题有没有帮助

PER 可能让吃食物、死亡等关键经验更频繁地被学习，但它**不能从根本上解决奖励函数导致的绕圈问题**。

如果奖励函数认为：

\[
\text{安全绕圈的回报}
>
\text{冒险吃食物的期望回报}
\]

那么即使使用 PER，模型仍然可能学习绕圈。

PER 解决的是：

\[
\boxed{\text{哪些经验应该更频繁地学习}}
\]

它不解决：

\[
\boxed{\text{什么行为才是任务真正希望的行为}}
\]

---

# 总结

PER 的完整核心可以浓缩为四步：

\[
\boxed{
p_i=|\delta_i|+\varepsilon
}
\]

\[
\boxed{
P(i)=
\frac{p_i^\alpha}
{\sum_jp_j^\alpha}
}
\]

\[
\boxed{
w_i=
\left(NP(i)\right)^{-\beta}
}
\]

\[
\boxed{
L=
\frac1B
\sum_i
\tilde w_i\,
\ell(\delta_i)
}
\]

直观上就是：

> 预测错误越大的经验，越优先学习；但为了避免这种非均匀采样改变训练分布，需要用重要性采样权重进行修正。
