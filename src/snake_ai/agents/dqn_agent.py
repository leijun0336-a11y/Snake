# 处理自引用问题
from __future__ import annotations

import random
from pathlib import Path

import torch
from torch import nn, optim

from snake_ai.agents.replay_buffer import ReplayBuffer
from snake_ai.models.q_network import QNetwork


class DQNAgent:
    def __init__(
        self,
        # 状态向量维度，当前贪吃蛇环境是 11 个输入特征。
        state_size: int,
        # 动作数量，当前为 3：直行、右转、左转。
        action_size: int,
        # Q 网络隐藏层神经元数量。
        hidden_size: int = 128,
        # Adam 优化器学习率，控制神经网络参数更新步幅。
        learning_rate: float = 1e-3,
        # 折扣因子，越接近 1 越重视未来奖励。
        gamma: float = 0.9,
        # 经验回放池容量，最多保存多少条 Transition。
        replay_buffer_size: int = 100_000,
        # 每次训练从经验池随机采样多少条经验。
        batch_size: int = 64,
        # 初始探索率，训练早期按这个概率随机选动作。
        epsilon_start: float = 1.0,
        # 最低探索率，防止后期完全不探索。
        epsilon_end: float = 0.01,
        # 每局结束后 epsilon 的衰减系数。
        epsilon_decay: float = 0.995,
        # 每隔多少次学习步骤，把 policy_net 的参数复制一份给 target_net.
        target_update_interval: int = 1000,
        # 随机种子，用于让探索、采样和网络初始化尽量可复现。
        seed: int = 42,
        # 计算设备；不传时优先使用 cuda，否则使用 cpu。
        device: str | None = None,
    ) -> None:
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_interval = target_update_interval
        self.learn_steps = 0
        self.random = random.Random(seed)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        
        # 生成随机数种子
        torch.manual_seed(seed)
        self.policy_net = QNetwork(state_size, hidden_size, action_size).to(self.device)
        self.target_net = QNetwork(state_size, hidden_size, action_size).to(self.device)
        # load_state_dict() 是一个用来加载模型参数的方法，这里用于参数复制
        self.target_net.load_state_dict(self.policy_net.state_dict())
        # 切换到评估模式，关闭dropout和batch_norm，因为目标网络本身不训练只接受参数
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        self.replay_buffer = ReplayBuffer(replay_buffer_size, seed=seed)

    # 动作采样，DQN采用epsilon-greedy
    def act(self, state: list[float], training: bool = True) -> int:
        # state: 输入的状态；training:是否训练，评估模式不探索。
        
        # 如果正在训练且落入epsilon概率内
        if training and self.random.random() < self.epsilon:
            return self.random.randrange(self.action_size)

        # act()只用于选动作，不参与梯度计算; learn()才需要梯度，因为要训练网络
        with torch.no_grad():
            state_tensor = torch.tensor([state], dtype=torch.float32, device=self.device)
            q_values = self.policy_net(state_tensor)
            # 返回Q网络输出向量中数值最大值对应的索引
            return int(q_values.argmax(dim=1).item())

    # 往经验回放池放东西
    def remember(
        self,
        # 执行动作前的当前状态。
        state: list[float],
        # 在当前状态下执行的动作。
        action: int,
        # 执行动作后环境返回的奖励。
        reward: float,
        # 执行动作后环境返回的下一个状态。
        next_state: list[float],
        # 执行动作后这一局是否结束。
        done: bool,
    ) -> None:
        self.replay_buffer.push(state, action, reward, next_state, done)

    # 更新动作函数
    def learn(self) -> float | None:  # 返回值是float或者None
        
        # 如果经验回放池的样本数量还不足一个batch_size，不进行更新
        if len(self.replay_buffer) < self.batch_size:
            return None

        # 从经验采样池随机采样作为一个batch，一个列表，每个列表元素是一条“经验”
        batch = self.replay_buffer.sample(self.batch_size)

        # 提取batch中的信息
        states = torch.tensor([item.state for item in batch], dtype=torch.float32, device=self.device)
        actions = torch.tensor([item.action for item in batch], dtype=torch.long, device=self.device)
        rewards = torch.tensor([item.reward for item in batch], dtype=torch.float32, device=self.device)
        next_states = torch.tensor(
            [item.next_state for item in batch], dtype=torch.float32, device=self.device
        )
        dones = torch.tensor([item.done for item in batch], dtype=torch.float32, device=self.device)

        # policy_net(states): 每个状态下，所有动作的 Q 值[batch_size, action_dim(可选动作数)]
        # actions.unsqueeze(1): 把实际执行过的动作编号变成 gather 需要的形状[batch_size, 1]
        # gather(1, ...): 取出每条经验中实际执行动作的 Q 值
        # squeeze(1): 把结果从 [batch_size, 1] 变成 [batch_size]
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # 基于时序差分学习和贝尔曼最优方程
        # 将当前 $Q$ 值拆解为当前奖励与未来最大 $Q$ 值的组合，并利用时序差分学习（TD）通过时序差分（TD Error）来逐步修正和逼近这个最优目标
        # 计算td_target作为标签，计算标签的过程不加入计算图
        with torch.no_grad():
            # max_a' (Q(s', a'))，挑出下一个可能动作a'中Q值最大的并提取Q值
            next_q = self.target_net(next_states).max(dim=1).values
            target_q = rewards + self.gamma * next_q * (1.0 - dones)

        # 均方差损失
        loss = self.loss_fn(current_q, target_q)

        # 清空梯度
        self.optimizer.zero_grad()
        # 反向传播，计算所有参数的梯度
        loss.backward()
        # 梯度裁剪，按比例缩小参数，让它的最大大番薯不超过10.0
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        # 更新参数
        self.optimizer.step()

        # 在经验池完全充足的情况下，蛇每走一格就learn_steps+1
        self.learn_steps += 1
        
        # 每隔一段时间更新目标网络
        if self.learn_steps % self.target_update_interval == 0:
            self.update_target_network()

        # 返回浮点数类型的loss
        return float(loss.item())

    # 逐渐减少随机探索，让智能体从“多尝试”过渡到“多利用已学到的策略”
    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    # 更新目标网络
    def update_target_network(self) -> None:
        self.target_net.load_state_dict(self.policy_net.state_dict())

    # 把当前训练状态存到文件里
    def save(self, path: str | Path) -> None:
        path = Path(path)
        # 创建保存模型文件所在的文件夹
        # 如果父目录不存在，连带创建父母，如果当前目录已经存在也不要报错
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                # 正在训练和决策使用的 Q 网络参数。
                "policy_net": self.policy_net.state_dict(),
                # 用于计算 DQN 目标值的目标网络参数。
                "target_net": self.target_net.state_dict(),
                # 当前探索率，恢复训练时可以接着当前探索进度继续。
                "epsilon": self.epsilon,
                # 已完成的神经网络更新次数，用于恢复目标网络同步节奏。
                "learn_steps": self.learn_steps,
                # 状态向量维度，方便加载时检查环境和模型是否匹配。
                "state_size": self.state_size,
                # 动作数量，方便加载时检查环境和模型是否匹配。
                "action_size": self.action_size,
            },
            path,
        )

    # 从文件里恢复模型状态
    def load(self, path: str | Path) -> None:
        # 把断点中的数据加载到当前设备上
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint.get("target_net", checkpoint["policy_net"]))
        self.epsilon = float(checkpoint.get("epsilon", self.epsilon_end))
        self.learn_steps = int(checkpoint.get("learn_steps", 0))
