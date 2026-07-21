# 处理自引用问题
from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn, optim

from snake_ai.agents.replay_buffer import NStepAccumulator, ReplayBuffer, Transition
from snake_ai.models.q_network import QNetwork


class DQNAgent:
    ARCHITECTURE_VERSION = 3

    def __init__(
        self,
        # 状态维度，如果是vector模式输入向量，CNN模式和hibrid模式都输入网格。
        state_size: int | tuple[int, int, int],
        # 动作数量，当前为 3：直行、右转、左转。
        action_size: int,
        # Q 网络隐藏层神经元数量。
        hidden_size: int = 128,
        # Adam 优化器学习率，控制神经网络参数更新步幅。
        learning_rate: float = 1e-4,
        # 折扣因子，越接近 1 越重视未来奖励。
        gamma: float = 0.99,
        # TD target 使用的真实奖励步数；1 保持传统 one-step DQN。
        n_step: int = 1,
        # 经验回放池容量，最多保存多少条 Transition。
        replay_buffer_size: int = 100_000,
        # 每次训练从经验池随机采样多少条经验。
        batch_size: int = 64,
        # 初始探索率，训练早期按这个概率随机选动作。
        epsilon_start: float = 1.0,
        # 最低探索率，防止后期完全不探索。
        epsilon_end: float = 0.01,
        # 是否使用指数衰减；False 表示使用线性衰减。
        epsilon_exp_decay: bool = False,
        # 每局结束后 epsilon 的指数衰减系数；仅在指数衰减模式下使用。
        epsilon_exp_factor: float = 0.995,
        # 线性衰减到 epsilon_end 需要的 episode 数。
        epsilon_linear_episodes: int = 7500,
        # 每隔多少次学习步骤，把 policy_net 的参数复制一份给 target_net.
        target_update_interval: int = 1000,
        # 是否使用 Dueling DQN 架构，分离状态值和优势值。
        dueling: bool = True,
        # 状态输入模式：vector 使用人工低维状态，grid 使用多通道网格状态。
        state_mode: str = "vector",
        # Hybrid 模式下与 CNN 特征拼接的人工状态维度。
        auxiliary_size: int = 20,
        # Grid/Hybrid CNN 主干通道数。
        cnn_channels: int = 32,
        # 1x1 卷积压缩后的通道数。
        cnn_output_channels: int = 8,
        # 空洞残差块的 dilation 序列。
        cnn_dilations: tuple[int, ...] = (1, 1, 2),
        # 随机种子，用于让探索、采样和网络初始化尽量可复现。
        seed: int = 42,
        # 计算设备；不传时优先使用 cuda，否则使用 cpu。
        device: str | None = None,
    ) -> None:
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.state_mode = state_mode
        self.auxiliary_size = auxiliary_size
        self.cnn_channels = cnn_channels
        self.cnn_output_channels = cnn_output_channels
        self.cnn_dilations = tuple(cnn_dilations)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.gamma = gamma
        if n_step < 1:
            raise ValueError("n_step must be at least 1")
        self.n_step = n_step
        self.epsilon_start = epsilon_start
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_exp_decay = epsilon_exp_decay
        self.epsilon_exp_factor = epsilon_exp_factor
        self.epsilon_linear_episodes = epsilon_linear_episodes
        self.target_update_interval = target_update_interval
        self.dueling = dueling
        self.learn_steps = 0
        self.random = random.Random(seed)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        # 生成随机数种子
        torch.manual_seed(seed)
        self.policy_net = self._build_network()
        self.target_net = self._build_network()
        # load_state_dict() 是一个用来加载模型参数的方法，这里用于参数复制
        self.target_net.load_state_dict(self.policy_net.state_dict())
        # 切换到评估模式，关闭dropout和batch_norm，因为目标网络本身不训练只接受参数
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        # DQN 常用 Huber loss；相比 MSE，它会降低大 TD error 对梯度的冲击。
        self.loss_fn = nn.SmoothL1Loss()
        self.replay_buffer = ReplayBuffer(replay_buffer_size, seed=seed)
        self.n_step_accumulator = NStepAccumulator(n_step, gamma)

    # 动作采样，DQN采用epsilon-greedy
    def act(self, state: Any, training: bool = True) -> int:
        # state: 输入的状态；training:是否训练，评估模式不探索。

        # 如果正在训练且落入epsilon概率内
        if training and self.random.random() < self.epsilon:
            return self.random.randrange(self.action_size)

        # act()只用于选动作，不参与梯度计算; learn()才需要梯度，因为要训练网络
        with torch.no_grad():
            state_tensor = self._state_batch_to_tensor([state])
            q_values = self.policy_net(state_tensor)
            # 返回Q网络输出向量中数值最大值对应的索引
            return int(q_values.argmax(dim=1).item())

    # 往经验回放池放东西
    def remember(
        self,
        # 执行动作前的当前状态。
        state: Any,
        # 在当前状态下执行的动作。
        action: int,
        # 执行动作后环境返回的奖励。
        reward: float,
        # 执行动作后环境返回的下一个状态。
        next_state: Any,
        # 执行动作后这一局是否结束。
        done: bool,
    ) -> None:
        ready = self.n_step_accumulator.append(state, action, reward, next_state, done)
        self._push_aggregated_transitions(ready)

    def finish_episode(self) -> None:
        """在自然终止或步数截断后冲刷不足 n 步的 episode 尾部。"""

        self._push_aggregated_transitions(self.n_step_accumulator.flush())

    def _push_aggregated_transitions(self, transitions: tuple[Transition, ...]) -> None:
        for transition in transitions:
            self.replay_buffer.push(
                transition.state,
                transition.action,
                transition.reward,
                transition.next_state,
                transition.done,
                transition.n_steps,
            )

    # 更新动作函数
    def learn(self) -> float | None:  # 返回值是float或者None
        # 如果经验回放池的样本数量还不足一个batch_size，不进行更新
        if len(self.replay_buffer) < self.batch_size:
            return None

        # 从经验采样池随机采样作为一个batch，一个列表，每个列表元素是一条“经验”
        batch = self.replay_buffer.sample(self.batch_size)

        # 提取batch中的信息
        states = self._state_batch_to_tensor([item.state for item in batch])
        actions = torch.tensor(
            [item.action for item in batch], dtype=torch.long, device=self.device
        )
        rewards = torch.tensor(
            [item.reward for item in batch], dtype=torch.float32, device=self.device
        )
        next_states = self._state_batch_to_tensor([item.next_state for item in batch])
        dones = torch.tensor([item.done for item in batch], dtype=torch.float32, device=self.device)
        sampled_n_steps = torch.tensor(
            [item.n_steps for item in batch], dtype=torch.float32, device=self.device
        )

        # policy_net(states): 每个状态下，所有动作的 Q 值[batch_size, action_dim(可选动作数)]
        # actions.unsqueeze(1): 把实际执行过的动作编号变成 gather 需要的形状[batch_size, 1]
        # gather(1, ...): 取出每条经验中实际执行动作的 Q 值
        # squeeze(1): 把结果从 [batch_size, 1] 变成 [batch_size]
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # 基于时序差分学习和贝尔曼最优方程
        # 将当前 $Q$ 值拆解为当前奖励与未来最大 $Q$ 值的组合，并利用时序差分学习（TD）通过时序差分（TD Error）来逐步修正和逼近这个最优目标
        # 计算td_target作为标签，计算标签的过程不加入计算图
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(dim=1)
            # Double DQN: 把 next_states 传给 target_net 来计算 Q(s', a')，而不是直接用 policy_net 的最大 Q 值
            # Y_t = R_{t+1} + gamma * Q_target(s_{t+1}, argmax_a Q_net(s_{t+1}, a))
            next_q = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            # [batch_size, 1]，如果 done 为 True，则不考虑未来奖励
            target_q = self._calculate_td_target(rewards, next_q, dones, sampled_n_steps)

        # Huber loss 用于降低离群 TD error 对训练稳定性的影响。
        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        # 反向传播，计算所有参数的梯度
        loss.backward()
        # 梯度裁剪，按比例缩小参数，让它的最大2范数不超过10.0
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

    def _calculate_td_target(
        self,
        rewards: torch.Tensor,
        next_q: torch.Tensor,
        dones: torch.Tensor,
        sampled_n_steps: torch.Tensor,
    ) -> torch.Tensor:
        """计算 one-step 或 n-step Double DQN 监督目标。"""

        if self.n_step == 1:
            # 保留历史 one-step 的同一运算表达式，默认参数不改变原训练数值路径。
            return rewards + self.gamma * next_q * (1.0 - dones)
        gamma_tensor = torch.full_like(rewards, self.gamma)
        bootstrap_discounts = torch.pow(gamma_tensor, sampled_n_steps)
        return rewards + bootstrap_discounts * next_q * (1.0 - dones)

    # 逐渐减少随机探索，让智能体从“多尝试”过渡到“多利用已学到的策略”
    def decay_epsilon(self, episode: int | None = None) -> None:
        if self.epsilon_exp_decay:
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_exp_factor)
            return
        if episode is None:
            raise ValueError("episode is required when using linear epsilon decay")
        progress = min(max(episode, 0) / self.epsilon_linear_episodes, 1.0)
        epsilon_range = self.epsilon_start - self.epsilon_end
        self.epsilon = max(self.epsilon_end, self.epsilon_start - epsilon_range * progress)

    # 更新目标网络
    def update_target_network(self) -> None:
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def _build_network(self) -> QNetwork:
        # Policy/Target 网络都通过此方法构建，确保它们使用完全相同的状态模式和 CNN 配置。
        return QNetwork(
            self.state_size,
            self.hidden_size,
            self.action_size,
            dueling=self.dueling,
            state_mode=self.state_mode,
            auxiliary_size=self.auxiliary_size,
            cnn_channels=self.cnn_channels,
            cnn_output_channels=self.cnn_output_channels,
            cnn_dilations=self.cnn_dilations,
        ).to(self.device)

    def _state_batch_to_tensor(
        self, states: list[Any]
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if self.state_mode == "hybrid":
            # Hybrid replay 中每条 state 都是 (grid, 20维人工状态)，分别组成两个 batch。
            grids = self._grid_batch_to_tensor([state[0] for state in states])
            auxiliary_array = np.asarray([state[1] for state in states], dtype=np.float32)
            auxiliary_states = self._numpy_to_tensor(auxiliary_array)
            return grids, auxiliary_states
        if self.state_mode == "grid":
            return self._grid_batch_to_tensor(states)

        # Vector 状态很小，先一次性组成连续 NumPy 数组，再交给 PyTorch。
        vector_array = np.asarray(states, dtype=np.float32)
        return self._numpy_to_tensor(vector_array)

    def _grid_batch_to_tensor(self, grids: list[Any]) -> torch.Tensor:
        if len(grids) == 1:
            # act() 的单状态只增加 batch 维度，不复制底层 NumPy 网格。
            grid_array = np.expand_dims(grids[0], axis=0)
        else:
            # learn() 只进行一次连续内存复制，替代 torch.tensor 对嵌套列表的递归遍历。
            grid_array = np.stack(grids, axis=0)
        grid_array = np.asarray(grid_array, dtype=np.float32)
        return self._numpy_to_tensor(grid_array)

    def _numpy_to_tensor(self, array: np.ndarray) -> torch.Tensor:
        # CPU 上 from_numpy 与数组共享内存；CUDA 模式只在最后一步统一复制到显存。
        contiguous_array = np.ascontiguousarray(array)
        return torch.from_numpy(contiguous_array).to(self.device)

    # 把当前训练状态存到文件里
    def save(
        self,
        path: str | Path,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        path = Path(path)
        # 创建保存模型文件所在的文件夹
        # 如果父目录不存在，连带创建父母，如果当前目录已经存在也不要报错
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            # 正在训练和决策使用的 Q 网络参数。
            "policy_net": self.policy_net.state_dict(),
            # 用于计算 DQN 目标值的目标网络参数。
            "target_net": self.target_net.state_dict(),
            # 当前探索率，恢复训练时可以接着当前探索进度继续。
            "epsilon_start": self.epsilon_start,
            "epsilon": self.epsilon,
            "epsilon_end": self.epsilon_end,
            "epsilon_exp_decay": self.epsilon_exp_decay,
            "epsilon_exp_factor": self.epsilon_exp_factor,
            "epsilon_linear_episodes": self.epsilon_linear_episodes,
            # 已完成的神经网络更新次数，用于恢复目标网络同步节奏。
            "learn_steps": self.learn_steps,
            # n-step 只改变训练目标，不改变网络结构；旧 checkpoint 缺失时按 1 处理。
            "n_step": self.n_step,
            "gamma": self.gamma,
            # 状态向量维度，方便加载时检查环境和模型是否匹配。
            "state_size": self.state_size,
            "state_mode": self.state_mode,
            # 保存完整 CNN 架构，评估自定义结构时才能准确重建网络。
            "auxiliary_size": self.auxiliary_size,
            "cnn_channels": self.cnn_channels,
            "cnn_output_channels": self.cnn_output_channels,
            "cnn_dilations": self.cnn_dilations,
            # 动作数量，方便加载时检查环境和模型是否匹配。
            "hidden_size": self.hidden_size,
            "action_size": self.action_size,
            "dueling": self.dueling,
            "architecture_version": self.ARCHITECTURE_VERSION,
        }
        if metadata is not None:
            checkpoint["run_config"] = metadata
        torch.save(checkpoint, path)

    # 从文件里恢复模型状态
    def load(self, path: str | Path) -> None:
        # 把断点中的数据加载到当前设备上
        checkpoint = torch.load(path, map_location=self.device)
        checkpoint_version = checkpoint.get("architecture_version")
        if checkpoint_version != self.ARCHITECTURE_VERSION:
            raise ValueError(
                f"Unsupported checkpoint architecture_version={checkpoint_version!r}; "
                f"current code requires architecture_version={self.ARCHITECTURE_VERSION}. "
                "Use the matching historical Git revision for older checkpoints."
            )

        required_fields = {
            "policy_net",
            "target_net",
            "epsilon_start",
            "epsilon",
            "epsilon_end",
            "epsilon_exp_decay",
            "epsilon_exp_factor",
            "epsilon_linear_episodes",
            "learn_steps",
            "state_size",
            "state_mode",
            "auxiliary_size",
            "cnn_channels",
            "cnn_output_channels",
            "cnn_dilations",
            "hidden_size",
            "action_size",
            "dueling",
        }
        missing_fields = required_fields.difference(checkpoint)
        if missing_fields:
            missing_text = ", ".join(sorted(missing_fields))
            raise ValueError(
                f"Architecture v{self.ARCHITECTURE_VERSION} checkpoint is missing "
                f"required fields: {missing_text}."
            )

        policy_state = checkpoint["policy_net"]
        checkpoint_dueling = bool(checkpoint["dueling"])
        checkpoint_hidden_size = int(checkpoint["hidden_size"])
        checkpoint_state_size = checkpoint["state_size"]
        checkpoint_state_mode = str(checkpoint["state_mode"])
        if checkpoint_state_size != self.state_size:
            raise ValueError(
                f"Checkpoint state_size={checkpoint_state_size} does not match "
                f"current agent state_size={self.state_size}. Retrain the model after "
                "changing the environment state features."
            )
        if checkpoint_state_mode != self.state_mode:
            raise ValueError(
                f"Checkpoint state_mode={checkpoint_state_mode!r} does not match "
                f"current agent state_mode={self.state_mode!r}."
            )

        checkpoint_action_size = int(checkpoint["action_size"])
        if checkpoint_action_size != self.action_size:
            raise ValueError(
                f"Checkpoint action_size={checkpoint_action_size} does not match "
                f"current agent action_size={self.action_size}."
            )
        checkpoint_auxiliary_size = int(checkpoint["auxiliary_size"])
        checkpoint_cnn_channels = int(checkpoint["cnn_channels"])
        checkpoint_cnn_output_channels = int(checkpoint["cnn_output_channels"])
        checkpoint_cnn_dilations = tuple(int(value) for value in checkpoint["cnn_dilations"])
        if checkpoint_state_mode == "hybrid" and checkpoint_auxiliary_size != self.auxiliary_size:
            raise ValueError(
                f"Checkpoint auxiliary_size={checkpoint_auxiliary_size} does not match "
                f"current agent auxiliary_size={self.auxiliary_size}."
            )
        architecture_changed = (
            checkpoint_dueling != self.dueling
            or checkpoint_hidden_size != self.hidden_size
            or checkpoint_cnn_channels != self.cnn_channels
            or checkpoint_cnn_output_channels != self.cnn_output_channels
            or checkpoint_cnn_dilations != self.cnn_dilations
        )
        # 按 checkpoint 中记录的完整架构参数重建网络，再加载对应权重。
        if architecture_changed:
            self.dueling = checkpoint_dueling
            self.hidden_size = checkpoint_hidden_size
            self.cnn_channels = checkpoint_cnn_channels
            self.cnn_output_channels = checkpoint_cnn_output_channels
            self.cnn_dilations = checkpoint_cnn_dilations
            self.policy_net = self._build_network()
            self.target_net = self._build_network()
            self.target_net.eval()
            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)

        # 正常加载权重到网络。
        self.policy_net.load_state_dict(policy_state)
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.epsilon_start = float(checkpoint["epsilon_start"])
        self.epsilon = float(checkpoint["epsilon"])
        self.epsilon_end = float(checkpoint["epsilon_end"])
        self.epsilon_exp_decay = bool(checkpoint["epsilon_exp_decay"])
        self.epsilon_exp_factor = float(checkpoint["epsilon_exp_factor"])
        self.epsilon_linear_episodes = int(checkpoint["epsilon_linear_episodes"])
        self.learn_steps = int(checkpoint["learn_steps"])
        self.n_step = int(checkpoint.get("n_step", 1))
        self.gamma = float(checkpoint.get("gamma", self.gamma))
        self.n_step_accumulator = NStepAccumulator(self.n_step, self.gamma)
