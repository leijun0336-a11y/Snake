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
        state_size: int,
        action_size: int,
        hidden_size: int = 128,
        learning_rate: float = 1e-3,
        gamma: float = 0.9,
        replay_buffer_size: int = 100_000,
        batch_size: int = 64,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        target_update_interval: int = 1000,
        seed: int = 42,
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

        torch.manual_seed(seed)
        self.policy_net = QNetwork(state_size, hidden_size, action_size).to(self.device)
        self.target_net = QNetwork(state_size, hidden_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        self.replay_buffer = ReplayBuffer(replay_buffer_size, seed=seed)

    def act(self, state: list[float], training: bool = True) -> int:
        if training and self.random.random() < self.epsilon:
            return self.random.randrange(self.action_size)

        with torch.no_grad():
            state_tensor = torch.tensor([state], dtype=torch.float32, device=self.device)
            q_values = self.policy_net(state_tensor)
            return int(q_values.argmax(dim=1).item())

    def remember(
        self,
        state: list[float],
        action: int,
        reward: float,
        next_state: list[float],
        done: bool,
    ) -> None:
        self.replay_buffer.push(state, action, reward, next_state, done)

    def learn(self) -> float | None:
        if len(self.replay_buffer) < self.batch_size:
            return None

        batch = self.replay_buffer.sample(self.batch_size)

        states = torch.tensor([item.state for item in batch], dtype=torch.float32, device=self.device)
        actions = torch.tensor([item.action for item in batch], dtype=torch.long, device=self.device)
        rewards = torch.tensor([item.reward for item in batch], dtype=torch.float32, device=self.device)
        next_states = torch.tensor(
            [item.next_state for item in batch], dtype=torch.float32, device=self.device
        )
        dones = torch.tensor([item.done for item in batch], dtype=torch.float32, device=self.device)

        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.target_net(next_states).max(dim=1).values
            target_q = rewards + self.gamma * next_q * (1.0 - dones)

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.learn_steps += 1
        if self.learn_steps % self.target_update_interval == 0:
            self.update_target_network()

        return float(loss.item())

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def update_target_network(self) -> None:
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "policy_net": self.policy_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "epsilon": self.epsilon,
                "learn_steps": self.learn_steps,
                "state_size": self.state_size,
                "action_size": self.action_size,
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint.get("target_net", checkpoint["policy_net"]))
        self.epsilon = float(checkpoint.get("epsilon", self.epsilon_end))
        self.learn_steps = int(checkpoint.get("learn_steps", 0))
