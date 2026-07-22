from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn, optim
from torch.distributions import Categorical

from snake_ai.models.q_network import QNetwork


# 继承QNetwork类作为ActorCriticNetwork，
# Value 分支给Critic，Advantage 分支给 Actor，外形复用但出入参含义不同
class ActorCriticNetwork(QNetwork):

    def __init__(
        self,
        input_size: int | tuple[int, int, int],
        hidden_size: int,
        action_size: int,
        *,
        state_mode: str = "vector",
        auxiliary_size: int = 20,
        cnn_channels: int = 32,
        cnn_output_channels: int = 8,
        cnn_dilations: tuple[int, ...] = (1, 1, 2),
    ) -> None:
        super().__init__(
            input_size,
            hidden_size,
            action_size,
            dueling=True,
            state_mode=state_mode,
            auxiliary_size=auxiliary_size,
            cnn_channels=cnn_channels,
            cnn_output_channels=cnn_output_channels,
            cnn_dilations=cnn_dilations,
        )

    def forward(
        self, x: torch.Tensor | tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encode(x)
        logits = self.advantage_stream(features)
        value = self.value_stream(features).squeeze(-1)
        return logits, value


@dataclass
class RolloutTransition:
    state: Any
    action: int
    reward: float
    done: bool
    episode_end: bool
    log_prob: float
    value: float
    next_value: float


@dataclass(frozen=True)
class PPOMetrics:
    loss: float
    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float
    clip_fraction: float
    explained_variance: float
    epochs: int
    samples: int


class PPOAgent:

    ALGORITHM = "ppo"
    ARCHITECTURE_VERSION = 1

    def __init__(
        self,
        state_size: int | tuple[int, int, int],
        action_size: int,
        hidden_size: int = 256,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        rollout_steps: int = 2048,
        batch_size: int = 128,
        update_epochs: int = 4,
        gae_lambda: float = 0.95,
        clip_coefficient: float = 0.2,
        value_clip_coefficient: float = 0.2,
        entropy_coefficient: float = 0.01,
        value_loss_coefficient: float = 0.5,
        max_grad_norm: float = 0.5,
        target_kl: float | None = 0.02,
        normalize_advantage: bool = True,
        normalize_returns: bool = True,
        state_mode: str = "vector",
        auxiliary_size: int = 20,
        cnn_channels: int = 32,
        cnn_output_channels: int = 8,
        cnn_dilations: tuple[int, ...] = (1, 1, 2),
        seed: int = 42,
        device: str | None = None,
    ) -> None:
        if rollout_steps < 1:
            raise ValueError("rollout_steps must be at least 1")
        if batch_size < 1 or batch_size > rollout_steps:
            raise ValueError("batch_size must satisfy 1 <= batch_size <= rollout_steps")
        if rollout_steps % batch_size != 0:
            raise ValueError("rollout_steps must be divisible by batch_size")
        if update_epochs < 1:
            raise ValueError("update_epochs must be at least 1")
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be between 0 and 1")
        if not 0.0 <= gae_lambda <= 1.0:
            raise ValueError("gae_lambda must be between 0 and 1")
        if clip_coefficient <= 0.0 or value_clip_coefficient <= 0.0:
            raise ValueError("PPO clip coefficients must be positive")
        if entropy_coefficient < 0.0 or value_loss_coefficient < 0.0:
            raise ValueError("loss coefficients must be non-negative")
        if max_grad_norm <= 0.0:
            raise ValueError("max_grad_norm must be positive")
        if target_kl is not None and target_kl <= 0.0:
            raise ValueError("target_kl must be positive when provided")

        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.rollout_steps = rollout_steps
        self.batch_size = batch_size
        self.update_epochs = update_epochs
        self.gae_lambda = gae_lambda
        self.clip_coefficient = clip_coefficient
        self.value_clip_coefficient = value_clip_coefficient
        self.entropy_coefficient = entropy_coefficient
        self.value_loss_coefficient = value_loss_coefficient
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.normalize_advantage = normalize_advantage
        self.normalize_returns = normalize_returns
        self.state_mode = state_mode
        self.auxiliary_size = auxiliary_size
        self.cnn_channels = cnn_channels
        self.cnn_output_channels = cnn_output_channels
        self.cnn_dilations = tuple(cnn_dilations)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.update_steps = 0

        torch.manual_seed(seed)
        self.policy_net = self._build_network()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.rollout: list[RolloutTransition] = []

    def _build_network(self) -> ActorCriticNetwork:
        return ActorCriticNetwork(
            self.state_size,
            self.hidden_size,
            self.action_size,
            state_mode=self.state_mode,
            auxiliary_size=self.auxiliary_size,
            cnn_channels=self.cnn_channels,
            cnn_output_channels=self.cnn_output_channels,
            cnn_dilations=self.cnn_dilations,
        ).to(self.device)

    def act(self, state: Any, training: bool = True) -> int:
        with torch.no_grad():
            logits, _ = self.policy_net(self._state_batch_to_tensor([state]))
            if training:
                return int(Categorical(logits=logits).sample().item())
            return int(logits.argmax(dim=1).item())

    def remember(
        self,
        state: Any,
        action: int,
        reward: float,
        next_state: Any,
        done: bool,
    ) -> None:
        with torch.no_grad():
            logits, value = self.policy_net(self._state_batch_to_tensor([state]))
            distribution = Categorical(logits=logits)
            action_tensor = torch.tensor([action], dtype=torch.long, device=self.device)
            log_prob = distribution.log_prob(action_tensor)
            if done:
                next_value = torch.zeros(1, device=self.device)
            else:
                _, next_value = self.policy_net(self._state_batch_to_tensor([next_state]))

        self.rollout.append(
            RolloutTransition(
                state=self._copy_state(state),
                action=int(action),
                reward=float(reward),
                done=bool(done),
                episode_end=bool(done),
                log_prob=float(log_prob.item()),
                value=float(value.item()),
                next_value=float(next_value.item()),
            )
        )

    # 在环境重置时切断 GAE 的递归计算，同时保留截断状态下的自举（Bootstrap）。
    def finish_episode(self) -> None:

        if self.rollout:
            self.rollout[-1].episode_end = True

    def learn(self) -> PPOMetrics | None:
        if len(self.rollout) < self.rollout_steps:
            return None

        batch = self.rollout[: self.rollout_steps]
        del self.rollout[: self.rollout_steps]
        advantages, returns = self._calculate_gae(batch)
        states = self._state_batch_to_tensor([item.state for item in batch])
        actions = torch.tensor(
            [item.action for item in batch], dtype=torch.long, device=self.device
        )
        old_log_probs = torch.tensor(
            [item.log_prob for item in batch], dtype=torch.float32, device=self.device
        )
        old_values = torch.tensor(
            [item.value for item in batch], dtype=torch.float32, device=self.device
        )
        advantages_tensor = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns_tensor = torch.tensor(returns, dtype=torch.float32, device=self.device)
        if self.normalize_returns and len(batch) > 1:
            ret_mean = returns_tensor.mean()
            ret_std = returns_tensor.std(unbiased=False) + 1e-8
            returns_tensor = (returns_tensor - ret_mean) / ret_std
        if self.normalize_advantage and len(batch) > 1:
            advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (
                advantages_tensor.std(unbiased=False) + 1e-8
            )

        totals = {
            "loss": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
        }
        minibatches = 0
        completed_epochs = 0
        stop_for_kl = False

        for epoch in range(self.update_epochs):
            indices = torch.randperm(len(batch), device=self.device)
            for start in range(0, len(batch), self.batch_size):
                minibatch_indices = indices[start : start + self.batch_size]
                minibatch_states = self._index_state_batch(states, minibatch_indices)
                logits, new_values = self.policy_net(minibatch_states)
                distribution = Categorical(logits=logits)
                new_log_probs = distribution.log_prob(actions[minibatch_indices])
                entropy = distribution.entropy().mean()

                log_ratio = new_log_probs - old_log_probs[minibatch_indices]
                ratio = log_ratio.exp()
                minibatch_advantages = advantages_tensor[minibatch_indices]
                unclipped_policy_loss = -minibatch_advantages * ratio
                clipped_policy_loss = -minibatch_advantages * ratio.clamp(
                    1.0 - self.clip_coefficient,
                    1.0 + self.clip_coefficient,
                )
                policy_loss = torch.maximum(
                    unclipped_policy_loss, clipped_policy_loss
                ).mean()

                old_minibatch_values = old_values[minibatch_indices]
                minibatch_returns = returns_tensor[minibatch_indices]
                unclipped_value_loss = (new_values - minibatch_returns).pow(2)
                clipped_values = old_minibatch_values + (
                    new_values - old_minibatch_values
                ).clamp(
                    -self.value_clip_coefficient,
                    self.value_clip_coefficient,
                )
                clipped_value_loss = (clipped_values - minibatch_returns).pow(2)
                value_loss = 0.5 * torch.maximum(
                    unclipped_value_loss, clipped_value_loss
                ).mean()
                loss = (
                    policy_loss
                    + self.value_loss_coefficient * value_loss
                    - self.entropy_coefficient * entropy
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()
                    clip_fraction = (
                        (ratio - 1.0).abs() > self.clip_coefficient
                    ).float().mean()
                values = {
                    "loss": loss,
                    "policy_loss": policy_loss,
                    "value_loss": value_loss,
                    "entropy": entropy,
                    "approx_kl": approx_kl,
                    "clip_fraction": clip_fraction,
                }
                for name, value_tensor in values.items():
                    totals[name] += float(value_tensor.item())
                minibatches += 1

                if self.target_kl is not None and float(approx_kl.item()) > self.target_kl:
                    stop_for_kl = True
                    break
            completed_epochs = epoch + 1
            if stop_for_kl:
                break

        self.update_steps += 1
        with torch.no_grad():
            _, final_values = self.policy_net(states)
            return_variance = torch.var(returns_tensor, unbiased=False)
            explained_variance = (
                torch.tensor(0.0, device=self.device)
                if return_variance <= 1e-8
                else 1.0
                - torch.var(returns_tensor - final_values, unbiased=False) / return_variance
            )
        divisor = max(minibatches, 1)
        return PPOMetrics(
            loss=totals["loss"] / divisor,
            policy_loss=totals["policy_loss"] / divisor,
            value_loss=totals["value_loss"] / divisor,
            entropy=totals["entropy"] / divisor,
            approx_kl=totals["approx_kl"] / divisor,
            clip_fraction=totals["clip_fraction"] / divisor,
            explained_variance=float(explained_variance.item()),
            epochs=completed_epochs,
            samples=len(batch),
        )

    def _calculate_gae(
        self, batch: list[RolloutTransition]
    ) -> tuple[np.ndarray, np.ndarray]:
        advantages = np.zeros(len(batch), dtype=np.float32)
        gae = 0.0
        for index in range(len(batch) - 1, -1, -1):
            transition = batch[index]
            bootstrap = 0.0 if transition.done else transition.next_value
            delta = transition.reward + self.gamma * bootstrap - transition.value
            continuation = 0.0 if transition.episode_end else 1.0
            gae = delta + self.gamma * self.gae_lambda * continuation * gae
            advantages[index] = gae
        values = np.asarray([item.value for item in batch], dtype=np.float32)
        return advantages, advantages + values

    def _state_batch_to_tensor(
        self, states: list[Any]
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if self.state_mode == "hybrid":
            grids = self._grid_batch_to_tensor([state[0] for state in states])
            auxiliary = np.asarray([state[1] for state in states], dtype=np.float32)
            return grids, self._numpy_to_tensor(auxiliary)
        if self.state_mode == "grid":
            return self._grid_batch_to_tensor(states)
        return self._numpy_to_tensor(np.asarray(states, dtype=np.float32))

    def _grid_batch_to_tensor(self, grids: list[Any]) -> torch.Tensor:
        array = np.expand_dims(grids[0], axis=0) if len(grids) == 1 else np.stack(grids)
        return self._numpy_to_tensor(np.asarray(array, dtype=np.float32))

    def _numpy_to_tensor(self, array: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(array)).to(self.device)

    @staticmethod
    def _index_state_batch(
        states: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        indices: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if isinstance(states, tuple):
            return states[0][indices], states[1][indices]
        return states[indices]

    def _copy_state(self, state: Any) -> Any:
        if self.state_mode == "hybrid":
            return np.array(state[0], dtype=np.float32, copy=True), np.array(
                state[1], dtype=np.float32, copy=True
            )
        return np.array(state, dtype=np.float32, copy=True)

    def save(self, path: str | Path, metadata: dict[str, Any] | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "algorithm": self.ALGORITHM,
            "policy_net": self.policy_net.state_dict(),
            "state_size": self.state_size,
            "state_mode": self.state_mode,
            "auxiliary_size": self.auxiliary_size,
            "cnn_channels": self.cnn_channels,
            "cnn_output_channels": self.cnn_output_channels,
            "cnn_dilations": self.cnn_dilations,
            "hidden_size": self.hidden_size,
            "action_size": self.action_size,
            "architecture_version": self.ARCHITECTURE_VERSION,
            "update_steps": self.update_steps,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "rollout_steps": self.rollout_steps,
            "batch_size": self.batch_size,
            "update_epochs": self.update_epochs,
            "gae_lambda": self.gae_lambda,
            "clip_coefficient": self.clip_coefficient,
            "value_clip_coefficient": self.value_clip_coefficient,
            "entropy_coefficient": self.entropy_coefficient,
            "value_loss_coefficient": self.value_loss_coefficient,
            "max_grad_norm": self.max_grad_norm,
            "target_kl": self.target_kl,
            "normalize_advantage": self.normalize_advantage,
            "normalize_returns": self.normalize_returns,
        }
        if metadata is not None:
            checkpoint["run_config"] = metadata
        torch.save(checkpoint, path)

    def load(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        if checkpoint.get("algorithm") != self.ALGORITHM:
            raise ValueError("Checkpoint is not a PPO checkpoint")
        if checkpoint.get("architecture_version") != self.ARCHITECTURE_VERSION:
            raise ValueError(
                "Unsupported PPO checkpoint architecture_version="
                f"{checkpoint.get('architecture_version')!r}; current code requires "
                f"architecture_version={self.ARCHITECTURE_VERSION}."
            )
        required = {
            "policy_net",
            "state_size",
            "state_mode",
            "auxiliary_size",
            "cnn_channels",
            "cnn_output_channels",
            "cnn_dilations",
            "hidden_size",
            "action_size",
        }
        missing = required.difference(checkpoint)
        if missing:
            raise ValueError(f"PPO checkpoint is missing required fields: {', '.join(sorted(missing))}")
        if checkpoint["state_size"] != self.state_size:
            raise ValueError(
                f"Checkpoint state_size={checkpoint['state_size']} does not match "
                f"current agent state_size={self.state_size}."
            )
        if str(checkpoint["state_mode"]) != self.state_mode:
            raise ValueError("Checkpoint state_mode does not match current agent state_mode")
        if int(checkpoint["action_size"]) != self.action_size:
            raise ValueError("Checkpoint action_size does not match current agent action_size")

        architecture = (
            int(checkpoint["hidden_size"]),
            int(checkpoint["auxiliary_size"]),
            int(checkpoint["cnn_channels"]),
            int(checkpoint["cnn_output_channels"]),
            tuple(int(value) for value in checkpoint["cnn_dilations"]),
        )
        current_architecture = (
            self.hidden_size,
            self.auxiliary_size,
            self.cnn_channels,
            self.cnn_output_channels,
            self.cnn_dilations,
        )
        if architecture != current_architecture:
            (
                self.hidden_size,
                self.auxiliary_size,
                self.cnn_channels,
                self.cnn_output_channels,
                self.cnn_dilations,
            ) = architecture
            self.policy_net = self._build_network()
            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.update_steps = int(checkpoint.get("update_steps", 0))
