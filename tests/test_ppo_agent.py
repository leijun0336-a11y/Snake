from pathlib import Path

import numpy as np
import pytest
import torch

from snake_ai.agents.dqn_agent import DQNAgent
from snake_ai.agents.ppo_agent import PPOAgent, PPOMetrics, RolloutTransition
from snake_ai.train import count_agent_parameters


def test_ppo_agent_uses_current_default_learning_rate() -> None:
    agent = PPOAgent(state_size=20, action_size=3, rollout_steps=4, batch_size=2, seed=1)

    assert agent.learning_rate == pytest.approx(1e-4)
    assert agent.optimizer.param_groups[0]["lr"] == pytest.approx(1e-4)


def test_hybrid_ppo_policy_matches_dueling_dqn_policy_capacity() -> None:
    common = dict(
        state_size=(9, 10, 10),
        action_size=3,
        hidden_size=256,
        state_mode="hybrid",
        auxiliary_size=20,
        cnn_channels=32,
        cnn_output_channels=8,
        cnn_dilations=(1, 1, 2),
        seed=1,
    )
    dqn = DQNAgent(**common)
    ppo = PPOAgent(**common, rollout_steps=4, batch_size=2)

    assert sum(parameter.numel() for parameter in dqn.policy_net.parameters()) == sum(
        parameter.numel() for parameter in ppo.policy_net.parameters()
    )
    dqn_counts = count_agent_parameters(dqn)
    ppo_counts = count_agent_parameters(ppo)
    assert dqn_counts["optimized"] == dqn_counts["policy"]
    assert dqn_counts["all_networks"] == 2 * dqn_counts["policy"]
    assert ppo_counts["optimized"] == ppo_counts["policy"]
    assert ppo_counts["all_networks"] == ppo_counts["policy"]


@pytest.mark.parametrize("state_mode", ["vector", "grid", "hybrid"])
def test_ppo_agent_outputs_action_for_every_state_mode(state_mode: str) -> None:
    state_size: int | tuple[int, int, int]
    if state_mode == "vector":
        state_size = 20
        state = np.zeros(20, dtype=np.float32)
    else:
        state_size = (9, 6, 6)
        grid = np.zeros((9, 6, 6), dtype=np.float32)
        grid[3, 3, 3] = 1.0
        state = (grid, np.zeros(20, dtype=np.float32)) if state_mode == "hybrid" else grid

    agent = PPOAgent(
        state_size=state_size,
        action_size=3,
        state_mode=state_mode,
        rollout_steps=4,
        batch_size=2,
        seed=1,
    )

    assert agent.act(state, training=False) in (0, 1, 2)


def test_gae_stops_at_episode_boundary_but_bootstraps_truncation() -> None:
    agent = PPOAgent(
        state_size=2,
        action_size=2,
        gamma=0.5,
        gae_lambda=1.0,
        rollout_steps=2,
        batch_size=1,
        seed=1,
    )
    batch = [
        RolloutTransition([0, 0], 0, 1.0, False, True, 0.0, 2.0, 4.0),
        RolloutTransition([1, 0], 0, 100.0, True, True, 0.0, 0.0, 0.0),
    ]

    advantages, returns = agent._calculate_gae(batch)

    assert advantages == pytest.approx([1.0, 100.0])
    assert returns == pytest.approx([3.0, 100.0])


def test_ppo_update_consumes_rollout_and_changes_parameters() -> None:
    agent = PPOAgent(
        state_size=2,
        action_size=2,
        hidden_size=16,
        rollout_steps=4,
        batch_size=2,
        update_epochs=2,
        target_kl=None,
        seed=1,
    )
    before = [parameter.detach().clone() for parameter in agent.policy_net.parameters()]
    for index in range(4):
        state = np.asarray([index % 2, (index + 1) % 2], dtype=np.float32)
        next_state = np.asarray([(index + 1) % 2, index % 2], dtype=np.float32)
        done = index == 3
        action = agent.act(state, training=True)
        agent.remember(state, action, float(index + 1), next_state, done)

    metrics = agent.learn()

    assert isinstance(metrics, PPOMetrics)
    assert metrics.samples == 4
    assert metrics.epochs == 2
    assert np.isfinite(metrics.loss)
    assert len(agent.rollout) == 0
    assert any(
        not torch.equal(old, new)
        for old, new in zip(before, agent.policy_net.parameters(), strict=True)
    )


def test_ppo_checkpoint_round_trip(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "ppo.pt"
    source = PPOAgent(
        state_size=(9, 6, 6),
        action_size=3,
        state_mode="hybrid",
        hidden_size=32,
        cnn_channels=8,
        cnn_output_channels=4,
        cnn_dilations=(1,),
        rollout_steps=4,
        batch_size=2,
        seed=1,
    )
    source.save(checkpoint_path, metadata={"algorithm": "ppo"})
    loaded = PPOAgent(
        state_size=(9, 6, 6),
        action_size=3,
        state_mode="hybrid",
        rollout_steps=4,
        batch_size=2,
        seed=2,
    )

    loaded.load(checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    assert checkpoint["algorithm"] == "ppo"
    assert checkpoint["run_config"] == {"algorithm": "ppo"}
    assert checkpoint["entropy_coefficient"] == pytest.approx(0.05)
    assert checkpoint["entropy_coefficient_start"] == pytest.approx(0.05)
    assert checkpoint["entropy_coefficient_end"] == pytest.approx(0.001)
    assert checkpoint["entropy_anneal_episodes"] == 15_000
    assert loaded.hidden_size == 32
    assert loaded.cnn_channels == 8
    assert loaded.cnn_output_channels == 4
    assert loaded.cnn_dilations == (1,)


def test_entropy_coefficient_anneals_linearly_to_configured_floor() -> None:
    agent = PPOAgent(
        state_size=20,
        action_size=3,
        rollout_steps=4,
        batch_size=2,
        entropy_anneal_episodes=5,
        seed=1,
    )

    assert agent.entropy_coefficient_start == pytest.approx(0.05)
    assert agent.entropy_coefficient_end == pytest.approx(0.001)
    assert agent.set_entropy_for_episode(1) == pytest.approx(0.05)
    assert agent.set_entropy_for_episode(3) == pytest.approx(0.0255)
    assert agent.set_entropy_for_episode(5) == pytest.approx(0.001)
    assert agent.set_entropy_for_episode(6) == pytest.approx(0.001)


def test_argmax_cycle_fallback_tries_ranked_alternatives_and_resets() -> None:
    class FixedPolicy(torch.nn.Module):
        def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            logits = torch.tensor(
                [[3.0, 2.0, 1.0]],
                device=state.device,
            )
            batch_size = state.shape[0]
            return logits.repeat(batch_size, 1), torch.zeros(batch_size, device=state.device)

    state = np.zeros(20, dtype=np.float32)
    default_agent = PPOAgent(
        state_size=20,
        action_size=3,
        rollout_steps=4,
        batch_size=2,
        seed=1,
    )
    default_agent.policy_net = FixedPolicy()
    assert default_agent.argmax_cycle_fallback is False
    assert [default_agent.act(state, training=False) for _ in range(3)] == [0, 0, 0]

    agent = PPOAgent(
        state_size=20,
        action_size=3,
        rollout_steps=4,
        batch_size=2,
        argmax_cycle_fallback=True,
        seed=1,
    )
    agent.policy_net = FixedPolicy()
    state = np.zeros(20, dtype=np.float32)

    assert [agent.act(state, training=False) for _ in range(4)] == [0, 1, 2, 1]
    agent.reset_evaluation_state()
    assert agent.act(state, training=False) == 0
    assert agent.act(np.ones(20, dtype=np.float32), training=False) == 0
