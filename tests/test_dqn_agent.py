import numpy as np
import pytest
import torch
from torch import nn

from snake_ai.agents.dqn_agent import DQNAgent
from snake_ai.agents.replay_buffer import PrioritizedReplayBuffer, ReplayBuffer
from snake_ai.models.q_network import QNetwork


def test_dueling_agent_outputs_action() -> None:
    agent = DQNAgent(state_size=20, action_size=3, seed=1)

    action = agent.act([0.0] * 20, training=False)

    assert action in (0, 1, 2)


def test_agent_default_learning_rate_is_one_e_minus_four() -> None:
    agent = DQNAgent(state_size=20, action_size=3, seed=1)

    assert agent.learning_rate == pytest.approx(1e-4)
    assert agent.optimizer.param_groups[0]["lr"] == pytest.approx(1e-4)
    assert agent.learning_starts == 2_000
    assert agent.per is False
    assert type(agent.replay_buffer) is ReplayBuffer


def test_per_agent_uses_prioritized_replay_and_updates() -> None:
    agent = DQNAgent(
        state_size=2,
        action_size=2,
        batch_size=2,
        learning_starts=0,
        per=True,
        per_beta_anneal_steps=10,
        seed=1,
    )
    agent.remember([0.0, 0.0], 0, 0.0, [1.0, 0.0], False)
    agent.remember([1.0, 0.0], 1, 1.0, [0.0, 1.0], True)

    assert type(agent.replay_buffer) is PrioritizedReplayBuffer
    assert agent._per_beta() == pytest.approx(0.4)
    assert isinstance(agent.learn(), float)
    assert agent.learn_steps == 1

    agent.learn_steps = 10
    assert agent._per_beta() == pytest.approx(1.0)


def test_grid_agent_outputs_action() -> None:
    agent = DQNAgent(
        state_size=(9, 6, 6),
        action_size=3,
        state_mode="grid",
        seed=1,
    )
    grid = np.zeros((9, 6, 6), dtype=np.float32)
    grid[3, 3, 3] = 1.0

    action = agent.act(grid, training=False)

    assert action in (0, 1, 2)


def test_learning_waits_until_replay_warmup_finishes() -> None:
    agent = DQNAgent(
        state_size=2,
        action_size=2,
        batch_size=2,
        learning_starts=3,
        seed=1,
    )
    agent.advance_environment_step()
    agent.remember([0.0, 0.0], 0, 0.0, [1.0, 0.0], False)
    agent.advance_environment_step()
    agent.remember([1.0, 0.0], 1, 1.0, [0.0, 1.0], False)

    assert len(agent.replay_buffer) == 2
    assert agent.learn() is None
    assert agent.learn_steps == 0

    agent.advance_environment_step()
    agent.remember([0.0, 1.0], 0, 0.5, [1.0, 1.0], True)

    assert isinstance(agent.learn(), float)
    assert agent.learn_steps == 1


def test_epsilon_decay_strategies_are_explicit() -> None:
    step_agent = DQNAgent(
        state_size=20,
        action_size=3,
        epsilon_start=1.0,
        epsilon_end=0.0,
        epsilon_linear_steps=10,
        seed=1,
    )
    for _ in range(5):
        step_agent.advance_environment_step()
    assert step_agent.environment_steps == 5
    assert step_agent.epsilon == pytest.approx(0.5)
    for _ in range(6):
        step_agent.advance_environment_step()
    assert step_agent.environment_steps == 11
    assert step_agent.epsilon == pytest.approx(0.0)

    episode_agent = DQNAgent(
        state_size=20,
        action_size=3,
        epsilon_start=1.0,
        epsilon_end=0.0,
        epsilon_decay_unit="episode",
        epsilon_linear_episodes=10,
        seed=1,
    )
    episode_agent.decay_epsilon(5)
    assert episode_agent.epsilon == pytest.approx(0.5)

    exp_agent = DQNAgent(
        state_size=20,
        action_size=3,
        epsilon_start=1.0,
        epsilon_end=0.0,
        epsilon_exp_decay=True,
        epsilon_exp_factor=0.8,
        epsilon_decay_unit="step",
        seed=1,
    )
    exp_agent.decay_epsilon()
    assert exp_agent.epsilon == pytest.approx(0.8)
    assert exp_agent.epsilon_decay_unit == "episode"


def test_grid_agent_crops_local_features_at_board_edge() -> None:
    agent = DQNAgent(
        state_size=(9, 6, 6),
        action_size=3,
        state_mode="grid",
        seed=1,
    )
    grid = np.zeros((9, 6, 6), dtype=np.float32)
    grid[4, 0, 0] = 1.0

    action = agent.act(grid, training=False)

    assert action in (0, 1, 2)


def test_hybrid_agent_outputs_action() -> None:
    agent = DQNAgent(
        state_size=(9, 6, 6),
        action_size=3,
        state_mode="hybrid",
        auxiliary_size=20,
        seed=1,
    )
    grid = np.zeros((9, 6, 6), dtype=np.float32)
    grid[3, 3, 3] = 1.0
    vector_state = [0.0] * 20

    action = agent.act((grid, vector_state), training=False)

    assert action in (0, 1, 2)


@pytest.mark.parametrize(
    ("height", "width"),
    [(6, 6), (10, 10), (20, 20), (7, 11)],
)
@pytest.mark.parametrize("state_mode", ["grid", "hybrid"])
def test_cnn_feature_size_follows_grid_dimensions(
    height: int,
    width: int,
    state_mode: str,
) -> None:
    network = QNetwork(
        input_size=(9, height, width),
        hidden_size=32,
        output_size=3,
        state_mode=state_mode,
        auxiliary_size=20,
        cnn_channels=8,
        cnn_output_channels=8,
        cnn_dilations=(1,),
    )
    expected_features = 8 * height * width + 8 * network.local_crop_size**2
    if state_mode == "hybrid":
        expected_features += 20

    assert not hasattr(network, "global_pool")
    assert isinstance(network.feature[0], nn.Linear)
    assert network.feature[0].in_features == expected_features
    assert not any(isinstance(module, nn.AdaptiveAvgPool2d) for module in network.modules())

    grid = torch.zeros((2, 9, height, width), dtype=torch.float32)
    grid[:, 3, height // 2, width // 2] = 1.0
    inputs = (grid, torch.zeros((2, 20), dtype=torch.float32)) if state_mode == "hybrid" else grid
    assert network(inputs).shape == (2, 3)


@pytest.mark.parametrize("state_mode", ["grid", "hybrid"])
def test_cnn_agent_learns_from_numpy_replay_batch(state_mode: str) -> None:
    agent = DQNAgent(
        state_size=(9, 6, 6),
        action_size=3,
        state_mode=state_mode,
        auxiliary_size=20,
        batch_size=2,
        learning_starts=0,
        seed=1,
    )
    grid = np.zeros((9, 6, 6), dtype=np.float32)
    grid[3, 3, 3] = 1.0
    vector_state = [0.0] * 20
    state = (grid, vector_state) if state_mode == "hybrid" else grid

    # 两条经验恰好填满一个 batch，覆盖 np.stack -> from_numpy -> learn() 完整路径。
    agent.remember(state, 0, 0.0, state, False)
    agent.remember(state, 1, 1.0, state, True)

    assert isinstance(agent.learn(), float)


def test_agent_n_step_remember_and_finish_episode_write_aggregated_replay() -> None:
    agent = DQNAgent(
        state_size=2,
        action_size=2,
        gamma=0.5,
        n_step=3,
        batch_size=8,
        seed=1,
    )
    agent.remember([0, 0], 0, 1.0, [1, 0], False)
    agent.remember([1, 0], 1, 2.0, [2, 0], False)

    assert len(agent.replay_buffer) == 0
    agent.finish_episode()
    batch = sorted(agent.replay_buffer.sample(2), key=lambda item: item.state[0])

    assert [(item.reward, item.n_steps, item.done) for item in batch] == [
        (pytest.approx(2.0), 2, False),
        (pytest.approx(2.0), 1, False),
    ]


def test_n_step_td_target_uses_gamma_to_actual_horizon_and_stops_at_terminal() -> None:
    agent = DQNAgent(state_size=2, action_size=2, gamma=0.5, n_step=3, seed=1)
    rewards = torch.tensor([2.75, 2.0])
    next_q = torch.tensor([4.0, 100.0])
    dones = torch.tensor([0.0, 1.0])
    sampled_n_steps = torch.tensor([3.0, 2.0])

    target = agent._calculate_td_target(rewards, next_q, dones, sampled_n_steps)

    assert torch.allclose(target, torch.tensor([3.25, 2.0]))


def test_default_one_step_td_target_keeps_traditional_formula() -> None:
    agent = DQNAgent(state_size=2, action_size=2, gamma=0.5, seed=1)
    rewards = torch.tensor([1.0, 2.0])
    next_q = torch.tensor([4.0, 100.0])
    dones = torch.tensor([0.0, 1.0])

    target = agent._calculate_td_target(
        rewards,
        next_q,
        dones,
        torch.ones(2),
    )

    assert agent.n_step == 1
    assert torch.allclose(target, torch.tensor([3.0, 2.0]))


def test_load_restores_non_dueling_architecture(tmp_path) -> None:
    checkpoint_path = tmp_path / "non_dueling.pt"
    source_agent = DQNAgent(state_size=20, action_size=3, dueling=False, seed=1)
    source_agent.save(checkpoint_path)

    agent = DQNAgent(state_size=20, action_size=3, dueling=True, seed=1)
    agent.load(checkpoint_path)

    assert agent.dueling is False
    assert agent.act([0.0] * 20, training=False) in (0, 1, 2)


def test_load_rejects_incompatible_state_size(tmp_path) -> None:
    checkpoint_path = tmp_path / "state_11.pt"
    old_agent = DQNAgent(state_size=11, action_size=3, seed=1)
    old_agent.save(checkpoint_path)

    agent = DQNAgent(state_size=20, action_size=3, seed=1)

    with pytest.raises(ValueError, match="state_size"):
        agent.load(checkpoint_path)


def test_load_rejects_incompatible_state_mode(tmp_path) -> None:
    checkpoint_path = tmp_path / "vector.pt"
    vector_agent = DQNAgent(state_size=20, action_size=3, seed=1)
    vector_agent.save(checkpoint_path)

    grid_agent = DQNAgent(
        state_size=(9, 6, 6),
        action_size=3,
        state_mode="grid",
        seed=1,
    )

    with pytest.raises(ValueError, match="state_size|state_mode"):
        grid_agent.load(checkpoint_path)


def test_load_restores_custom_cnn_architecture(tmp_path) -> None:
    checkpoint_path = tmp_path / "custom_cnn.pt"
    trained_agent = DQNAgent(
        state_size=(9, 8, 8),
        action_size=3,
        state_mode="hybrid",
        auxiliary_size=20,
        cnn_channels=24,
        cnn_output_channels=12,
        cnn_dilations=(1, 3),
        local_crop_size=5,
        use_local_crop=False,
        epsilon_exp_decay=True,
        epsilon_exp_factor=0.8,
        learning_starts=321,
        epsilon_linear_steps=456,
        epsilon_linear_episodes=123,
        seed=1,
    )
    trained_agent.advance_environment_step()
    trained_agent.save(checkpoint_path)

    loaded_agent = DQNAgent(
        state_size=(9, 8, 8),
        action_size=3,
        state_mode="hybrid",
        auxiliary_size=20,
        seed=1,
    )
    loaded_agent.load(checkpoint_path)

    assert loaded_agent.cnn_channels == 24
    assert loaded_agent.cnn_output_channels == 12
    assert loaded_agent.cnn_dilations == (1, 3)
    assert loaded_agent.local_crop_size == 5
    assert loaded_agent.use_local_crop is False
    assert loaded_agent.epsilon_exp_decay is True
    assert loaded_agent.epsilon_exp_factor == 0.8
    assert loaded_agent.learning_starts == 321
    assert loaded_agent.epsilon_decay_unit == "episode"
    assert loaded_agent.epsilon_linear_steps == 456
    assert loaded_agent.epsilon_linear_episodes == 123
    assert loaded_agent.environment_steps == 1


def test_save_embeds_resolved_run_config(tmp_path) -> None:
    checkpoint_path = tmp_path / "configured.pt"
    agent = DQNAgent(state_size=20, action_size=3, seed=1)
    run_config = {
        "reward": {"profile": "experiment8", "starvation_comparison": "gt"},
        "training": {"max_steps_per_episode": None},
    }

    agent.save(checkpoint_path, metadata=run_config)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    assert checkpoint["run_config"] == run_config
    assert checkpoint["architecture_version"] == 3
    assert "network_type" not in checkpoint
    assert "cnn_pool_size" not in checkpoint


def test_checkpoint_restores_n_step_training_semantics(tmp_path) -> None:
    checkpoint_path = tmp_path / "n_step.pt"
    source_agent = DQNAgent(
        state_size=20,
        action_size=3,
        gamma=0.95,
        n_step=4,
        seed=1,
    )
    source_agent.save(checkpoint_path)

    loaded_agent = DQNAgent(state_size=20, action_size=3, seed=2)
    loaded_agent.load(checkpoint_path)

    assert loaded_agent.n_step == 4
    assert loaded_agent.gamma == pytest.approx(0.95)
    assert loaded_agent.n_step_accumulator.n_step == 4
    assert loaded_agent.n_step_accumulator.gamma == pytest.approx(0.95)


def test_legacy_checkpoint_defaults_to_episode_epsilon_decay(tmp_path) -> None:
    checkpoint_path = tmp_path / "legacy_epsilon.pt"
    source_agent = DQNAgent(
        state_size=20,
        action_size=3,
        epsilon_decay_unit="episode",
        seed=1,
    )
    source_agent.save(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint.pop("epsilon_decay_unit")
    checkpoint.pop("epsilon_linear_steps")
    checkpoint.pop("environment_steps")
    checkpoint.pop("learning_starts")
    torch.save(checkpoint, checkpoint_path)

    loaded_agent = DQNAgent(state_size=20, action_size=3, seed=2)
    loaded_agent.load(checkpoint_path)

    assert loaded_agent.epsilon_decay_unit == "episode"
    assert loaded_agent.epsilon_linear_steps == 300_000
    assert loaded_agent.environment_steps == 0
    assert loaded_agent.learning_starts == 0


def test_legacy_dqn_checkpoint_without_crop_metadata_uses_historical_5x5(tmp_path) -> None:
    checkpoint_path = tmp_path / "legacy_crop.pt"
    source = DQNAgent(
        state_size=(9, 6, 6),
        action_size=3,
        state_mode="grid",
        local_crop_size=5,
        seed=1,
    )
    source.save(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint.pop("local_crop_size")
    checkpoint.pop("use_local_crop")
    torch.save(checkpoint, checkpoint_path)

    loaded = DQNAgent(
        state_size=(9, 6, 6),
        action_size=3,
        state_mode="grid",
        seed=2,
    )
    loaded.load(checkpoint_path)

    assert loaded.local_crop_size == 5
    assert loaded.use_local_crop is True


@pytest.mark.parametrize("architecture_version", [None, 1, 2, 4])
def test_load_rejects_non_current_architecture(
    tmp_path,
    architecture_version: int | None,
) -> None:
    checkpoint_path = tmp_path / f"architecture_{architecture_version}.pt"
    source_agent = DQNAgent(state_size=20, action_size=3, seed=1)
    source_agent.save(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if architecture_version is None:
        checkpoint.pop("architecture_version")
    else:
        checkpoint["architecture_version"] = architecture_version
    torch.save(checkpoint, checkpoint_path)

    loaded_agent = DQNAgent(state_size=20, action_size=3, seed=2)
    with pytest.raises(ValueError, match="requires architecture_version=3"):
        loaded_agent.load(checkpoint_path)


def test_load_rejects_incomplete_current_checkpoint_without_fallback(tmp_path) -> None:
    checkpoint_path = tmp_path / "incomplete_v3.pt"
    source_agent = DQNAgent(state_size=20, action_size=3, seed=1)
    source_agent.save(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint.pop("target_net")
    torch.save(checkpoint, checkpoint_path)

    loaded_agent = DQNAgent(state_size=20, action_size=3, seed=2)
    with pytest.raises(ValueError, match="missing required fields: target_net"):
        loaded_agent.load(checkpoint_path)
