import numpy as np
import pytest
import torch
from torch import nn

from snake_ai.agents.dqn_agent import DQNAgent
from snake_ai.models.q_network import QNetwork
from snake_ai.models.q_network_old import QNetworkOld


def test_dueling_agent_outputs_action() -> None:
    agent = DQNAgent(state_size=20, action_size=3, seed=1)

    action = agent.act([0.0] * 20, training=False)

    assert action in (0, 1, 2)


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


def test_epsilon_decay_strategies_are_explicit() -> None:
    linear_agent = DQNAgent(
        state_size=20,
        action_size=3,
        epsilon_start=1.0,
        epsilon_end=0.0,
        epsilon_exp_decay=False,
        epsilon_linear_episodes=10,
        seed=1,
    )
    linear_agent.decay_epsilon(5)
    assert linear_agent.epsilon == pytest.approx(0.5)

    exp_agent = DQNAgent(
        state_size=20,
        action_size=3,
        epsilon_start=1.0,
        epsilon_end=0.0,
        epsilon_exp_decay=True,
        epsilon_exp_factor=0.8,
        seed=1,
    )
    exp_agent.decay_epsilon()
    assert exp_agent.epsilon == pytest.approx(0.8)


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
    expected_features = 8 * height * width + 200
    if state_mode == "hybrid":
        expected_features += 20

    assert not hasattr(network, "global_pool")
    assert isinstance(network.feature[0], nn.Linear)
    assert network.feature[0].in_features == expected_features
    assert not any(isinstance(module, nn.AdaptiveAvgPool2d) for module in network.modules())

    grid = torch.zeros((2, 9, height, width), dtype=torch.float32)
    grid[:, 3, height // 2, width // 2] = 1.0
    inputs = (
        (grid, torch.zeros((2, 20), dtype=torch.float32))
        if state_mode == "hybrid"
        else grid
    )
    assert network(inputs).shape == (2, 3)


@pytest.mark.parametrize("state_mode", ["grid", "hybrid"])
def test_cnn_agent_learns_from_numpy_replay_batch(state_mode: str) -> None:
    agent = DQNAgent(
        state_size=(9, 6, 6),
        action_size=3,
        state_mode=state_mode,
        auxiliary_size=20,
        batch_size=2,
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


def test_load_legacy_non_dueling_checkpoint(tmp_path) -> None:
    checkpoint_path = tmp_path / "legacy.pt"
    legacy_agent = DQNAgent(state_size=11, action_size=3, dueling=False, seed=1)
    legacy_agent.save(checkpoint_path)

    agent = DQNAgent(state_size=11, action_size=3, dueling=True, seed=1)
    agent.load(checkpoint_path)

    assert agent.dueling is False
    assert agent.act([0.0] * 11, training=False) in (0, 1, 2)


def test_load_migrates_legacy_epsilon_decay_names(tmp_path) -> None:
    checkpoint_path = tmp_path / "legacy_epsilon.pt"
    source_agent = DQNAgent(state_size=11, action_size=3, seed=1)
    source_agent.save(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint.pop("epsilon_exp_decay")
    checkpoint.pop("epsilon_exp_factor")
    checkpoint.pop("epsilon_linear_episodes")
    checkpoint["epsilon_decay"] = 0.8
    checkpoint["epsilon_decay_episodes"] = None
    torch.save(checkpoint, checkpoint_path)

    loaded_agent = DQNAgent(state_size=11, action_size=3, seed=1)
    loaded_agent.load(checkpoint_path)

    assert loaded_agent.epsilon_exp_decay is True
    assert loaded_agent.epsilon_exp_factor == 0.8


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
        epsilon_exp_decay=True,
        epsilon_exp_factor=0.8,
        epsilon_linear_episodes=123,
        seed=1,
    )
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
    assert loaded_agent.epsilon_exp_decay is True
    assert loaded_agent.epsilon_exp_factor == 0.8
    assert loaded_agent.epsilon_linear_episodes == 123


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
    assert "cnn_pool_size" not in checkpoint


def test_load_migrates_version2_checkpoint_when_grid_matches_pool_size(tmp_path) -> None:
    checkpoint_path = tmp_path / "version2_10x10.pt"
    source_agent = DQNAgent(
        state_size=(9, 10, 10),
        action_size=3,
        state_mode="hybrid",
        auxiliary_size=20,
        seed=1,
    )
    source_agent.save(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint["architecture_version"] = 2
    checkpoint["cnn_pool_size"] = (10, 10)
    torch.save(checkpoint, checkpoint_path)

    loaded_agent = DQNAgent(
        state_size=(9, 10, 10),
        action_size=3,
        state_mode="hybrid",
        auxiliary_size=20,
        seed=2,
    )
    loaded_agent.load(checkpoint_path)

    assert torch.equal(
        loaded_agent.policy_net.feature[0].weight,
        source_agent.policy_net.feature[0].weight,
    )


def test_load_rejects_version2_checkpoint_with_resized_grid(tmp_path) -> None:
    checkpoint_path = tmp_path / "version2_6x6.pt"
    source_agent = DQNAgent(
        state_size=(9, 6, 6),
        action_size=3,
        state_mode="hybrid",
        auxiliary_size=20,
        seed=1,
    )
    source_agent.save(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint["architecture_version"] = 2
    checkpoint["cnn_pool_size"] = (10, 10)
    torch.save(checkpoint, checkpoint_path)

    loaded_agent = DQNAgent(
        state_size=(9, 6, 6),
        action_size=3,
        state_mode="hybrid",
        auxiliary_size=20,
        seed=2,
    )

    with pytest.raises(ValueError, match="fixed pooled size"):
        loaded_agent.load(checkpoint_path)


def test_q_network_old_loads_version2_resized_checkpoint_for_evaluation(
    tmp_path,
) -> None:
    checkpoint_path = tmp_path / "version2_old_6x6.pt"
    source_agent = DQNAgent(
        state_size=(9, 6, 6),
        action_size=3,
        state_mode="hybrid",
        network_type="q_network_old",
        auxiliary_size=20,
        seed=1,
    )
    torch.save(
        {
            "policy_net": source_agent.policy_net.state_dict(),
            "target_net": source_agent.target_net.state_dict(),
            "state_size": (9, 6, 6),
            "state_mode": "hybrid",
            "auxiliary_size": 20,
            "cnn_channels": 32,
            "cnn_output_channels": 8,
            "cnn_dilations": (1, 1, 2),
            "cnn_pool_size": (10, 10),
            "hidden_size": 128,
            "action_size": 3,
            "dueling": True,
            "architecture_version": 2,
        },
        checkpoint_path,
    )

    loaded_agent = DQNAgent(
        state_size=(9, 6, 6),
        action_size=3,
        state_mode="hybrid",
        network_type="q_network_old",
        auxiliary_size=20,
        seed=2,
    )
    loaded_agent.load(checkpoint_path)

    assert isinstance(loaded_agent.policy_net, QNetworkOld)
    assert isinstance(loaded_agent.policy_net.global_pool, nn.AdaptiveAvgPool2d)
    grid = np.zeros((9, 6, 6), dtype=np.float32)
    grid[3, 3, 3] = 1.0
    assert loaded_agent.act((grid, [0.0] * 20), training=False) in (0, 1, 2)
    with pytest.raises(RuntimeError, match="evaluation-only"):
        loaded_agent.learn()


def test_dqn_agent_rejects_unknown_network_type() -> None:
    with pytest.raises(ValueError, match="network_type"):
        DQNAgent(state_size=20, action_size=3, network_type="unknown")
