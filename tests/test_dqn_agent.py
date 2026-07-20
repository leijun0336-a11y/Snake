import numpy as np
import pytest
import torch
from torch import nn

from snake_ai.agents.dqn_agent import DQNAgent, MaskedDQNAgent
from snake_ai.agents.replay_buffer import MaskedReplayBuffer, ReplayBuffer
from snake_ai.models.q_network import QNetwork


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


class ConstantQNetwork(nn.Module):
    def __init__(self, values: tuple[float, float, float]) -> None:
        super().__init__()
        self.values = nn.Parameter(torch.tensor(values, dtype=torch.float32))

    def forward(self, state):
        batch_size = state[0].shape[0] if isinstance(state, tuple) else state.shape[0]
        return self.values.unsqueeze(0).expand(batch_size, -1)


def test_masked_agent_exploration_and_greedy_action_never_leave_safe_set() -> None:
    agent = MaskedDQNAgent(
        state_size=20,
        action_size=3,
        epsilon_start=1.0,
        epsilon_end=0.0,
        seed=1,
        device="cpu",
    )
    state = [0.0] * 20

    assert isinstance(agent.replay_buffer, MaskedReplayBuffer)
    assert agent.act_masked(state, (False, False, True), training=True) == 2

    agent.epsilon = 0.0
    agent.policy_net = ConstantQNetwork((2.0, 100.0, 4.0))
    assert agent.act_masked(state, (True, False, True), training=False) == 2
    with pytest.raises(ValueError, match="at least one"):
        agent.act_masked(state, (False, False, False), training=False)


def test_masked_double_dqn_target_ignores_unsafe_highest_q() -> None:
    agent = MaskedDQNAgent(
        state_size=20,
        action_size=3,
        batch_size=1,
        gamma=1.0,
        seed=1,
        device="cpu",
    )
    agent.policy_net = ConstantQNetwork((1.0, 100.0, 2.0))
    agent.target_net = ConstantQNetwork((10.0, 1000.0, 20.0))
    agent.optimizer = torch.optim.SGD(agent.policy_net.parameters(), lr=0.01)
    state = [0.0] * 20
    mask = (True, False, True)
    agent.remember_masked(state, 0, 0.0, state, False, mask, mask)

    loss = agent.learn()

    # 在线网络在安全动作 0/2 中选择动作 2，目标值为 20；Huber(1, 20)=18.5。
    assert loss == pytest.approx(18.5)


def test_plain_agent_still_uses_plain_replay_buffer() -> None:
    agent = DQNAgent(state_size=20, action_size=3, seed=1)

    assert type(agent.replay_buffer) is ReplayBuffer


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
    assert "network_type" not in checkpoint
    assert "cnn_pool_size" not in checkpoint
    assert "mask_enabled" not in checkpoint


def test_masked_checkpoint_is_explicit_and_rejects_plain_checkpoint(tmp_path) -> None:
    masked_path = tmp_path / "masked.pt"
    plain_path = tmp_path / "plain.pt"
    masked = MaskedDQNAgent(state_size=20, action_size=3, seed=1, device="cpu")
    plain = DQNAgent(state_size=20, action_size=3, seed=1, device="cpu")

    masked.save(masked_path, metadata={"mask": {"enabled": True}})
    plain.save(plain_path)
    checkpoint = torch.load(masked_path, map_location="cpu")

    assert checkpoint["mask_enabled"] is True
    assert checkpoint["mask_version"] == 2
    assert checkpoint["mask_planner"] == "hamiltonian_viability"
    MaskedDQNAgent(state_size=20, action_size=3, seed=2, device="cpu").load(masked_path)
    with pytest.raises(ValueError, match="mask_enabled=true"):
        MaskedDQNAgent(state_size=20, action_size=3, seed=2, device="cpu").load(plain_path)


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
