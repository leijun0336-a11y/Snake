import numpy as np
import pytest

from snake_ai.agents.dqn_agent import DQNAgent


def test_dueling_agent_outputs_action() -> None:
    agent = DQNAgent(state_size=19, action_size=3, seed=1)

    action = agent.act([0.0] * 19, training=False)

    assert action in (0, 1, 2)


def test_grid_agent_outputs_action() -> None:
    agent = DQNAgent(
        state_size=(5, 6, 6),
        action_size=3,
        state_mode="grid",
        seed=1,
    )
    grid = np.zeros((5, 6, 6), dtype=np.float32)

    action = agent.act(grid, training=False)

    assert action in (0, 1, 2)


def test_hybrid_agent_outputs_action() -> None:
    agent = DQNAgent(
        state_size=(5, 6, 6),
        action_size=3,
        state_mode="hybrid",
        auxiliary_size=19,
        seed=1,
    )
    grid = np.zeros((5, 6, 6), dtype=np.float32)
    vector_state = [0.0] * 19

    action = agent.act((grid, vector_state), training=False)

    assert action in (0, 1, 2)


@pytest.mark.parametrize("state_mode", ["grid", "hybrid"])
def test_cnn_agent_learns_from_numpy_replay_batch(state_mode: str) -> None:
    agent = DQNAgent(
        state_size=(5, 6, 6),
        action_size=3,
        state_mode=state_mode,
        auxiliary_size=19,
        batch_size=2,
        seed=1,
    )
    grid = np.zeros((5, 6, 6), dtype=np.float32)
    vector_state = [0.0] * 19
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


def test_load_rejects_incompatible_state_size(tmp_path) -> None:
    checkpoint_path = tmp_path / "state_11.pt"
    old_agent = DQNAgent(state_size=11, action_size=3, seed=1)
    old_agent.save(checkpoint_path)

    agent = DQNAgent(state_size=19, action_size=3, seed=1)

    with pytest.raises(ValueError, match="state_size"):
        agent.load(checkpoint_path)


def test_load_rejects_incompatible_state_mode(tmp_path) -> None:
    checkpoint_path = tmp_path / "vector.pt"
    vector_agent = DQNAgent(state_size=19, action_size=3, seed=1)
    vector_agent.save(checkpoint_path)

    grid_agent = DQNAgent(
        state_size=(5, 6, 6),
        action_size=3,
        state_mode="grid",
        seed=1,
    )

    with pytest.raises(ValueError, match="state_size|state_mode"):
        grid_agent.load(checkpoint_path)


def test_load_restores_custom_cnn_architecture(tmp_path) -> None:
    checkpoint_path = tmp_path / "custom_cnn.pt"
    trained_agent = DQNAgent(
        state_size=(5, 8, 8),
        action_size=3,
        state_mode="hybrid",
        auxiliary_size=19,
        cnn_channels=24,
        cnn_output_channels=12,
        cnn_dilations=(1, 3),
        cnn_pool_size=(4, 4),
        seed=1,
    )
    trained_agent.save(checkpoint_path)

    loaded_agent = DQNAgent(
        state_size=(5, 8, 8),
        action_size=3,
        state_mode="hybrid",
        auxiliary_size=19,
        seed=1,
    )
    loaded_agent.load(checkpoint_path)

    assert loaded_agent.cnn_channels == 24
    assert loaded_agent.cnn_output_channels == 12
    assert loaded_agent.cnn_dilations == (1, 3)
    assert loaded_agent.cnn_pool_size == (4, 4)
