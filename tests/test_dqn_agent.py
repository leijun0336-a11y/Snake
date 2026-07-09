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
    grid = [[[0.0 for _ in range(6)] for _ in range(6)] for _ in range(5)]
    direction = [0.0, 1.0, 0.0, 0.0]

    action = agent.act((grid, direction), training=False)

    assert action in (0, 1, 2)


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
