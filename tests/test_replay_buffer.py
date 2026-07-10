import pytest

from snake_ai.agents.replay_buffer import ReplayBuffer


def test_replay_buffer_respects_capacity() -> None:
    buffer = ReplayBuffer(capacity=2, seed=1)

    buffer.push([0], 0, 0.0, [1], False)
    buffer.push([1], 1, 1.0, [2], False)
    buffer.push([2], 2, 2.0, [3], True)

    assert len(buffer) == 2
    # 环形列表写满后应覆盖最旧的 state=[0]，保留最近两条经验。
    assert {item.state[0] for item in buffer.sample(2)} == {1, 2}


def test_sample_returns_requested_batch_size() -> None:
    buffer = ReplayBuffer(capacity=4, seed=1)
    for index in range(4):
        buffer.push([index], 0, 0.0, [index + 1], False)

    batch = buffer.sample(3)

    assert len(batch) == 3


def test_sample_too_large_raises_error() -> None:
    buffer = ReplayBuffer(capacity=2, seed=1)
    buffer.push([0], 0, 0.0, [1], False)

    with pytest.raises(ValueError):
        buffer.sample(2)
