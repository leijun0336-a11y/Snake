import pytest

from snake_ai.agents.replay_buffer import NStepAccumulator, PrioritizedReplayBuffer, ReplayBuffer


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


def test_prioritized_replay_samples_large_td_errors_more_often() -> None:
    buffer = PrioritizedReplayBuffer(capacity=2, alpha=1.0, seed=1)
    buffer.push([0], 0, 0.0, [1], False)
    buffer.push([1], 0, 0.0, [2], False)
    buffer.update_priorities((0, 1), (0.0, 9.0))

    counts = [0, 0]
    for _ in range(1000):
        sample = buffer.sample(batch_size=1, beta=0.4)
        counts[sample.transitions[0].state[0]] += 1

    assert counts[1] > 900


def test_prioritized_replay_returns_normalized_importance_weights() -> None:
    buffer = PrioritizedReplayBuffer(capacity=4, seed=1)
    for index in range(4):
        buffer.push([index], 0, 0.0, [index + 1], False)
    buffer.update_priorities((0, 1, 2, 3), (0.0, 1.0, 2.0, 3.0))

    sample = buffer.sample(batch_size=4, beta=0.4)

    assert len(sample.transitions) == len(sample.indices) == len(sample.weights) == 4
    assert max(sample.weights) == pytest.approx(1.0)
    assert all(0.0 < weight <= 1.0 for weight in sample.weights)


def test_prioritized_replay_samples_partial_non_power_of_two_capacity() -> None:
    buffer = PrioritizedReplayBuffer(capacity=5, seed=1)
    for index in range(3):
        buffer.push([index], 0, 0.0, [index + 1], False)

    for _ in range(20):
        sample = buffer.sample(batch_size=3, beta=0.4)
        assert all(0 <= index < 3 for index in sample.indices)
        assert all(transition.state[0] in (0, 1, 2) for transition in sample.transitions)


def test_prioritized_replay_rejects_non_finite_td_error() -> None:
    buffer = PrioritizedReplayBuffer(capacity=1)
    buffer.push([0], 0, 0.0, [1], False)

    with pytest.raises(ValueError, match="finite"):
        buffer.update_priorities((0,), (float("nan"),))


def test_n_step_accumulator_emits_discounted_full_horizon() -> None:
    accumulator = NStepAccumulator(n_step=3, gamma=0.5)

    assert accumulator.append("s0", 0, 1.0, "s1", False) == ()
    assert accumulator.append("s1", 1, 2.0, "s2", False) == ()
    ready = accumulator.append("s2", 2, 3.0, "s3", False)

    assert len(ready) == 1
    transition = ready[0]
    assert transition.state == "s0"
    assert transition.action == 0
    assert transition.reward == pytest.approx(1.0 + 0.5 * 2.0 + 0.5**2 * 3.0)
    assert transition.next_state == "s3"
    assert transition.done is False
    assert transition.n_steps == 3
    assert len(accumulator) == 2


def test_n_step_terminal_flushes_tail_without_bootstrap() -> None:
    accumulator = NStepAccumulator(n_step=3, gamma=0.5)
    accumulator.append("s0", 0, 1.0, "s1", False)

    ready = accumulator.append("s1", 1, 2.0, "terminal", True)

    assert [(item.state, item.reward, item.done, item.n_steps) for item in ready] == [
        ("s0", pytest.approx(2.0), True, 2),
        ("s1", pytest.approx(2.0), True, 1),
    ]
    assert all(item.next_state == "terminal" for item in ready)
    assert len(accumulator) == 0


def test_n_step_truncation_flushes_actual_horizon_and_keeps_bootstrap() -> None:
    accumulator = NStepAccumulator(n_step=3, gamma=0.5)
    accumulator.append("s0", 0, 1.0, "s1", False)
    accumulator.append("s1", 1, 2.0, "s2", False)

    ready = accumulator.flush()

    assert [(item.state, item.reward, item.done, item.n_steps) for item in ready] == [
        ("s0", pytest.approx(2.0), False, 2),
        ("s1", pytest.approx(2.0), False, 1),
    ]
    assert all(item.next_state == "s2" for item in ready)
    assert len(accumulator) == 0


def test_one_step_accumulator_matches_original_transition_immediately() -> None:
    accumulator = NStepAccumulator(n_step=1, gamma=0.99)

    ready = accumulator.append([0], 2, 4.5, [1], False)

    assert len(ready) == 1
    assert ready[0].reward == pytest.approx(4.5)
    assert ready[0].n_steps == 1
    assert len(accumulator) == 0
