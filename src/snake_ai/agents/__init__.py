from snake_ai.agents.replay_buffer import ReplayBuffer, Transition

__all__ = ["DQNAgent", "ReplayBuffer", "Transition"]


def __getattr__(name: str):
    if name == "DQNAgent":
        from snake_ai.agents.dqn_agent import DQNAgent

        return DQNAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
