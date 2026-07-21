from snake_ai.agents.replay_buffer import ReplayBuffer, Transition

__all__ = ["DQNAgent", "PPOAgent", "ReplayBuffer", "Transition"]


def __getattr__(name: str):
    if name == "DQNAgent":
        from snake_ai.agents.dqn_agent import DQNAgent

        return DQNAgent
    if name == "PPOAgent":
        from snake_ai.agents.ppo_agent import PPOAgent

        return PPOAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
