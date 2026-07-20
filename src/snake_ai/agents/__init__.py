from snake_ai.agents.replay_buffer import (
    MaskedReplayBuffer,
    MaskedTransition,
    ReplayBuffer,
    SafeMask,
    Transition,
)

__all__ = [
    "DQNAgent",
    "MaskedDQNAgent",
    "MaskedReplayBuffer",
    "MaskedTransition",
    "ReplayBuffer",
    "SafeMask",
    "Transition",
]


def __getattr__(name: str):
    if name == "DQNAgent":
        from snake_ai.agents.dqn_agent import DQNAgent

        return DQNAgent
    if name == "MaskedDQNAgent":
        from snake_ai.agents.dqn_agent import MaskedDQNAgent

        return MaskedDQNAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
