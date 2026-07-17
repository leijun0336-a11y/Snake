from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Protocol

from snake_ai.agents import DQNAgent
from snake_ai.game.ai_profiles import AIProfile
from snake_ai.game.snake_env import Direction, Observation, SnakeEnv


@dataclass(frozen=True)
class ControlContext:
    observation: Observation
    direction: Direction


class Controller(Protocol):
    def reset(self) -> None: ...

    def handle_event(self, event: Any) -> None: ...

    def choose_action(self, context: ControlContext) -> int: ...


class HumanController:
    """Translate buffered absolute key directions into relative environment actions."""

    _MAX_BUFFERED_DIRECTIONS = 2

    def __init__(self) -> None:
        self._directions: deque[Direction] = deque()

    def reset(self) -> None:
        self._directions.clear()

    def handle_event(self, event: Any) -> None:
        import pygame

        if event.type != pygame.KEYDOWN:
            return
        direction_by_key = {
            pygame.K_RIGHT: Direction.RIGHT,
            pygame.K_d: Direction.RIGHT,
            pygame.K_DOWN: Direction.DOWN,
            pygame.K_s: Direction.DOWN,
            pygame.K_LEFT: Direction.LEFT,
            pygame.K_a: Direction.LEFT,
            pygame.K_UP: Direction.UP,
            pygame.K_w: Direction.UP,
        }
        direction = direction_by_key.get(event.key)
        if direction is None:
            return
        if len(self._directions) >= self._MAX_BUFFERED_DIRECTIONS:
            self._directions.popleft()
        self._directions.append(direction)

    def choose_action(self, context: ControlContext) -> int:
        directions = list(Direction)
        current_index = directions.index(context.direction)
        while self._directions:
            requested = self._directions.popleft()
            delta = (directions.index(requested) - current_index) % len(directions)
            if delta == 1:
                return 1
            if delta == 3:
                return 2
            if delta == 0:
                return 0
            # A direct reversal is invalid; consume it and keep looking.
        return 0


class DQNController:
    def __init__(self, profile: AIProfile, agent: DQNAgent) -> None:
        self.profile = profile
        self.agent = agent
        self.agent.policy_net.eval()

    @classmethod
    def load(cls, profile: AIProfile) -> DQNController:
        import torch

        if not profile.checkpoint_path.is_file():
            raise FileNotFoundError(f"AI checkpoint does not exist: {profile.checkpoint_path}")
        if profile.state_mode not in ("vector", "grid", "hybrid"):
            raise ValueError(f"Unsupported AI state mode: {profile.state_mode}")

        checkpoint = torch.load(profile.checkpoint_path, map_location="cpu")
        run_config = checkpoint.get("run_config")
        if not isinstance(run_config, dict):
            raise ValueError(f"AI checkpoint has no complete run_config: {profile.checkpoint_path}")
        environment = run_config.get("environment")
        reward = run_config.get("reward")
        if not isinstance(environment, dict) or not isinstance(reward, dict):
            raise ValueError("AI checkpoint run_config is missing environment or reward metadata")
        expected = {
            "width": profile.width,
            "height": profile.height,
            "state_mode": profile.state_mode,
        }
        actual = {key: environment.get(key) for key in expected}
        if actual != expected:
            raise ValueError(
                f"AI profile environment {expected} does not match checkpoint {actual}"
            )
        if reward.get("profile") != profile.reward_profile:
            raise ValueError(
                f"AI profile reward {profile.reward_profile!r} does not match checkpoint "
                f"{reward.get('profile')!r}"
            )
        if checkpoint.get("network_type", "q_network") != profile.network_type:
            raise ValueError("AI profile network_type does not match checkpoint metadata")

        state_size: int | tuple[int, int, int]
        if profile.state_mode in ("grid", "hybrid"):
            state_size = (SnakeEnv.grid_channels, profile.height, profile.width)
        else:
            state_size = SnakeEnv.state_size

        agent = DQNAgent(
            state_size=state_size,
            action_size=SnakeEnv.action_size,
            epsilon_start=0.0,
            epsilon_end=0.0,
            state_mode=profile.state_mode,
            network_type=profile.network_type,
            auxiliary_size=SnakeEnv.state_size,
        )
        # load() performs strict state, mode, version and architecture checks.
        agent.load(profile.checkpoint_path)
        return cls(profile, agent)

    def reset(self) -> None:
        pass

    def handle_event(self, event: Any) -> None:
        del event

    def choose_action(self, context: ControlContext) -> int:
        return self.agent.act(context.observation, training=False)


class AIControllerRegistry:
    """Load each configured AI once; never substitute another profile."""

    def __init__(self) -> None:
        self._controllers: dict[str, DQNController] = {}

    def get(self, profile: AIProfile) -> DQNController:
        controller = self._controllers.get(profile.id)
        if controller is None:
            controller = DQNController.load(profile)
            self._controllers[profile.id] = controller
        return controller
