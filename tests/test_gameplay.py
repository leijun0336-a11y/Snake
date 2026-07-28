from __future__ import annotations

import random
from concurrent.futures import Future
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pygame
import pytest

from snake_ai.game.ai_profiles import DEFAULT_AI_ID, get_ai_profile
from snake_ai.game.controllers import ControlContext, DQNController, HumanController
from snake_ai.game.food_policy import SeededRaceFoodPolicy
from snake_ai.game.game_app import AppScene, GameSettings, ModeName, SnakeGameApp
from snake_ai.game.modes.ai_viewer import AIViewerMode
from snake_ai.game.modes.race import RaceMode, RaceResult
from snake_ai.game.session import GameSession
from snake_ai.game.snake_env import Direction, Point, SnakeEnv


class StraightController:
    def reset(self) -> None:
        pass

    def handle_event(self, event: Any) -> None:
        del event

    def choose_action(self, context: ControlContext) -> int:
        del context
        return 0


class SilentAudio:
    def play_click(self) -> None:
        pass

    def play_eat(self) -> None:
        pass

    def play_finish(self) -> None:
        pass


def make_solo_app() -> SnakeGameApp:
    app = SnakeGameApp.__new__(SnakeGameApp)
    app.pygame = pygame
    app.settings = GameSettings()
    app.audio = SilentAudio()
    app._seed = 1
    app.mode_name = ModeName.SOLO
    app.active_mode = None
    app.error_message = ""
    app.accumulator = 0.0
    app.countdown = 0.0
    app.game_over_remaining = 0.0
    app.scene = AppScene.RESULT
    return app


def test_game_speed_has_slow_fine_grained_levels() -> None:
    assert tuple(range(1, 21)) == SnakeGameApp.SPEEDS
    assert GameSettings().tick_rate == 6


def test_settings_arrow_keys_adjust_and_clamp_speed() -> None:
    app = SnakeGameApp.__new__(SnakeGameApp)
    app.pygame = pygame
    app.settings = GameSettings()
    app.audio = SilentAudio()
    app.scene = AppScene.SETTINGS
    app.running = True

    app._handle_event(SimpleNamespace(type=pygame.KEYDOWN, key=pygame.K_UP))
    assert app.settings.tick_rate == 7
    app._handle_event(SimpleNamespace(type=pygame.KEYDOWN, key=pygame.K_DOWN))
    assert app.settings.tick_rate == 6

    app.settings.tick_rate = 20
    app._handle_event(SimpleNamespace(type=pygame.KEYDOWN, key=pygame.K_UP))
    assert app.settings.tick_rate == 20
    app.settings.tick_rate = 1
    app._handle_event(SimpleNamespace(type=pygame.KEYDOWN, key=pygame.K_DOWN))
    assert app.settings.tick_rate == 1


def test_play_again_accepts_wasd_during_countdown() -> None:
    app = make_solo_app()
    app._start(ModeName.SOLO)
    previous_mode = app.active_mode
    app.scene = AppScene.RESULT
    play_again = app._buttons()[0]

    app._handle_event(
        SimpleNamespace(
            type=pygame.MOUSEBUTTONUP,
            button=1,
            pos=play_again.rect.center,
        )
    )
    assert app.active_mode is not previous_mode
    assert app.scene == AppScene.PLAYING
    assert app.countdown == 3.0

    app._handle_event(SimpleNamespace(type=pygame.KEYDOWN, key=pygame.K_w))
    app._update(3.0)
    app._update(1.0 / app.settings.tick_rate)

    assert app.active_mode is not None
    assert app.active_mode.session.snapshot.direction == Direction.UP


def test_pause_resume_keeps_movement_controls_active() -> None:
    app = make_solo_app()
    app._start(ModeName.SOLO)
    app.countdown = 0.0

    app._handle_event(SimpleNamespace(type=pygame.KEYDOWN, key=pygame.K_p))
    assert app.scene == AppScene.PAUSED
    app._handle_event(SimpleNamespace(type=pygame.KEYDOWN, key=pygame.K_p))
    assert app.scene == AppScene.PLAYING

    app._handle_event(SimpleNamespace(type=pygame.KEYDOWN, key=pygame.K_s))
    app._update(1.0 / app.settings.tick_rate)

    assert app.active_mode is not None
    assert app.active_mode.session.snapshot.direction == Direction.DOWN


def test_ai_start_uses_loading_scene_without_consuming_a_seed() -> None:
    app = make_solo_app()
    app.ai_future = Future()
    initial_seed = app._seed

    app._start(ModeName.RACE)

    assert app.scene == AppScene.LOADING
    assert app.pending_mode == ModeName.RACE
    assert app._seed == initial_seed


def test_game_over_scene_waits_two_seconds_before_result() -> None:
    app = SnakeGameApp.__new__(SnakeGameApp)
    app.scene = AppScene.GAME_OVER
    app.game_over_remaining = app.GAME_OVER_DURATION

    app._update(1.25)
    assert app.scene == AppScene.GAME_OVER
    assert app.game_over_remaining == pytest.approx(0.75)

    app._update(0.75)
    assert app.scene == AppScene.RESULT
    assert app.game_over_remaining == 0.0


@pytest.mark.parametrize(
    ("result", "reason_text"),
    [
        (RaceResult("player", "board_completed"), "YOU REACHED FULL SCORE FIRST"),
        (RaceResult("ai", "collision_wall"), "YOU HIT THE WALL"),
        (RaceResult("player", "collision_body"), "AI HIT ITS BODY"),
        (RaceResult("player", "max_steps"), "400 STEPS: HIGHER SCORE"),
        (RaceResult("ai", "max_steps_earlier_score"), "400 STEPS: EARLIER FINAL SCORE"),
        (RaceResult("draw", "max_steps"), "400 STEPS: SAME SCORE AND TIMING"),
    ],
)
def test_race_result_explains_the_outcome(result: RaceResult, reason_text: str) -> None:
    assert result.reason_text == reason_text


def test_seeded_race_food_policy_is_deterministic_and_uses_legal_fallback() -> None:
    all_cells = [Point(x, y) for x in range(6) for y in range(6)]
    policy = SeededRaceFoodPolicy(1234)
    rng = random.Random(0)

    first = policy.choose(all_cells, all_cells, food_index=5, rng=rng)
    repeated = policy.choose(all_cells, all_cells, food_index=5, rng=rng)
    available_without_first = [cell for cell in all_cells if cell != first]
    fallback = policy.choose(
        all_cells,
        available_without_first,
        food_index=5,
        rng=rng,
    )

    assert repeated == first
    assert fallback != first
    assert fallback in available_without_first


def test_race_food_policy_reproduces_identical_environments() -> None:
    first = SnakeEnv(
        width=6,
        height=6,
        seed=11,
        starvation_enabled=False,
        food_policy=SeededRaceFoodPolicy(99),
    )
    second = SnakeEnv(
        width=6,
        height=6,
        seed=11,
        starvation_enabled=False,
        food_policy=SeededRaceFoodPolicy(99),
    )

    assert first.food == second.food
    for action in (2, 1, 0):
        first_result = first.step(action)
        second_result = second.step(action)
        assert first.snake == second.snake
        assert first.food == second.food
        assert first_result[1:] == second_result[1:]


def test_human_controller_buffers_turns_and_rejects_reversal() -> None:
    controller = HumanController()
    controller.handle_event(SimpleNamespace(type=pygame.KEYDOWN, key=pygame.K_UP))
    controller.handle_event(SimpleNamespace(type=pygame.KEYDOWN, key=pygame.K_LEFT))

    assert controller.choose_action(ControlContext([], Direction.RIGHT)) == 2
    assert controller.choose_action(ControlContext([], Direction.UP)) == 2

    controller.handle_event(SimpleNamespace(type=pygame.KEYDOWN, key=pygame.K_RIGHT))
    assert controller.choose_action(ControlContext([], Direction.LEFT)) == 0


@pytest.mark.parametrize(
    ("key", "current_direction", "expected_action"),
    [
        (pygame.K_w, Direction.RIGHT, 2),
        (pygame.K_s, Direction.RIGHT, 1),
        (pygame.K_d, Direction.RIGHT, 0),
        (pygame.K_a, Direction.UP, 2),
        (pygame.K_UP, Direction.RIGHT, 2),
        (pygame.K_DOWN, Direction.RIGHT, 1),
        (pygame.K_RIGHT, Direction.RIGHT, 0),
        (pygame.K_LEFT, Direction.UP, 2),
    ],
)
def test_all_movement_keys_map_to_actions(
    key: int,
    current_direction: Direction,
    expected_action: int,
) -> None:
    controller = HumanController()
    controller.handle_event(SimpleNamespace(type=pygame.KEYDOWN, key=key))

    assert controller.choose_action(ControlContext([], current_direction)) == expected_action


def test_session_snapshot_tracks_steps_without_starvation() -> None:
    env = SnakeEnv(width=6, height=6, seed=1, starvation_enabled=False)
    session = GameSession(env, StraightController(), tick_rate=10, seed=1)

    step = session.tick()

    assert step.previous.steps == 0
    assert step.current.steps == 1
    assert step.current.elapsed_seconds == 0.1
    assert step.current.hunger_ratio == 0.0


def test_session_stops_at_configured_max_steps() -> None:
    env = SnakeEnv(width=6, height=6, seed=1, starvation_enabled=False)
    session = GameSession(
        env,
        StraightController(),
        tick_rate=10,
        seed=1,
        max_steps=1,
    )

    session.tick()

    assert session.done is True
    assert session.snapshot.steps == 1
    assert session.snapshot.termination_reason == "max_steps"


def test_race_advances_both_sides_before_same_tick_draw() -> None:
    policy = SeededRaceFoodPolicy(7)
    player = GameSession(
        SnakeEnv(
            width=6,
            height=6,
            seed=7,
            starvation_enabled=False,
            food_policy=policy,
        ),
        StraightController(),
        tick_rate=10,
        seed=7,
    )
    ai = GameSession(
        SnakeEnv(
            width=6,
            height=6,
            seed=7,
            starvation_enabled=False,
            food_policy=policy,
        ),
        StraightController(),
        tick_rate=10,
        seed=7,
    )
    race = RaceMode(player, ai, max_steps=20)

    while not race.finished:
        race.tick()

    assert race.full_score == 33
    assert player.snapshot.steps == ai.snapshot.steps == race.race_steps
    assert race.result is not None
    assert race.result.winner == "draw"
    assert race.result.reason == "both_collided"


def test_default_ai_profile_strictly_loads_selected_checkpoint() -> None:
    profile = get_ai_profile(DEFAULT_AI_ID)
    controller = DQNController.load(profile)
    env = SnakeEnv(
        width=profile.width,
        height=profile.height,
        state_mode=profile.state_mode,
        starvation_enabled=False,
        seed=1,
    )

    action = controller.choose_action(ControlContext(env.reset(seed=1), env.direction))

    assert profile.checkpoint_path.name == "best.pt"
    assert action in (0, 1, 2)

    viewer = AIViewerMode.create(profile, controller, seed=1, tick_rate=6)
    race = RaceMode.create(profile, controller, race_seed=1, tick_rate=6)
    assert viewer.ai_id == profile.id
    assert race.ai_id == profile.id


def test_ai_profile_mismatch_is_rejected_without_fallback() -> None:
    profile = replace(get_ai_profile(DEFAULT_AI_ID), width=8)

    with pytest.raises(ValueError, match="does not match checkpoint"):
        DQNController.load(profile)
