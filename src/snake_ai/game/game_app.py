from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum

from snake_ai.game.ai_profiles import DEFAULT_AI_ID, get_ai_profile
from snake_ai.game.controllers import AIControllerRegistry, DQNController
from snake_ai.game.modes import AIViewerMode, RaceMode, SoloMode
from snake_ai.ui.audio import AudioManager
from snake_ai.ui.game_renderer import GameRenderer
from snake_ai.ui.widgets import Button


class AppScene(Enum):
    MENU = "menu"
    SETTINGS = "settings"
    RULES = "rules"
    LOADING = "loading"
    PLAYING = "playing"
    PAUSED = "paused"
    GAME_OVER = "game_over"
    RESULT = "result"


class ModeName(Enum):
    SOLO = "solo"
    AI_VIEWER = "ai_viewer"
    RACE = "race"


@dataclass
class GameSettings:
    tick_rate: int = 6
    sound_enabled: bool = True


ActiveMode = SoloMode | AIViewerMode | RaceMode


class SnakeGameApp:
    WIDTH = 1120
    HEIGHT = 720
    RENDER_FPS = 60
    GAME_OVER_DURATION = 2.0
    SPEEDS = tuple(range(1, 21))

    def __init__(self) -> None:
        import pygame

        pygame.init()
        pygame.display.set_caption("Snake Protocol")
        self.pygame = pygame
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.renderer = GameRenderer(self.screen)
        self.settings = GameSettings()
        self.audio = AudioManager(self.settings.sound_enabled)
        self.ai_registry = AIControllerRegistry()
        self.ai_loader = ThreadPoolExecutor(max_workers=1, thread_name_prefix="snake-ai")
        self.ai_future: Future[DQNController] | None = None
        self.pending_mode: ModeName | None = None
        self.scene = AppScene.MENU
        self.mode_name: ModeName | None = None
        self.active_mode: ActiveMode | None = None
        self.running = True
        self.error_message = ""
        self.rules_return_scene = AppScene.MENU
        self.accumulator = 0.0
        self.elapsed = 0.0
        self.countdown = 0.0
        self.game_over_remaining = 0.0
        self._seed = 42

    def run(self) -> None:
        try:
            while self.running:
                dt = min(self.clock.tick(self.RENDER_FPS) / 1000.0, 0.1)
                self.elapsed += dt
                for event in self.pygame.event.get():
                    self._handle_event(event)
                self._update(dt)
                self._draw(dt)
                self.pygame.display.flip()
                self._begin_ai_load()
        finally:
            self.ai_loader.shutdown(wait=False, cancel_futures=True)
            self.pygame.quit()

    def _handle_event(self, event: object) -> None:
        pygame = self.pygame
        if event.type == pygame.QUIT:
            self.running = False
            return

        if self.scene == AppScene.PLAYING:
            if event.type == pygame.KEYDOWN and event.key in (pygame.K_p, pygame.K_ESCAPE):
                self.scene = AppScene.PAUSED
                return
            if self.active_mode is not None:
                self.active_mode.handle_event(event)
            return

        if self.scene == AppScene.SETTINGS and event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                self._adjust_speed(1)
                self.audio.play_click()
                return
            if event.key == pygame.K_DOWN:
                self._adjust_speed(-1)
                self.audio.play_click()
                return
            if event.key == pygame.K_ESCAPE:
                self.scene = AppScene.MENU
                return

        if self.scene == AppScene.LOADING and event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.pending_mode = None
                self.scene = AppScene.MENU
                return

        if self.scene == AppScene.RULES and event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.scene = self.rules_return_scene
                return

        for button in self._buttons():
            if button.clicked(event):
                self.audio.play_click()
                self._perform(button.action)
                return

        if self.scene == AppScene.PAUSED and event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_p, pygame.K_ESCAPE):
                self.scene = AppScene.PLAYING

    def _perform(self, action: str) -> None:
        if action == "solo":
            self._start(ModeName.SOLO)
        elif action == "ai":
            self._start(ModeName.AI_VIEWER)
        elif action == "race":
            self._start(ModeName.RACE)
        elif action == "settings":
            self.scene = AppScene.SETTINGS
        elif action == "rules":
            self.rules_return_scene = (
                AppScene.PAUSED if self.scene == AppScene.PAUSED else AppScene.MENU
            )
            self.scene = AppScene.RULES
        elif action == "rules_back":
            self.scene = self.rules_return_scene
        elif action == "back":
            self.scene = AppScene.MENU
        elif action == "resume":
            self.scene = AppScene.PLAYING
        elif action == "restart":
            if self.mode_name is None:
                raise RuntimeError("Cannot restart without an active game mode")
            self._start(self.mode_name)
        elif action == "speed_up":
            self._adjust_speed(1)
        elif action == "speed_down":
            self._adjust_speed(-1)
        elif action == "sound":
            self.settings.sound_enabled = not self.settings.sound_enabled
            self.audio.set_enabled(self.settings.sound_enabled)
        elif action == "quit":
            self.running = False
        else:
            raise ValueError(f"Unknown UI action: {action}")

    def _adjust_speed(self, change: int) -> None:
        if change not in (-1, 1):
            raise ValueError("Speed change must be -1 or 1")
        minimum, maximum = self.SPEEDS[0], self.SPEEDS[-1]
        self.settings.tick_rate = min(max(self.settings.tick_rate + change, minimum), maximum)

    def _begin_ai_load(self) -> None:
        if self.ai_future is None:
            profile = get_ai_profile(DEFAULT_AI_ID)
            self.ai_future = self.ai_loader.submit(self.ai_registry.get, profile)

    def _request_ai_controller(self, mode_name: ModeName) -> DQNController | None:
        self._begin_ai_load()
        if self.ai_future is None:
            raise RuntimeError("AI loader did not start")
        if not self.ai_future.done():
            self.pending_mode = mode_name
            self.scene = AppScene.LOADING
            return None
        return self.ai_future.result()

    def _start(self, mode_name: ModeName) -> None:
        self.error_message = ""
        try:
            controller: DQNController | None = None
            if mode_name != ModeName.SOLO:
                controller = self._request_ai_controller(mode_name)
                if controller is None:
                    return

            seed = self._seed
            self._seed += 1
            if mode_name == ModeName.SOLO:
                mode: ActiveMode = SoloMode.create(seed=seed, tick_rate=self.settings.tick_rate)
            else:
                profile = get_ai_profile(DEFAULT_AI_ID)
                if mode_name == ModeName.AI_VIEWER:
                    mode = AIViewerMode.create(
                        profile,
                        controller,
                        seed=seed,
                        tick_rate=self.settings.tick_rate,
                    )
                elif mode_name == ModeName.RACE:
                    mode = RaceMode.create(
                        profile,
                        controller,
                        race_seed=seed,
                        tick_rate=self.settings.tick_rate,
                    )
                else:
                    raise ValueError(f"Unsupported game mode: {mode_name}")
        except (FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
            self.error_message = str(exc)
            self.scene = AppScene.MENU
            return

        self.mode_name = mode_name
        self.pending_mode = None
        self.active_mode = mode
        self.accumulator = 0.0
        self.countdown = 3.0 if mode_name in (ModeName.SOLO, ModeName.RACE) else 0.0
        self.game_over_remaining = 0.0
        self.scene = AppScene.PLAYING

    def _update(self, dt: float) -> None:
        if self.scene == AppScene.LOADING:
            if self.ai_future is not None and self.ai_future.done():
                pending_mode = self.pending_mode
                self.pending_mode = None
                if pending_mode is not None:
                    self._start(pending_mode)
            return
        if self.scene == AppScene.GAME_OVER:
            self.game_over_remaining = max(self.game_over_remaining - dt, 0.0)
            if self.game_over_remaining == 0.0:
                self.scene = AppScene.RESULT
            return
        if self.scene != AppScene.PLAYING or self.active_mode is None:
            return
        if self.countdown > 0.0:
            self.countdown = max(self.countdown - dt, 0.0)
            return

        interval = 1.0 / self.settings.tick_rate
        self.accumulator += dt
        ticks = 0
        while self.accumulator >= interval and ticks < 4:
            before_scores = self._scores()
            self.active_mode.tick()
            after_scores = self._scores()
            if sum(after_scores) > sum(before_scores):
                self.audio.play_eat()
            self.accumulator -= interval
            ticks += 1
            if self.active_mode.finished:
                self.audio.play_finish()
                self.scene = AppScene.GAME_OVER
                self.game_over_remaining = self.GAME_OVER_DURATION
                self.accumulator = 0.0
                break
        if ticks == 4 and self.accumulator >= interval:
            self.accumulator = 0.0

    def _scores(self) -> tuple[int, ...]:
        if isinstance(self.active_mode, RaceMode):
            return (
                self.active_mode.player_session.snapshot.score,
                self.active_mode.ai_session.snapshot.score,
            )
        if isinstance(self.active_mode, (SoloMode, AIViewerMode)):
            return (self.active_mode.session.snapshot.score,)
        return ()

    def _draw(self, dt: float) -> None:
        self.renderer.clear()
        if self.scene == AppScene.MENU:
            self._draw_menu()
        elif self.scene == AppScene.SETTINGS:
            self._draw_settings()
        elif self.scene == AppScene.RULES:
            self._draw_rules()
        elif self.scene == AppScene.LOADING:
            self._draw_loading()
        else:
            self._draw_game(dt)
            if self.scene == AppScene.PAUSED:
                self._draw_overlay("PAUSED")
            elif self.scene == AppScene.GAME_OVER:
                self._draw_game_over()
            elif self.scene == AppScene.RESULT:
                result = self.active_mode.result_text if self.active_mode is not None else ""
                self._draw_overlay(result)

    def _draw_menu(self) -> None:
        self.renderer.text(
            "SNAKE PROTOCOL",
            (self.WIDTH // 2, 105),
            font=self.renderer.font_title,
            color=self.renderer.theme.player,
        )
        self.renderer.text(
            "HUMAN  /  AI  /  RACE",
            (self.WIDTH // 2, 150),
            font=self.renderer.font_small,
            color=self.renderer.theme.muted_text,
        )
        for button in self._buttons():
            button.draw(self.screen, self.renderer.font_button, self.renderer.theme)
        if self.error_message:
            message = self.error_message
            if len(message) > 92:
                message = message[:89] + "..."
            self.renderer.text(
                message,
                (self.WIDTH // 2, 650),
                font=self.renderer.font_small,
                color=self.renderer.theme.warning,
            )

    def _draw_settings(self) -> None:
        import pygame

        self.renderer.text("SETTINGS", (self.WIDTH // 2, 145), font=self.renderer.font_title)
        self.renderer.text(
            "Game speed changes logical ticks; rendering remains 60 FPS.",
            (self.WIDTH // 2, 205),
            font=self.renderer.font_small,
            color=self.renderer.theme.muted_text,
        )
        speed_panel = pygame.Rect(self.WIDTH // 2 - 190, 255, 285, 100)
        self.renderer.panel(speed_panel)
        self.renderer.text(
            f"GAME SPEED  {self.settings.tick_rate}",
            speed_panel.center,
            font=self.renderer.font_button,
        )
        self.renderer.text(
            "UP / DOWN KEYS TO ADJUST",
            (self.WIDTH // 2, 375),
            font=self.renderer.font_small,
            color=self.renderer.theme.muted_text,
        )
        for button in self._buttons():
            button.draw(self.screen, self.renderer.font_button, self.renderer.theme)

    def _draw_loading(self) -> None:
        import pygame

        self._draw_menu()
        veil = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        veil.fill((4, 9, 18, 190))
        self.screen.blit(veil, (0, 0))
        self.renderer.panel(pygame.Rect(self.WIDTH // 2 - 220, 265, 440, 180))
        dots = "." * (int(self.elapsed * 3) % 4)
        self.renderer.text(
            f"LOADING AI{dots}",
            (self.WIDTH // 2, 335),
            font=self.renderer.font_button,
            color=self.renderer.theme.ai,
        )
        self.renderer.text(
            "ESC TO RETURN",
            (self.WIDTH // 2, 390),
            font=self.renderer.font_small,
            color=self.renderer.theme.muted_text,
        )

    def _draw_rules(self) -> None:
        import pygame

        self.renderer.text("GAME RULES", (self.WIDTH // 2, 70), font=self.renderer.font_title)
        self.renderer.panel(pygame.Rect(190, 115, 740, 480))
        rules = [
            "BOARD",
            "6 x 6 grid  |  Initial length 3  |  Full score 33",
            "A run ends on collision, board completion, or 400 steps.",
            "",
            "HUMAN VS AI",
            "Both boards advance once per logical tick.",
            "The first side to reach full score wins; collision loses.",
            "At 400 steps, higher score wins; earlier final score breaks a tie.",
            "Food uses the same seeded candidate order for each score index.",
            "",
            "CONTROLS",
            "Arrow keys / WASD: turn     P / Esc: pause",
            "Direct reversal is ignored. Human games start with a countdown.",
            "HUD shows SCORE and STEPS only. Starvation is disabled.",
        ]
        headings = {"BOARD", "HUMAN VS AI", "CONTROLS"}
        y = 150
        for line in rules:
            if not line:
                y += 10
                continue
            self.renderer.text(
                line,
                (self.WIDTH // 2, y),
                font=(self.renderer.font_body if line in headings else self.renderer.font_small),
                color=(
                    self.renderer.theme.player if line in headings else self.renderer.theme.text
                ),
            )
            y += 31
        for button in self._buttons():
            button.draw(self.screen, self.renderer.font_button, self.renderer.theme)

    def _draw_game(self, dt: float) -> None:
        import pygame

        if self.active_mode is None or self.mode_name is None:
            raise RuntimeError("A gameplay scene requires an active mode")
        interval = 1.0 / self.settings.tick_rate
        alpha = min(self.accumulator / interval, 1.0)
        title_by_mode = {
            ModeName.SOLO: "SOLO",
            ModeName.AI_VIEWER: "AI VIEWER",
            ModeName.RACE: "HUMAN VS AI",
        }
        self.renderer.text(
            title_by_mode[self.mode_name],
            (self.WIDTH // 2, 42),
            font=self.renderer.font_button,
        )
        self.renderer.text(
            "P / ESC  PAUSE",
            (self.WIDTH - 105, 42),
            font=self.renderer.font_small,
            color=self.renderer.theme.muted_text,
        )

        if isinstance(self.active_mode, RaceMode):
            player = self.active_mode.player_session
            ai = self.active_mode.ai_session
            self.renderer.draw_board(
                "race-player",
                player.previous_snapshot,
                player.snapshot,
                pygame.Rect(90, 145, 380, 380),
                accent=self.renderer.theme.player,
                label="YOU",
                alpha=alpha,
                elapsed=self.elapsed,
                dt=dt,
            )
            self.renderer.draw_board(
                "race-ai",
                ai.previous_snapshot,
                ai.snapshot,
                pygame.Rect(650, 145, 380, 380),
                accent=self.renderer.theme.ai,
                label="AI",
                subtitle=f"AI ID  {self.active_mode.ai_id}",
                alpha=alpha,
                elapsed=self.elapsed,
                dt=dt,
            )
        else:
            session = self.active_mode.session
            accent = (
                self.renderer.theme.ai
                if isinstance(self.active_mode, AIViewerMode)
                else self.renderer.theme.player
            )
            label = "AI" if isinstance(self.active_mode, AIViewerMode) else "YOU"
            self.renderer.draw_board(
                "single",
                session.previous_snapshot,
                session.snapshot,
                pygame.Rect(340, 135, 420, 420),
                accent=accent,
                label=label,
                subtitle=(
                    f"AI ID  {self.active_mode.ai_id}"
                    if isinstance(self.active_mode, AIViewerMode)
                    else None
                ),
                alpha=alpha,
                elapsed=self.elapsed,
                dt=dt,
            )

        if self.countdown > 0.0:
            self.renderer.text(
                str(max(int(self.countdown) + 1, 1)),
                (self.WIDTH // 2, self.HEIGHT // 2),
                font=self.renderer.font_result,
                color=self.renderer.theme.warning,
            )

    def _draw_game_over(self) -> None:
        import pygame

        veil = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        veil.fill((4, 9, 18, 150))
        self.screen.blit(veil, (0, 0))
        self.renderer.text(
            "GAME OVER",
            (self.WIDTH // 2, self.HEIGHT // 2),
            font=self.renderer.font_result,
            color=self.renderer.theme.warning,
        )

    def _draw_overlay(self, title: str) -> None:
        import pygame

        veil = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        veil.fill((4, 9, 18, 190))
        self.screen.blit(veil, (0, 0))
        panel = pygame.Rect(self.WIDTH // 2 - 250, 165, 500, 390)
        self.renderer.panel(panel)
        self.renderer.text(
            title,
            (self.WIDTH // 2, 245),
            font=self.renderer.font_result,
            color=self.renderer.theme.player,
        )
        if self.active_mode is not None:
            scores = self._scores()
            if len(scores) == 1:
                session = self.active_mode.session
                summary = f"SCORE  {scores[0]}     STEPS  {session.snapshot.steps}"
            else:
                race = self.active_mode
                summary = (
                    f"YOU  SCORE {scores[0]}  STEPS {race.player_session.snapshot.steps}     "
                    f"AI  SCORE {scores[1]}  STEPS {race.ai_session.snapshot.steps}"
                )
            self.renderer.text(summary, (self.WIDTH // 2, 300), font=self.renderer.font_small)
            if self.scene == AppScene.RESULT:
                self.renderer.text(
                    f"REASON  {self.active_mode.result_reason_text}",
                    (self.WIDTH // 2, 345),
                    font=self.renderer.font_small,
                    color=self.renderer.theme.muted_text,
                )
        for button in self._buttons():
            button.draw(self.screen, self.renderer.font_button, self.renderer.theme)

    def _buttons(self) -> list[Button]:
        pygame = self.pygame
        center_x = self.WIDTH // 2
        if self.scene == AppScene.MENU:
            entries = [
                ("PLAY SOLO", "solo"),
                ("WATCH AI", "ai"),
                ("HUMAN VS AI", "race"),
                ("RULES", "rules"),
                ("SETTINGS", "settings"),
                ("QUIT", "quit"),
            ]
            return [
                Button(pygame.Rect(center_x - 170, 190 + index * 66, 340, 50), label, action)
                for index, (label, action) in enumerate(entries)
            ]
        if self.scene == AppScene.SETTINGS:
            sound = "ON" if self.settings.sound_enabled else "OFF"
            return [
                Button(
                    pygame.Rect(center_x + 110, 255, 80, 47),
                    "+",
                    "speed_up",
                ),
                Button(
                    pygame.Rect(center_x + 110, 308, 80, 47),
                    "-",
                    "speed_down",
                ),
                Button(
                    pygame.Rect(center_x - 190, 405, 380, 58),
                    f"SOUND  {sound}",
                    "sound",
                ),
                Button(pygame.Rect(center_x - 190, 500, 380, 58), "BACK", "back"),
            ]
        if self.scene == AppScene.PAUSED:
            return [
                Button(pygame.Rect(center_x - 150, 320, 300, 48), "RESUME", "resume"),
                Button(pygame.Rect(center_x - 150, 378, 300, 48), "RULES", "rules"),
                Button(pygame.Rect(center_x - 150, 436, 300, 48), "RESTART", "restart"),
                Button(pygame.Rect(center_x - 150, 494, 300, 48), "MAIN MENU", "back"),
            ]
        if self.scene == AppScene.RULES:
            return [Button(pygame.Rect(center_x - 140, 625, 280, 52), "BACK", "rules_back")]
        if self.scene == AppScene.RESULT:
            return [
                Button(pygame.Rect(center_x - 150, 390, 300, 52), "PLAY AGAIN", "restart"),
                Button(pygame.Rect(center_x - 150, 460, 300, 52), "MAIN MENU", "back"),
            ]
        return []


def main() -> None:
    SnakeGameApp().run()


if __name__ == "__main__":
    main()
