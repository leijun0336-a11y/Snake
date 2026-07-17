from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

from snake_ai.game.session import GameSnapshot
from snake_ai.game.snake_env import Direction, Point
from snake_ai.ui.theme import Color, Theme


@dataclass
class Particle:
    x: float
    y: float
    vx: float
    vy: float
    life: float
    max_life: float


class GameRenderer:
    def __init__(self, screen: Any, theme: Theme | None = None) -> None:
        import pygame

        self.screen = screen
        self.theme = theme or Theme()
        self.font_small = pygame.font.Font(None, 24)
        self.font_body = pygame.font.Font(None, 30)
        self.font_button = pygame.font.Font(None, 34)
        self.font_title = pygame.font.Font(None, 64)
        self.font_result = pygame.font.Font(None, 78)
        self._particles: dict[str, list[Particle]] = {}
        self._last_scores: dict[str, int] = {}
        self._random = random.Random(42)

    def clear(self) -> None:
        self.screen.fill(self.theme.background)
        self._draw_background()

    def text(
        self,
        value: str,
        center: tuple[int, int],
        *,
        font: Any | None = None,
        color: Color | None = None,
    ) -> None:
        rendered = (font or self.font_body).render(value, True, color or self.theme.text)
        self.screen.blit(rendered, rendered.get_rect(center=center))

    def panel(self, rect: Any) -> None:
        import pygame

        pygame.draw.rect(self.screen, self.theme.panel, rect, border_radius=16)
        pygame.draw.rect(self.screen, self.theme.border, rect, width=2, border_radius=16)

    def draw_board(
        self,
        key: str,
        previous: GameSnapshot,
        current: GameSnapshot,
        rect: Any,
        *,
        accent: Color,
        label: str,
        alpha: float,
        elapsed: float,
        dt: float,
    ) -> None:
        import pygame

        board_size = min(rect.width, rect.height)
        cell_size = board_size / 6.0
        board = pygame.Rect(rect.x, rect.y, board_size, board_size)
        self.panel(board.inflate(20, 20))
        pygame.draw.rect(self.screen, (12, 24, 38), board, border_radius=10)
        for index in range(7):
            offset = round(index * cell_size)
            pygame.draw.line(
                self.screen,
                self.theme.grid,
                (board.left + offset, board.top),
                (board.left + offset, board.bottom),
            )
            pygame.draw.line(
                self.screen,
                self.theme.grid,
                (board.left, board.top + offset),
                (board.right, board.top + offset),
            )

        self._maybe_spawn_food_particles(key, previous, current, board, cell_size)
        self._draw_food(current.food, board, cell_size, elapsed)
        self._draw_snake(previous, current, board, cell_size, accent, alpha)
        self._update_and_draw_particles(key, dt)

        hud_y = board.bottom + 34
        self.text(label, (board.centerx, board.top - 34), color=accent)
        self.text(
            f"SCORE  {current.score}     STEPS  {current.steps}",
            (board.centerx, hud_y),
            font=self.font_small,
        )

    def _draw_food(self, food: Point, board: Any, cell_size: float, elapsed: float) -> None:
        import pygame

        center = self._cell_center(food, board, cell_size)
        radius = max(int(cell_size * (0.18 + 0.025 * math.sin(elapsed * 5.0))), 4)
        glow = pygame.Surface((radius * 6, radius * 6), pygame.SRCALPHA)
        glow_center = (radius * 3, radius * 3)
        pygame.draw.circle(glow, (*self.theme.food, 35), glow_center, radius * 3)
        pygame.draw.circle(glow, (*self.theme.food, 80), glow_center, radius * 2)
        self.screen.blit(glow, glow.get_rect(center=center))
        pygame.draw.circle(self.screen, self.theme.food, center, radius)

    def _draw_snake(
        self,
        previous: GameSnapshot,
        current: GameSnapshot,
        board: Any,
        cell_size: float,
        accent: Color,
        alpha: float,
    ) -> None:
        import pygame

        alpha = max(0.0, min(alpha, 1.0))
        previous_snake = previous.snake or current.snake
        inset = max(int(cell_size * 0.12), 3)
        for index in range(len(current.snake) - 1, -1, -1):
            current_point = current.snake[index]
            old_point = previous_snake[min(index, len(previous_snake) - 1)]
            old_x, old_y = self._cell_origin(old_point, board, cell_size)
            new_x, new_y = self._cell_origin(current_point, board, cell_size)
            x = old_x + (new_x - old_x) * alpha
            y = old_y + (new_y - old_y) * alpha
            segment = pygame.Rect(
                round(x) + inset,
                round(y) + inset,
                round(cell_size) - inset * 2,
                round(cell_size) - inset * 2,
            )
            shade = 1.0 if index == 0 else max(0.58, 0.88 - index * 0.025)
            color = tuple(min(round(channel * shade), 255) for channel in accent)
            pygame.draw.rect(self.screen, color, segment, border_radius=max(6, inset))
            if index == 0:
                self._draw_eyes(segment, current.direction)

    def _draw_eyes(self, head: Any, direction: Direction) -> None:
        import pygame

        cx, cy = head.center
        spread = max(head.width // 5, 2)
        forward = max(head.width // 5, 2)
        if direction in (Direction.RIGHT, Direction.LEFT):
            sign = 1 if direction == Direction.RIGHT else -1
            points = [(cx + sign * forward, cy - spread), (cx + sign * forward, cy + spread)]
        else:
            sign = 1 if direction == Direction.DOWN else -1
            points = [(cx - spread, cy + sign * forward), (cx + spread, cy + sign * forward)]
        for point in points:
            pygame.draw.circle(self.screen, self.theme.background, point, max(head.width // 12, 2))

    def _maybe_spawn_food_particles(
        self,
        key: str,
        previous: GameSnapshot,
        current: GameSnapshot,
        board: Any,
        cell_size: float,
    ) -> None:
        last_score = self._last_scores.get(key, previous.score)
        if current.score < last_score:
            self._particles.pop(key, None)
        if current.score > last_score:
            x, y = self._cell_center(previous.food, board, cell_size)
            particles = self._particles.setdefault(key, [])
            for _ in range(12):
                angle = self._random.random() * math.tau
                speed = self._random.uniform(45.0, 100.0)
                particles.append(
                    Particle(
                        float(x),
                        float(y),
                        math.cos(angle) * speed,
                        math.sin(angle) * speed,
                        0.42,
                        0.42,
                    )
                )
        self._last_scores[key] = current.score

    def _update_and_draw_particles(self, key: str, dt: float) -> None:
        import pygame

        alive: list[Particle] = []
        for particle in self._particles.get(key, []):
            particle.life -= dt
            if particle.life <= 0.0:
                continue
            particle.x += particle.vx * dt
            particle.y += particle.vy * dt
            radius = max(round(4 * particle.life / particle.max_life), 1)
            pygame.draw.circle(
                self.screen, self.theme.food, (round(particle.x), round(particle.y)), radius
            )
            alive.append(particle)
        self._particles[key] = alive

    @staticmethod
    def _cell_origin(point: Point, board: Any, cell_size: float) -> tuple[float, float]:
        return board.left + point.x * cell_size, board.top + point.y * cell_size

    @classmethod
    def _cell_center(cls, point: Point, board: Any, cell_size: float) -> tuple[int, int]:
        x, y = cls._cell_origin(point, board, cell_size)
        return round(x + cell_size / 2), round(y + cell_size / 2)

    def _draw_background(self) -> None:
        import pygame

        width, height = self.screen.get_size()
        for x in range(0, width, 48):
            pygame.draw.line(self.screen, (12, 25, 42), (x, 0), (x - height // 3, height))
