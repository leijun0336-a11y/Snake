from __future__ import annotations

from snake_ai.game.snake_env import Point


class SnakeRenderer:
    def __init__(self, width: int, height: int, cell_size: int = 24, fps: int = 30) -> None:
        import pygame

        self.pygame = pygame
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.fps = fps
        self.screen_width = width * cell_size
        self.screen_height = height * cell_size + 44

        pygame.init()
        pygame.display.set_caption("Snake AI")
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("arial", 24)

    def render(self, snake: list[Point], food: Point, score: int) -> None:
        pygame = self.pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                raise SystemExit

        self.screen.fill((20, 24, 28))
        self._draw_grid()
        self._draw_food(food)
        self._draw_snake(snake)

        score_text = self.font.render(f"Score: {score}", True, (235, 239, 244))
        self.screen.blit(score_text, (12, self.height * self.cell_size + 9))

        pygame.display.flip()
        self.clock.tick(self.fps)

    def close(self) -> None:
        self.pygame.quit()

    def _cell_rect(self, point: Point):
        return self.pygame.Rect(
            point.x * self.cell_size,
            point.y * self.cell_size,
            self.cell_size,
            self.cell_size,
        )

    def _draw_grid(self) -> None:
        pygame = self.pygame
        color = (35, 40, 46)
        for x in range(self.width + 1):
            px = x * self.cell_size
            pygame.draw.line(self.screen, color, (px, 0), (px, self.height * self.cell_size))
        for y in range(self.height + 1):
            py = y * self.cell_size
            pygame.draw.line(self.screen, color, (0, py), (self.screen_width, py))

    def _draw_food(self, food: Point) -> None:
        self.pygame.draw.rect(self.screen, (230, 70, 70), self._cell_rect(food).inflate(-6, -6))

    def _draw_snake(self, snake: list[Point]) -> None:
        for index, point in enumerate(snake):
            color = (80, 210, 120) if index == 0 else (55, 160, 95)
            self.pygame.draw.rect(self.screen, color, self._cell_rect(point).inflate(-4, -4))
