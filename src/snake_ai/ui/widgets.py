from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from snake_ai.ui.theme import Theme


@dataclass
class Button:
    rect: Any
    label: str
    action: str
    enabled: bool = True

    def clicked(self, event: Any) -> bool:
        import pygame

        return bool(
            self.enabled
            and event.type == pygame.MOUSEBUTTONUP
            and event.button == 1
            and self.rect.collidepoint(event.pos)
        )

    def draw(self, screen: Any, font: Any, theme: Theme) -> None:
        import pygame

        hovered = self.enabled and self.rect.collidepoint(pygame.mouse.get_pos())
        color = theme.panel_hover if hovered else theme.panel
        border = theme.player if hovered else theme.border
        pygame.draw.rect(screen, color, self.rect, border_radius=12)
        pygame.draw.rect(screen, border, self.rect, width=2, border_radius=12)
        text_color = theme.text if self.enabled else theme.muted_text
        label = font.render(self.label, True, text_color)
        screen.blit(label, label.get_rect(center=self.rect.center))
