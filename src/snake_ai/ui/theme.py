from __future__ import annotations

from dataclasses import dataclass

Color = tuple[int, int, int]


@dataclass(frozen=True)
class Theme:
    background: Color = (9, 17, 31)
    panel: Color = (17, 29, 46)
    panel_hover: Color = (28, 46, 70)
    grid: Color = (35, 55, 76)
    border: Color = (58, 82, 106)
    player: Color = (53, 230, 193)
    ai: Color = (169, 112, 255)
    food: Color = (255, 100, 124)
    warning: Color = (255, 181, 71)
    text: Color = (238, 244, 255)
    muted_text: Color = (145, 164, 184)
