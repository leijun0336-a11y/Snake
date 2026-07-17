from __future__ import annotations

import math

import numpy as np


class AudioManager:
    """Small procedural sound set with no external asset dependency."""

    SAMPLE_RATE = 44_100

    def __init__(self, enabled: bool = True) -> None:
        import pygame

        pygame.mixer.init(frequency=self.SAMPLE_RATE, size=-16, channels=2)
        self.enabled = enabled
        self._eat = self._tone(660.0, 0.07, 0.20)
        self._finish = self._tone(220.0, 0.22, 0.24)
        self._click = self._tone(440.0, 0.04, 0.12)

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = enabled

    def play_eat(self) -> None:
        self._play(self._eat)

    def play_finish(self) -> None:
        self._play(self._finish)

    def play_click(self) -> None:
        self._play(self._click)

    def _play(self, sound: object) -> None:
        if self.enabled:
            sound.play()

    def _tone(self, frequency: float, duration: float, volume: float):
        import pygame

        count = max(int(self.SAMPLE_RATE * duration), 1)
        times = np.arange(count, dtype=np.float64) / self.SAMPLE_RATE
        envelope = np.linspace(1.0, 0.0, count, dtype=np.float64)
        waveform = np.sin(2.0 * math.pi * frequency * times) * envelope * volume
        mono = np.asarray(waveform * np.iinfo(np.int16).max, dtype=np.int16)
        stereo = np.column_stack((mono, mono))
        return pygame.sndarray.make_sound(np.ascontiguousarray(stereo))
