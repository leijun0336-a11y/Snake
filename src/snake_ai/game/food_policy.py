from __future__ import annotations

import hashlib
import random
from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from typing import Protocol, TypeVar

CellT = TypeVar("CellT", bound=Hashable)


class FoodPolicy(Protocol[CellT]):
    """Select a legal food cell without owning environment state."""

    def choose(
        self,
        all_cells: Sequence[CellT],
        available_cells: Sequence[CellT],
        *,
        food_index: int,
        rng: random.Random,
    ) -> CellT: ...


@dataclass(frozen=True)
class RandomFoodPolicy:
    """Preserve the original uniformly random food placement."""

    def choose(
        self,
        all_cells: Sequence[CellT],
        available_cells: Sequence[CellT],
        *,
        food_index: int,
        rng: random.Random,
    ) -> CellT:
        del all_cells, food_index
        if not available_cells:
            raise ValueError("available_cells must not be empty")
        return rng.choice(available_cells)


@dataclass(frozen=True)
class SeededRaceFoodPolicy:
    """Use the same deterministic candidate order for each food index."""

    race_seed: int

    def choose(
        self,
        all_cells: Sequence[CellT],
        available_cells: Sequence[CellT],
        *,
        food_index: int,
        rng: random.Random,
    ) -> CellT:
        del rng
        if not available_cells:
            raise ValueError("available_cells must not be empty")

        payload = f"snake-race:{self.race_seed}:{food_index}".encode("ascii")
        local_seed = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")
        candidates = list(all_cells)
        random.Random(local_seed).shuffle(candidates)
        available = set(available_cells)
        return next(cell for cell in candidates if cell in available)
