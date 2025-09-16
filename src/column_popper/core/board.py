from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass
class Board:
    height: int = 12
    width: int = 3
    number_pool: Sequence[int] = (1, 2, 3)
    seed: int | None = None

    def __post_init__(self) -> None:
        self.rng = np.random.Generator(np.random.PCG64(self.seed))
        self.grid = np.zeros((self.height, self.width), dtype=np.int32)

    # Core utilities for tests
    def pop_triples_in_column(self, col: int) -> int:
        """Remove vertical triples in the specified column. Returns popped cell count.

        Only contiguous sequences of exactly three identical non-zero values are removed.
        If multiple disjoint triples exist, remove all of them.
        """
        assert 0 <= col < self.width
        column = self.grid[:, col]
        popped = 0

        # Scan for any run-lengths of >=3 with equal values; remove triples greedily.
        i = 0
        while i <= self.height - 3:
            v = column[i]
            if v != 0 and column[i + 1] == v and column[i + 2] == v:
                # Pop exactly three here
                column[i : i + 3] = 0
                popped += 3
                # After popping, do not compress here; compression/gravity is driven by game tick
                i += 3
            else:
                i += 1

        self.grid[:, col] = column
        return popped

    def spawn_value_for_column(self, col: int) -> int:
        """Sample a spawn value avoiding an immediate triple on the top three cells.

        This function is called AFTER the column has been shifted down by one during
        a fall tick. The new value will be placed at row 0. To avoid creating an
        immediate vertical triple at rows [0,1,2], we check the two directly beneath
        (rows 1 and 2). If they are identical and non-zero, avoid that value.
        """
        assert 0 <= col < self.width
        pool = list(self.number_pool)
        column = self.grid[:, col]

        avoid = None
        if self.height >= 3:
            a, b = int(column[1]), int(column[2])
            if a != 0 and a == b:
                avoid = a

        if avoid is not None and avoid in pool and len(pool) > 1:
            pool = [p for p in pool if p != avoid]

        return int(self.rng.choice(pool))
