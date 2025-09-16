from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class Schedule:
    """Simple falling schedule and timer.

    - Advances time in fixed dt steps (default 1.0 per env.step call)
    - Emits automatic fall events based on `fall_interval`
    - Supports changing interval over elapsed time via `curve`
    """

    game_duration: float = 60.0
    initial_interval: float = 1.0
    curve: List[Tuple[float, float]] = field(default_factory=list)
    # curve items are (elapsed_time_threshold, new_interval), applied when elapsed >= threshold

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.elapsed: float = 0.0
        self.time_left: float = float(self.game_duration)
        self.fall_interval: float = float(self.initial_interval)
        self._accum: float = 0.0

    def _update_interval(self) -> None:
        for t, interval in sorted(self.curve):
            if self.elapsed >= t:
                self.fall_interval = float(interval)

    def advance_step(self, dt: float = 1.0) -> int:
        """Advance time by dt and return the number of auto fall events to apply."""
        self.elapsed += dt
        self.time_left = max(0.0, self.game_duration - self.elapsed)
        self._update_interval()
        self._accum += dt

        falls = 0
        # Guard against zero or negative intervals
        interval = max(1e-6, float(self.fall_interval))
        while self._accum >= interval:
            self._accum -= interval
            falls += 1
        return falls

    @property
    def truncated(self) -> bool:
        return self.time_left <= 0.0

