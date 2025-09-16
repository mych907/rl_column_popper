from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RewardPreset:
    step_cost: float = -0.01
    valid_action: float = 1.0
    pop_cell: float = 3.0
    overflow: float = -1.0
    invalid_full_drop: float = -1.0
    time_up: float = -0.5
    manual_fall_bonus: float = 1.0


def get_preset(name: str | None) -> RewardPreset:
    # Only default for now; placeholder for future variants
    return RewardPreset()
