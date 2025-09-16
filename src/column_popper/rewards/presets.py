from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RewardPreset:
    # Small step penalty encourages purpose, not spam
    step_cost: float = -0.01
    # Do not reward arbitrary button presses; rely on pops
    valid_action: float = 0.0
    # Reward per popped cell (triple → 9)
    pop_cell: float = 3.0
    # Penalize overflow so reckless falling is discouraged
    overflow: float = -3.0
    invalid_full_drop: float = -1.0
    time_up: float = -1.5
    # Small intrinsic bonus for manual fall; less than time_up penalty
    manual_fall_bonus: float = 0.1


def get_preset(name: str | None) -> RewardPreset:
    # Only default for now; placeholder for future variants
    return RewardPreset()
