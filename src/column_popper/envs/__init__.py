import gymnasium as gym

from .column_popper_env import ColumnPopperEnv

_ENV_ID = "SpecKitAI/ColumnPopper-v1"

try:
    # Register once; ignore if already registered
    gym.spec(_ENV_ID)
except gym.error.Error:
    gym.register(
        id=_ENV_ID,
        entry_point="column_popper.envs.column_popper_env:ColumnPopperEnv",
    )

__all__ = ["ColumnPopperEnv"]
