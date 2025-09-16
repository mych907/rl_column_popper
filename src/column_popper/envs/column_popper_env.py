from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..core.board import Board
from ..core.schedule import Schedule
from ..rewards.presets import RewardPreset, get_preset
from ..version import __version__ as PKG_VERSION


class ColumnPopperEnv(gym.Env[dict[str, Any], int]):
    metadata = {"render_modes": ["ansi"], "render_fps": 30}

    def __init__(
        self,
        *,
        seed: int | None = None,
        game_duration: float = 60.0,
        strict_invalid: bool = False,
        include_time_left_norm: bool = False,
        reward_preset: RewardPreset | None = None,
        use_wall_time: bool = False,
        initial_fall_interval: float = 3.0,
        schedule_curve: list[tuple[float, float]] | None = None,
    ) -> None:
        super().__init__()
        self._seed = seed
        self.strict_invalid = strict_invalid
        self.game_duration = float(game_duration)
        self.include_time_left_norm = include_time_left_norm
        self.rewards = reward_preset or get_preset("default")
        self.use_wall_time = use_wall_time
        self._initial_fall_interval = float(initial_fall_interval)
        # Default ramp: 3s -> 2s at 20s, then 1s at 40s
        self._schedule_curve = schedule_curve or [(20.0, 2.0), (40.0, 1.0)]

        self.board = Board(seed=seed)
        self.selection = np.zeros((2,), dtype=np.int32)  # [is_selected, value]
        self.score = 0.0
        self.schedule = Schedule(
            game_duration=self.game_duration,
            initial_interval=self._initial_fall_interval,
            curve=list(self._schedule_curve),
        )
        self._last_wall_time = 0.0
        self._terminated = False
        self._sel_col = -1
        self._sel_row = -1

        # Observation: Dict(board:int32[8,3], selection:int32[2], optional time_left_norm)
        obs_spaces: dict[str, spaces.Space[Any]] = {
            "board": spaces.Box(low=0, high=9, shape=(12, 3), dtype=np.int32),
            "selection": spaces.Box(low=0, high=9, shape=(2,), dtype=np.int32),
            # Selected position for renderers/policies: [row, col], -1 indicates none
            "sel_pos": spaces.Box(
                low=np.array([-1, -1], dtype=np.int32),
                high=np.array([self.board.height - 1, self.board.width - 1], dtype=np.int32),
                dtype=np.int32,
            ),
        }
        if include_time_left_norm:
            obs_spaces["time_left_norm"] = spaces.Box(
                low=0.0, high=1.0, shape=(1,), dtype=np.float32
            )
        self.observation_space = spaces.Dict(obs_spaces)
        self.action_space = spaces.Discrete(4)  # type: ignore[assignment]

        # RNG for any stochasticity (kept minimal here)
        self._rng = np.random.Generator(np.random.PCG64(seed))

    # Gym API
    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if seed is not None:
            self._seed = seed
            self._rng = np.random.Generator(np.random.PCG64(seed))
        # Reset game state
        self.board = Board(seed=self._seed)
        self.selection = np.zeros((2,), dtype=np.int32)
        self.score = 0.0
        self.schedule = Schedule(
            game_duration=self.game_duration,
            initial_interval=self._initial_fall_interval,
            curve=list(self._schedule_curve),
        )
        if self.use_wall_time:
            import time

            self._last_wall_time = time.perf_counter()
        self._terminated = False
        self._sel_col = -1
        self._sel_row = -1
        # Ensure first row is visible
        self._fall_tick()
        return self._obs(), self._info(pops_this_step=0)

    def step(self, action: int) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:  # noqa: C901
        assert self.action_space.contains(action)

        reward = 0.0
        reward += self.rewards.step_cost

        terminated = False
        truncated = False
        pops = 0

        # Actions 0,1,2 operate on columns
        if action in (0, 1, 2):
            col = int(action)
            is_sel, value = int(self.selection[0]), int(self.selection[1])
            column = self.board.grid[:, col]

            if is_sel == 0:
                # pick bottom-most existing number if any (do not remove yet)
                nz = np.nonzero(column)[0]
                if nz.size > 0:
                    bottom_idx = nz[-1]
                    val = int(column[bottom_idx])
                    self.selection[:] = (1, val)
                    self._sel_col = col
                    self._sel_row = int(bottom_idx)
                    reward += self.rewards.valid_action
                else:
                    # invalid (empty column pick) – already has step cost applied
                    pass
            else:
                # drop into target column: remove from source pos then place
                src_c = self._sel_col
                src_r = self._sel_row
                # If same column, remove first to compute correct top-empty
                if col == src_c and 0 <= src_r < self.board.height:
                    self.board.grid[src_r, src_c] = 0
                    column = self.board.grid[:, col]
                zeros = np.where(column == 0)[0]
                if zeros.size > 0:
                    top_empty = int(zeros[0])
                    column[top_empty] = value
                    # If different column, remove after placement
                    if col != src_c and 0 <= src_r < self.board.height and 0 <= src_c < self.board.width:
                        self.board.grid[src_r, src_c] = 0
                    self.selection[:] = (0, 0)
                    self._sel_col = -1
                    self._sel_row = -1
                    reward += self.rewards.valid_action
                    pops = self.board.pop_triples_in_column(col)
                    if pops:
                        reward += self.rewards.pop_cell * pops
                        self.score += self.rewards.pop_cell * pops
                else:
                    # invalid full-column drop
                    reward += self.rewards.invalid_full_drop
                    if self.strict_invalid:
                        terminated = True
            self.board.grid[:, col] = column

        elif action == 3:
            # Manual fall – valid action plus bonus; apply one fall tick only
            reward += self.rewards.valid_action
            reward += self.rewards.manual_fall_bonus
            self.score += self.rewards.manual_fall_bonus

        # Advance time and handle falling
        if self.use_wall_time:
            import time

            now = time.perf_counter()
            dt = max(0.0, now - self._last_wall_time)
            self._last_wall_time = now
        else:
            dt = 1.0
        # In non-wall-time mode, model action duration as 0.1s per action
        if not self.use_wall_time:
            dt = 0.1
        falls = self.schedule.advance_step(dt=dt)

        if action == 3:
            # Manual fall: ignore scheduled falls this step; apply exactly one row fall
            if self._fall_tick():
                reward += self.rewards.overflow
                terminated = True
                self._terminated = True
        else:
            # Apply scheduled falls for this step
            for _ in range(falls):
                if self._fall_tick():
                    reward += self.rewards.overflow
                    terminated = True
                    self._terminated = True
                    break

        # Truncation check
        if self.schedule.truncated and not terminated:
            truncated = True
            reward += self.rewards.time_up

        obs = self._obs()
        info = self._info(pops_this_step=pops)
        return obs, float(reward), bool(terminated), bool(truncated), info

    # Rendering (ANSI minimal placeholder)
    def render(self) -> str:  # type: ignore[override]
        lines = []
        for r in range(self.board.height):
            row = self.board.grid[r, :]
            lines.append(" ".join(str(int(x)) for x in row))
        return "\n".join(lines)

    # Helpers
    def _obs(self) -> dict[str, Any]:
        obs: dict[str, Any] = {
            "board": self.board.grid.astype(np.int32, copy=False),
            "selection": self.selection.astype(np.int32, copy=False),
        }
        if self.include_time_left_norm:
            norm = np.array(
                [max(0.0, min(1.0, self.schedule.time_left / self.game_duration))],
                dtype=np.float32,
            )
            obs["time_left_norm"] = norm
        # Include selected position for renderers
        obs["sel_pos"] = np.array([self._sel_row, self._sel_col], dtype=np.int32)
        return obs

    def _info(self, *, pops_this_step: int) -> dict[str, Any]:
        return {
            "score": float(self.score),
            "time_left": float(max(0.0, self.schedule.time_left)),
            "pops_this_step": int(pops_this_step),
            "fall_interval": float(self.schedule.fall_interval),
            "seed": int(self._seed) if self._seed is not None else None,
            "version": PKG_VERSION,
        }

    # Wall-time advancement for human UI
    def wall_time_tick(self) -> None:
        if not self.use_wall_time:
            return
        import time

        now = time.perf_counter()
        dt = max(0.0, now - self._last_wall_time)
        self._last_wall_time = now
        falls = self.schedule.advance_step(dt=dt)
        for _ in range(falls):
            if self._fall_tick():
                self._terminated = True
                break

    # Peek current observation and info without stepping
    def peek(self) -> tuple[dict[str, Any], dict[str, Any]]:
        return self._obs(), self._info(pops_this_step=0)

    # Mechanics
    def _fall_tick(self) -> bool:
        """Advance the board by one falling tick.

        For each column, shift all cells down by one row and spawn a new value at the top
        using the board's spawn generator with the per-column constraint. If the bottom-most
        cell is already occupied before shifting, this tick causes an overflow.

        Returns True if overflow occurred.
        """
        overflow = False
        for col in range(self.board.width):
            column = self.board.grid[:, col]
            if column[-1] != 0:
                overflow = True
            # shift down by one
            column[1:] = column[:-1]
            # spawn at top with constraint
            column[0] = self.board.spawn_value_for_column(col)
            # update selection row when in this column
            if self._sel_col == col and self._sel_row >= 0:
                self._sel_row = min(self._sel_row + 1, self.board.height - 1)
            self.board.grid[:, col] = column
        return overflow
