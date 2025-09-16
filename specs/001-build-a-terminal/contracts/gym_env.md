# Gymnasium Environment Contract

This document defines the contract for the Column Popper Gymnasium environment, aligned with the Constitution.

## Environment Name

`SpecKitAI/ColumnPopper-v1`

## Observation Space

The observation space is a `gym.spaces.Dict` with the following keys:

- `board`: `gym.spaces.Box(low=0, high=9, shape=(12, 3), dtype=np.int32)` where 0 = empty and values come from `number_pool` (default [1,2,3]).
- `selection`: `gym.spaces.Box(low=0, high=9, shape=(2,), dtype=np.int32)` interpreted as `[is_selected, value]`.
- Optional (feature-flagged): `time_left_norm`: `gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)`.

## Action Space

`gym.spaces.Discrete(4)` with the following semantics:

- `0, 1, 2`: Select/release on the target column index (toggle pick/place depending on selection state and column occupancy).
- `3`: Manual fall (advance time by one row according to the schedule).

Invalid actions are treated as no-ops with a small penalty by default. When `strict_invalid=True`, an invalid full-column drop terminates the episode with a fault code.

## Reward

Default reward preset (constitutional):
- `+1` per valid player action
- `+3` per popped cell (i.e., `+9` for a triple)
- `−0.01` step cost each step
- `−1` on overflow or invalid full-column drop
- `−0.5` on time-up truncation

Alternate reward presets may be offered behind explicit configuration.

Reward range: `(-inf, inf)`

## Initial State & Determinism

- Board shape is fixed at 12×3. The generator enforces a per-column constraint: a newly spawned top cell avoids creating an immediate three-in-a-column from two identical cells directly beneath.
- With a fixed seed and action sequence, the observation sequence MUST be identical.

## Termination & Truncation

Termination conditions:
- Overflow (a new number spawns with no available space in its column)
- Invalid drop when `strict_invalid=True`

Truncation condition:
- Timer reaches zero (`GAME_DURATION`, default 60s)

## Info Keys

The `info` dict MUST include at least:
- `score`: current score
- `time_left`: time remaining in seconds
- `pops_this_step`: number of cells popped in the last step
- `fall_interval`: current fall interval per the schedule
- `seed`: environment seed
- `version`: observation schema/environment version tag
