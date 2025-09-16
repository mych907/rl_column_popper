# Research: Column Popper Game

This document addresses the open questions from the feature specification.

## Scoring Logic

**Decision**: Adopt constitutional defaults. Score gains are driven by the reward function: +1 per valid action, +3 per popped cell (i.e., +9 for a triple), with a −0.01 step cost; penalties are −1 on overflow/invalid full-column drop and −0.5 on time-up truncation.

**Rationale**: Aligns with the Constitution’s fairness/safety and determinism goals, balances sparse pop rewards with action shaping, and provides consistent signals for RL baselines.

**Alternatives considered**:
- Fixed +10 per pop only (no shaping). Rejected for weaker learning signal and divergence from constitutional defaults.
- Value/Combo-based scoring. Kept as a future optional preset to avoid early complexity and preserve reproducibility.

## Default Time Limit

**Decision**: Default `GAME_DURATION = 60s`, configurable via CLI and API. Fall rate follows a schedule; manual fall action is supported.

**Rationale**: Matches the Constitution and ensures short, reproducible episodes suitable for CI and benchmarking. Configurability supports curricula and human variability.

**Alternatives considered**:
- 180s default. Rejected to keep runs short and aligned with CI performance budgets.
- No time limit. Conflicts with evaluation fairness and truncation requirements.

## Observation Format

**Decision**: The observation is a Dict with:
- `board`: int32 array shape (8,3), 0=empty, values in `number_pool` (default [1,2,3])
- `selection`: int32 array shape (2,) → `[is_selected, value]`
- Optional (feature-gated): `time_left_norm` in [0,1]

**Rationale**: Aligns exactly with the Constitution; preserves spatial locality, minimal feature set, and backwards-compatible extension via optional flags.

**Alternatives considered**:
- `held_number` and absolute `time_remaining` fields. Replaced with `selection` vector and optional normalized time to match constitutional contract and testing needs.

## Reward Structure

**Decision**: Use constitutional default reward preset: `+1` per valid action, `+3` per popped cell, `−0.01` step cost, `−1` overflow/invalid full-column drop, `−0.5` time-up truncation. Provide additional presets behind explicit configuration.

**Rationale**: Encourages meaningful interaction, rewards clearing, discourages stalling and unsafe actions, and is proven suitable for RL benchmarking per Constitution.

**Alternatives considered**:
- Pure sparse rewards (e.g., +10 pop only). Deferred as optional preset.
- Heavier penalties (e.g., −100 overflow). Not aligned with constitutional calibration and can destabilize early learning.
