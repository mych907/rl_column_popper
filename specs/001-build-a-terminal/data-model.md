# Data Model: Column Popper

This document describes the key entities and data structures for the Column Popper game.

## Game Board

- **Description**: A fixed 8×3 grid representing the play area.
- **Fields**:
    - `grid`: A 2D NumPy array of dtype `int32`. `0` represents an empty space; non-zero values are from `number_pool`.
    - `width`: The number of columns (constant: `3`).
    - `height`: The number of rows (constant: `8`).
    - `number_pool`: Allowed piece values (default `[1,2,3]`).
- **Validation**:
    - The grid MUST be shape `(8, 3)` at all times.
    - No negative numbers or values outside `number_pool` may appear.
    - Pops only remove triples and only in the acted column.
    - Spawn generator enforces: a new top cell avoids creating an immediate three-in-a-column from two identical cells directly beneath (per-column constraint).

## Player/Agent

- **Description**: The actor interacting with the game.
- **Fields**:
    - `selection`: A pair `[is_selected, value]` where `is_selected ∈ {0,1}`, and `value ∈ {0} ∪ number_pool`.

## Number

- **Description**: The game pieces that fall and are moved.
- **Fields**:
    - `value`: An integer representing the number's value.

## Score

- **Description**: A numerical value representing the player's points.
- **Fields**:
    - `value`: An integer.

## Timer & Schedule

- **Description**: A countdown mechanism to limit the game duration.
- **Fields**:
    - `time_remaining`: A float representing the time remaining in seconds (default start: `60.0`).
    - `fall_interval`: Current fall interval per a schedule (may change over time).

## Game State

- **Description**: A container for all the game's data at a single point in time.
- **Fields**:
    - `board`: A `GameBoard` object.
    - `player`: A `Player` object.
    - `score`: A `Score` object.
    - `timer`: A `Timer & Schedule` object.
    - `is_game_over`: A boolean.
