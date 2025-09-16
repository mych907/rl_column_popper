# Column Popper

Column Popper is a terminal-playable number puzzle that also functions as a Gymnasium-compatible reinforcement learning environment. Numbers fall into three columns on a schedule; you pick up from a column and drop into another to make vertical triples that pop for points. The game ends on overflow or when time runs out. The environment is deterministic when seeded.

- Env ID: `SpecKitAI/ColumnPopper-v1`
- Board: `8 x 3` (int32, `0` = empty, values from `[1,2,3]`)
- Actions: `Discrete(4)` → `0,1,2` select/drop on target column; `3` manual fall
- Observation: `Dict(board:int32[8,3], selection:int32[2], optional time_left_norm:float32[1])`

## Install

```bash
pip install -e .[dev]
```

On Windows, you may also want `windows-curses` for the curses UI.

## Play (Terminal UI)

```bash
# Auto-select curses UI with ANSI fallback
column_popper --mode=play --seed=42

# Force curses UI
column_popper --mode=play --ui=curses --seed=42

# Force ANSI UI
column_popper --mode=play --ui=ansi --seed=42
```

Controls: `a` = column 0, `s` = column 1, `d` = column 2, `f` = manual fall, `q` = quit.

## Headless Rollout (JSONL)

Provide actions on stdin (0–3), receive JSONL frames on stdout.

```bash
python - << 'PY'
import random
for _ in range(200):
    print(random.choice([0,1,2,3]))
PY
\
| column_popper --mode=rollout --episodes=5 --format=jsonl --seed=42
```

Each line includes: `episode`, `step`, `action`, `reward`, `terminated`, `truncated`, `info`, and `obs`.

## Streaming Protocol (Interactive JSONL)

```bash
column_popper --mode=stream --episodes=1 --seed=42
```

Sends `meta`, `reset`, `step_request` messages and expects one integer action per line. Emits `step_result` and `done`.

## Gym Usage

```python
import gymnasium as gym
import column_popper.envs  # Ensure env registration

env = gym.make("SpecKitAI/ColumnPopper-v1", seed=42)
obs, info = env.reset()
for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
env.close()
```

## Development

```bash
# Lint and type-check
ruff check .
mypy .

# Run tests
pytest -q
```

Project structure is defined in `specs/001-build-a-terminal/plan.md` and progress in `specs/001-build-a-terminal/tasks.md`.
