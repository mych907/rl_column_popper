# Quickstart: Column Popper

This guide provides a quick way to get started with the Column Popper game and RL environment.

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/column_popper.git
cd column_popper

# Install the package
pip install -e .
```

## Playing the Game (Human Mode)

To play the game in your terminal (curses UI; falls back to ASCII when needed):

```bash
column_popper --mode=play --seed=42
```

## Training an RL Agent

To train a simple RL agent using the provided environment, you can use the following Python script:

```python
import gymnasium as gym
import column_popper

# Create the environment
env = gym.make("SpecKitAI/ColumnPopper-v1", seed=42)

# Reset the environment
obs, info = env.reset()

# Run the environment for 1000 steps
for _ in range(1000):
    # Take a random action
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

## Running a Headless Rollout

Run headless evaluation that streams JSONL observations/rewards to stdout. Actions are read from stdin (one integer per line, 0–3):

```bash
# example: random policy piped into rollout
python - << 'PY'
import random, sys
for _ in range(200):
    print(random.choice([0,1,2,3]))
PY
\
| column_popper --mode=rollout \
  --episodes=5 --format=jsonl --seed=42 --reward=preset:default
```

JSONL frames include observation fields, reward, terminated/truncated flags, and info (score, time_left, pops_this_step, fall_interval, seed, version).
