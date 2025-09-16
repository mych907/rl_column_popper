from __future__ import annotations

import argparse
import sys

import gymnasium as gym
import column_popper.envs  # noqa: F401 ensure registration


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Play Column Popper (headless stub)")
    parser.add_argument("--mode", choices=["play", "rollout", "stream"], default="play")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    env = gym.make("SpecKitAI/ColumnPopper-v1", disable_env_checker=True, seed=args.seed)
    try:
        obs, info = env.reset(seed=args.seed)
        # Simple stub loop: take random actions
        for _ in range(100):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())

