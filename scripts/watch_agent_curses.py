import argparse
import time

import gymnasium as gym
import column_popper.envs  # register env


def load_policy(model_path: str):
    try:
        from stable_baselines3 import PPO  # type: ignore

        model = PPO.load(model_path)

        def policy(obs):
            action, _ = model.predict(obs, deterministic=True)
            return int(action)

        return policy
    except Exception:
        import random

        print("Warning: Could not load PPO model; using random policy instead.")

        def policy(_obs):
            return random.choice([0, 1, 2, 3])

        return policy


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch a trained agent play in curses UI")
    parser.add_argument("--model", type=str, default="models/ppo_column_popper.zip")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    policy = load_policy(args.model)

    env = gym.make("SpecKitAI/ColumnPopper-v1", seed=args.seed, use_wall_time=False)

    def _loop(stdscr):
        from column_popper.render.curses_ui import _draw_board

        obs, info = env.reset()
        _draw_board(stdscr, obs, info)
        while True:
            action = policy(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            _draw_board(stdscr, obs, info)
            if terminated or truncated:
                break
            time.sleep(0.1)

    try:
        import curses

        curses.wrapper(_loop)
    finally:
        env.close()


if __name__ == "__main__":
    main()

