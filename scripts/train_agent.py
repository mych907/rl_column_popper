import argparse
import os
from pathlib import Path

import gymnasium as gym
import column_popper.envs  # register env


def parse_curve(s: str) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for seg in s.split(","):
        seg = seg.strip()
        if not seg:
            continue
        try:
            t_str, i_str = seg.split(":", 1)
            out.append((float(t_str), float(i_str)))
        except Exception:
            continue
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO on Column Popper")
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--initial-fall", type=float, default=3.0)
    parser.add_argument("--fall-curve", type=str, default="20:2,40:1")
    parser.add_argument("--epsilon-fall", type=float, default=0.0, help="With probability epsilon, override action to manual fall (3) to encourage exploration.")
    parser.add_argument(
        "--model-out",
        type=Path,
        default=Path("models/ppo_column_popper"),
        help="Output path (without .zip)",
    )
    args = parser.parse_args()

    try:
        from stable_baselines3 import PPO  # type: ignore
    except Exception as e:  # pragma: no cover
        print("stable-baselines3 not installed. Install with: pip install stable-baselines3[extra]")
        raise

    os.makedirs(args.model_out.parent, exist_ok=True)

    base_env = gym.make(
        "SpecKitAI/ColumnPopper-v1",
        seed=args.seed,
        use_wall_time=False,
        initial_fall_interval=args.initial_fall,
        schedule_curve=parse_curve(args.fall_curve),
    )

    # Optional exploration wrapper to ensure agent experiences manual fall
    if args.epsilon_fall > 0:
        import numpy as np

        class ManualFallEpsilonWrapper(gym.Wrapper):
            def __init__(self, env: gym.Env, eps: float):
                super().__init__(env)
                self.eps = float(eps)
                self.np_random = np.random.default_rng(args.seed)

            def step(self, action):  # type: ignore[override]
                if self.np_random.random() < self.eps:
                    action = 3
                return self.env.step(action)

        env = ManualFallEpsilonWrapper(base_env, args.epsilon_fall)
    else:
        env = base_env

    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        n_steps=2048,
        batch_size=512,
        learning_rate=3e-4,
        seed=args.seed,
    )
    model.learn(total_timesteps=args.timesteps)
    model.save(str(args.model_out))
    env.close()
    print(f"Saved model to {args.model_out}.zip")


if __name__ == "__main__":
    main()
