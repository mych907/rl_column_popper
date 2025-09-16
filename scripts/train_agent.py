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
    parser.add_argument("--epsilon-fall", type=float, default=0.05, help="With probability epsilon, override action to manual fall (3) to encourage exploration. Decays to 0 over training.")
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
        include_time_left_norm=True,
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

    # Optional: decay epsilon linearly to 0 over training
    callback = None
    if args.epsilon_fall > 0:
        try:
            from stable_baselines3.common.callbacks import BaseCallback  # type: ignore

            class EpsilonDecayCallback(BaseCallback):
                def __init__(self, wrapped_env: gym.Env, initial_eps: float, total_timesteps: int):
                    super().__init__()
                    self.wrapped_env = wrapped_env
                    self.initial_eps = float(initial_eps)
                    self.total_timesteps = max(1, int(total_timesteps))

                def _on_step(self) -> bool:
                    progress = min(1.0, self.num_timesteps / self.total_timesteps)
                    new_eps = self.initial_eps * (1.0 - progress)
                    if hasattr(self.wrapped_env, "eps"):
                        setattr(self.wrapped_env, "eps", float(new_eps))
                    return True

            callback = EpsilonDecayCallback(env, args.epsilon_fall, args.timesteps)
        except Exception:
            callback = None

    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        n_steps=2048,
        batch_size=512,
        learning_rate=3e-4,
        ent_coef=0.01,
        seed=args.seed,
    )
    model.learn(total_timesteps=args.timesteps, callback=callback)
    model.save(str(args.model_out))
    env.close()
    print(f"Saved model to {args.model_out}.zip")


if __name__ == "__main__":
    main()
