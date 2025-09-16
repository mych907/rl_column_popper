import argparse
import os
from pathlib import Path
from typing import Optional

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

    # Optional: set up a CSV logger for SB3 progress (progress.csv)
    # This records keys like train/entropy_loss and rollout/ep_rew_mean
    # alongside time/total_timesteps. We'll also write our own metrics CSV below.
    try:
        from stable_baselines3.common.logger import configure  # type: ignore

        log_dir = Path("models") / "logs"
        os.makedirs(log_dir, exist_ok=True)
        new_logger = configure(str(log_dir), ["stdout", "csv"])
    except Exception:
        new_logger = None  # type: ignore[assignment]

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
    # Attach logger if available so SB3 writes progress.csv
    if new_logger is not None:
        model.set_logger(new_logger)

    # Create a metrics writer that logs every 1000 timesteps
    # It records train/entropy_loss (from SB3 logger) and evaluation mean reward
    # computed via evaluate_policy on a separate eval env.
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList  # type: ignore
    from stable_baselines3.common.evaluation import evaluate_policy  # type: ignore
    import csv

    metrics_path = args.model_out.parent / f"{args.model_out.name}_metrics.csv"

    class MetricsCallback(BaseCallback):
        def __init__(self, eval_env: gym.Env, eval_freq: int = 1000, n_eval_episodes: int = 5):
            super().__init__()
            self.eval_freq = int(max(1, eval_freq))
            self.n_eval_episodes = int(max(1, n_eval_episodes))
            self._last_dump: int = 0
            self._csv_initialized = False
            self.eval_env = eval_env

        def _init_callback(self) -> None:
            # Ensure CSV header exists
            if not self._csv_initialized:
                with open(metrics_path, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["total_timesteps", "entropy_loss", "eval_mean_reward"])  # header
                self._csv_initialized = True

        def _on_step(self) -> bool:
            # Log every eval_freq timesteps
            if (self.num_timesteps - self._last_dump) >= self.eval_freq:
                self._last_dump = self.num_timesteps

                # Try to read the latest entropy loss from SB3 logger if present
                entropy_loss: Optional[float] = None
                try:
                    log_dict = self.model.logger.get_log_dict()  # type: ignore[attr-defined]
                    # train/entropy_loss is typically recorded by PPO.train()
                    if "train/entropy_loss" in log_dict:
                        val = log_dict["train/entropy_loss"]
                        if isinstance(val, (int, float)):
                            entropy_loss = float(val)
                except Exception:
                    entropy_loss = None

                # Compute evaluation mean reward deterministically over N episodes
                eval_mean_reward: Optional[float] = None
                try:
                    mean_r, _ = evaluate_policy(self.model, self.eval_env, n_eval_episodes=self.n_eval_episodes, deterministic=True, render=False)
                    eval_mean_reward = float(mean_r)
                except Exception:
                    eval_mean_reward = None

                # Append to CSV
                with open(metrics_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        int(self.num_timesteps),
                        ("" if entropy_loss is None else f"{entropy_loss:.6f}"),
                        ("" if eval_mean_reward is None else f"{eval_mean_reward:.6f}"),
                    ])
            return True

        def _on_training_end(self) -> None:
            # Optionally perform a final eval at the end
            try:
                mean_r, _ = evaluate_policy(self.model, self.eval_env, n_eval_episodes=self.n_eval_episodes, deterministic=True, render=False)
                with open(metrics_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([int(self.num_timesteps), "", f"{mean_r:.6f}"])
            except Exception:
                pass

    # Build a fresh eval env (deterministic) for evaluation snapshots
    eval_env = gym.make(
        "SpecKitAI/ColumnPopper-v1",
        seed=args.seed,
        include_time_left_norm=True,
        use_wall_time=False,
        initial_fall_interval=args.initial_fall,
        schedule_curve=parse_curve(args.fall_curve),
    )
    try:
        from stable_baselines3.common.monitor import Monitor  # type: ignore
        eval_env = Monitor(eval_env)
    except Exception:
        pass
    metrics_cb = MetricsCallback(eval_env, eval_freq=1000, n_eval_episodes=5)
    # Combine callbacks: epsilon decay (if any) and metrics collection
    if callback is not None:
        cb = CallbackList([callback, metrics_cb])
    else:
        cb = CallbackList([metrics_cb])
    model.learn(total_timesteps=args.timesteps, callback=cb)
    model.save(str(args.model_out))
    env.close()
    print(f"Saved model to {args.model_out}.zip")


if __name__ == "__main__":
    main()
